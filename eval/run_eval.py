r"""评测：量化"模型用工具用得对不对"。

契约测试（tests/）证明的是**工具写对了**，这个脚本证明的是**模型会用**。
两者必须分开：混在一起时，模型答错你无法判断是工具描述有歧义还是模型太笨。

## 五个指标

| 指标 | 定义 | 为什么需要它 |
|---|---|---|
| 工具选择正确率 | need 全部命中且 forbid 未命中 | 选错工具但答案对，是最难发现的坏 |
| 参数正确率 | conf 降到位、detail 是否 full | 工具对了参数错了，等于没对 |
| 错误恢复率 | 注入错误后仍给出有效答案 | 检验 error.hint 有没有被真读进去 |
| 幻觉率 | 回答中的数字无法在任何观测中找到 | 唯一能自动发现编造的手段 |
| 平均步数 | 完成任务所需工具调用数 | 反映请求效率，不是越低越好 |

## 数字溯源怎么判

回答里出现 `\d+` 就要求它能被某个观测里的数字解释。三个放宽条件，
都是为了不冤枉合理的推理：

1. 0~3 的整数豁免（"两张图""差 2 株"这类结构用法）
2. 允许派生值：任意两个观测数字的差、比值的百分数（"差 +1""降了 2%"）
3. 允许 0.005 的绝对误差与 0.5% 的相对误差（浮点格式化）

剩下的就是编造。`G1`（断言没得病）和 `G2`（给出每亩株数）专门测这种。

## 用法

    python eval/run_eval.py --provider mock            # 自检管线，不需要 key
    python eval/run_eval.py --provider dashscope --only recover
    CVAGENT_EVAL_DATA=/path/to/test/images python eval/run_eval.py --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PLACEHOLDER_RE = re.compile(r"\{([a-z_0-9]+)\}")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXEMPT_INTS = {0.0, 1.0, 2.0, 3.0}


# ─────────────────────────── 图片占位符 ───────────────────────────
def build_fixtures(data_root: Path, tmp: Path) -> tuple[dict[str, str], set[str]]:
    """占位符 → 真实路径。缺数据的角色记入 missing，相关任务标 SKIP。

    错误恢复类任务的图是**合成**的，所以这部分在任何机器上都能跑；
    只有单图/批量类任务需要真实数据集。
    """
    fx: dict[str, str] = {}
    missing: set[str] = set()

    imgs: list[Path] = []
    if data_root and data_root.is_dir():
        imgs = sorted(p for p in data_root.iterdir() if p.suffix.lower() in IMG_EXTS)

    if imgs:
        fx["single"] = str(imgs[0])
    else:
        missing.add("single")
    if len(imgs) >= 2:
        fx["single2"] = str(imgs[1])
    else:
        missing.add("single2")
    if data_root and data_root.is_dir() and imgs:
        fx["dir"] = str(data_root)
    else:
        missing.add("dir")

    from PIL import Image

    tiny = tmp / "tiny.png"
    Image.new("RGB", (8, 8), (128, 128, 128)).save(tiny)
    fx["tiny"] = str(tiny)

    dark = tmp / "dark.jpg"
    Image.new("RGB", (1024, 1024), (2, 2, 2)).save(dark, quality=90)
    fx["dark"] = str(dark)

    corrupt = tmp / "fake.jpg"
    corrupt.write_bytes(b"not an image, only the extension lies\n")
    fx["corrupt"] = str(corrupt)

    fx["missing"] = str(tmp / "no_such_image.jpg")  # 故意不创建
    fx["no_dir"] = str(tmp / "no_such_dir")         # 故意不创建
    return fx, missing


# ─────────────────────────── 数字溯源 ───────────────────────────
def _walk_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_numbers(v, out)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            try:
                out.append(float(m))
            except ValueError:
                pass


def _derived(vals: set[float]) -> set[float]:
    """观测数字的差与百分比。允许"差 +1""降了 2%"这类合理推算。"""
    base = list(vals)
    if len(base) > 120:
        return set()
    out: set[float] = set()
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            out.add(abs(a - b))
            if b:
                out.add(round(a / b * 100, 1))
                out.add(float(round(a / b * 100)))
            if a:
                out.add(round(b / a * 100, 1))
                out.add(float(round(b / a * 100)))
    return out


def _explained(x: float, pool: set[float]) -> bool:
    if x in pool:
        return True
    for p in pool:
        if abs(x - p) <= 0.005:
            return True
        if p and abs(x - p) / abs(p) <= 0.005:
            return True
    return False


def find_unsourced(answer: str, observations: list[dict]) -> list[float]:
    """回答里的数字，减去所有观测（含派生）能解释的部分，剩下的就是编造。"""
    if not answer:
        return []
    nums = [float(m) for m in NUM_RE.findall(answer)]
    if not nums:
        return []

    pool: set[float] = set()
    for obs in observations:
        buf: list[float] = []
        _walk_numbers(obs, buf)
        pool.update(round(v, 2) for v in buf)
    pool |= _derived(pool)

    bad: list[float] = []
    for x in nums:
        if x in EXEMPT_INTS:
            continue
        if not _explained(x, pool):
            bad.append(x)
    return bad


# ─────────────────────────── 判定 ───────────────────────────
def _strings(obj, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)


def _norm(p: str) -> str:
    return str(p).replace("\\", "/").lower()


def _check_paths(names: list[str], steps, fx: dict[str, str]) -> tuple[bool, list[str]]:
    """多图任务必须真的碰过每一张。

    这不是多余检查：C3 问"两张图哪张更密"，模型只看一张就作答，
    前面的 need/calls 检查全都会通过——只有这里能抓住。
    """
    seen: list[str] = []
    for s in steps:
        _strings(s.args, seen)
    seen_n = [_norm(x) for x in seen]
    bad = []
    for name in names:
        p = fx.get(name)
        if not p:
            continue
        if _norm(p) not in seen_n:
            bad.append(f"没处理 {name}")
    return (not bad), bad


def _check_args(checks: list[dict], steps) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for chk in checks:
        name = chk.get("tool")
        st = next((s for s in steps if s.tool == name and s.ok), None)
        if st is None:
            bad.append(f"{name} 未成功调用")
            continue
        for key, want in chk.items():
            if key == "tool":
                continue
            if key.endswith("_max"):
                real = key[:-4]
                got = st.args.get(real)
                if not isinstance(got, (int, float)) or got > float(want) + 1e-9:
                    bad.append(f"{real}={got} 应 <= {want}")
            elif key.endswith("_min"):
                real = key[:-4]
                got = st.args.get(real)
                if not isinstance(got, (int, float)) or got < float(want) - 1e-9:
                    bad.append(f"{real}={got} 应 >= {want}")
            else:
                got = st.args.get(key)
                if got != want:
                    bad.append(f"{key}={got!r} 应 == {want!r}")
    return (not bad), bad


def judge(task: dict, traj, fx: dict[str, str]) -> dict:
    called = [s.tool for s in traj.steps]
    need = task.get("need") or []
    forbid = task.get("forbid") or []

    tool_ok = all(any(t in called for t in grp) for grp in need) and not any(t in called for t in forbid)
    n = len(called)
    calls_ok = task.get("min_calls", 0) <= n <= task.get("max_calls", 99)
    param_ok, param_bad = _check_args(task.get("arg_checks") or [], traj.steps)
    path_ok, path_bad = _check_paths(task.get("paths") or [], traj.steps, fx)

    ans = (traj.answer or "").strip()
    spec = task.get("answer") or {}
    missing_pat = [p for p in spec.get("match", []) if not re.search(p, ans)]
    hit_pat = [p for p in spec.get("not_match", []) if re.search(p, ans)]
    unsourced = find_unsourced(ans, [s.observation for s in traj.steps])
    answer_ok = not missing_pat and not hit_pat and not unsourced

    recover_ok = True
    if task.get("expect_error"):
        saw_error = any(not s.ok for s in traj.steps)
        recover_ok = saw_error and bool(ans) and not unsourced

    reasons: list[str] = []
    if not tool_ok:
        miss = ["/".join(grp) for grp in need if not any(t in called for t in grp)]
        if miss:
            reasons.append(f"没调 {'、'.join(miss)}")
        forb = [t for t in forbid if t in called]
        if forb:
            reasons.append(f"不该调 {'、'.join(forb)}")
    if not calls_ok:
        reasons.append(f"调用数 {n} 超出 [{task.get('min_calls', 0)},{task.get('max_calls', 99)}]")
    if not param_ok:
        reasons.extend(param_bad)
    if not path_ok:
        reasons.extend(path_bad)
    if missing_pat:
        reasons.append(f"回答缺关键词 {missing_pat}")
    if hit_pat:
        reasons.append(f"回答命中禁语 {hit_pat}")
    if unsourced:
        reasons.append(f"编造数字 {unsourced[:4]}")
    if task.get("expect_error") and not recover_ok:
        reasons.append("错误未被处理（没出现失败步骤或其后没有有效结论）")

    return {
        "tool_ok": tool_ok,
        "calls_ok": calls_ok,
        "param_ok": param_ok,
        "path_ok": path_ok,
        "answer_ok": answer_ok,
        "recover_ok": recover_ok,
        "passed": tool_ok and calls_ok and param_ok and path_ok and answer_ok and recover_ok,
        "called": called,
        "n_calls": n,
        "answer": ans,
        "unsourced": unsourced,
        "n_answer_numbers": len(NUM_RE.findall(ans)),
        "stopped_reason": traj.stopped_reason,
        "elapsed_s": round(traj.elapsed_s, 2),
        "reasons": reasons,
    }


# ─────────────────────────── 汇总 ───────────────────────────
def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if r.get(key)) / len(rows)


def aggregate(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    num_rows = [r for r in ok_rows if r.get("n_answer_numbers", 0) > 0]
    rec_rows = [r for r in ok_rows if r.get("expect_error")]
    par_rows = [r for r in ok_rows if r.get("has_arg_checks")]

    cats: dict[str, list[dict]] = {}
    for r in ok_rows:
        cats.setdefault(r["cat"], []).append(r)

    return {
        "n_tasks": len(rows),
        "n_ok": len(ok_rows),
        "n_skip": len([r for r in rows if r.get("status") == "skip"]),
        "overall_pass": _rate(ok_rows, "passed"),
        "tool_selection": _rate(ok_rows, "tool_ok"),
        "call_budget": _rate(ok_rows, "calls_ok"),
        "param_accuracy": _rate(par_rows, "param_ok"),
        "answer_ok": _rate(ok_rows, "answer_ok"),
        "error_recovery": _rate(rec_rows, "recover_ok"),
        "hallucination_rate": (None if not num_rows else
                               sum(1 for r in num_rows if r["unsourced"]) / len(num_rows)),
        "n_hallucinated_tasks": sum(1 for r in num_rows if r["unsourced"]),
        "avg_steps": statistics.mean([r["n_calls"] for r in ok_rows]) if ok_rows else 0.0,
        "by_category": {
            c: {
                "n": len(rs),
                "pass": _rate(rs, "passed"),
                "tool": _rate(rs, "tool_ok"),
                "avg_steps": statistics.mean([r["n_calls"] for r in rs]),
            }
            for c, rs in sorted(cats.items())
        },
    }


def render_md(meta: dict, agg: dict, rows: list[dict]) -> str:
    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    L: list[str] = []
    L.append(f"# 评测结果：{meta['provider']}")
    L.append("")
    L.append(f"- 时间：{meta['time']}")
    L.append(f"- 模型：`{meta['model']}`")
    L.append(f"- 数据根目录：`{meta['data_root']}`")
    L.append(f"- 任务：{agg['n_tasks']} 条，执行 {agg['n_ok']}，跳过 {agg['n_skip']}")
    L.append("")
    if meta["provider"] == "mock":
        L.append("> **这份分数不代表模型能力。** `mock` 是按预设脚本行动的替身")
        L.append("> （检测→看 n_low→复核→收束），不读问题。用它跑是为了验证")
        L.append("> 评测管线本身正确：如果 mock 在所有任务上都通过，说明判定太松。")
        L.append("")

    L.append("## 总览")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| 任务通过率 | {pct(agg['overall_pass'])} | 五层判定全过 |")
    L.append(f"| 工具选择正确率 | {pct(agg['tool_selection'])} | need 命中且 forbid 未命中 |")
    L.append(f"| 参数正确率 | {pct(agg['param_accuracy'])} | 仅统计有参数期望的任务 |")
    L.append(f"| 调用数合规 | {pct(agg['call_budget'])} | 未超出步数预算 |")
    L.append(f"| 回答合规 | {pct(agg['answer_ok'])} | 正则约束 + 数字可溯源 |")
    L.append(f"| 错误恢复率 | {pct(agg['error_recovery'])} | 仅统计注入错误的任务 |")
    L.append(f"| **幻觉率** | {pct(agg['hallucination_rate'])} | 有数字的回答中，含无法溯源数字的比例"
             f"（{agg['n_hallucinated_tasks']} 条）|")
    L.append(f"| 平均步数 | {agg['avg_steps']:.2f} | 工具调用数均值 |")
    L.append("")

    L.append("## 分类")
    L.append("")
    L.append("| 分类 | 任务数 | 通过 | 工具选择 | 平均步数 |")
    L.append("|---|---|---|---|---|")
    for cat, s in agg["by_category"].items():
        L.append(f"| {cat} | {s['n']} | {pct(s['pass'])} | {pct(s['tool'])} | {s['avg_steps']:.2f} |")
    L.append("")

    fails = [r for r in rows if r.get("status") == "ok" and not r["passed"]]
    if fails:
        L.append(f"## 失败明细（{len(fails)} 条）")
        L.append("")
        L.append("| id | 分类 | 实际调用 | 卡在哪 |")
        L.append("|---|---|---|---|")
        for r in fails:
            call = " → ".join(r["called"]) or "（未调用）"
            L.append(f"| {r['id']} | {r['cat']} | `{call}` | {'；'.join(r['reasons'])} |")
        L.append("")

    skips = [r for r in rows if r.get("status") == "skip"]
    if skips:
        L.append(f"## 跳过（{len(skips)} 条）")
        L.append("")
        for r in skips:
            L.append(f"- {r['id']}：{r['reason']}")
        L.append("")
    return "\n".join(L)


# ─────────────────────────── 主流程 ───────────────────────────
def default_data_root() -> Path:
    env = os.getenv("CVAGENT_EVAL_DATA")
    if env:
        return Path(env)
    return ROOT.parent / "cabbage_project" / "dataset" / "test" / "images"


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 工具使用评测")
    ap.add_argument("--provider", default="mock", help="mock / openai / dashscope / deepseek / moonshot")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data-root", default=None, help="测试图目录，默认 ../cabbage_project/dataset/test/images")
    ap.add_argument("--tasks", default=str(ROOT / "eval" / "tasks.yaml"))
    ap.add_argument("--only", default=None, help="只跑某个分类：basic/ambiguous/verify/batch/quality/recover/refuse")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条，用于冒烟")
    ap.add_argument("--device", default=None, help="cpu / cuda:0")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out-dir", default=str(ROOT / "eval_reports"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import yaml

    spec = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["cat"] == args.only]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("没有匹配的任务", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root else default_data_root()

    try:
        from cvagent.agent.loop import Agent
        from cvagent.llm.registry import build_provider
        from cvagent.tools.registry import default_registry
        from cvagent.vision.engine import YoloEngine
    except ImportError as e:
        print(f"导入失败：{e}\n先执行 pip install -e .", file=sys.stderr)
        return 2

    try:
        provider = build_provider(args.provider, args.model)
    except Exception as e:  # noqa: BLE001 - 缺 key 时给清楚的提示，不要堆栈
        print(f"provider 构建失败：{e}", file=sys.stderr)
        return 2

    engine = YoloEngine(weights=None, device=args.device)
    try:
        ms = engine.warmup()
        if not args.quiet:
            print(f"权重预热 {ms / 1000:.1f}s（不计入任务耗时）")
    except Exception as e:  # noqa: BLE001
        print(f"权重加载失败：{e}\n先执行 cvagent setup-weights --from <best.pt>", file=sys.stderr)
        return 2

    registry = default_registry(engine)

    tmp = Path(tempfile.mkdtemp(prefix="cvagent_eval_"))
    fx, missing = build_fixtures(data_root, tmp)
    if missing and not args.quiet:
        print(f"缺真实图（{sorted(missing)}），相关任务将跳过；数据根目录：{data_root}")

    rows: list[dict] = []
    for i, task in enumerate(tasks, 1):
        q = task["q"]
        holes = set(PLACEHOLDER_RE.findall(q))
        if any(h not in fx for h in holes):
            rows.append({"id": task["id"], "cat": task["cat"], "status": "skip",
                         "reason": f"缺数据 {sorted(h for h in holes if h not in fx)}"})
            if not args.quiet:
                print(f"[{i}/{len(tasks)}] {task['id']} SKIP")
            continue
        for h in holes:
            q = q.replace("{" + h + "}", fx[h])

        try:
            # 每条任务都重建 provider：mock 的 stage 是有状态的一次性替身，
            # 复用会让第二轮起全部空转而误判为"模型不调工具"。
            # 真实 provider 无状态，重建的代价只是一次对象构造。
            engine.clear_cache()  # 上一条任务的阈值复核不能泄漏到下一条
            traj = Agent(
                build_provider(args.provider, args.model),
                registry,
                max_steps=args.max_steps,
            ).run(q)
            res = judge(task, traj, fx)
            res.update({"id": task["id"], "cat": task["cat"], "status": "ok",
                        "question": q, "note": task.get("note", ""),
                        "expect_error": bool(task.get("expect_error")),
                        "has_arg_checks": bool(task.get("arg_checks"))})
        except Exception as e:  # noqa: BLE001 - 单条任务崩了不该中断整轮
            res = {"id": task["id"], "cat": task["cat"], "status": "ok", "passed": False,
                   "tool_ok": False, "calls_ok": False, "param_ok": False, "path_ok": False,
                   "answer_ok": False, "recover_ok": False, "called": [], "n_calls": 0,
                   "answer": "", "unsourced": [], "n_answer_numbers": 0,
                   "stopped_reason": "exception", "elapsed_s": 0.0,
                   "reasons": [f"运行异常：{type(e).__name__}: {e}"],
                   "expect_error": bool(task.get("expect_error")),
                   "has_arg_checks": bool(task.get("arg_checks")), "question": q}
        rows.append(res)
        if not args.quiet:
            mark = "PASS" if res.get("passed") else "FAIL"
            call = " → ".join(res.get("called") or []) or "-"
            why = "" if res.get("passed") else "  " + "；".join(res.get("reasons", []))
            print(f"[{i}/{len(tasks)}] {res['id']} {mark}  {call}{why}")

    agg = aggregate(rows)
    meta = {
        "provider": args.provider,
        "model": getattr(provider, "model", ""),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "max_steps": args.max_steps,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.provider}" + (f"_{args.only}" if args.only else "")
    (out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "aggregate": agg, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / f"{stem}.md").write_text(render_md(meta, agg, rows), encoding="utf-8")

    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    print()
    print(f"通过 {pct(agg['overall_pass'])}  工具选择 {pct(agg['tool_selection'])}  "
          f"参数 {pct(agg['param_accuracy'])}  错误恢复 {pct(agg['error_recovery'])}  "
          f"幻觉 {pct(agg['hallucination_rate'])}  平均步数 {agg['avg_steps']:.2f}")
    print(f"报告：{out / (stem + '.md')}")

    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
