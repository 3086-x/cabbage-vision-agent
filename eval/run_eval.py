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
    python eval/run_eval.py --provider pipeline        # 确定性管线（0 次 LLM），用于对照
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

try:  # 供应商侧故障（额度、网络）与"模型不会用工具"要分开处理
    from cvagent.llm.base import ProviderError as _InfraError
except ImportError:  # pragma: no cover - 未安装时仍能给出友好提示
    class _InfraError(Exception):
        code = None


_DEFAULTS_CACHE: str | None = None


def _known_defaults_text() -> str:
    """系统提示与工具 schema 里出现的数字，属于文档化默认值。

    模型说"conf=0.25 是默认值"时，0.25 不在任何一次观测里，但它写在我们
    自己的工具描述里——复述文档不算编造。把这段文本并入溯源来源，
    比在词表里逐个堆默认值可靠。
    """
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        try:
            from cvagent.agent.loop import SYSTEM_PROMPT
            from cvagent.tools.spec import build_tools

            schemas = json.dumps([t.parameters for t in build_tools(None)], ensure_ascii=False)
            _DEFAULTS_CACHE = SYSTEM_PROMPT + "\n" + schemas
        except Exception:  # noqa: BLE001 - 取不到就不放宽，宁可严一点
            _DEFAULTS_CACHE = ""
    return _DEFAULTS_CACHE


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
    # 上限从 120 提到 600：detail=full 时一张图就有 49 框 x 5 个数值，
    # 原来的上限会让整张图的派生值全部失效，导致合法推算被误判。
    if len(base) > 600:
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
    """x 能否被观测解释。

    EPS 不是随手加的：0.805 这类值做 round(v, 2) 会变成 0.81，
    而 |0.805-0.81| 在浮点下等于 0.0050000000000000044，刚好越过 0.005 边界。
    这是"先改坏数字再判它对不上"的典型误判，所以扣掉浮点噪声再比较。

    另外允许 ×/÷10、100、1000：把覆盖率 0.448 说成 44.8%、把 0.87 说成 87%
    是单位换算，不是编造。真实幻觉（观测是 49 却说 52）离任何缩放值都很远，
    仍然会被抓住。
    """
    EPS = 1e-9
    candidates = [x, x * 10, x / 10, x * 100, x / 100, x * 1000, x / 1000]
    for c in candidates:
        for p in pool:
            if abs(c - p) <= 0.005 + EPS:
                return True
            if p and abs(c - p) / abs(p) <= 0.005 + EPS:
                return True
    return False


def find_unsourced(answer: str, observations: list[dict], question: str = "") -> list[float]:
    """回答里的数字，减去所有观测（含派生）能解释的部分，剩下的就是无出处。

    question 也算来源：用户问"用 conf=1.5 检测"，回答复述 1.5、0.05~1.0
    这些边界值是完全正常的，只比对观测会把它判成编造。
    """
    if not answer:
        return []
    nums = [float(m) for m in NUM_RE.findall(answer)]
    if not nums:
        return []

    pool: set[float] = set()
    for obs in observations:
        buf: list[float] = []
        _walk_numbers(obs, buf)
        # 只洗掉浮点噪声，不能降到 2 位小数——那等于先改坏观测再判对错
        pool.update(round(v, 6) for v in buf)
    if question:
        pool.update(float(m) for m in NUM_RE.findall(question))
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


def _arg_violations(st, chk: dict) -> list[str]:
    bad: list[str] = []
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
    return bad


def _check_args(checks: list[dict], steps) -> tuple[bool, list[str]]:
    """参数校验。tool 写成 "*" 表示"任意工具的一次调用满足即可"。

    需要 "*" 的理由是 C1/C2：用户要求"把阈值降到 0.15 再数一遍"，
    用 refine_detection 或再调一次 count_cabbage 都能做到。写死工具名
    会把正确答案判错，而这里真正想约束的是"conf 有没有真的降下去"。
    """
    bad: list[str] = []
    for chk in checks:
        name = chk.get("tool")
        if name == "*":
            if not any(not _arg_violations(s, chk) for s in steps if s.ok):
                bad.append(f"没有任何调用满足 {dict(chk)}")
            continue
        st = next((s for s in steps if s.tool == name and s.ok), None)
        if st is None:
            bad.append(f"{name} 未成功调用")
            continue
        bad.extend(_arg_violations(st, chk))
    return (not bad), bad


def judge(task: dict, traj, fx: dict[str, str]) -> dict:
    called = [s.tool for s in traj.steps]
    need = task.get("need") or []
    forbid = task.get("forbid") or []

    tool_ok = all(any(t in called for t in grp) for grp in need) and not any(t in called for t in forbid)
    n_calls = len(called)
    n_turns = getattr(traj, "n_turns", n_calls)
    # 两个方向用两个口径，刻意不对称：
    #   min_calls 是"覆盖够不够"——两张图要调两次，并行在一轮里也是两次，按调用数
    #   max_calls 是"浪不浪费"——按轮次，一轮并行多个工具不该被罚
    # 统一用轮次会让 C2（模型一轮并行跑两个阈值）被误判成没做够。
    calls_ok = (n_calls >= task.get("min_calls", 0)
                and n_turns <= task.get("max_calls", 99))
    param_ok, param_bad = _check_args(task.get("arg_checks") or [], traj.steps)
    path_ok, path_bad = _check_paths(task.get("paths") or [], traj.steps, fx)

    ans = (traj.answer or "").strip()
    # 明确标注了是假设/示意的回答，不按禁语和无出处判——模型给了 GSD 代入的
    # 量级示意并写明"不是检测结果"，这比拒绝回答更有用，不该罚。
    # 词表判不准（模型这次写的是"仅为公式演示"，词表里没有"演示"），
    # 所以主判据用正则，词表只作补充。
    exempt = (any(t in ans for t in (task.get("exempt_if") or []))
              or any(re.search(p, ans) for p in (task.get("exempt_if_re") or [])))
    spec = task.get("answer") or {}
    missing_pat = [p for p in spec.get("match", []) if not re.search(p, ans)]
    hit_pat = [] if exempt else [p for p in spec.get("not_match", []) if re.search(p, ans)]
    unsourced = [] if exempt else find_unsourced(
        # 来源有两类：
        #   traj.question —— 不能用 task["q"]，后者还带着 {single} 占位符，
        #     里面没有真实路径，回答引用 img_106 会被判成无出处
        #   系统提示与工具 schema —— 里面写的 conf=0.25 这类默认值是文档化的，
        #     模型复述它不算编造
        ans, [s.observation for s in traj.steps],
        question=traj.question + "\n" + _known_defaults_text())
    answer_ok = not missing_pat and not hit_pat and not unsourced

    # 错误恢复：只要最后给了有效回答就算恢复。
    # 原先还要求"必须出现过失败步骤"，这惩罚了更好的行为——模型先用
    # detect 探到合法缓存再 refine（F3），或主动把越界 conf 收敛到上限（F2），
    # 根本没触发错误，却被判成"没处理错误"。
    saw_error = any(not s.ok for s in traj.steps)
    recover_ok = bool(ans) and not unsourced if task.get("expect_error") else True

    reasons: list[str] = []
    if not tool_ok:
        miss = ["/".join(grp) for grp in need if not any(t in called for t in grp)]
        if miss:
            reasons.append(f"没调 {'、'.join(miss)}")
        forb = [t for t in forbid if t in called]
        if forb:
            reasons.append(f"不该调 {'、'.join(forb)}")
    if not calls_ok:
        reasons.append(
            f"轮次 {n_turns} 超出 [{task.get('min_calls', 0)},{task.get('max_calls', 99)}]"
            f"（共 {n_calls} 次调用）")
    if not param_ok:
        reasons.extend(param_bad)
    if not path_ok:
        reasons.extend(path_bad)
    if missing_pat:
        reasons.append(f"回答缺关键词 {missing_pat}")
    if hit_pat:
        reasons.append(f"回答命中禁语 {hit_pat}")
    if unsourced:
        # 措辞是"无出处"而不是"编造"：其中一部分是模型自己算出来的派生值，
        # 计算过程可能合理但不可核验，与凭空捏造不是一回事。
        reasons.append(f"数字无出处 {unsourced[:4]}")
    if task.get("expect_error") and not recover_ok:
        reasons.append("没有给出有效结论")

    return {
        "tool_ok": tool_ok,
        "calls_ok": calls_ok,
        "param_ok": param_ok,
        "path_ok": path_ok,
        "answer_ok": answer_ok,
        "recover_ok": recover_ok,
        "passed": tool_ok and calls_ok and param_ok and path_ok
                  and answer_ok and recover_ok,
        "called": called,
        "n_calls": n_calls,
        "n_turns": n_turns,
        "saw_error": saw_error,
        "exempt": exempt,
        "answer": ans,
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
        "n_error": len([r for r in rows if r.get("status") == "error"]),
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
        "avg_turns": statistics.mean([r.get("n_turns", r["n_calls"]) for r in ok_rows]) if ok_rows else 0.0,
        "n_error_recovery_exercised": sum(1 for r in rec_rows if r.get("saw_error")),
        # 分层路由的分布。只有 auto 模式有区分度（pipeline 恒为 rule、agent 恒为 agent），
        # 但照算不误，渲染时按 provider 决定要不要展示。
        "n_rule": sum(1 for r in ok_rows if r.get("route") == "rule"),
        "n_agent": sum(1 for r in ok_rows if r.get("route") == "agent"),
        "escalation_rate": (None if not ok_rows else
                            sum(1 for r in ok_rows if r.get("route") == "agent") / len(ok_rows)),
        "pass_rule": _rate([r for r in ok_rows if r.get("route") == "rule"], "passed"),
        "pass_agent": _rate([r for r in ok_rows if r.get("route") == "agent"], "passed"),
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
    L.append(f"- 任务：{agg['n_tasks']} 条，计入 {agg['n_ok']}，跳过 {agg['n_skip']}，"
             f"环境故障 {agg['n_error']}")
    L.append("")
    if meta["provider"] == "mock":
        L.append("> **这份分数不代表模型能力。** `mock` 是按预设脚本行动的替身")
        L.append("> （检测→看 n_low→复核→收束），不读问题。用它跑是为了验证")
        L.append("> 评测管线本身正确：如果 mock 在所有任务上都通过，说明判定太松。")
        L.append("")
    if meta["provider"] == "pipeline":
        L.append("> **这份分数不代表泛化能力。** `pipeline` 是零 LLM 的确定性路径：")
        L.append("> 正则路由意图 → 固定动作序列 → 模板拼接答案，全过程 0 次模型调用。")
        L.append("> 规则是看着 `tasks.yaml` 的措辞写的，所以在这批任务上的分数是**上界**；")
        L.append("> 衡量泛化要看 `tasks_holdout.yaml`（换措辞、同一批意图，写规则时没参考过）。")
        L.append("")
    if meta["provider"] == "auto":
        L.append("> **这是线上那条路径。** `auto` = 规则优先，判断不了才升级到 Agent。")
        L.append("> 与单独跑 `pipeline` / 单独跑模型的区别在于：判据、工具、交接文本完全一致，")
        L.append("> 所以分数可以和那两份直接对照，差异只来自「这条请求走了哪条路」。")
        L.append("")

    if meta["provider"] == "auto":
        L.append("## 分层路由")
        L.append("")
        L.append("| 路径 | 条数 | 占比 | 通过率 | 延迟 |")
        L.append("|---|---|---|---|---|")
        for key, label in (("rule", "规则（0 次 LLM）"), ("agent", "升级到 Agent")):
            n = agg["n_rule"] if key == "rule" else agg["n_agent"]
            rate = agg["n_rule"] / agg["n_ok"] if key == "rule" and agg["n_ok"] else \
                   (agg["n_agent"] / agg["n_ok"] if agg["n_ok"] else None)
            rows_k = [r for r in rows if r.get("status") == "ok" and r.get("route") == key]
            lat = statistics.median([r["elapsed_s"] for r in rows_k]) if rows_k else 0.0
            L.append(f"| {label} | {n} | {pct(rate)} | "
                     f"{pct(agg['pass_' + key])} | {lat:.2f}s |")
        L.append("")
        L.append(f"升级率 **{pct(agg['escalation_rate'])}**。这个数不是缺陷指标：")
        L.append("规则只接它读得懂的措辞，读不懂就交给循环——升级率高说明任务集的措辞")
        L.append("本身超出规则覆盖，低则说明规则在泛化。两边都不该被优化成 0 或 100%。")
        L.append("")

    L.append("## 总览")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| 任务通过率 | {pct(agg['overall_pass'])} | 五层判定全过 |")
    L.append(f"| 工具选择正确率 | {pct(agg['tool_selection'])} | need 命中且 forbid 未命中 |")
    L.append(f"| 参数正确率 | {pct(agg['param_accuracy'])} | 仅统计有参数期望的任务 |")
    L.append(f"| 调用数合规 | {pct(agg['call_budget'])} | 未超出轮次预算（并行调用算一轮） |")
    L.append(f"| 回答合规 | {pct(agg['answer_ok'])} | 正则约束 + 数字可溯源 |")
    L.append(f"| 错误恢复率 | {pct(agg['error_recovery'])} | 仅统计注入错误的任务；"
             f"{agg['n_error_recovery_exercised']} 条真的触发了错误 |")
    L.append(f"| **幻觉率** | {pct(agg['hallucination_rate'])} | 有数字的回答中，含无法溯源数字的比例"
             f"（{agg['n_hallucinated_tasks']} 条）|")
    L.append(f"| 平均步数 | {agg['avg_steps']:.2f} | 工具调用数均值 |")
    L.append(f"| 平均轮次 | {agg['avg_turns']:.2f} | 决策轮数均值（并行调用算一轮） |")
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
        from cvagent.pipeline import Pipeline, escalate_reason, handoff
        from cvagent.tools.registry import default_registry
        from cvagent.vision.engine import YoloEngine
    except ImportError as e:
        print(f"导入失败：{e}\n先执行 pip install -e .", file=sys.stderr)
        return 2

    is_pipeline = args.provider == "pipeline"
    is_auto = args.provider == "auto"
    if is_pipeline and args.model:
        print("pipeline 不走模型，--model 被忽略", file=sys.stderr)
    try:
        provider = None if is_pipeline else build_provider(args.provider, args.model)
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
            if is_pipeline:
                # 同一个 registry、同一个 dispatch、同一条判据，只换决策方式
                traj = Pipeline(registry).run(q)
                route = "rule"
            elif is_auto:
                # 线上那条路径：先规则，判断不了才升级。交接文本与 CLI 同一份，
                # 否则这里测的就不是部署出去的那个系统。
                pre = Pipeline(registry).run(q)
                reason = escalate_reason(pre)
                if reason is None:
                    traj, route = pre, "rule"
                else:
                    traj = Agent(
                        build_provider(args.provider, args.model),
                        registry,
                        max_steps=args.max_steps,
                    ).run(q, extra_context=handoff(pre))
                    route = "agent"
            else:
                traj = Agent(
                    build_provider(args.provider, args.model),
                    registry,
                    max_steps=args.max_steps,
                ).run(q)
                route = "agent"
            res = judge(task, traj, fx)
            res.update({"id": task["id"], "cat": task["cat"], "status": "ok",
                        "question": q, "note": task.get("note", ""),
                        "route": route,
                        "expect_error": bool(task.get("expect_error")),
                        "has_arg_checks": bool(task.get("arg_checks"))})
        except _InfraError as e:
            # 额度耗尽、网络超时这类是环境问题，不是模型不会用工具。
            # 算成失败会把报告污染成"模型很差"，而真正的原因在账户余额。
            res = {"id": task["id"], "cat": task["cat"], "status": "error", "passed": False,
                   "called": [], "n_calls": 0, "n_turns": 0, "saw_error": False, "exempt": False,
                   "answer": "", "unsourced": [], "n_answer_numbers": 0,
                   "stopped_reason": "infra", "elapsed_s": 0.0,
                   "reasons": [f"环境故障，不计入分母：{getattr(e, 'code', None) or e}"],
                   "expect_error": bool(task.get("expect_error")),
                   "has_arg_checks": bool(task.get("arg_checks")), "question": q}
        except Exception as e:  # noqa: BLE001 - 单条任务崩了不该中断整轮
            res = {"id": task["id"], "cat": task["cat"], "status": "ok", "passed": False,
                   "tool_ok": False, "calls_ok": False, "param_ok": False, "path_ok": False,
                   "answer_ok": False, "recover_ok": False,
                   "called": [], "n_calls": 0, "n_turns": 0, "saw_error": False, "exempt": False,
                   "answer": "", "unsourced": [], "n_answer_numbers": 0,
                   "stopped_reason": "exception", "elapsed_s": 0.0,
                   "reasons": [f"运行异常：{type(e).__name__}: {e}"],
                   "expect_error": bool(task.get("expect_error")),
                   "has_arg_checks": bool(task.get("arg_checks")), "question": q}
        rows.append(res)
        if not args.quiet:
            mark = ("PASS" if res.get("passed")
                    else "ERR " if res.get("status") == "error" else "FAIL")
            call = " → ".join(res.get("called") or []) or "-"
            why = "" if res.get("passed") else "  " + "；".join(res.get("reasons", []))
            print(f"[{i}/{len(tasks)}] {res['id']} {mark}  {call}{why}")

    agg = aggregate(rows)
    meta = {
        "provider": args.provider,
        "model": "pipeline/rules" if is_pipeline else getattr(provider, "model", ""),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "max_steps": args.max_steps,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.provider}" + (f"_{args.only}" if args.only else "")
    # 换任务集时要分开存，否则留出集的报告会覆盖主任务集的报告，
    # 两份数字混在一个文件里，事后没法说清哪个是哪个。
    tstem = Path(args.tasks).stem
    if tstem not in ("tasks",):
        stem += f"_{tstem}"
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
