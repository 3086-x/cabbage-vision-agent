"""VLM 直接计数基线：回答「为什么不直接让多模态大模型数」。

这是本项目最容易被问倒的一问，而在此之前没有任何数据。这个脚本把它变成一张表：

  同一批图，三个计数来源 ——
    GT            标注框数（`dataset/*/labels/*.txt` 行数）
    detector      专用检测器（yolo 权重，conf 阈值可调）
    VLM-direct    把整图丢给多模态模型，让它直接报数字
    VLM-grid      切成 2×2 交给模型分块数，再把四块相加

为什么要加 grid 这一档：只测 direct 会得到「VLM 数不准」这个过于便宜的结论，
而对方的反驳是「你提示词没写好」。分块数才是 VLM 能做到的最好水平，所以两档
都要有，结论只能建立在后者上——如果分块也救不回来，那「需要专用检测器」才算
被证明。

用法（在 cabbage-vision-agent 下）：
    python scripts/vlm_baseline.py --n 12
    python scripts/vlm_baseline.py --n 75 --mode both --out runs/vlm_baseline/full

成本上要注意：一张 1024×1024 大约 1000+ image tokens，grid 模式一张图要 5 次
调用。先用 --n 3 --mode direct 探一次单张价格，再决定跑多大。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DATA = Path(r"D:\cabbage_project")
DEFAULT_WEIGHTS = DEFAULT_DATA / "weights" / "yolo26n.pt"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

PROMPT_DIRECT = (
    "这是一张甘蓝田的俯拍图。请数出图中有多少株甘蓝（整株幼苗，不是叶片）。"
    '只输出 JSON，格式为 {"count": 数字}，不要任何解释。'
)
PROMPT_GRID = (
    "这是甘蓝田俯拍图的一个局部区域。请数出这个区域里有多少株甘蓝（整株幼苗）。"
    '只输出 JSON，格式为 {"count": 数字}，不要任何解释。'
)


def encode_image(path: Path, max_side: int = 1024, quality: int = 85) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def ask(b64: str, prompt: str, model: str, key: str, timeout: float = 180) -> dict:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}", "elapsed": time.perf_counter() - t0}
    except Exception as e:  # 超时、连接重置
        return {"error": f"{type(e).__name__}: {str(e)[:150]}", "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    msg = d["choices"][0]["message"]["content"] or ""
    usage = d.get("usage") or {}
    return {
        "text": msg,
        "count": parse_count(msg),
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def parse_count(text: str) -> int | None:
    m = re.search(r'"count"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else None


def split_tiles(path: Path, n: int = 2):
    """切成 n×n 块。n 可调是因为 n=2 每块仍有 25 株，会被质疑"切得不够细"。
    测 n=4（每块 3 株左右）才能说明问题不在视野大小，而在模型本身。"""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        out = {}
        for r in range(n):
            for c in range(n):
                box = (c * w // n, r * h // n, (c + 1) * w // n, (r + 1) * h // n)
                buf = io.BytesIO()
                im.crop(box).save(buf, "JPEG", quality=85)
                out[f"r{r}c{c}"] = base64.b64encode(buf.getvalue()).decode()
    return out


def gt_counts(labels_dir: Path, stem: str) -> int:
    f = labels_dir / f"{stem}.txt"
    if not f.exists():
        return 0
    return sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())


def detector_counts(weights: Path, images: list[Path], conf: float, device: str) -> list[int]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model([str(p) for p in images], conf=conf, verbose=False, device=device)
    return [int(len(r.boxes)) for r in results]


def stratified_pick(rows: list[dict], n: int) -> list[dict]:
    """按 GT 数排序后等间隔取，保证覆盖稀疏到密集，不靠随机运气。"""
    if n >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda r: r["gt"])
    step = (len(ordered) - 1) / (n - 1) if n > 1 else 0
    return [ordered[round(i * step)] for i in range(n)]


def summarize(name: str, pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"name": name, "n": 0}
    diffs = [p - g for g, p in pairs]
    abs_ = [abs(d) for d in diffs]
    gt_sum = sum(g for g, _ in pairs)
    pr_sum = sum(p for _, p in pairs)
    return {
        "name": name,
        "n": len(pairs),
        "gt_total": gt_sum,
        "pred_total": pr_sum,
        "bias": round(statistics.mean(diffs), 2),
        "mae": round(statistics.mean(abs_), 2),
        "mape": round(100 * statistics.mean([abs(d) / g for (g, _), d in zip(pairs, diffs)]), 2),
        "exact": round(100 * sum(1 for d in diffs if d == 0) / len(diffs), 1),
        "within3": round(100 * sum(1 for d in abs_ if d <= 3) / len(abs_), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="cabbage_project 根目录")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--n", type=int, default=12, help="抽样张数（按 GT 分层等间隔取）")
    ap.add_argument("--mode", choices=["direct", "grid", "both"], default="direct")
    ap.add_argument("--grid", type=int, default=2, help="分块数 n（n×n），2 或 4")
    ap.add_argument("--model", default=os.getenv("CVAGENT_VLM", "qwen3.5-omni-plus"))
    ap.add_argument("--conf", type=float, default=0.25, help="检测器 conf 阈值")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        print("缺少 DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    images_dir = args.data / "dataset" / "test" / "images"
    labels_dir = args.data / "dataset" / "test" / "labels"
    if not images_dir.is_dir():
        print(f"找不到测试图目录：{images_dir}", file=sys.stderr)
        return 2

    rows = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rows.append({"path": p, "stem": p.stem, "gt": gt_counts(labels_dir, p.stem)})
    rows = [r for r in rows if r["gt"] > 0]
    picked = stratified_pick(rows, args.n)
    print(f"数据集 {len(rows)} 张，抽 {len(picked)} 张（GT {min(r['gt'] for r in picked)}~{max(r['gt'] for r in picked)}）")

    det = detector_counts(args.weights, [r["path"] for r in picked], args.conf, args.device)
    for r, c in zip(picked, det):
        r["detector"] = c
    print("检测器完成")

    total_pt = total_ct = 0
    for i, r in enumerate(picked, 1):
        b64 = encode_image(r["path"])

        if args.mode in {"direct", "both"}:
            res = ask(b64, PROMPT_DIRECT, args.model, key)
            r["vlm_direct"] = res.get("count")
            r["direct_meta"] = {k: res.get(k) for k in ("elapsed", "prompt_tokens", "completion_tokens", "error", "text")}
            total_pt += res.get("prompt_tokens") or 0
            total_ct += res.get("completion_tokens") or 0
        else:
            r["vlm_direct"] = None

        if args.mode in {"grid", "both"}:
            quads = split_tiles(r["path"], args.grid)
            parts, t0, errs = [], time.perf_counter(), []
            for name, b64q in quads.items():
                res = ask(b64q, PROMPT_GRID, args.model, key)
                total_pt += res.get("prompt_tokens") or 0
                total_ct += res.get("completion_tokens") or 0
                if res.get("count") is None:
                    errs.append(f"{name}:{res.get('error') or res.get('text', '')[:40]}")
                else:
                    parts.append(res["count"])
            expected = args.grid ** 2
            r["vlm_grid"] = sum(parts) if len(parts) == expected else None
            r["grid_parts"] = parts
            r["grid_meta"] = {"elapsed": round(time.perf_counter() - t0, 2), "errors": errs}
        else:
            r["vlm_grid"] = None

        got = " ".join(
            f"{k}={r.get(k)}" for k in ("vlm_direct", "vlm_grid") if r.get(k) is not None
        )
        print(f"  [{i}/{len(picked)}] {r['stem']}  gt={r['gt']} det={r['detector']} {got}")

    pairs_det = [(r["gt"], r["detector"]) for r in picked]
    pairs_dir = [(r["gt"], r["vlm_direct"]) for r in picked if r.get("vlm_direct") is not None]
    pairs_grid = [(r["gt"], r["vlm_grid"]) for r in picked if r.get("vlm_grid") is not None]

    sums = [summarize("detector", pairs_det)]
    if pairs_dir:
        sums.append(summarize("vlm_direct", pairs_dir))
    if pairs_grid:
        sums.append(summarize(f"vlm_grid{args.grid}x{args.grid}", pairs_grid))

    out_dir = args.out or (Path("runs") / "vlm_baseline" / time.strftime("run_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "model": args.model,
            "mode": args.mode,
            "n": len(picked),
            "grid": args.grid,
            "conf": args.conf,
            "weights": str(args.weights),
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": sums,
        "rows": [
            {
                "stem": r["stem"],
                "gt": r["gt"],
                "detector": r["detector"],
                "vlm_direct": r.get("vlm_direct"),
                "vlm_grid": r.get("vlm_grid"),
                "grid_parts": r.get("grid_parts"),
            }
            for r in picked
        ],
    }
    (out_dir / "vlm_baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VLM 直接计数基线",
        "",
        f"- 模型：`{args.model}`，抽样 {len(picked)} 张，conf={args.conf}，分块 {args.grid}×{args.grid}",
        f"- token：prompt {total_pt} / completion {total_ct}",
        "",
        "| 来源 | n | 预测总数 | bias | MAE | MAPE | 完全数对 | 误差≤3 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sums:
        lines.append(
            f"| {s['name']} | {s['n']} | {s['pred_total']} | {s['bias']:+} | {s['mae']} | "
            f"{s['mape']}% | {s['exact']}% | {s['within3']}% |"
        )
    (out_dir / "vlm_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\n输出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
