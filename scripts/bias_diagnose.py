"""计数系统性多检的归因实验：mAP50 0.9718，为什么计数 MAE 是 15.6%。

`eval_report.md` 里有一组互相矛盾的数字：

    mAP50 0.9718 / Recall 0.9515        → 检测看着很好
    计数 MAE 7.57 个/图 / MAPE 15.59%   → 计数很难看
    完全数对 2.7% / 误差≤3 仅 16.0%
    bias +7.52（预测 4329 vs 真实 3765，多 15%）

而且 **bias 几乎等于 MAE**（7.52 / 7.57），意味着误差有 99% 来自系统性多检，
不是随机抖动。多出来的那 15% 到底是模型的错还是标注的错，决定了下一步该改什么：

    假设 A（模型过检）  低置信框是误检 → 抬高 conf 应该让 bias 收敛到 0
    假设 B（GT 漏标）   模型检对了但没被标 → 抬阈值后 bias 仍为正，且 FP 紧贴 GT

区分办法不是看总数，是看每个多余框落在哪：与 GT 重叠但 IoU 不够（口径/框回归问题），
还是与任何 GT 都不沾边（真误检）。

用法：
    python scripts/bias_diagnose.py --weights /path/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

DEFAULT_DATA = Path(r"D:\cabbage_project")
DEFAULT_WEIGHTS = DEFAULT_DATA / "runs" / "detect" / "base_v2_repro" / "weights" / "best.pt"

# 扫描的 conf 阈值。0.25 是项目默认值，0.001 用来收全部候选
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
         0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.80]


def read_gt(label_path: Path, w: int, h: int):
    """YOLO txt → 像素 xyxy。"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_dist(a, b) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def diag(b) -> float:
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("runs") / "bias_diagnose")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from PIL import Image

    images_dir = args.data / "dataset" / args.split / "images"
    labels_dir = args.data / "dataset" / args.split / "labels"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print(f"没有图片：{images_dir}", file=sys.stderr)
        return 2

    print(f"{args.split}: {len(images)} 张，权重 {args.weights.name}")
    model = YOLO(str(args.weights))
    t0 = time.perf_counter()
    results = model([str(p) for p in images], conf=0.001, verbose=False, device=args.device)
    print(f"推理完成 {time.perf_counter() - t0:.1f}s")

    per_image = []
    for img_path, res in zip(images, results):
        with Image.open(img_path) as im:
            w, h = im.size
        gt = read_gt(labels_dir / f"{img_path.stem}.txt", w, h)
        preds = [
            (list(map(float, b.xyxy[0].tolist())), float(b.conf[0]))
            for b in res.boxes
        ]
        per_image.append({"stem": img_path.stem, "gt": gt, "preds": preds})

    total_gt = sum(len(r["gt"]) for r in per_image)
    print(f"GT 合计 {total_gt} 株，候选框 {sum(len(r['preds']) for r in per_image)} 个（conf≥0.001）")

    # ---- 1. 阈值扫描 ----
    sweep_rows = []
    for conf in SWEEP:
        gt_sum = pred_sum = 0
        diffs, abs_diffs, exact, within3 = [], [], 0, 0
        tp = fp = fn = 0
        for r in per_image:
            kept = [p for p in r["preds"] if p[1] >= conf]
            cnt = len(kept)
            g = len(r["gt"])
            gt_sum += g
            pred_sum += cnt
            d = cnt - g
            diffs.append(d)
            abs_diffs.append(abs(d))
            exact += d == 0
            within3 += abs(d) <= 3

            # 贪心匹配，IoU≥0.5 记 TP（与 mAP50 同口径）
            used = set()
            for box, _ in sorted(kept, key=lambda x: -x[1]):
                best_i, best_iou = -1, 0.5
                for j, gb in enumerate(r["gt"]):
                    if j in used:
                        continue
                    v = iou(box, gb)
                    if v >= best_iou:
                        best_i, best_iou = j, v
                if best_i >= 0:
                    used.add(best_i)
                    tp += 1
                else:
                    fp += 1
            fn += len(r["gt"]) - len(used)

        n = len(per_image)
        sweep_rows.append({
            "conf": conf,
            "pred_total": pred_sum,
            "gt_total": gt_sum,
            "bias": round(statistics.mean(diffs), 2),
            "mae": round(statistics.mean(abs_diffs), 2),
            "mape": round(100 * statistics.mean([abs(d) / len(r["gt"]) for d, r in zip(diffs, per_image)]), 2),
            "exact": round(100 * exact / n, 1),
            "within3": round(100 * within3 / n, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(100 * tp / max(1, tp + fp), 2),
            "recall": round(100 * tp / max(1, tp + fn), 2),
        })

    # ---- 2. 多余框归因（在默认 conf 上）----
    CONF = 0.25
    orphan, near_gt, matched = [], [], []
    for r in per_image:
        kept = [p for p in r["preds"] if p[1] >= CONF]
        used = set()
        for box, c in sorted(kept, key=lambda x: -x[1]):
            best_j, best_iou = -1, 0.5
            for j, gb in enumerate(r["gt"]):
                if j in used:
                    continue
                v = iou(box, gb)
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0:
                used.add(best_j)
                matched.append({"conf": c, "area": (box[2] - box[0]) * (box[3] - box[1])})
                continue
            # 没匹配上：看它离最近的 GT 有多近
            if r["gt"]:
                j = min(range(len(r["gt"])), key=lambda k: center_dist(box, r["gt"][k]))
                dist_ratio = center_dist(box, r["gt"][j]) / max(diag(box), 1e-6)
                ov = iou(box, r["gt"][j])
            else:
                dist_ratio, ov = 99.0, 0.0
            rec = {"conf": c, "dist_ratio": dist_ratio, "iou": ov,
                   "area": (box[2] - box[0]) * (box[3] - box[1])}
            (near_gt if dist_ratio <= 0.5 else orphan).append(rec)

    def stat(xs):
        if not xs:
            return None
        return {"n": len(xs), "conf_median": round(statistics.median([x["conf"] for x in xs]), 3),
                "area_median": round(statistics.median([x["area"] for x in xs]), 1)}

    attribution = {
        "conf": CONF,
        "matched": stat(matched),
        "near_gt": stat(near_gt),
        "orphan": stat(orphan),
        "near_gt_median_iou": round(statistics.median([x["iou"] for x in near_gt]), 3) if near_gt else None,
    }

    # ---- 3. 密度分组的 bias ----
    by_density = []
    for label, lo, hi in [("sparse", 0, 44), ("medium", 45, 54), ("dense", 55, 999)]:
        group = [r for r in per_image if lo <= len(r["gt"]) <= hi]
        if not group:
            continue
        ds = [len([p for p in r["preds"] if p[1] >= CONF]) - len(r["gt"]) for r in group]
        by_density.append({"bucket": label, "n": len(group),
                           "gt_mean": round(statistics.mean([len(r["gt"]) for r in group]), 1),
                           "bias": round(statistics.mean(ds), 2),
                           "mae": round(statistics.mean([abs(d) for d in ds]), 2)})

    payload = {
        "meta": {"split": args.split, "n_images": len(images), "weights": str(args.weights),
                 "conf_default": CONF, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "sweep": sweep_rows,
        "attribution": attribution,
        "by_density": by_density,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bias_diagnose.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 计数多检归因（{args.split}，{len(images)} 张）", "",
        "## 阈值扫描", "",
        "| conf | 预测 | bias | MAE | MAPE | 完全数对 | TP | FP | FN | P | R |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sweep_rows:
        lines.append(
            f"| {r['conf']:.2f} | {r['pred_total']} | {r['bias']:+} | {r['mae']} | {r['mape']}% | "
            f"{r['exact']}% | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']}% | {r['recall']}% |")
    lines += ["", f"## 多余框落在哪（conf={CONF}）", ""]
    for k in ("matched", "near_gt", "orphan"):
        s = attribution[k]
        if s:
            lines.append(f"- **{k}**：{s['n']} 个，置信度中位 {s['conf_median']}，面积中位 {s['area_median']} px²")
    if attribution["near_gt_median_iou"] is not None:
        lines.append(f"- 紧贴 GT 那批的中位 IoU：{attribution['near_gt_median_iou']}")
    lines += ["", "## 按密度", "", "| 密度 | n | GT 均值 | bias | MAE |", "|---|---|---|---|---|"]
    for d in by_density:
        lines.append(f"| {d['bucket']} | {d['n']} | {d['gt_mean']} | {d['bias']:+} | {d['mae']} |")

    (args.out / "bias_diagnose.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\n输出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
