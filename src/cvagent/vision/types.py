"""检测结果的数据类型。

设计约束：这个对象**不是**给人看的标注结果，而是给模型看的证据。
所以它必须自带两套视图：
    - 摘要视图（默认）：统计量 + 少量代表框，几十 token
    - 完整视图：全部框，仅在模型显式要求时给出

理由：一张密集图能有 80+ 个框，每个框 5 个浮点数，直接序列化会让单次工具
返回吃掉上千 token，几轮下来就挤爆上下文。摘要 + 按需下钻是 Agent 工程的
基本约束，不是优化项。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """一个检测框。坐标保留到整像素，置信度保留三位小数。"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_list(self) -> list[float]:
        """紧凑序列化：[cx, cy, w, h, conf]，5 个数即可复原位置。"""
        return [
            round(self.cx, 1),
            round(self.cy, 1),
            round(self.w, 1),
            round(self.h, 1),
            round(self.conf, 3),
        ]


@dataclass
class DetectionResult:
    image_id: str
    path: str
    width: int
    height: int
    conf_threshold: float
    iou_threshold: float
    boxes: list[Box] = field(default_factory=list)
    elapsed_ms: float = 0.0
    device: str = "cpu"

    # ---- 派生统计 ----
    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def concurrency(self) -> float:
        """拥挤度：框面积总和 / 图像面积。比绝对计数更能反映遮挡。"""
        if self.width <= 0 or self.height <= 0:
            return 0.0
        covered = sum(b.area for b in self.boxes)
        return min(covered / (self.width * self.height), 1.5)

    def conf_stats(self) -> dict:
        if not self.boxes:
            return {"min": 0.0, "median": 0.0, "max": 0.0, "n_low": 0}
        cs = sorted(b.conf for b in self.boxes)
        n = len(cs)
        median = cs[n // 2] if n % 2 else (cs[n // 2 - 1] + cs[n // 2]) / 2
        return {
            "min": round(cs[0], 3),
            "median": round(median, 3),
            "max": round(cs[-1], 3),
            "n_low": sum(1 for c in cs if c < 0.5),
        }

    def size_stats(self) -> dict:
        """框尺寸分布，单位像素。用于判断目标是否过小（小目标 mAP 天然偏低）。"""
        if not self.boxes:
            return {"w_median": 0.0, "h_median": 0.0}
        ws = sorted(b.w for b in self.boxes)
        hs = sorted(b.h for b in self.boxes)
        return {
            "w_median": round(ws[len(ws) // 2], 1),
            "h_median": round(hs[len(hs) // 2], 1),
        }

    def grid(self, n: int = 4) -> dict:
        """把画面均分成 n×n 格，统计每格株数，用于回答"分布均匀吗"。

        口径四件事，缺一个就会变成另一组数字：

        1 **按框中心点归属**，不按面积分摊。一个框只落一格，所以
           counts 求和恒等于 count——用户拿总株数一加就能核对。按面积分摊
           会出现 3.4 株这种值，反而没法核。
        2 **坐标轴是图像，不是地块**。图像本身是裁出来的，图内均匀不代表
           田间均匀，所以只给"相对本图"的判断，不给农学结论。
        3 **空格要显式列出来，同时给出随机基线**。"有没有哪块是空的"问的
           就是这个；但空格本身不是证据——格子越小、株数越少，随机撒点也会留空。
           所以同时给 empty_if_random，让"3 格空着"能被对照成"随机本来就会空 6 格"。
        4 **判断用 VMR 而不是变异系数**。VMR = 方差 / 均值，是空间点格局里
           数格子（quadrat count）的标准检验量：随机分布约为 1，<1 比随机更规则，
           >1 有聚集。CV 会随 n 变而漂——实测同一张图 n=4 时 CV=0.30、
           n=6 时 0.52，同一套阈值会把"格变小"误判成"分布变差"；VMR 在两种
           粒度下分别是 0.35 和 0.47，结论一致。
        """
        n = max(2, min(int(n), 8))
        cells = [[0] * n for _ in range(n)]
        if self.width > 0 and self.height > 0:
            for b in self.boxes:
                col = min(max(int(b.cx / self.width * n), 0), n - 1)
                row = min(max(int(b.cy / self.height * n), 0), n - 1)
                cells[row][col] += 1

        flat = [c for r in cells for c in r]
        n_cells = n * n
        total = sum(flat)
        mean = total / n_cells
        var = sum((c - mean) ** 2 for c in flat) / n_cells
        vmr = (var / mean) if mean > 0 else 0.0

        empty = [[r, c] for r in range(n) for c in range(n) if cells[r][c] == 0]
        mx, mn = max(flat), min(flat)
        mx_pos = next([r, c] for r in range(n) for c in range(n) if cells[r][c] == mx)
        mn_pos = next([r, c] for r in range(n) for c in range(n) if cells[r][c] == mn)

        out: dict = {
            "n": n,
            "counts": cells,
            "cell_sum": total,
            "mean_per_cell": round(mean, 1),
            "max_cell": [*mx_pos, mx],
            "min_cell": [*mn_pos, mn],
            "empty_cells": len(empty),
            # 随机撒同样的点数，期望空几格：n² · e^(-λ)。空格判断的基线。
            "empty_if_random": round(n_cells * _EXP_NEG(mean), 1),
            "vmr": round(vmr, 2),
            "pattern": _pattern_label(vmr, mean),
            "basis": "框中心点归属；vmr=方差/均值，随机=1；坐标轴为图像，非地块",
        }
        if empty:
            # 只列前 8 个：n=8 时最多 64 格，全列会挤掉别的字段
            out["empty_positions"] = empty[:8]
            if len(empty) > 8:
                out["empty_positions_omitted"] = len(empty) - 8
        return out

    # ---- 两套视图 ----
    def summary(self, head: int = 3) -> dict:
        """默认视图：统计 + 少量代表框 + 4×4 网格。

        网格默认就给：模型回答"分布均匀吗"时若手上没有这个字段，会拿
        boxes_head 的 3 个坐标去脑补出一张 4×4 表（实测发生过）。补数据
        比在提示词里反复叮嘱有用——约束能拦住编造，拦不住不会。
        """
        return {
            "image_id": self.image_id,
            "size": [self.width, self.height],
            "count": self.count,
            "density": self.density(),
            "mean_density_per_mpx": round(self.count / max(self.width * self.height / 1e6, 1e-6), 1),
            "coverage": round(self.concurrency, 3),
            "conf_threshold": self.conf_threshold,
            "conf_stats": self.conf_stats(),
            "box_size_px": self.size_stats(),
            "grid": self.grid(4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "device": self.device,
            "boxes_head": [b.to_list() for b in self.boxes[:head]],
            "boxes_truncated": max(0, self.count - head),
        }

    def full(self) -> dict:
        payload = self.summary(head=0)
        payload.pop("boxes_truncated", None)
        payload["boxes"] = [b.to_list() for b in self.boxes]
        return payload

    def density(self) -> str:
        """密度分级。阈值的来源见 docs/DESIGN.md：
        取训练集逐图株数的四分位点，单图 1024x1024 尺度下
        sparse < Q1 <= moderate < Q2 <= dense < Q3 <= very_dense。
        """
        from ..config import DENSITY_QUARTILES

        q1, q2, q3 = DENSITY_QUARTILES
        if self.count < q1:
            return "sparse"
        if self.count < q2:
            return "moderate"
        if self.count < q3:
            return "dense"
        return "very_dense"


def _EXP_NEG(x: float) -> float:
    import math

    return math.exp(-x)


def _pattern_label(vmr: float, mean: float) -> str:
    """把 VMR 翻成一句人话。

    阈值 0.8 / 1.3 不是农学标准，是照着实测分布定的：30 张验证图的 VMR
    在 n=4 时落在 0.12~0.51、n=6 时 0.17~0.92，**没有一张超过 1**。
    也就是说这批图全都比随机撒点更规整（甘蓝是定植的，株行距本身就齐），
    "有聚集"这一档在验证集上一次都没触发——它是对超出已验证范围的情况
    报警，不是"看起来不均"的形容词。

    mean < 1.5 时不给判断：格子数远多于株数时，VMR 的抽样波动比信号大，
    这时候报"规则"或"聚集"都是噪声。宁可说判断不了。
    """
    if mean < 1.5:
        return "样本过少，不足以判断格局"
    if vmr < 0.8:
        return "比随机更规则"
    if vmr <= 1.3:
        return "接近随机"
    return "有聚集"
