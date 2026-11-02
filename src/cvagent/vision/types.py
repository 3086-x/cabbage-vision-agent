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

    # ---- 两套视图 ----
    def summary(self, head: int = 3) -> dict:
        """默认视图：统计 + 少量代表框。"""
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
