"""网格统计的纯逻辑测试。不加载权重。

这一层锁的是"模型手上有什么数据"。它对应的失败是真实发生过的：
`detect_cabbage` 只回前 3 个框，模型要回答"分布均匀吗"，于是自己造了
一张 4×4 的表。补数据的正确性靠这里守，提示词只能守"拿到数据后别越界"。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.vision.types import Box, DetectionResult  # noqa: E402


def make(boxes, w=1024, h=1024, conf=0.25) -> DetectionResult:
    return DetectionResult(
        image_id="x" * 12, path="mem.jpg", width=w, height=h,
        conf_threshold=conf, iou_threshold=0.45, boxes=boxes, device="cpu",
    )


def at(cx, cy, size=10) -> Box:
    """以 (cx, cy) 为中心造一个框。"""
    return Box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, 0.9)


class TestGridCounts(unittest.TestCase):
    def test_shape_and_sum(self):
        res = make([at(100, 100), at(900, 900), at(500, 500)])
        g = res.grid(4)
        self.assertEqual(len(g["counts"]), 4)
        self.assertTrue(all(len(r) == 4 for r in g["counts"]))
        # 求和必须等于 count：用户拿总株数一加就能核对，这是这套字段唯一的自证
        self.assertEqual(g["cell_sum"], res.count)

    def test_corner_and_boundary_clamp(self):
        # 中心点正好在右/下边界上：不能落到越界索引里丢框
        res = make([at(1024, 1024), at(0, 0)])
        g = res.grid(4)
        self.assertEqual(g["counts"][0][0], 1)   # 左上
        self.assertEqual(g["counts"][3][3], 1)   # 右下
        self.assertEqual(g["cell_sum"], 2)

    def test_clamps_invalid_n(self):
        self.assertEqual(make([at(1, 1)]).grid(99)["n"], 8)
        self.assertEqual(make([at(1, 1)]).grid(0)["n"], 2)

    def test_empty_cells_listed(self):
        # 只有左上角一格有株，其余 15 格空
        res = make([at(100, 100)] * 6)
        g = res.grid(4)
        self.assertEqual(g["empty_cells"], 15)
        self.assertEqual(len(g["empty_positions"]), 8)      # 只列前 8
        self.assertEqual(g["empty_positions_omitted"], 7)
        self.assertNotIn([0, 0], g["empty_positions"])

    def test_empty_cells_absent_when_full(self):
        boxes = [at(128 + 256 * c, 128 + 256 * r) for r in range(4) for c in range(4)]
        g = make(boxes).grid(4)
        self.assertEqual(g["empty_cells"], 0)
        self.assertNotIn("empty_positions", g)

    def test_max_min_positions(self):
        boxes = [at(100, 100)] * 5 + [at(900, 900)]
        g = make(boxes).grid(4)
        # [row, col, count]
        self.assertEqual(g["max_cell"], [0, 0, 5])
        self.assertEqual(g["min_cell"][2], 0)


class TestPatternLabel(unittest.TestCase):
    def test_regular_grid_is_more_regular_than_random(self):
        # 每格一样多（4 株/格）-> 方差 0 -> vmr 0。株数要够，否则会先撞上"样本过少"
        boxes = [at(128 + 256 * c, 128 + 256 * r) for r in range(4) for c in range(4)] * 4
        g = make(boxes).grid(4)
        self.assertEqual(g["vmr"], 0.0)
        self.assertEqual(g["pattern"], "比随机更规则")
        self.assertEqual(g["empty_if_random"], 0.3)  # 16*e^-4 ≈ 0.29

    def test_single_pile_is_clustered(self):
        g = make([at(100, 100)] * 48).grid(4)
        self.assertGreater(g["vmr"], 1.3)
        self.assertEqual(g["pattern"], "有聚集")

    def test_too_few_points_refuses_judgement(self):
        # 4 株撒在 16 格里，均值 0.25：这时候报"规则"是噪声不是结论
        g = make([at(100, 100), at(900, 100), at(100, 900), at(900, 900)]).grid(4)
        self.assertEqual(g["pattern"], "样本过少，不足以判断格局")

    def test_empty_below_random_baseline_is_visible(self):
        # 空格数要跟随机期望一起看：少于期望说明空格不是"空地"
        boxes = [at(128 + 256 * c, 128 + 256 * r) for r in range(3) for c in range(3)]
        g = make(boxes).grid(4)
        self.assertEqual(g["empty_cells"], 7)
        self.assertLess(g["empty_cells"], g["empty_if_random"])


class TestSummaryCarriesGrid(unittest.TestCase):
    def test_summary_has_grid(self):
        res = make([at(100, 100), at(500, 500)])
        s = res.summary()
        self.assertIn("grid", s)
        self.assertEqual(s["grid"]["cell_sum"], s["count"])
        self.assertEqual(s["grid"]["n"], 4)

    def test_full_keeps_grid(self):
        res = make([at(100, 100), at(500, 500)])
        f = res.full()
        self.assertIn("grid", f)
        self.assertEqual(len(f["boxes"]), 2)


if __name__ == "__main__":
    unittest.main()
