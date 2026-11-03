"""工具层契约测试。不加载权重、不联网。

这一层的测试价值最高：它锁住的是"模型会看到什么"，而模型行为本身是不可
测试的。工具返回结构或错误码一变，这里先红，不用等真模型跑偏才发现。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import build_tools  # noqa: E402
from cvagent.vision.types import Box, DetectionResult  # noqa: E402


class FakeEngine:
    """最小替身：只实现工具层用到的三个方法。"""

    def __init__(self, count: int = 48, n_low: int = 0):
        self.count = count
        self.n_low = n_low
        self._cache: dict[str, DetectionResult] = {}
        self.calls: list[tuple] = []

    def validate(self, path):
        p = Path(path)
        if not p.exists():
            from cvagent.errors import ImageNotFound

            raise ImageNotFound(str(p))
        from PIL import Image

        with Image.open(p) as im:
            w, h = im.size
        return p, w, h, "png"

    def detect(self, image_path, conf=0.25, iou=0.45, use_cache=True):
        self.calls.append(("detect", str(image_path), conf))
        boxes = []
        for i in range(self.count):
            c = 0.3 if i < self.n_low else 0.9
            boxes.append(Box(10 * i, 10, 10 * i + 40, 50, c))
        res = DetectionResult(
            image_id="abc123def456", path=str(image_path), width=1024, height=1024,
            conf_threshold=conf, iou_threshold=iou, boxes=boxes, elapsed_ms=12.5, device="cpu",
        )
        self._cache[res.image_id] = res
        return res

    def get_cached(self, image_id):
        return self._cache.get(image_id)

    def get_last(self, image_id):
        return self._cache.get(image_id)


def make_registry(**kw) -> tuple[ToolRegistry, FakeEngine]:
    eng = FakeEngine(**kw)
    return ToolRegistry(build_tools(eng)), eng


class TestDispatchContract(unittest.TestCase):
    def setUp(self):
        self.reg, self.eng = make_registry()

    def test_unknown_tool_returns_hint_not_exception(self):
        out = self.reg.dispatch("rm_rf", {})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "UNKNOWN_TOOL")
        self.assertIn("detect_cabbage", out["error"]["hint"])

    def test_missing_required_arg(self):
        out = self.reg.dispatch("detect_cabbage", {})
        self.assertEqual(out["error"]["code"], "INVALID_ARG")
        self.assertTrue(out["error"]["retryable"])

    def test_conf_out_of_range(self):
        out = self.reg.dispatch("detect_cabbage", {"image_path": "a.jpg", "conf": 2.5})
        self.assertEqual(out["error"]["code"], "INVALID_ARG")
        self.assertIn("1.0", out["error"]["message"])

    def test_unknown_arg_rejected(self):
        out = self.reg.dispatch("count_cabbage", {"image_path": "a.jpg", "threshold": 0.3})
        self.assertEqual(out["error"]["code"], "INVALID_ARG")

    def test_malformed_json_arguments(self):
        out = self.reg.dispatch("detect_cabbage", "{not json")
        self.assertEqual(out["error"]["code"], "INVALID_ARG")

    def test_empty_arguments_string_parsed_as_empty_object(self):
        out = self.reg.dispatch("inspect_image", "")
        self.assertEqual(out["error"]["code"], "INVALID_ARG")  # 缺 image_path

    def test_handler_crash_is_contained(self):
        def boom(**kwargs):
            raise RuntimeError("显存炸了")

        from cvagent.tools.spec import Tool, _schema

        reg = ToolRegistry([Tool(name="boom", description="d", parameters=_schema({}, []), handler=boom)])
        out = reg.dispatch("boom", {})
        self.assertEqual(out["error"]["code"], "TOOL_CRASH")
        self.assertFalse(out["error"]["retryable"])


class TestDetectionPayload(unittest.TestCase):
    def setUp(self):
        self.reg, self.eng = make_registry(count=48, n_low=2)

    def test_summary_has_stats_and_bounded_boxes(self):
        out = self.reg.dispatch("detect_cabbage", {"image_path": "x.jpg", "conf": 0.25})
        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 48)
        self.assertEqual(len(out["boxes_head"]), 3)
        self.assertEqual(out["boxes_truncated"], 45)
        self.assertIn("n_low", out["conf_stats"])
        self.assertEqual(out["conf_stats"]["n_low"], 2)
        self.assertIn("density", out)

    def test_full_detail_returns_all_boxes(self):
        out = self.reg.dispatch("detect_cabbage", {"image_path": "x.jpg", "detail": "full"})
        self.assertEqual(len(out["boxes"]), 48)

    def test_big_payload_gets_truncated(self):
        reg, _ = make_registry(count=300)
        out = reg.dispatch("detect_cabbage", {"image_path": "x.jpg", "detail": "full"})
        self.assertTrue(out.get("truncated"))
        self.assertIn("boxes", out["dropped_fields"])
        self.assertIn("count", out)  # 结论字段必须保留

    def test_count_tool_is_lean(self):
        out = self.reg.dispatch("count_cabbage", {"image_path": "x.jpg"})
        self.assertEqual(out["count"], 48)
        self.assertNotIn("boxes", out)


class TestRefineFlow(unittest.TestCase):
    def test_refine_requires_prior_detection(self):
        reg, _ = make_registry()
        out = reg.dispatch("refine_detection", {"image_id": "deadbeef0000", "conf": 0.15})
        self.assertEqual(out["error"]["code"], "INVALID_ARG")
        self.assertIn("detect_cabbage", out["error"]["hint"])

    def test_refine_after_detect_reports_comparison(self):
        reg, _ = make_registry(count=48, n_low=3)
        reg.dispatch("detect_cabbage", {"image_path": "x.jpg", "conf": 0.25})
        out = reg.dispatch("refine_detection", {"image_id": "abc123def456", "conf": 0.15})
        self.assertTrue(out["ok"])
        self.assertEqual(out["comparison"]["before"]["count"], 48)
        self.assertIn(out["stability"], {"stable", "unstable", "no_baseline"})


class TestInspectImage(unittest.TestCase):
    def test_rejects_missing_file(self):
        reg, _ = make_registry()
        out = reg.dispatch("inspect_image", {"image_path": "C:/definitely/not/here.jpg"})
        self.assertEqual(out["error"]["code"], "IMAGE_NOT_FOUND")

    def test_reads_real_image(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.png"
            Image.new("RGB", (320, 240), (120, 130, 110)).save(p)
            reg, _ = make_registry()
            out = reg.dispatch("inspect_image", {"image_path": str(p)})
            self.assertTrue(out["ok"])
            self.assertEqual(out["size"], [320, 240])
            self.assertEqual(out["format"], "png")
            self.assertIn("assessment", out)


class TestBatchPagination(unittest.TestCase):
    """批量检测必须能翻页。

    起因是评测里发现模型在 75 张的目录上把同一个调用重试了 12 次：
    工具说"仅处理前 N 张"却没有任何参数能继续，模型只能干瞪眼。
    """

    def _dir(self, n: int) -> Path:
        d = Path(tempfile.mkdtemp())
        for i in range(n):
            (d / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        return d

    def test_first_page_reports_next_offset(self):
        reg, _ = make_registry(count=5)
        d = self._dir(12)
        out = reg.dispatch("batch_detect", {"directory": str(d), "limit": 5})
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_processed"], 5)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["next_offset"], 5)
        self.assertIn("offset=5", out["note"])

    def test_last_page_closes_the_loop(self):
        reg, _ = make_registry(count=5)
        d = self._dir(12)
        out = reg.dispatch("batch_detect", {"directory": str(d), "limit": 10, "offset": 10})
        self.assertEqual(out["n_processed"], 2)
        self.assertFalse(out["truncated"])
        self.assertIsNone(out["next_offset"], "没有下一页时必须是 None，不能让模型继续翻")

    def test_pages_do_not_overlap(self):
        reg, _ = make_registry(count=1)
        d = self._dir(6)
        a = reg.dispatch("batch_detect", {"directory": str(d), "limit": 3})
        b = reg.dispatch("batch_detect", {"directory": str(d), "limit": 3, "offset": a["next_offset"]})
        names_a = [r["file"] for r in a["rows_top"]]
        names_b = [r["file"] for r in b["rows_top"]]
        self.assertEqual(set(names_a) & set(names_b), set())
        self.assertEqual(len(set(names_a) | set(names_b)), 6)

    def test_offset_out_of_range_is_explained(self):
        reg, _ = make_registry(count=1)
        d = self._dir(3)
        out = reg.dispatch("batch_detect", {"directory": str(d), "offset": 99})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "INVALID_ARG")
        self.assertIn("3", out["error"]["message"], "要告诉模型目录到底有多少张，它才能修正")


if __name__ == "__main__":
    unittest.main(verbosity=2)
