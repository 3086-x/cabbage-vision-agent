"""推理后端测试：只测校验与缓存逻辑，不加载权重。

校验顺序刻意放在模型加载**之前**——一张打不开的文件不该花 10 秒等模型。
这里断言的就是这个顺序带来的行为：错误在几百毫秒内抛出，且分类正确。
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.errors import ImageNotFound, InvalidArgument, UnsupportedFormat  # noqa: E402
from cvagent.vision.engine import YoloEngine, image_id_of, sniff_format  # noqa: E402


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.eng = YoloEngine(weights=Path("does_not_exist.pt"))

    def test_missing_file(self):
        with self.assertRaises(ImageNotFound):
            self.eng.validate("C:/nope/missing.jpg")

    def test_directory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InvalidArgument) as ctx:
                self.eng.validate(d)
            self.assertIn("目录", ctx.exception.message)

    def test_bad_extension(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.gif"
            p.write_bytes(b"GIF89a")
            with self.assertRaises(UnsupportedFormat):
                self.eng.validate(p)

    def test_fake_jpeg_rejected_by_magic_bytes(self):
        """扩展名是 .jpg，内容却是文本——必须靠魔术字节抓住。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.jpg"
            p.write_text("这不是图片", encoding="utf-8")
            with self.assertRaises(UnsupportedFormat) as ctx:
                self.eng.validate(p)
            self.assertIn("损坏", ctx.exception.message)

    def test_tiny_image_rejected(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.png"
            Image.new("RGB", (8, 8)).save(p)
            with self.assertRaises(InvalidArgument) as ctx:
                self.eng.validate(p)
            self.assertIn("过小", ctx.exception.message)

    def test_validation_is_fast(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.jpg"
            p.write_text("x", encoding="utf-8")
            t0 = time.perf_counter()
            try:
                self.eng.validate(p)
            except UnsupportedFormat:
                pass
            self.assertLess(time.perf_counter() - t0, 0.5, "校验不应触发模型加载")

    def test_sniff_format(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            for fmt, name in (("PNG", "a.png"), ("JPEG", "b.jpg"), ("BMP", "c.bmp")):
                p = Path(d) / name
                Image.new("RGB", (64, 64)).save(p, fmt)
                self.assertEqual(sniff_format(p), {"PNG": "png", "JPEG": "jpeg", "BMP": "bmp"}[fmt])


class TestImageId(unittest.TestCase):
    def test_id_changes_with_content(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGB", (64, 64), (10, 10, 10)).save(p)
            first = image_id_of(p)
            self.assertEqual(len(first), 12)
            Image.new("RGB", (64, 64), (200, 200, 200)).save(p)
            self.assertNotEqual(image_id_of(p), first)

    def test_id_stable_for_same_file(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGB", (64, 64)).save(p)
            self.assertEqual(image_id_of(p), image_id_of(p))


class TestConfGuard(unittest.TestCase):
    def test_conf_out_of_range_raises_before_model_load(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGB", (128, 128)).save(p)
            eng = YoloEngine(weights=Path("does_not_exist.pt"))
            for bad in (0, -0.1, 1.5):
                with self.assertRaises(InvalidArgument):
                    eng.detect(p, conf=bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
