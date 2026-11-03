"""服务层的契约测试。不加载权重、不联网、不起端口。

这里锁的不是"功能能跑"，是三条**出错了不会有报错提示**的边界：

1. **路径隔离**。pipeline 会从问题文本里抽绝对路径去读盘，如果前端能把路径
   写进问题，这个功能就是一次任意文件读取。所以「用户写的路径必须被抹掉、
   服务端注入的那条必须保留」要在函数层面钉死，而不是靠端到端手工点一遍。
2. **格式判定只看内容**。Content-Type 和文件名都能伪造，而 engine 后面是按
   扩展名白名单放行的，两者对不上就等于让坏数据进模型。
3. **模型不可用时状态码要正确**。页面能打开、能上传、能降级，比"服务起不来"
   好得多；但 503 和 500 必须分开，否则运维看不出是没加载还是崩了。
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient

    from cvagent import serve

    HAS_WEB = True
except ImportError:  # pragma: no cover - 没装 web extra 时整组跳过
    HAS_WEB = False


@unittest.skipUnless(HAS_WEB, "未安装 fastapi / httpx")
class PathIsolationTest(unittest.TestCase):
    """用户文本里的路径不能进入工具参数。"""

    def test_scrub_removes_posix_and_windows_paths(self):
        s = serve._scrub_paths("数一下 /etc/passwd 和 C:\\Windows\\win.ini 有多少株")
        self.assertNotIn("/etc/passwd", s)
        self.assertNotIn("C:\\Windows", s)
        self.assertIn("有多少株", s)  # 只抹路径，不动问题本身

    def test_user_path_is_dropped_but_server_path_kept(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"x")
            serve._uploads.clear()
            serve._remember_upload("aaaa1111bbbb", p)
            try:
                req = serve.AskRequest(
                    question="数一下 D:\\secret\\other.jpg 有多少株", image_id="aaaa1111bbbb"
                )
                q = serve._build_question(req)
            finally:
                serve._uploads.clear()

        self.assertIn(str(p), q)          # 服务端注入的在
        self.assertNotIn("secret", q)      # 用户写的不在
        self.assertEqual(q.count("图片路径："), 1)

    def test_no_image_leaves_question_untouched(self):
        req = serve.AskRequest(question="为什么默认阈值是 0.25？")
        self.assertEqual(serve._build_question(req), "为什么默认阈值是 0.25？")

    def test_unknown_image_id_is_410_not_500(self):
        from fastapi import HTTPException

        req = serve.AskRequest(question="多少株", image_id="000000000000")
        with self.assertRaises(HTTPException) as ctx:
            serve._build_question(req)
        self.assertEqual(ctx.exception.status_code, 410)

    def test_session_falls_back_to_last_image(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.jpg"
            p.write_bytes(b"x")
            serve._uploads.clear()
            serve._sessions.clear()
            serve._remember_upload("cccc2222dddd", p)
            try:
                serve._build_question(serve.AskRequest(question="多少株", image_id="cccc2222dddd", session="s1"))
                q = serve._build_question(serve.AskRequest(question="再复核一下", session="s1"))
            finally:
                serve._uploads.clear()
                serve._sessions.clear()
        self.assertIn(str(p), q)


@unittest.skipUnless(HAS_WEB, "未安装 fastapi / httpx")
class SniffTest(unittest.TestCase):
    def test_magic_bytes_win_over_extension(self):
        self.assertEqual(serve._sniff_ext(b"\xff\xd8\xff\xe0" + b"0" * 20), ".jpg")
        self.assertEqual(serve._sniff_ext(b"\x89PNG\r\n\x1a\n" + b"0" * 20), ".png")
        self.assertEqual(serve._sniff_ext(b"RIFF\x00\x00\x00\x00WEBPVP8 "), ".webp")

    def test_non_image_rejected(self):
        self.assertIsNone(serve._sniff_ext(b"%PDF-1.7"))
        self.assertIsNone(serve._sniff_ext(b""))


@unittest.skipUnless(HAS_WEB, "未安装 fastapi / httpx")
class ObsSlimTest(unittest.TestCase):
    def test_long_lists_and_strings_are_summarised(self):
        slim = serve._slim_obs({"count": 42, "detections": list(range(300)), "note": "x" * 900})
        self.assertEqual(slim["count"], 42)
        self.assertEqual(slim["detections"], "<300 项省略>")
        self.assertTrue(slim["note"].endswith("…"))
        self.assertLess(len(slim["note"]), 500)


@unittest.skipUnless(HAS_WEB, "未安装 fastapi / httpx")
class EndpointTest(unittest.TestCase):
    """端点级：只覆盖不依赖模型的那部分。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_dir = serve.UPLOAD_DIR
        serve.UPLOAD_DIR = Path(self.tmp.name)
        serve._uploads.clear()
        # 不进 with 就不触发 lifespan，模型保持未加载——正好用来测降级路径
        self.client = TestClient(serve.app)

    def tearDown(self):
        serve.UPLOAD_DIR = self._old_dir
        serve._uploads.clear()
        self.tmp.cleanup()

    def test_upload_rejects_non_image(self):
        r = self.client.post(
            "/api/upload", files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}
        )
        self.assertEqual(r.status_code, 415)

    def test_upload_accepts_png(self):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 120, 60)).save(buf, format="PNG")
        r = self.client.post(
            "/api/upload", files={"file": ("x.png", io.BytesIO(buf.getvalue()), "image/png")}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["width"], 64)
        self.assertEqual(body["height"], 48)
        self.assertEqual(len(body["image_id"]), 12)
        self.assertTrue((Path(self.tmp.name) / f"{body['image_id']}.png").exists())

    def test_ask_without_model_is_503(self):
        old = dict(serve._state)
        serve._state.update(ready=False, error="")
        try:
            r = self.client.post("/api/ask", json={"question": "多少株"})
        finally:
            serve._state.update(old)
        self.assertEqual(r.status_code, 503)
        self.assertIn("不可用", r.json()["detail"])

    def test_health_reflects_state(self):
        old = dict(serve._state)
        serve._state.update(ready=False, error="权重不存在")
        try:
            h = self.client.get("/api/health").json()
        finally:
            serve._state.update(old)
        self.assertEqual(h["status"], "degraded")
        self.assertFalse(h["ready"])


if __name__ == "__main__":
    unittest.main()
