"""会话记忆测试：落盘、截断、非法名、损坏文件容错。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.errors import InvalidArgument  # noqa: E402
from cvagent.session import Session, Turn  # noqa: E402


class TestSessionRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_load(self):
        s = Session.load("demo", root=self.root)
        s.append(Turn(q="有多少株", a="49 株", image="x.jpg", count=49, model="m"))
        path = s.save()
        self.assertTrue(path.exists())

        again = Session.load("demo", root=self.root)
        self.assertEqual(len(again.turns), 1)
        self.assertEqual(again.turns[0].count, 49)
        self.assertEqual(again.turns[0].a, "49 株")

    def test_context_uses_last_three_turns(self):
        s = Session.load("demo", root=self.root)
        for i in range(5):
            s.append(Turn(q=f"q{i}", a=f"a{i}", count=i))
        ctx = s.context_for("新问题")
        self.assertIn("q4", ctx)
        self.assertIn("q2", ctx)
        self.assertNotIn("q1", ctx)
        self.assertNotIn("q0", ctx)

    def test_context_budget_and_field_clip(self):
        s = Session.load("demo", root=self.root)
        for i in range(3):
            s.append(Turn(q="长" * 500, a="答" * 500))
        ctx = s.context_for("新问题")
        self.assertLessEqual(len(ctx), 1210)  # 1200 + 结尾提示的少量余量
        self.assertIn("…", ctx)

    def test_same_question_skips_history(self):
        s = Session.load("demo", root=self.root)
        s.append(Turn(q="q", a="a"))
        self.assertEqual(s.context_for("q"), "")

    def test_retention_capped_at_20(self):
        s = Session.load("demo", root=self.root)
        for i in range(25):
            s.append(Turn(q=f"q{i}", a="a"))
        self.assertEqual(len(s.turns), 20)
        self.assertEqual(s.turns[-1].q, "q24")

    def test_invalid_id_rejected(self):
        for bad in ("../x", "", "a" * 33, "含 空格"):
            with self.assertRaises(InvalidArgument):
                Session.load(bad, root=self.root)
        self.assertFalse(any(self.root.glob("*.json")))

    def test_corrupt_file_is_backed_up(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "demo.json").write_text("{not json", encoding="utf-8")
        s = Session.load("demo", root=self.root)
        self.assertEqual(s.turns, [])
        self.assertTrue(list(self.root.glob("demo.bak-*")))

    def test_blank_session_context(self):
        s = Session.load("fresh", root=self.root)
        self.assertEqual(s.context_for("q"), "")
        self.assertFalse(s.path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
