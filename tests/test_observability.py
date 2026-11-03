"""可观测性测试：token、耗时、重试必须能被分离与汇总。"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvagent.agent.loop import Agent  # noqa: E402
from cvagent.llm.base import Completion, Message, ToolCall  # noqa: E402
from cvagent.report import render_markdown  # noqa: E402
from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import Tool, _schema, build_tools  # noqa: E402

from test_loop import ScriptedProvider, detect_call  # noqa: E402
from test_tools import FakeEngine  # noqa: E402


class SlowEngine(FakeEngine):
    """在引擎层注入 10ms 延迟，确保工具耗时非 0。"""

    def detect(self, *a, **kw):  # noqa: D401
        time.sleep(0.01)
        return super().detect(*a, **kw)


class TestTurnStats(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry(build_tools(FakeEngine(count=48)))

    def test_turn_stats_collect_usage(self):
        a = detect_call("c1")
        a.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        b = Completion(content="共 48 株。", usage={"prompt_tokens": 20, "completion_tokens": 7})
        traj = Agent(ScriptedProvider([a, b]), self.reg).run("q")
        self.assertEqual(len(traj.turns), 2)
        self.assertEqual(traj.totals()["prompt_tokens"], 30)
        self.assertEqual(traj.totals()["completion_tokens"], 12)

    def test_retries_propagate(self):
        a = detect_call("c1")
        a.retries = 2
        traj = Agent(ScriptedProvider([a, Completion(content="ok")]), self.reg).run("q")
        self.assertEqual(traj.totals()["retries"], 2)
        self.assertEqual(traj.steps[0].retries, 2)

    def test_tool_and_llm_time_separated(self):
        reg = ToolRegistry([
            Tool(name="wait", description="slow", parameters=_schema({}, []),
                 handler=lambda **kw: (time.sleep(0.01) or {"ok": True, "n": 1})),
        ])
        call = Completion(content=None,
                          tool_calls=[ToolCall(id="c1", name="wait", arguments="{}")],
                          finish_reason="tool_calls")
        traj = Agent(ScriptedProvider([call, Completion(content="done")]), reg).run("q")
        tot = traj.totals()
        self.assertGreater(tot["tool_ms"], 0)
        self.assertGreaterEqual(tot["llm_ms"], 0)
        self.assertGreaterEqual(tot["wall_ms"], tot["tool_ms"])

    def test_max_steps_conclusion_is_counted(self):
        prov = ScriptedProvider([detect_call(f"c{i}") for i in range(4)])
        traj = Agent(prov, self.reg, max_steps=2).run("q")
        self.assertEqual(traj.stopped_reason, "max_steps")
        # 2 轮工具 + 1 轮强制收束
        self.assertEqual(len(traj.turns), 3)

    def test_to_dict_has_totals(self):
        traj = Agent(ScriptedProvider([detect_call("c1"), Completion(content="ok")]), self.reg).run("q")
        d = traj.to_dict()
        for key in ("llm_ms", "tool_ms", "wall_ms", "prompt_tokens",
                    "completion_tokens", "retries", "n_turns",
                    "n_tool_calls", "n_failed_calls"):
            self.assertIn(key, d["totals"])
        self.assertIn("turns", d)
        self.assertIn("retries", d["steps"][0])

    def test_markdown_contains_cost_table(self):
        traj = Agent(ScriptedProvider([detect_call("c1"), Completion(content="ok")]), self.reg).run("q")
        md = render_markdown(traj)
        self.assertIn("## 开销", md)
        self.assertIn("**合计**", md)
        self.assertIn("LLM 耗时", md)


class _SeqRequests:
    """第一次 429、第二次 200，验证重试次数被记录。"""

    RequestException = Exception

    def __init__(self):
        self.calls = 0

    def post(self, *a, **kw):
        self.calls += 1
        if self.calls == 1:
            return _Resp(429, json.dumps({"error": {"code": "limit_requests", "message": "slow down"}}))
        return _Resp(200, json.dumps({
            "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }))


class _Resp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text

    def json(self):
        return json.loads(self.text)


class TestProviderRetries(unittest.TestCase):
    def test_retry_count_returned(self):
        import types

        from cvagent.llm.openai_compat import OpenAICompatProvider

        fake = _SeqRequests()
        sys.modules["requests"] = types.ModuleType("requests")
        sys.modules["requests"].post = fake.post
        sys.modules["requests"].RequestException = Exception
        try:
            prov = OpenAICompatProvider("https://example.invalid/v1", "sk-test", "m",
                                        name="dashscope", max_retries=2)
            comp = prov.complete([Message("user", "hi")], [])
        finally:
            sys.modules.pop("requests", None)
        self.assertEqual(comp.retries, 1)
        self.assertEqual(comp.usage["prompt_tokens"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
