"""循环层测试：用脚本化 provider 驱动，验证与模型无关的工程保证。

重点覆盖三件容易出错的事：
    - 步数打满时必须收束，不能返回半句话
    - 工具报错必须原样进入上下文，由模型决定
    - 历史压缩不能破坏 tool_call_id 配对（否则真供应商会 400）
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvagent.agent.loop import Agent  # noqa: E402
from cvagent.llm.base import Completion, Message, ToolCall  # noqa: E402
from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import Tool, _schema, build_tools  # noqa: E402

from test_tools import FakeEngine  # noqa: E402


class ScriptedProvider:
    """按预设顺序返回 Completion，记录被调用时的消息快照。

    tools 为空说明是收束请求（循环打满步数后强制作答），这时返回 conclusion，
    模拟真实模型"没有工具可用了就根据已有观测总结"的行为。
    """

    name = "scripted"
    model = "scripted-1"

    def __init__(self, script: list[Completion], conclusion: str = "基于已有信息：10 株。") -> None:
        self.script = script
        self.conclusion = conclusion
        self.i = 0
        self.seen: list[list[Message]] = []

    def complete(self, messages, tools):
        self.seen.append([Message(m.role, m.content, list(m.tool_calls), m.tool_call_id, m.name) for m in messages])
        # 收束请求的特征是最后一条 user 消息在要求"不要再调工具"
        last = messages[-1]
        if last.role == "user" and "上限" in (last.content or ""):
            return Completion(content=self.conclusion, finish_reason="stop")
        if self.i >= len(self.script):
            return Completion(content="（脚本耗尽）", finish_reason="stop")
        out = self.script[self.i]
        self.i += 1
        return out


def detect_call(cid="c1", args=None) -> Completion:
    return Completion(
        content=None,
        tool_calls=[ToolCall(id=cid, name="detect_cabbage", arguments=json.dumps(args or {"image_path": "x.jpg"}))],
        finish_reason="tool_calls",
    )


class TestLoopBasics(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry(build_tools(FakeEngine(count=48)))

    def test_single_tool_then_answer(self):
        prov = ScriptedProvider([detect_call(), Completion(content="共 48 株。")])
        traj = Agent(prov, self.reg).run("有多少株？")
        self.assertEqual(traj.answer, "共 48 株。")
        self.assertEqual(traj.n_tool_calls, 1)
        self.assertEqual(traj.stopped_reason, "stop")
        self.assertEqual(traj.steps[0].observation["count"], 48)
        self.assertTrue(traj.steps[0].ok)

    def test_tool_observation_is_json_in_context(self):
        prov = ScriptedProvider([detect_call(), Completion(content="ok")])
        Agent(prov, self.reg).run("q")
        tool_msgs = [m for m in prov.seen[1] if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        payload = json.loads(tool_msgs[0].content)
        self.assertEqual(payload["count"], 48)
        self.assertEqual(tool_msgs[0].tool_call_id, "c1")

    def test_no_tool_calls_stops_immediately(self):
        prov = ScriptedProvider([Completion(content="不需要工具。")])
        traj = Agent(prov, self.reg).run("你好")
        self.assertEqual(traj.n_tool_calls, 0)
        self.assertEqual(traj.answer, "不需要工具。")

    def test_usage_accumulates(self):
        a = detect_call("c1")
        a.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        b = Completion(content="done", usage={"prompt_tokens": 20, "completion_tokens": 7})
        traj = Agent(ScriptedProvider([a, b]), self.reg).run("q")
        self.assertEqual(traj.usage["prompt_tokens"], 30)
        self.assertEqual(traj.usage["completion_tokens"], 12)


class TestErrorPropagation(unittest.TestCase):
    def test_tool_error_reaches_model_and_loop_continues(self):
        reg = ToolRegistry(build_tools(FakeEngine()))
        bad = Completion(
            content=None,
            tool_calls=[ToolCall(id="c1", name="detect_cabbage", arguments="{}")],  # 缺参数
            finish_reason="tool_calls",
        )
        prov = ScriptedProvider([bad, Completion(content="参数缺失，已说明。")])
        traj = Agent(prov, reg).run("q")
        self.assertEqual(traj.n_failed_calls, 1)
        err_payload = json.loads([m for m in prov.seen[1] if m.role == "tool"][0].content)
        self.assertEqual(err_payload["error"]["code"], "INVALID_ARG")
        self.assertEqual(traj.answer, "参数缺失，已说明。")

    def test_unknown_tool_does_not_crash_loop(self):
        reg = ToolRegistry([])
        bad = Completion(content=None, tool_calls=[ToolCall(id="c9", name="nope", arguments="{}")],
                         finish_reason="tool_calls")
        traj = Agent(ScriptedProvider([bad, Completion(content="工具不存在")]), reg).run("q")
        self.assertFalse(traj.steps[0].ok)
        self.assertIn("UNKNOWN_TOOL", json.dumps(traj.steps[0].observation))


class TestMaxSteps(unittest.TestCase):
    def test_forces_conclusion_when_budget_exhausted(self):
        reg = ToolRegistry(build_tools(FakeEngine(count=10)))
        prov = ScriptedProvider([detect_call(f"c{i}") for i in range(4)])
        traj = Agent(prov, reg, max_steps=3).run("q")
        self.assertEqual(traj.stopped_reason, "max_steps")
        self.assertEqual(traj.n_tool_calls, 3)
        self.assertIn("10", traj.answer)

    def test_conclude_request_sends_no_tools(self):
        reg = ToolRegistry(build_tools(FakeEngine()))
        prov = ScriptedProvider([detect_call(f"c{i}") for i in range(3)])
        Agent(prov, reg, max_steps=2).run("q")
        # 最后一次调用的消息里应有一句"不要再请求调用工具"
        last = prov.seen[-1]
        self.assertIn("上限", last[-1].content)


class TestHistoryCompaction(unittest.TestCase):
    def test_old_observations_compacted_but_structure_kept(self):
        # 8 次调用 > MAX_HISTORY_OBSERVATIONS(6)，最早的若干条应被压缩
        reg = ToolRegistry(build_tools(FakeEngine(count=48)))
        calls = [detect_call(f"c{i}") for i in range(8)]
        prov = ScriptedProvider(calls)
        Agent(prov, reg, max_steps=8).run("q")

        final_msgs = prov.seen[-1]
        tool_msgs = [m for m in final_msgs if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 8)
        # 所有 tool 消息都必须保留 tool_call_id 且是合法 JSON
        for m in tool_msgs:
            self.assertTrue(m.tool_call_id)
            json.loads(m.content)
        # 最早的若干条应被压缩
        compacted = [m for m in tool_msgs if json.loads(m.content).get("_compacted")]
        self.assertTrue(compacted, "超过预算后应出现压缩标记")
        # 最近一条必须保持完整，否则模型看不到刚查到的结果
        self.assertFalse(json.loads(tool_msgs[-1].content).get("_compacted"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
