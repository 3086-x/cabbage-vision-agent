"""流式契约测试。

约束只有一条：无论供应商怎么切块，回调拼接必须等于最终 content；
有 tool_calls 时不能因为流式而丢调用。
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvagent.agent.loop import Agent  # noqa: E402
from cvagent.llm.base import Message  # noqa: E402
from cvagent.llm.mock import MockProvider  # noqa: E402
from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import build_tools  # noqa: E402

from test_tools import FakeEngine  # noqa: E402


class TestMockStream(unittest.TestCase):
    def test_deltas_concatenate_to_answer(self):
        pieces: list[str] = []
        reg = ToolRegistry(build_tools(FakeEngine(count=48, n_low=3)))
        traj = Agent(MockProvider(), reg).run("这张图有多少株？\n图片路径：x.jpg",
                                             on_delta=pieces.append)
        self.assertTrue(pieces)
        self.assertEqual("".join(pieces), traj.answer)
        self.assertTrue(len(pieces) > 1, "应按块回调，而不是一次性给整段")

    def test_non_stream_path_unchanged(self):
        reg = ToolRegistry(build_tools(FakeEngine(count=48, n_low=3)))
        traj = Agent(MockProvider(), reg).run("这张图有多少株？\n图片路径：x.jpg")
        self.assertIn("48", traj.answer)


class _FakeStream:
    """模拟 SSE：content 分片 + tool_calls 分片 + usage。"""

    def __init__(self, lines):
        self.lines = lines
        self.status_code = 200
        self.text = ""

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)


class _StreamRequests(types.ModuleType):
    RequestException = Exception

    def __init__(self, lines):
        super().__init__("requests")
        self.lines = lines

    def post(self, *a, **kw):
        return _FakeStream(self.lines)


class TestOpenAIStreamParsing(unittest.TestCase):
    def _run(self, lines, on_delta):
        from cvagent.llm.openai_compat import OpenAICompatProvider

        sys.modules["requests"] = _StreamRequests(lines)
        try:
            prov = OpenAICompatProvider("https://example.invalid/v1", "sk-test", "m")
            return prov.complete([Message("user", "hi")], [], on_delta=on_delta)
        finally:
            sys.modules.pop("requests", None)

    def test_content_deltas(self):
        got = []
        lines = [
            'data: {"choices":[{"delta":{"content":"共 "}}]}',
            'data: {"choices":[{"delta":{"content":"48 株"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2}}',
            "data: [DONE]",
        ]
        comp = self._run(lines, got.append)
        self.assertEqual("".join(got), "共 48 株")
        self.assertEqual(comp.content, "共 48 株")
        self.assertEqual(comp.usage["prompt_tokens"], 9)

    def test_tool_call_fragments_merge_by_index(self):
        got = []
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"detect_cabbage","arguments":"{\\"image"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"_path\\": \\"x.jpg\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        comp = self._run(lines, got.append)
        self.assertEqual(len(comp.tool_calls), 1)
        self.assertEqual(comp.tool_calls[0].name, "detect_cabbage")
        self.assertEqual(json.loads(comp.tool_calls[0].arguments)["image_path"], "x.jpg")
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
