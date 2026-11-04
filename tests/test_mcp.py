"""MCP 服务器的契约测试。不加载权重、不起进程、不碰 GPU。

这一层锁的是**协议形状**和两条容易静默失效的约定：

1. **工具表只有一份。** `tools/list` 必须和 `tools/spec.py` 里写的是同一份
   description / schema。分成两份之后，模型在 MCP 客户端里看到的使用条件会和
   在 CLI / Web 里看到的不一样，而且没有任何报错——只是行为不同。
2. **工具失败不是协议失败。** 参数越界要走 `isError: true` 的正常返回，让客户端
   里的模型看到 `INVALID_ARG` 后改参数重试；写成 JSON-RPC error，客户端不一定会
   把内容交给模型，"能自我纠正"这件事就没了。
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.errors import InvalidArgument  # noqa: E402
from cvagent.mcp_server import (  # noqa: E402
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    McpServer,
    serve_stdio,
)
from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import (  # noqa: E402
    Tool,
    _CONF,
    _IMAGE_PATH,
    _schema,
    build_tools,
)


def _fake_registry() -> ToolRegistry:
    """两件工具：一个总是成功，一个参数越界就抛 InvalidArgument。

    不注入真引擎：这里要验的是协议与错误映射，推理有它自己的测试。
    """

    def count_cabbage(image_path: str, conf: float = 0.25) -> dict:
        return {"ok": True, "count": 62, "image_path": image_path, "conf_threshold": conf}

    def detect_cabbage(image_path: str, conf: float = 0.25) -> dict:
        raise InvalidArgument("conf", conf, "不得大于 1.0")

    return ToolRegistry([
        Tool(
            name="count_cabbage",
            description="数这张图里有多少株甘蓝。",
            parameters=_schema({"image_path": _IMAGE_PATH, "conf": _CONF}, ["image_path"]),
            handler=count_cabbage,
        ),
        Tool(
            name="detect_cabbage",
            description="检测并返回密度与分布。",
            parameters=_schema({"image_path": _IMAGE_PATH, "conf": _CONF}, ["image_path"]),
            handler=detect_cabbage,
        ),
    ])


def _server() -> McpServer:
    return McpServer(_fake_registry)


def _rpc(server: McpServer, method: str, params: dict | None = None, mid: int = 1) -> dict:
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


class TestHandshake(unittest.TestCase):
    def test_initialize_reports_capabilities(self):
        r = _rpc(_server(), "initialize", {"protocolVersion": "2025-06-18"})
        self.assertEqual(r["id"], 1)
        res = r["result"]
        self.assertEqual(res["protocolVersion"], "2025-06-18")
        self.assertEqual(res["capabilities"]["tools"]["listChanged"], False)
        self.assertEqual(res["serverInfo"]["name"], "cabbage-vision")
        self.assertTrue(res["serverInfo"]["version"])

    def test_unknown_protocol_falls_back(self):
        # 不回显客户端瞎写的版本：那句回显等于宣称"我能说这个版本"。
        r = _rpc(_server(), "initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")

    def test_notification_gets_no_reply(self):
        # notifications/initialized 没有 id，回一帧就是协议违规。
        self.assertIsNone(_server().handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_is_an_error(self):
        r = _rpc(_server(), "resources/list")
        self.assertEqual(r["error"]["code"], METHOD_NOT_FOUND)


class TestToolList(unittest.TestCase):
    def test_schema_comes_from_spec(self):
        r = _rpc(_server(), "tools/list")
        tools = {t["name"]: t for t in r["result"]["tools"]}
        self.assertEqual(set(tools), {"count_cabbage", "detect_cabbage"})
        t = tools["count_cabbage"]
        self.assertEqual(set(t["inputSchema"]["properties"]), {"image_path", "conf"})
        self.assertEqual(t["inputSchema"]["required"], ["image_path"])
        # MCP 用 inputSchema，OpenAI 用 parameters：键名不同，内容必须同源
        self.assertEqual(t["description"], "数这张图里有多少株甘蓝。")

    def test_real_spec_has_no_duplicate_names(self):
        # 用真工具表跑一遍映射：名字重复会让客户端只看到其中一个，且不报错。
        class _Null:
            pass

        names = [t.name for t in build_tools(_Null())]
        self.assertEqual(len(names), len(set(names)))


class TestToolCall(unittest.TestCase):
    def test_success_returns_text_content(self):
        r = _rpc(_server(), "tools/call",
                 {"name": "count_cabbage", "arguments": {"image_path": "D:/a.jpg"}})
        res = r["result"]
        self.assertEqual(res["isError"], False)
        self.assertEqual(res["content"][0]["type"], "text")
        payload = json.loads(res["content"][0]["text"])
        self.assertEqual(payload["count"], 62)

    def test_tool_failure_is_content_not_rpc_error(self):
        r = _rpc(_server(), "tools/call",
                 {"name": "detect_cabbage", "arguments": {"image_path": "D:/a.jpg", "conf": 0.5}})
        self.assertNotIn("error", r)          # 协议层没有错
        self.assertEqual(r["result"]["isError"], True)
        payload = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(payload["error"]["code"], "INVALID_ARG")
        self.assertTrue(payload["error"]["hint"])   # 模型靠 hint 知道怎么改

    def test_unknown_tool_is_a_request_error(self):
        r = _rpc(_server(), "tools/call", {"name": "rm_rf", "arguments": {}})
        self.assertEqual(r["error"]["code"], INVALID_PARAMS)

    def test_missing_args_surfaces_as_invalid_arg(self):
        r = _rpc(_server(), "tools/call", {"name": "count_cabbage", "arguments": {}})
        self.assertEqual(r["result"]["isError"], True)
        payload = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(payload["error"]["code"], "INVALID_ARG")


class TestTransport(unittest.TestCase):
    def _run(self, lines: list[str]) -> list[dict]:
        out = io.StringIO()
        serve_stdio(_fake_registry, stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
        return [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]

    def test_one_reply_per_request_and_nothing_else(self):
        # 帧边界：多一个空行、多一句日志，客户端就解析不到下一条。
        replies = self._run([
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            '{"jsonrpc":"2.0","method":"notifications/initialized"}',
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
        ])
        self.assertEqual([r["id"] for r in replies], [1, 2])

    def test_bad_json_is_answered_not_crashed(self):
        replies = self._run(["{不是 json"])
        self.assertEqual(replies[0]["error"]["code"], PARSE_ERROR)
        self.assertIsNone(replies[0]["id"])

    def test_blank_lines_are_skipped(self):
        replies = self._run(["", "   ", '{"jsonrpc":"2.0","id":9,"method":"ping"}'])
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["result"], {})

    def test_shutdown_ends_the_stream(self):
        replies = self._run([
            '{"jsonrpc":"2.0","id":1,"method":"shutdown"}',
            '{"jsonrpc":"2.0","id":2,"method":"ping"}',
        ])
        self.assertEqual(len(replies), 1)

    def test_registry_is_built_lazily(self):
        # initialize / tools/list 之前不构造注册表：真跑起来时那一步要加载 torch。
        calls = []

        def factory():
            calls.append(1)
            return _fake_registry()

        server = McpServer(factory)
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(calls, [])
        server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(calls, [1])   # 只构造一次，不重复加载
        server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
