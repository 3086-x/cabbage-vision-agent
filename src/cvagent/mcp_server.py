"""把工具表暴露成 MCP 服务器（stdio 传输）。

## 这一层解决什么

前面几层解决的是"能力"：能测、能问、能在界面上看见框。这一层解决的是"别人能不能
用"——MCP 之后，任何支持它的客户端（Claude Desktop、Cursor、IDE 插件）都能把
这个项目当成一组原生工具，不需要知道它有没有 CLI、有没有 Web 界面。

## 三个必须在这一层做对的事

**1. 工具表只有一份。**
`tools/list` 的 schema 直接从 `Tool` 取，不另写一份给 MCP 用。喂给模型的
`description` 就是"什么时候该调用我"的判据——两份描述一漂移，同一个问题在两个
客户端里就会走出两种行为，而且**不会报错**。这个项目在别处已经踩过同类的坑
（评测的禁语正则、抽取与抹除的路径正则）。

**2. stdout 只放协议消息。**
stdio 传输是一行一个 JSON-RPC。任何一句残留的 `print` 都会破坏帧边界，客户端
表现是"连上了但不回话"，排查极费时间。所以启动时把真正的 stdout 收进变量自己写，
再把 `sys.stdout` 指向 stderr：库里的 print 顶多污染日志，不会污染协议。

**3. 权重不能在这里加载。**
`initialize` 必须秒回，客户端等二三十秒就会判定启动失败。注册表在第一次真正需要
时才构造；而权重更晚——`YoloEngine` 的模型是首次推理才加载的，所以 `tools/list`
不会碰 GPU，第一次 `tools/call` 才付这个代价。

## 刻意没做的

- **没有 resources / prompts / sampling。** 这个项目的能力全在"动作"里，硬凑成
  资源只是把 JSON 换个壳。
- **不做 `listChanged` 订阅。** 工具表是编译期固定的，发个 never-fired 的通知不如不发。
- **不处理 JSON-RPC 批量请求。** MCP 不用它，实现了也没有调用方能验证。
- **不并发。** stdio 是单连接顺序流；要并发起 `cvagent serve`，那层的锁只包推理。
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Iterable

from . import __version__
from .tools.registry import ToolRegistry

SERVER_NAME = "cabbage-vision"

# 收客户端要的版本：支持就照回，不支持就回自己最新的。不猜、不改写参数。
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
FALLBACK_PROTOCOL = PROTOCOL_VERSIONS[0]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class RpcError(Exception):
    """协议层错误。工具内部的失败不算——那些是正常返回里的 `ok: false`。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(mid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _content(payload: dict) -> list[dict]:
    return [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]


class McpServer:
    """把 `handle()` 做成纯函数，传输层单独一个循环——测试不用起管道。"""

    def __init__(self, registry_factory: Callable[[], ToolRegistry]) -> None:
        self._factory = registry_factory
        self._registry: ToolRegistry | None = None

    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            self._registry = self._factory()
        return self._registry

    # ---- 单条消息 ----
    def handle(self, msg: Any) -> dict | None:
        if not isinstance(msg, dict):
            return _error(None, INVALID_REQUEST, "消息必须是 JSON 对象")
        mid = msg.get("id")
        method = msg.get("method")
        if not isinstance(method, str):
            # 客户端发来的 response（对 server→client 请求的回复）不处理；
            # 其它没有 method 的东西是坏消息，但如果是通知就没什么可回的。
            return _error(mid, INVALID_REQUEST, "缺少 method") if "id" in msg else None
        if "id" not in msg:  # 通知一律不回
            return None

        try:
            if method == "initialize":
                result = self._initialize(msg.get("params") or {})
            elif method == "tools/list":
                result = {"tools": self._tool_list()}
            elif method == "tools/call":
                result = self._tool_call(msg.get("params") or {})
            elif method in ("ping", "shutdown"):
                result = {}
            else:
                raise RpcError(METHOD_NOT_FOUND, f"不支持的方法：{method}")
        except RpcError as e:
            return _error(mid, e.code, str(e))
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    # ---- 方法实现 ----
    def _initialize(self, params: dict) -> dict:
        wanted = params.get("protocolVersion")
        return {
            "protocolVersion": wanted if wanted in PROTOCOL_VERSIONS else FALLBACK_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            # 客户端通常会把这段说明塞进系统提示。写"怎么用"而不是"我是什么"：
            # 工具描述已经写了能力，这里补的是路由建议。
            "instructions": (
                "甘蓝航拍图分析。工具都吃本机图片路径（image_path）："
                "只要株数用 count_cabbage，要看密度/分布/框尺寸用 detect_cabbage"
                "（空间分布只能引用它的 grid 字段），批量看目录用 batch_detect，"
                "先判断图能不能用用 inspect_image。计数口径与局限可用 search_docs 查。"
            ),
        }

    def _tool_list(self) -> list[dict]:
        out = []
        for schema in self.registry.schemas():
            fn = schema["function"]
            out.append({
                "name": fn["name"],
                "description": fn["description"],
                "inputSchema": fn["parameters"],
            })
        return out

    def _tool_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise RpcError(INVALID_PARAMS, "tools/call 缺少 name")
        if not isinstance(args, dict):
            raise RpcError(INVALID_PARAMS, "arguments 必须是 JSON 对象")
        if name not in self.registry.names():
            raise RpcError(INVALID_PARAMS, f"没有这个工具：{name}")

        # 参数校验与错误语义交给 dispatch：只在那一处定义，MCP 客户端自己校验
        # 一遍也无所谓，两边口径同源就不会出现"客户端放行、工具拒绝"。
        obs = self.registry.dispatch(name, args)
        # 工具失败走 isError，而不是 JSON-RPC error：协议层的 error 意味着
        # 这次请求本身不成立，客户端不一定会把内容交给模型；而这里模型**应该**
        # 看到 INVALID_ARG / NO_DETECTION 才能改参数重试，这正是循环的价值。
        return {"content": _content(obs), "isError": not obs.get("ok", False)}


def serve_stdio(
    registry_factory: Callable[[], ToolRegistry],
    stdin: Iterable[str] | None = None,
    stdout: Any = None,
    warmup: bool = False,
) -> int:
    """一行一条 JSON-RPC，读到 EOF 结束。stdout 只写协议。"""
    server = McpServer(registry_factory)
    out = stdout if stdout is not None else sys.stdout
    src = stdin if stdin is not None else sys.stdin

    if warmup:
        # 只在显式要求时做：默认让客户端先拿到 tools/list，别在 initialize 上等 28 秒。
        server.registry

    for line in src:
        line = line.strip()
        if not line:
            continue
        msg: Any = None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp: dict | None = _error(None, PARSE_ERROR, f"不是合法 JSON：{e}")
        else:
            resp = server.handle(msg)

        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
        if isinstance(msg, dict) and msg.get("method") == "shutdown":
            break
    return 0


def main(registry_factory: Callable[[], ToolRegistry], warmup: bool = False) -> int:
    """给 CLI 用的入口：先把 stdout 抢过来，再做任何别的事。

    顺序不能反。只要在抢之前有库 print 过一行，协议流里就多了一帧垃圾，
    而且这一帧不会报错，只会让客户端在解析到它时静默失联。
    """
    proto_out = sys.stdout
    sys.stdout = sys.stderr
    print(f"[mcp] {SERVER_NAME} 通过 stdio 提供工具，日志走 stderr", file=sys.stderr, flush=True)
    return serve_stdio(registry_factory, stdout=proto_out, warmup=warmup)
