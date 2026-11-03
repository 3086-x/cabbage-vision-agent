"""对话抽象。

刻意**不引入 langchain**。理由不是"手写显得厉害"，而是：
  - 这个项目的全部需求是 "多轮消息 + function calling"，一个 120 行的
    OpenAI 兼容客户端就能覆盖，链式框架反而把可调试性变差；
  - 供应商差异（字段名、错误码、tool_calls 格式）集中在一个文件里，
    换模型只改这里，agent 循环/工具层完全不动。

抽象边界只有三样：Message、ToolCall、Completion。任何 provider 只要能产出
Completion，就能接进循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，由工具层负责解析与容错


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict:
        """转成 OpenAI 兼容格式。空字段一律不发，避免某些供应商报错。"""
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content or ""}
        msg: dict = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        return msg


@dataclass
class Completion:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    model: str = ""
    # 本次调用在拿到 200 之前经历的重试次数。token/延迟之外，重试是第三个
    # 成本来源：限流时它会把一次调用的 wall time 拉长数倍，但不出现在 usage 里。
    retries: int = 0

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(RuntimeError):
    """供应商侧失败。

    code / hint 存在的理由：HTTP 状态码本身不含行动信息——401 和 403 都可能是
    "key 错了"，也可能是"额度没了"，而这两件事的处理方式完全不同。把供应商
    自己的错误码归一化成一个稳定的 code，再附一句能直接照做的 hint，
    调用方（doctor、评测脚本）就不需要各自维护一张错误码对照表。
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 code: str | None = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.code = code
        self.hint = hint

    @property
    def action(self) -> str:
        """人可直接照做的下一步。"""
        return self.hint or ""


class ChatProvider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], tools: list[dict], on_delta=None) -> Completion: ...
