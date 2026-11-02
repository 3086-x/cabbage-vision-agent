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

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ChatProvider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion: ...
