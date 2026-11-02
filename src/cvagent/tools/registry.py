"""工具注册与分发。

分发层是三件事的唯一入口，Agent 循环本身不碰异常处理：
    1. 参数校验（缺参、类型错、越界）→ INVALID_ARG，可纠正
    2. 执行并捕获异常 → 结构化 payload
    3. 观测截断 → 控制在字符预算内

截断放在这里而不是循环里，因为只有这里知道"这个工具的返回结构长什么样"。
"""

from __future__ import annotations

import json
from typing import Any

from .spec import Tool
from ..config import MAX_TOOL_CHARS
from ..errors import InvalidArgument, ToolError, UnknownTool

_KEY_FIELDS = ("ok", "error", "count", "total_count", "mean_count", "density", "image_id", "comparison", "n_processed", "note")


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    # ---- 暴露给 LLM ----
    def schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "required": t.parameters.get("required", []),
                "tags": list(t.tags),
            }
            for t in self._tools.values()
        ]

    # ---- 分发 ----
    def dispatch(self, name: str, raw_args: str | dict | None) -> dict:
        """永不抛异常。所有失败都变成 {"ok": false, "error": {...}}。"""
        tool = self._tools.get(name)
        if tool is None:
            return UnknownTool(name, self.names()).to_payload()

        try:
            args = self._parse_args(raw_args)
        except ToolError as e:
            return e.to_payload()

        try:
            self._validate(tool, args)
            result = tool.handler(**args)
        except ToolError as e:
            return e.to_payload()
        except TypeError as e:  # 参数签名不匹配
            return InvalidArgument("args", list(args), str(e)).to_payload()
        except Exception as e:  # noqa: BLE001 - 兜底，绝不让循环崩掉
            return {
                "ok": False,
                "error": {
                    "code": "TOOL_CRASH",
                    "message": f"{type(e).__name__}: {e}",
                    "retryable": False,
                    "hint": "该工具内部异常，换一种方式完成任务或直接说明无法处理。",
                },
            }

        return self._truncate(result, tool.max_chars or MAX_TOOL_CHARS)

    # ---- 内部 ----
    @staticmethod
    def _parse_args(raw: str | dict | None) -> dict:
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise InvalidArgument("arguments", raw[:80], f"不是合法 JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise InvalidArgument("arguments", type(parsed).__name__, "必须是 JSON 对象")
        return parsed

    @staticmethod
    def _validate(tool: Tool, args: dict) -> None:
        props: dict[str, Any] = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        missing = [k for k in required if k not in args]
        if missing:
            raise InvalidArgument("required", missing, f"{tool.name} 缺少必需参数")

        unknown = [k for k in args if k not in props]
        if unknown:
            raise InvalidArgument("unknown", unknown, f"{tool.name} 不接受这些参数")

        for key, value in args.items():
            spec = props[key]
            kind = spec.get("type")
            if kind == "number" and not isinstance(value, (int, float)):
                raise InvalidArgument(key, value, "应为数字")
            if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                raise InvalidArgument(key, value, "应为整数")
            if kind == "string" and not isinstance(value, str):
                raise InvalidArgument(key, value, "应为字符串")
            if kind in ("number", "integer"):
                if "minimum" in spec and value < spec["minimum"]:
                    raise InvalidArgument(key, value, f"不得小于 {spec['minimum']}")
                if "maximum" in spec and value > spec["maximum"]:
                    raise InvalidArgument(key, value, f"不得大于 {spec['maximum']}")
            if "enum" in spec and value not in spec["enum"]:
                raise InvalidArgument(key, value, f"只能取 {spec['enum']}")

    @staticmethod
    def _truncate(result: dict, budget: int) -> dict:
        """超预算时丢掉长列表，保留结论字段，并明确告知被截断。"""
        text = json.dumps(result, ensure_ascii=False)
        if len(text) <= budget:
            return result

        slim = {k: v for k, v in result.items() if k in _KEY_FIELDS}
        dropped = [k for k in result if k not in _KEY_FIELDS]
        slim["truncated"] = True
        slim["dropped_fields"] = dropped
        slim["note_truncation"] = (
            f"原始返回 {len(text)} 字符超出预算，已丢弃 {dropped}。"
            "若确实需要明细，请缩小范围（如 detail=summary 或减少 limit）后重新调用。"
        )
        return slim


def default_registry(engine=None) -> ToolRegistry:
    """构造带默认推理后端的注册表。engine 可注入以隔离测试。"""
    if engine is None:
        from ..vision.engine import YoloEngine

        engine = YoloEngine()
    from .spec import build_tools

    return ToolRegistry(build_tools(engine))
