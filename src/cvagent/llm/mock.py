"""离线回放 provider。

它不是为了"假装有 LLM"，而是为了让**决策链本身可测试**：

  - 工具契约、错误传播、观测截断、循环终止，这些逻辑与模型无关，
    用固定脚本驱动就能断言，不需要网络也不需要花 token；
  - 新工具加进来时，先在这里跑通，再接真模型，能把"工具坏了"和
    "模型不会用"两类问题分开。

脚本策略（对应真实模型在这个任务上应有的行为）：
  1. 首轮：调 detect_cabbage 拿到统计
  2. 若低置信框 > 0：调 refine_detection 降阈值复核
  3. 复核后：给出结论
"""

from __future__ import annotations

import json
import re

from .base import Completion, Message, ToolCall

_PATH_RE = re.compile(r"[\w./\\:-]+\.(?:jpg|jpeg|png|bmp|tif|tiff|webp)", re.IGNORECASE)


class MockProvider:
    """按脚本产出 tool_calls。同一实例内记住进度。"""

    name = "mock"

    def __init__(self, model: str = "mock-scripted", refine_conf: float = 0.15) -> None:
        self.model = model
        self.refine_conf = refine_conf
        self.stage = 0

    def complete(self, messages: list[Message], tools: list[dict], on_delta=None) -> Completion:
        comp = self._complete(messages, tools)
        if on_delta and comp.content:
            # 按 10 字符切块，模拟流式。真实 provider 的切块由供应商决定，
            # 契约只要求「拼起来等于最终 content」。
            for i in range(0, len(comp.content), 10):
                on_delta(comp.content[i:i + 10])
        return comp

    def _complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        if self.stage == 0:
            self.stage = 1
            path = self._extract_path(messages)
            return Completion(
                content=None,
                tool_calls=[ToolCall(id="call_detect_1", name="detect_cabbage",
                                     arguments=json.dumps({"image_path": path, "conf": 0.25}, ensure_ascii=False))],
                finish_reason="tool_calls",
            )

        last = self._last_tool_payload(messages)

        if last.get("ok") is False:
            # 真实模型此时应读 hint 决定是否重试；mock 直接终止并说明原因
            err = last.get("error", {})
            return Completion(
                content=f"无法完成：{err.get('message')}（{err.get('code')}）。建议：{err.get('hint', '无')}",
                finish_reason="stop",
            )

        n_low = (last.get("conf_stats") or {}).get("n_low", 0)
        already_refined = any(
            m.role == "tool" and m.name == "refine_detection" for m in messages
        )

        if n_low > 0 and not already_refined and last.get("image_id"):
            self.stage = 2
            return Completion(
                content=None,
                tool_calls=[ToolCall(
                    id="call_refine_1",
                    name="refine_detection",
                    arguments=json.dumps({"image_id": last["image_id"], "conf": self.refine_conf}, ensure_ascii=False),
                )],
                finish_reason="tool_calls",
            )

        return Completion(content=self._conclude(last, refined=already_refined), finish_reason="stop")

    # ---- helpers ----
    @staticmethod
    def _extract_path(messages: list[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user" and m.content:
                found = _PATH_RE.search(m.content)
                if found:
                    return found.group(0)
        return ""

    @staticmethod
    def _last_tool_payload(messages: list[Message]) -> dict:
        for m in reversed(messages):
            if m.role == "tool" and m.content:
                try:
                    return json.loads(m.content)
                except json.JSONDecodeError:
                    return {"ok": False, "error": {"code": "BAD_JSON", "message": m.content[:120]}}
        return {"ok": False, "error": {"code": "NO_OBSERVATION", "message": "没有工具结果"}}

    @staticmethod
    def _conclude(last: dict, *, refined: bool) -> str:
        count = last.get("count")
        density = last.get("density")
        cmp_ = last.get("comparison")
        if cmp_:
            before, after = cmp_["before"], cmp_["after"]
            return (
                f"该图共 {after['count']} 株，密度 {density}。"
                f"在 conf={before['conf']} 时检出 {before['count']} 株，"
                f"降到 conf={after['conf']} 后为 {after['count']} 株"
                f"（差 {cmp_['delta']:+d}，稳定性 {last.get('stability', 'unknown')}）。"
                f"结论取复核后的计数。"
            )
        if count is None:
            return "检测未返回有效结果，无法给出结论。"
        return (
            f"该图共 {count} 株，密度 {density}。"
            + ("低置信框较多，建议复核。" if (last.get("conf_stats") or {}).get("n_low", 0) else "置信度分布正常。")
        )
