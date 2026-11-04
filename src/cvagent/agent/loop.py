"""规划-执行-观察循环。

三个工程细节，都是实战里踩出来的，不是教科书内容：

1. **步数上限必须带收束动作**。到上限直接 return 会得到半句话；正确做法是
   再发一次不带 tools 的请求，强制模型用已有观测作答，并把 stopped_reason
   标为 "max_steps"，让人知道答案是被截断的。

2. **历史观测要压缩，不能删**。删消息会让 tool_call_id 失去配对的 assistant
   消息，多数供应商直接报 400。所以只替换 tool 消息的 content，保留结构。

3. **错误不拦截，交给模型**。工具返回 ok=false 时循环照常继续，由模型读
   hint 决定重试还是换路。人类预设的重试策略在真实任务里经常比模型差。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .. import config
from ..llm.base import ChatProvider, Message, ProviderError, ToolCall
from ..tools.registry import ToolRegistry

SYSTEM_PROMPT = """你是一个甘蓝（cabbage）航拍图像的视觉分析助手。你只能通过工具获取关于图像的事实。

## 工作方式
1 先想清楚用户到底要什么数字或判断，再选工具。
2 只调用必要的工具，避免重复检测同一张图。
3 工具返回 ok=false 时，读 error.hint 判断能否通过改参数解决；能改就改一次再试，
  不能改就如实告知用户，不要编造结果。
4 拿到足够信息后直接给出结论，不要为了凑步骤继续调用工具。

## 工具选择
- 只想了解图片质量（是否太暗/太糊/尺寸）→ inspect_image
- 想知道株数、密度、置信度分布 → detect_cabbage
- 问「分布均不均匀 / 哪块空着 / 哪块稀」→ detect_cabbage，用返回的 `grid` 字段；
  **不要**用 count_cabbage（它不返回网格），也不要用 boxes_head 的 3 个坐标自己分格
- 只需要一个株数、要最省时间 → count_cabbage
- 检测结果里 n_low（低置信框数）> 0，或用户质疑准确性 → refine_detection 换阈值复核
- 用户给的是文件夹 → batch_detect
- 问口径/依据/局限（为什么用这个阈值、密度怎么分级、能不能换算株/亩、评测怎么判）→ search_docs

## 回答要求
- 每个数字都必须来自工具返回，禁止估算或沿用常识。
- **工具没返回的量，直接说没提供，不要自己补。** 比如 `grid` 只有 4×4，
  就不要描述"第 3 行第 5 列"；没有逐株坐标就不要说"左下角那两株"；
  没有标定就不要给株/亩。宁可答"这些数据工具里没有"，也不要给一个像真的数。
- 空间分布只能引用 `grid`：`counts`（行优先，[0][0] 是左上格）、`pattern`、
  `vmr`（方差均值比，随机分布 = 1，就叫这个名字，别替换成"变异系数"）、
  `empty_cells` 与 `empty_if_random`。回答"均匀吗"要给具体格数
  （最密格 X 株、最稀格 Y 株），并说明结论是**相对本图**、不是田间分布——
  图像是裁出来的，两者不是一回事。
- "有空格"不等于"有空地"：如果 `empty_cells` 少于 `empty_if_random`，
  说明空格还没多到随机水平，不能说成缺苗或空地。
- 格坐标直接给行列号（"第 2 行第 1 列"），**不要翻成方位词**（右上/左下/中部）。
  4×4 的"第 3 列"不等于"右侧"，这种转译工具没给依据。
- 引用文档结论时保留出处（`path#heading`），例如 `docs/knowledge/density.md#阈值从哪来`。
- 如果做了阈值复核且两次结果差异超过 5%，必须同时说明两个数字和差异，不要只报一个。
- 密度分级（sparse/moderate/dense/very_dense）是相对本数据集的标签，不是农学绝对值；
  用户问"密不密"时可以给分级，问"多少株/亩"时说明缺少标定无法换算。
- 用中文回答，简洁，先给结论再给依据。"""


@dataclass
class Step:
    index: int
    tool: str
    args: dict
    observation: dict
    ok: bool
    # 第几轮决策产生的。模型可以在一轮里并行请求多个工具，
    # 这时"调用次数"会大于"轮次"——预算该按轮次算，否则并行调用的
    # 高效行为会被当成浪费步数惩罚。
    turn: int = 0
    elapsed_ms: float = 0.0
    # 本轮 provider 调用经历的重试次数。同一轮并行调用的工具共享这个值：
    # 重试发生在拿到响应之前，无法归因到某个具体工具。
    retries: int = 0

    def brief(self) -> str:
        if not self.ok:
            err = self.observation.get("error", {})
            return f"{self.tool} ✗ {err.get('code')}: {err.get('message', '')[:60]}"
        keys = ("count", "total_count", "mean_count", "density", "delta")
        bits = [f"{k}={self.observation[k]}" for k in keys if k in self.observation]
        if "comparison" in self.observation:
            c = self.observation["comparison"]
            bits.append(f"{c['before']['count']}→{c['after']['count']}")
        return f"{self.tool} ✓ " + (", ".join(bits) if bits else "ok")


@dataclass
class TurnStat:
    """一轮决策的开销。工具耗时与 LLM 耗时分开记，避免互相掩盖。"""
    turn: int
    llm_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    tool_ms: float = 0.0
    n_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "llm_ms": round(self.llm_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retries": self.retries,
            "tool_ms": round(self.tool_ms, 1),
            "n_calls": self.n_calls,
        }


@dataclass
class Trajectory:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "stop"
    elapsed_s: float = 0.0
    usage: dict = field(default_factory=dict)
    model: str = ""
    turns: list[TurnStat] = field(default_factory=list)

    @property
    def n_tool_calls(self) -> int:
        return len(self.steps)

    @property
    def n_turns(self) -> int:
        """决策轮数。并行调用多个工具算一轮。"""
        return max((s.turn for s in self.steps), default=-1) + 1

    @property
    def n_failed_calls(self) -> int:
        return sum(1 for s in self.steps if not s.ok)

    def totals(self) -> dict:
        """全链路开销汇总。token 取 provider 返回值，缺失时按 0 计。"""
        return {
            "llm_ms": round(sum(t.llm_ms for t in self.turns), 1),
            "tool_ms": round(sum(t.tool_ms for t in self.turns), 1),
            "wall_ms": round(self.elapsed_s * 1000, 1),
            "prompt_tokens": sum(t.prompt_tokens for t in self.turns),
            "completion_tokens": sum(t.completion_tokens for t in self.turns),
            "retries": sum(t.retries for t in self.turns),
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "n_failed_calls": self.n_failed_calls,
        }

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "stopped_reason": self.stopped_reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "n_tool_calls": self.n_tool_calls,
            "n_turns": self.n_turns,
            "n_failed_calls": self.n_failed_calls,
            "usage": self.usage,
            "totals": self.totals(),
            "turns": [t.to_dict() for t in self.turns],
            "steps": [
                {"index": s.index, "turn": s.turn, "tool": s.tool, "args": s.args, "ok": s.ok,
                 "elapsed_ms": round(s.elapsed_ms, 1), "retries": s.retries,
                 "observation": s.observation}
                for s in self.steps
            ],
        }


class Agent:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        *,
        max_steps: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = False,
        on_step=None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps or config.MAX_STEPS
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.on_step = on_step

    def run(self, question: str, *, extra_context: str = "", on_delta=None) -> Trajectory:
        t0 = time.perf_counter()
        user_content = question if not extra_context else f"{question}\n\n[附加上下文]\n{extra_context}"
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]
        steps: list[Step] = []
        turns: list[TurnStat] = []
        usage_total: dict = {}
        stopped = "stop"
        model = getattr(self.provider, "model", "")

        for i in range(self.max_steps):
            ts = TurnStat(turn=i)
            llm_t0 = time.perf_counter()
            if on_delta is None:
                comp = self.provider.complete(messages, self.tools.schemas())
            else:
                comp = self.provider.complete(messages, self.tools.schemas(), on_delta=on_delta)
            ts.llm_ms = (time.perf_counter() - llm_t0) * 1000
            ts.retries = getattr(comp, "retries", 0)
            ts.prompt_tokens = int((comp.usage or {}).get("prompt_tokens", 0) or 0)
            ts.completion_tokens = int((comp.usage or {}).get("completion_tokens", 0) or 0)
            turns.append(ts)
            model = comp.model or model
            _acc_usage(usage_total, comp.usage)

            if not comp.wants_tool:
                return Trajectory(
                    question=question, answer=(comp.content or "").strip(),
                    steps=steps, stopped_reason="stop", turns=turns,
                    elapsed_s=time.perf_counter() - t0, usage=usage_total, model=model,
                )

            messages.append(Message(role="assistant", content=comp.content, tool_calls=comp.tool_calls))

            for call in comp.tool_calls:
                st = time.perf_counter()
                obs = self.tools.dispatch(call.name, call.arguments)
                elapsed = (time.perf_counter() - st) * 1000
                ts.tool_ms += elapsed
                ts.n_calls += 1
                args = _safe_json(call.arguments)
                step = Step(index=len(steps), tool=call.name, args=args,
                            observation=obs, ok=bool(obs.get("ok")), elapsed_ms=elapsed,
                            turn=i, retries=ts.retries)
                steps.append(step)
                messages.append(Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=json.dumps(obs, ensure_ascii=False),
                ))
                if self.on_step:
                    self.on_step(step)
                elif self.verbose:
                    print(f"  [{step.index + 1}] {step.brief()}  ({elapsed:.0f}ms)")

            messages = _compact_history(messages)

        # 达到上限：强制收束，不要留下半句话
        stopped = "max_steps"
        conclude = TurnStat(turn=len(turns))
        c0 = time.perf_counter()
        final = self._force_conclude(messages)
        conclude.llm_ms = (time.perf_counter() - c0) * 1000
        turns.append(conclude)
        return Trajectory(
            question=question,
            answer=final,
            steps=steps,
            stopped_reason=stopped,
            elapsed_s=time.perf_counter() - t0,
            usage=usage_total,
            model=model,
            turns=turns,
        )

    def _force_conclude(self, messages: list[Message]) -> str:
        """不带 tools 再问一次，逼模型基于已有观测作答。"""
        messages = messages + [Message(
            role="user",
            content="已达到工具调用上限。请仅根据上面已获得的信息，直接给出结论；"
                    "如果信息不足，明确说明缺什么，不要再请求调用工具。",
        )]
        try:
            comp = self.provider.complete(messages, tools=[])
            if comp.content and comp.content.strip():
                return comp.content.strip()
        except ProviderError as e:
            return f"（已达到工具调用上限，且收束请求失败：{e}）"
        return "（已达到工具调用上限，未能得出结论）"


def _compact_history(messages: list[Message]) -> list[Message]:
    """把较早的 tool 观测替换成一行摘要，保留消息结构。

    只替换 content，不动 role/tool_call_id，避免破坏配对。
    """
    keep = config.MAX_HISTORY_OBSERVATIONS
    tool_idx = [i for i, m in enumerate(messages) if m.role == "tool"]
    if len(tool_idx) <= keep:
        return messages

    for i in tool_idx[:-keep]:
        m = messages[i]
        try:
            payload = json.loads(m.content or "{}")
        except json.JSONDecodeError:
            payload = {}
        if payload.get("ok"):
            summary = {k: payload[k] for k in ("count", "density", "image_id", "total_count") if k in payload}
            m.content = json.dumps({"ok": True, "_compacted": True, "summary": summary}, ensure_ascii=False)
        else:
            m.content = json.dumps({"ok": False, "_compacted": True,
                                    "error": payload.get("error", {})}, ensure_ascii=False)
    return messages


def _safe_json(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": (raw or "")[:120]}


def _acc_usage(total: dict, usage: dict) -> None:
    for k, v in (usage or {}).items():
        if isinstance(v, int):
            total[k] = total.get(k, 0) + v
