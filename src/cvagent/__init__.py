"""Cabbage vision agent: 把检测能力包装成可编排的 Agent。

分层：
    vision/  纯推理，不感知 LLM
    tools/   把推理能力暴露为带 schema 的工具，含错误语义
    llm/     供应商无关的对话抽象（OpenAI 兼容 / 离线回放）
    agent/   规划-执行-观察循环，上下文预算管理
"""

__version__ = "0.1.0"
