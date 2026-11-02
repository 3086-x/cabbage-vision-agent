"""Provider 工厂与可用性探测。"""

from __future__ import annotations

import os

from .base import ChatProvider, ProviderError
from .openai_compat import OpenAICompatProvider
from .. import config


def _env_key(env_name: str) -> str:
    return os.getenv(env_name, "").strip()


def available_providers() -> list[str]:
    """返回本机已配置 key 的 provider，按 config 中的声明顺序。"""
    return [name for name, meta in config.PROVIDERS.items() if _env_key(meta["api_key_env"])]


def build_provider(name: str | None = None, model: str | None = None) -> ChatProvider:
    """按名字或自动探测构造 provider。mock 不依赖网络。"""
    choice = (name or config.LLM_PROVIDER or "auto").lower()

    if choice == "mock":
        from .mock import MockProvider

        return MockProvider(model=model or "mock-scripted")

    if choice == "auto":
        candidates = available_providers()
        if not candidates:
            raise ProviderError(
                "未检测到任何 LLM API Key。请设置 DASHSCOPE_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY 之一，"
                "或用 CVAGENT_LLM=mock 以离线脚本模式运行。"
            )
        choice = candidates[0]

    meta = config.PROVIDERS.get(choice)
    if meta is None:
        raise ProviderError(
            f"未知 provider: {choice}。可选：{', '.join(list(config.PROVIDERS) + ['mock', 'auto'])}"
        )

    key = _env_key(meta["api_key_env"])
    if not key:
        raise ProviderError(f"{choice} 需要环境变量 {meta['api_key_env']}，当前为空")

    base_url = meta["base_url"]
    if not base_url:
        raise ProviderError(f"{choice} 缺少 base_url：请设置 CVAGENT_BASE_URL（如 https://your-gateway/v1）")

    # 代理优先级：provider 专属 > 通用
    proxy = os.getenv(f"CVAGENT_PROXY_{choice.upper()}", "").strip() or config.LLM_PROXY or None

    return OpenAICompatProvider(
        base_url=base_url,
        api_key=key,
        model=model or config.LLM_MODEL or meta["default_model"],
        name=choice,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        proxy=proxy,
    )


def probe(provider: ChatProvider, timeout: float = 20.0) -> tuple[bool, str]:
    """发一次最小请求验证可用性。返回 (ok, 说明)。

    只发 "ping" 一个词，成本可忽略，但能一次性验证 key、网络、模型名三件事。
    """
    from .base import Message

    try:
        out = provider.complete([Message(role="user", content="ping")], tools=[])
        snippet = (out.content or "").strip().replace("\n", " ")[:40]
        return True, f"{provider.name}/{provider.model} 可用（回显: {snippet or '空'}）"
    except ProviderError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
