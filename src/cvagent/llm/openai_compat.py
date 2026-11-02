"""OpenAI 兼容客户端。

覆盖 DashScope（qwen）、DeepSeek、Moonshot、OpenAI 官方以及任何自建的
兼容网关。协议差异全部收敛在这里：

  - 一律用 /chat/completions + tools 字段（不用各家私有的 agent 接口）
  - tool_calls 的 function.arguments 是**字符串**而非对象，不同供应商都可能
    返回空串或非法 JSON，所以这里只做透传，解析责任交给工具层
  - 429 / 5xx 标记为可重试，401 / 400 直接失败
"""

from __future__ import annotations

import json
import time

from .base import ChatProvider, Completion, Message, ProviderError, ToolCall


class OpenAICompatProvider(ChatProvider):
    def __init__(self, base_url: str, api_key: str, model: str, *, name: str = "openai-compat",
                 temperature: float = 0.0, timeout: float = 60.0, max_retries: int = 2,
                 proxy: str | None = None) -> None:
        if not api_key:
            raise ProviderError(f"{name}: 缺少 API Key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        # 代理按 provider 单独设置：国内端点直连更快，境外端点才需要代理，
        # 用全局环境变量会把两者一起带偏。
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        body = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post(body)
            except ProviderError as e:
                last_err = e
                if not e.retryable or attempt == self.max_retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last_err  # pragma: no cover

    def _post(self, body: dict) -> Completion:
        import requests

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
                proxies=self.proxies,
            )
        except requests.RequestException as e:
            raise ProviderError(f"{self.name}: 网络错误 {e}", retryable=True) from e

        if resp.status_code != 200:
            snippet = resp.text[:300]
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise ProviderError(
                f"{self.name}: HTTP {resp.status_code} — {snippet}",
                status=resp.status_code,
                retryable=retryable,
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: 响应中没有 choices — {json.dumps(data)[:300]}")

        msg = choices[0].get("message", {})
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            calls.append(
                ToolCall(
                    id=c.get("id") or f"call_{i}",
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or "{}",
                )
            )

        return Completion(
            content=msg.get("content"),
            tool_calls=calls,
            finish_reason=choices[0].get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            model=data.get("model", self.model),
        )
