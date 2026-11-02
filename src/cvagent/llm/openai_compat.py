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


def _extract_error(body: str) -> tuple[str | None, str]:
    """从各家的错误响应里取出 (code, message)。

    形状不统一：OpenAI 用 {"error": {"code", "message"}}，有的网关把出错信息
    直接放在顶层，还有返回纯文本的。取不到就原样返回，绝不因为解析失败而
    把真正的错误信息丢掉。
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None, body[:200].strip()

    if not isinstance(data, dict):
        return None, body[:200].strip()

    err = data.get("error")
    if isinstance(err, dict):
        code = err.get("code") or err.get("type")
        msg = err.get("message") or ""
        return (str(code) if code else None), str(msg)[:200]
    if isinstance(err, str):
        return None, err[:200]
    code = data.get("code") or data.get("error_code")
    msg = data.get("message") or ""
    return (str(code) if code else None), str(msg)[:200]


# 归一化规则：(匹配子串, 归一化 code, 可照做的下一步)
# 匹配的是“供应商原始错误码 + 错误文案”拼起来的串，全小写。
# 顺序有意义：先匹配越具体的越好，"free tier" 必须先于泛化的 "quota"。
_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("freetieronly", "free tier only", "free quota exhausted", "free tier quota"),
        "quota_free_tier_only",
        "免费额度已用尽，且账户处于「仅使用免费额度」模式。"
        "到供应商控制台关闭该开关并充值即可继续；或换用其他 provider。",
    ),
    (
        ("invalid_api_key", "invalidapikey", "incorrect api key", "invalid api-key", "authentication_error"),
        "invalid_api_key",
        "API Key 无效或已被删除。检查环境变量是否粘进了空格/换行，或到控制台重新生成。",
    ),
    (
        ("limit_requests", "rate_limit", "ratelimit", "too many requests", "requests per min"),
        "rate_limited",
        "触发速率限制。等几秒重试或降低并发；若账户免费额度已耗尽，"
        "也可能持续返回此码，需先确认额度状态。",
    ),
    (
        ("model_not_found", "does not exist", "invalid model", "no such model"),
        "model_not_found",
        "模型名不存在或当前区域未上线。用 --model 指定一个可用模型。",
    ),
    (
        ("access_denied", "permission_denied", "unauthorized_model"),
        "access_denied",
        "账号或 Key 没有该模型的权限，需在控制台先开通该模型。",
    ),
    (
        ("insufficient_quota", "quotaexhausted", "insufficient_balance", "arrears", "balance"),
        "insufficient_quota",
        "账户余额不足，需先充值。",
    ),
    (
        ("invalid_parameter", "invalid value", "unsupported parameter"),
        "bad_request",
        "请求参数不被该模型接受，常见于模型不支持 tools/function calling。"
        "换一个支持工具调用的模型。",
    ),
]


def diagnose(status: int, body: str) -> tuple[str | None, str | None, bool]:
    """把 HTTP 失败翻译成 (归一化 code, 可照做的 hint, 是否值得重试)。

    单独成函数是为了能在不联网的情况下测——错误映射写错的代价很高：
    把“额度用尽”误判成可重试，会让人对着 403 白等两轮退避。
    """
    raw_code, message = _extract_error(body)
    blob = f"{raw_code or ''} {message}".lower()

    retryable = status == 429 or status >= 500

    for keys, code, hint in _RULES:
        if any(k in blob for k in keys):
            # 额度类错误重试没有意义，无论状态码是什么
            if code in ("quota_free_tier_only", "insufficient_quota", "invalid_api_key",
                        "model_not_found", "access_denied"):
                retryable = False
            return code, hint, retryable

    if status == 401:
        return "unauthorized", "认证失败：检查 API Key 是否正确、是否已被删除。", False
    if status == 403:
        return "forbidden", "认证通过但无权限，多为额度用尽或未开通该模型。", False
    if status == 404:
        return "not_found", "端点或模型不存在：核对 base_url 与 model 名。", False
    if status == 429:
        return "rate_limited", "触发速率限制，稍后重试。", True
    if status >= 500:
        return "server_error", "供应商服务端错误，稍后重试。", True
    return (raw_code.lower() if raw_code else None), None, retryable


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
            code, hint, retryable = diagnose(resp.status_code, resp.text)
            _, detail = _extract_error(resp.text)
            parts = [f"{self.name}: HTTP {resp.status_code}"]
            if code:
                parts.append(f"[{code}]")
            if detail:
                parts.append(f"— {detail}")
            if hint:
                parts.append(f"\n  → {hint}")
            raise ProviderError(
                " ".join(parts),
                status=resp.status_code,
                retryable=retryable,
                code=code,
                hint=hint,
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
