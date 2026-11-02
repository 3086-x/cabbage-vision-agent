"""供应商错误的映射测试。不联网、不需要 key。

这些断言看着琐碎，但映射错一类错误的代价很高：
把"额度用尽"当成可重试，使用者会对着 403 白等两轮退避才看到真正原因；
把"key 无效"当成额度问题，则会被引向完全错误的排查方向。
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.llm.base import Message, ProviderError  # noqa: E402
from cvagent.llm.openai_compat import (  # noqa: E402
    OpenAICompatProvider,
    _extract_error,
    diagnose,
)

# DashScope 真实返回体（免费额度耗尽），原样保留便于日后回归
DASHSCOPE_FREE_TIER = json.dumps({
    "error": {
        "message": 'Free quota exhausted. To continue accessing the model on a paid basis, '
                   'please add funds or disable the "use free tier only" mode in the management console.',
        "type": "AllocationQuota.FreeTierOnly",
        "param": None,
        "code": "AllocationQuota.FreeTierOnly",
    },
    "id": "chatcmpl-55a6f252",
})

DASHSCOPE_RATE_LIMIT = json.dumps({
    "error": {"message": "You have exceeded your current request limit.", "type": "limit_requests",
              "param": None, "code": "limit_requests"}
})


class TestDiagnose(unittest.TestCase):
    def test_free_tier_exhausted_is_not_retryable(self):
        code, hint, retryable = diagnose(403, DASHSCOPE_FREE_TIER)
        self.assertEqual(code, "quota_free_tier_only")
        self.assertFalse(retryable, "额度用尽重试没有意义，必须立刻暴露给用户")
        self.assertIn("关闭该开关", hint or "")

    def test_free_tier_text_only_body_still_matched(self):
        # 有的网关不回 JSON，只剩文案
        body = "Free quota exhausted, please disable the free tier only mode"
        code, _, retryable = diagnose(403, body)
        self.assertEqual(code, "quota_free_tier_only")
        self.assertFalse(retryable)

    def test_invalid_key(self):
        body = json.dumps({"error": {"code": "invalid_api_key", "message": "Incorrect API key provided"}})
        code, hint, retryable = diagnose(401, body)
        self.assertEqual(code, "invalid_api_key")
        self.assertFalse(retryable)
        self.assertIn("空格", hint or "")

    def test_rate_limit_is_retryable(self):
        code, _, retryable = diagnose(429, DASHSCOPE_RATE_LIMIT)
        self.assertEqual(code, "rate_limited")
        self.assertTrue(retryable)

    def test_model_not_found(self):
        body = json.dumps({"error": {"code": "model_not_found", "message": "The model does not exist"}})
        code, _, retryable = diagnose(404, body)
        self.assertEqual(code, "model_not_found")
        self.assertFalse(retryable)

    def test_access_denied(self):
        body = json.dumps({"error": {"code": "access_denied", "message": "no permission"}})
        code, _, retryable = diagnose(403, body)
        self.assertEqual(code, "access_denied")
        self.assertFalse(retryable)

    def test_server_error_is_retryable(self):
        code, _, retryable = diagnose(502, "<html>Bad Gateway</html>")
        self.assertEqual(code, "server_error")
        self.assertTrue(retryable)

    def test_unknown_403_falls_back_to_forbidden(self):
        code, hint, retryable = diagnose(403, json.dumps({"error": {"message": "denied"}}))
        self.assertEqual(code, "forbidden")
        self.assertFalse(retryable)
        self.assertIn("额度", hint or "")

    def test_plain_text_body_kept_as_message(self):
        raw_code, msg = _extract_error("upstream connect error")
        self.assertIsNone(raw_code)
        self.assertEqual(msg, "upstream connect error")

    def test_non_dict_json_does_not_crash(self):
        raw_code, msg = _extract_error("[1, 2, 3]")
        self.assertIsNone(raw_code)
        self.assertIn("1, 2, 3", msg)


class _FakeResp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text

    def json(self):
        return json.loads(self.text)


class _FakeRequests(types.ModuleType):
    """替身 requests：只实现 post，返回预置响应并记录调用次数。"""

    RequestException = Exception

    def __init__(self, resp: _FakeResp):
        super().__init__("requests")
        self._resp = resp
        self.calls = 0

    def post(self, *a, **kw):
        self.calls += 1
        return self._resp


class TestProviderSurface(unittest.TestCase):
    def _run(self, status, text, max_retries=0):
        fake = _FakeRequests(_FakeResp(status, text))
        sys.modules["requests"] = fake
        try:
            prov = OpenAICompatProvider("https://example.invalid/v1", "sk-test", "m",
                                        name="dashscope", max_retries=max_retries)
            with self.assertRaises(ProviderError) as ctx:
                prov.complete([Message("user", "hi")], [])
            return ctx.exception, fake
        finally:
            sys.modules.pop("requests", None)

    def test_error_carries_code_and_hint(self):
        err, _ = self._run(403, DASHSCOPE_FREE_TIER)
        self.assertEqual(err.code, "quota_free_tier_only")
        self.assertTrue(err.hint)
        self.assertFalse(err.retryable)
        # 人看到的是三段：状态码、供应商原文、可照做的下一步
        text = str(err)
        self.assertIn("HTTP 403", text)
        self.assertIn("Free quota exhausted", text)
        self.assertIn("→", text)

    def test_non_retryable_does_not_retry(self):
        _, fake = self._run(403, DASHSCOPE_FREE_TIER, max_retries=2)
        self.assertEqual(fake.calls, 1, "403 额度错误不该重试，白等只会拖慢排查")

    def test_action_property(self):
        err = ProviderError("x", hint="去做这件事")
        self.assertEqual(err.action, "去做这件事")
        self.assertEqual(ProviderError("x").action, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
