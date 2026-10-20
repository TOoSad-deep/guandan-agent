"""Real SDK parsing paths under HTTP Mock; diagnostics never expose raw errors."""
import asyncio

import httpx
import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.model.deepseek import DeepSeekProvider
from tests.unit.test_model import completion, provider_input


@pytest.mark.parametrize("reason,code", [("length", "MODEL_OUTPUT_LIMIT"), ("content_filter", "MODEL_CONTENT_FILTER")])
def test_sdk_finish_error_preserves_reported_usage_without_partial_text(reason, code):
    async def scenario():
        value = completion(content='PRIVATE-partial-output',
                           usage={"prompt_tokens": 800, "completion_tokens": 1600, "total_tokens": 2400})
        value["choices"][0]["finish_reason"] = reason
        provider = DeepSeekProvider(api_key="fake-test-key", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=value)))
        try:
            reply = await provider.complete(provider_input(SystemClock()))
        finally:
            await provider.aclose()
        assert reply.error.code == code and reply.content is None
        assert reply.usage.total_tokens == 2400 and reply.transport_requests == 1
        assert 'PRIVATE' not in reply.model_dump_json()
    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_http_status_is_safe_and_never_retried(status):
    async def scenario():
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(status, json={"error": {"message": "PRIVATE-provider-body"}})
        provider = DeepSeekProvider(api_key="fake-test-key", transport=httpx.MockTransport(handle))
        try:
            reply = await provider.complete(provider_input(SystemClock()))
        finally:
            await provider.aclose()
        assert len(requests) == reply.transport_requests == 1
        assert str(status) in reply.error.safe_message and reply.error.code == "MODEL_TRANSPORT"
        assert 'PRIVATE' not in reply.model_dump_json() and reply.usage.total_tokens is None
    asyncio.run(scenario())
