import base64
import json

import httpx
import pytest

from app.llm.base import LLMMalformedOutputError, LLMUnavailableError
from app.llm.ollama_provider import OllamaProvider

# OllamaProvider had no dedicated unit tests at all -- only ever exercised
# indirectly through the real-Ollama integration tests, which never hit
# genuinely malformed-response edge cases since the real service always
# returns well-formed JSON on success. Found while auditing error-handling
# consistency: every other LLM failure mode in this app is wrapped into
# LLMError (see app/llm/base.py's docstring), but a non-JSON *top-level*
# response body (distinct from a malformed *inner* "response" field, which
# was already handled) propagated as a raw json.JSONDecodeError instead.


def _provider_with_transport(handler) -> OllamaProvider:
    provider = OllamaProvider(
        host="http://ollama.invalid",
        generation_model="test-model",
        embed_model="test-embed",
        embed_dim=8,
    )
    # No constructor seam for injecting a transport -- swapping the
    # client after construction is the simplest way to exercise _post()
    # against a fake backend without adding a test-only parameter to
    # production code.
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=provider._host
    )
    return provider


@pytest.mark.asyncio
async def test_non_json_response_body_raises_malformed_output_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMMalformedOutputError):
        await provider.generate("hello")


@pytest.mark.asyncio
async def test_non_200_status_raises_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailableError):
        await provider.generate("hello")


@pytest.mark.asyncio
async def test_connection_failure_raises_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailableError):
        await provider.generate("hello")


@pytest.mark.asyncio
async def test_well_formed_response_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hello back"}})

    provider = _provider_with_transport(handler)
    assert await provider.generate("hello") == "hello back"


@pytest.mark.asyncio
async def test_generate_calls_chat_endpoint_with_system_and_user_messages():
    """Regression: /api/generate's flat system+prompt fields were measured
    to truncate array-typed structured output for gemma4 (a 3-item array
    collapsed to 1 item; an array-of-strings schema came back empty) --
    /api/chat's proper messages list does not have this problem, confirmed
    directly against the live model (see docs/DECISIONS.md "Gemma 4
    migration"). This locks in that /api/chat is actually what gets
    called, not /api/generate."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    provider = _provider_with_transport(handler)
    await provider.generate("do the thing", system="be helpful")

    assert captured["path"] == "/api/chat"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "do the thing"},
    ]


@pytest.mark.asyncio
async def test_generate_attaches_images_to_the_user_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "a photo"}})

    provider = _provider_with_transport(handler)
    await provider.generate("describe this", images=[b"fake-image-bytes"])

    user_message = captured["body"]["messages"][-1]
    assert user_message["role"] == "user"
    assert user_message["images"] == [base64.b64encode(b"fake-image-bytes").decode("ascii")]
