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
        return httpx.Response(200, json={"response": "hello back"})

    provider = _provider_with_transport(handler)
    assert await provider.generate("hello") == "hello back"
