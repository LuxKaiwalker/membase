import json
import logging
from typing import Any

import httpx

from app.llm.base import (
    LLMMalformedOutputError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str,
        generation_model: str,
        embed_model: str,
        embed_dim: int,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._generation_model = generation_model
        self._embed_model = embed_model
        self._embed_dim = embed_dim
        self._client = httpx.AsyncClient(base_url=self._host, timeout=timeout_seconds)

    @property
    def embedding_model_name(self) -> str:
        return self._embed_model

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama timed out calling {path}") from exc
        except httpx.TransportError as exc:
            # Catches ConnectError plus everything else at the transport
            # level (connection reset mid-response, protocol errors, ...)
            # -- a bare `except httpx.ConnectError` here previously let
            # other real failure modes (e.g. Ollama crashing/restarting
            # mid-generation) propagate as a raw, unhandled httpx
            # exception instead of the LLMError the rest of the app
            # expects to always be able to catch (see LLMError's
            # docstring in app/llm/base.py). TimeoutException is itself a
            # TransportError subclass, so it's caught above, first.
            raise LLMUnavailableError(f"Could not reach Ollama at {self._host}: {exc}") from exc

        if resp.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {resp.status_code} for {path}: {resp.text[:500]}"
            )
        return resp.json()

    async def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.2
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._generation_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        data = await self._post("/api/generate", payload)
        response = data.get("response")
        if not isinstance(response, str):
            raise LLMMalformedOutputError("Ollama /api/generate response missing 'response' field")
        return response

    async def generate_structured(
        self, prompt: str, *, schema: dict[str, Any], system: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._generation_model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.0},
        }
        if system:
            payload["system"] = system
        data = await self._post("/api/generate", payload)
        raw = data.get("response")
        if not isinstance(raw, str):
            raise LLMMalformedOutputError("Ollama /api/generate response missing 'response' field")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(f"Model did not return valid JSON: {raw[:500]}") from exc
        if not isinstance(parsed, dict):
            raise LLMMalformedOutputError(f"Expected a JSON object, got: {type(parsed)}")
        return parsed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post("/api/embed", {"model": self._embed_model, "input": texts})
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise LLMMalformedOutputError("Ollama /api/embed returned an unexpected shape")
        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()
