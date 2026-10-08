from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """Base class for all local-LLM failures the rest of the app should
    handle explicitly (never let these surface as a generic 500)."""


class LLMUnavailableError(LLMError):
    """The backend (e.g. Ollama) could not be reached."""


class LLMTimeoutError(LLMError):
    """The backend was reached but did not respond in time."""


class LLMMalformedOutputError(LLMError):
    """The backend responded, but the output didn't match what was asked for
    (e.g. structured output that isn't valid JSON)."""


class LLMProvider(ABC):
    """Abstraction boundary between application logic and whatever local
    inference runtime is actually running. Business logic must depend only
    on this interface, never on a specific backend's client/SDK, so the
    backend can be swapped (e.g. Ollama -> llama.cpp server) without
    touching ingestion/retrieval code.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        images: list[bytes] | None = None,
    ) -> str:
        """Free-text completion. `images` (raw bytes, any format the
        backend's vision encoder accepts) is for a natively multimodal
        model -- passing it to a backend/model that doesn't support image
        input is a programming error, not a degrade-gracefully case,
        since this app now uses exactly one generation model everywhere
        and that model is always multimodal (see docs/DECISIONS.md)."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Completion constrained to a JSON schema, returned as a parsed dict.

        Raises LLMMalformedOutputError if the backend's output can't be
        parsed/doesn't match the schema shape.
        """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns one vector per input, in order."""

    @property
    @abstractmethod
    def embedding_model_name(self) -> str: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
