"""The only place that talks to the AI provider (Mistral, ADR-38).

The rest of the code calls get_llm().chat(), .embed() or .transcribe(). To change provider,
only this file changes. Tests use FakeLLM, so they are free and need no API key.
"""

import hashlib
import logging
from typing import Protocol

from app.config import get_settings
from app.core.errors import AppError

log = logging.getLogger("mechai.llm")


class LLMClient(Protocol):
    @property
    def embedding_model(self) -> str: ...

    def chat(self, messages: list[dict[str, str]], json_output: bool = False) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def transcribe(self, audio: bytes, file_name: str, language: str | None = None) -> str: ...


def ai_unavailable() -> AppError:
    return AppError("ASSISTANT_AI_UNAVAILABLE", "The AI service cannot be reached.", 503)


class MistralLLM:
    def __init__(self) -> None:
        from mistralai.client import Mistral

        s = get_settings()
        if not s.mistral_api_key:
            raise AppError("ASSISTANT_AI_DISABLED", "MISTRAL_API_KEY is not set.", 503)
        self.settings = s
        self.client = Mistral(api_key=s.mistral_api_key)

    @property
    def embedding_model(self) -> str:
        return self.settings.embed_model

    def chat(self, messages: list[dict[str, str]], json_output: bool = False) -> str:
        try:
            response = self.client.chat.complete(
                model=self.settings.chat_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.1,
                response_format={"type": "json_object"} if json_output else None,
            )
        except Exception as exc:
            raise ai_unavailable() from exc
        message = response.choices[0].message
        if message is None or not message.content:
            raise ai_unavailable()
        return str(message.content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(model=self.settings.embed_model, inputs=texts)
        except Exception as exc:
            raise AppError(
                "KNOWLEDGE_EMBEDDING_FAILED", "The embedding service failed.", 503
            ) from exc
        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts) or any(vector is None for vector in vectors):
            raise AppError("KNOWLEDGE_EMBEDDING_FAILED", "The embedding service failed.", 503)
        return vectors  # type: ignore[return-value]

    def transcribe(self, audio: bytes, file_name: str, language: str | None = None) -> str:
        try:
            response = self.client.audio.transcriptions.complete(
                model=self.settings.stt_model,
                file={"file_name": file_name, "content": audio},
                language=language,
            )
        except Exception as exc:
            raise AppError(
                "ASSISTANT_TRANSCRIBE_FAILED", "Could not transcribe the recording.", 503
            ) from exc
        return response.text


class FakeLLM:
    """Fixed answers for tests and for working without an API key."""

    embedding_model = "fake-word-count-v1"

    def chat(self, messages: list[dict[str, str]], json_output: bool = False) -> str:
        return '{"answer": "fake"}' if json_output else "fake answer"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """A word-count vector: texts that share words end up close. Free, no API call."""
        dims = get_settings().embed_dimensions
        vectors = []
        for text in texts:
            vector = [0.0] * dims
            for word in text.lower().split():
                word = word.strip(".,:;!?()\"'")
                if word:
                    vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
            vector[0] += 1e-6  # never an all-zero vector (cosine needs a length)
            vectors.append(vector)
        return vectors

    def transcribe(self, audio: bytes, file_name: str, language: str | None = None) -> str:
        return "fake transcript"


def get_llm() -> LLMClient:
    if not get_settings().mistral_api_key:
        log.warning("MISTRAL_API_KEY is empty: using the fake AI client")
        return FakeLLM()
    return MistralLLM()
