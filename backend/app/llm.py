"""The only place that talks to the AI provider (Mistral, ADR-38).

The rest of the code calls get_llm().chat(), .embed() or .transcribe(). To change provider,
only this file changes. Tests use FakeLLM, so they are free and need no API key.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import get_settings
from app.core.errors import AppError

log = logging.getLogger("mechai.llm")


# The result of one chat call: the answer text and the tokens used (for cost tracking).
@dataclass(frozen=True)
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int


# The interface every AI client has: chat, embeddings and speech-to-text.
class LLMClient(Protocol):
    @property
    def embedding_model(self) -> str: ...

    @property
    def chat_model(self) -> str: ...

    # Chat completion; json_output asks the model for a JSON object, max_tokens caps the answer.
    def chat(
        self,
        messages: list[dict[str, str]],
        json_output: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult: ...
    # Embeddings: one vector per text, in the same order.
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    # Speech-to-text for voice input (Voxtral).
    def transcribe(self, audio: bytes, file_name: str, language: str | None = None) -> str: ...


# Error returned when the provider cannot be reached or gives an empty answer.
def ai_unavailable() -> AppError:
    return AppError("ASSISTANT_AI_UNAVAILABLE", "The AI service cannot be reached.", 503)


# The real client. Every provider error becomes an AppError, so the api answers
# with a clean error code instead of a stack trace.
class MistralLLM:
    # Fails with ASSISTANT_AI_DISABLED when no API key is set.
    def __init__(self) -> None:
        from mistralai.client import Mistral

        s = get_settings()
        if s.mistral_api_key is None or not s.mistral_api_key.get_secret_value():
            raise AppError("ASSISTANT_AI_DISABLED", "MISTRAL_API_KEY is not set.", 503)
        self.settings = s
        self.client = Mistral(api_key=s.mistral_api_key.get_secret_value())

    @property
    def embedding_model(self) -> str:
        return self.settings.embed_model

    @property
    def chat_model(self) -> str:
        return self.settings.chat_model

    # One chat call at temperature 0 with a timeout. Any provider problem (error, no choices,
    # empty text) becomes ASSISTANT_AI_UNAVAILABLE; the provider's own text is never passed on.
    def chat(
        self,
        messages: list[dict[str, str]],
        json_output: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        try:
            response = self.client.chat.complete(
                model=self.settings.chat_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_output else None,
                timeout_ms=self.settings.ai_timeout_seconds * 1000,
            )
        except Exception as exc:
            raise ai_unavailable() from exc
        if not response.choices:
            raise ai_unavailable()
        message = response.choices[0].message
        if message is None or not message.content:
            raise ai_unavailable()
        usage = response.usage
        return ChatResult(
            text=str(message.content),
            tokens_in=(usage.prompt_tokens or 0) if usage else 0,
            tokens_out=(usage.completion_tokens or 0) if usage else 0,
        )

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


# The fixed JSON the fake client returns for json_output=True. It holds the keys of both
# assistant calls (understand the question, write the answer); each caller ignores the
# keys it does not know, so one object serves both.
FAKE_JSON_ANSWER: dict[str, object] = {
    "question_language": "en",
    "question_en": "",
    "search_terms_en": "",
    "keywords_en": [],
    "part_numbers": [],
    "safety_flag": "normal",
    "steps": [{"text": "Fake step.", "sources": ["S1"]}],
    "parts": [],
    "warnings": [],
    "confidence": "low",
    "missing_information": [],
}


class FakeLLM:
    """Fixed answers for tests and for working without an API key."""

    embedding_model = "fake-word-count-v1"
    chat_model = "fake-chat-v1"

    # Fixed answers, so code that calls the AI can run without a key. No tokens are used.
    def chat(
        self,
        messages: list[dict[str, str]],
        json_output: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if json_output:
            return ChatResult(json.dumps(FAKE_JSON_ANSWER), 0, 0)
        return ChatResult("fake answer", 0, 0)

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


# Pick the client: the fake one when no API key is set, otherwise Mistral.
def get_llm() -> LLMClient:
    key = get_settings().mistral_api_key
    if key is None or not key.get_secret_value():
        log.warning("MISTRAL_API_KEY is empty: using the fake AI client")
        return FakeLLM()
    return MistralLLM()
