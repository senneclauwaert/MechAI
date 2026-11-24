"""Tests for the AI client: the fake chat and MistralLLM.chat with a mocked provider client."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.errors import AppError
from app.llm import ChatResult, FakeLLM, MistralLLM


# A MistralLLM without the real SDK client: only the mocked client and the settings are set.
def make_llm(response: Any = None, error: Exception | None = None) -> MistralLLM:
    llm = MistralLLM.__new__(MistralLLM)
    llm.settings = SimpleNamespace(chat_model="m", ai_timeout_seconds=7)  # type: ignore[assignment]
    llm.client = MagicMock()
    if error:
        llm.client.chat.complete.side_effect = error
    else:
        llm.client.chat.complete.return_value = response
    return llm


# A provider response with one choice (content may be empty), optional usage and finish reason.
def response(
    content: str | None, usage: Any = None, choices: bool = True, finish: str = "stop"
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice] if choices else [], usage=usage)


def test_fake_chat_json_has_all_keys() -> None:
    result = FakeLLM().chat([], json_output=True)
    assert isinstance(result, ChatResult)
    data = json.loads(result.text)
    assert data["question_language"] == "en"
    assert data["steps"] == [{"text": "Fake step.", "sources": ["S1"]}]
    assert data["safety_flag"] == "normal"
    assert data["confidence"] == "low"
    assert {"keywords_en", "part_numbers", "parts", "warnings", "missing_information"} <= set(data)
    assert (result.tokens_in, result.tokens_out) == (0, 0)


def test_fake_chat_text_and_model() -> None:
    llm = FakeLLM()
    assert llm.chat([]).text == "fake answer"
    assert llm.chat_model == "fake-chat-v1"


def test_mistral_chat_returns_text_and_tokens() -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5)
    llm = make_llm(response("hello", usage))
    result = llm.chat([{"role": "user", "content": "hi"}], json_output=True, max_tokens=50)
    assert result == ChatResult("hello", 11, 5)
    kwargs = llm.client.chat.complete.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["max_tokens"] == 50
    assert kwargs["timeout_ms"] == 7000
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}


def test_mistral_chat_without_usage_gives_zero_tokens() -> None:
    result = make_llm(response("hello")).chat([])
    assert (result.tokens_in, result.tokens_out) == (0, 0)


@pytest.mark.parametrize(
    ("finish", "expected"), [("length", True), ("model_length", True), ("stop", False)]
)
def test_mistral_chat_marks_cut_off_answers(finish: str, expected: bool) -> None:
    result = make_llm(response("half an ans", finish=finish)).chat([])
    assert result.truncated is expected
    assert result.text == "half an ans"  # no error is raised


def test_fake_chat_is_never_truncated() -> None:
    assert FakeLLM().chat([]).truncated is False
    assert FakeLLM().chat([], json_output=True).truncated is False


def test_mistral_embed_uses_the_timeout() -> None:
    llm = make_llm()
    llm.settings.embed_model = "e"  # type: ignore[attr-defined]
    llm.client.embeddings.create.return_value = SimpleNamespace(  # type: ignore[attr-defined]
        data=[SimpleNamespace(embedding=[0.5])]
    )
    assert llm.embed(["x"]) == [[0.5]]
    kwargs = llm.client.embeddings.create.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["timeout_ms"] == 7000


@pytest.mark.parametrize(
    "llm",
    [
        make_llm(error=RuntimeError("secret provider text")),
        make_llm(response("x", choices=False)),
        make_llm(response("")),
        make_llm(response(None)),
    ],
)
def test_mistral_chat_failures_are_ai_unavailable(llm: MistralLLM) -> None:
    with pytest.raises(AppError) as info:
        llm.chat([])
    assert info.value.code == "ASSISTANT_AI_UNAVAILABLE"
    assert "secret" not in info.value.message
