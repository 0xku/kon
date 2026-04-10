from typing import Any, cast

import pytest

from kon.llm.base import ProviderConfig, is_local_base_url, resolve_api_key
from kon.llm.providers.openai_codex_responses import OpenAICodexResponsesProvider
from kon.llm.providers.openai_compat import supports_developer_role
from kon.llm.providers.openai_completions import OpenAICompletionsProvider, _detect_compat
from kon.llm.providers.openai_responses import OpenAIResponsesProvider


def test_supports_developer_role_for_openai_api() -> None:
    assert supports_developer_role("openai", "https://api.openai.com/v1") is True


def test_supports_developer_role_for_local_openai_compatible_api() -> None:
    assert supports_developer_role("openai", "http://127.0.0.1:1234/v1") is False


def test_supports_developer_role_for_zhipu() -> None:
    assert supports_developer_role("zhipu", "https://api.z.ai/api/coding/paas/v4") is False


def test_detect_compat_disables_developer_role_for_local_api() -> None:
    compat = _detect_compat("openai", "http://127.0.0.1:1234/v1")

    assert compat.supports_developer_role is False
    assert compat.supports_reasoning_effort is True


def test_detect_compat_uses_llama_gemma_for_local_gemma_models() -> None:
    compat = _detect_compat(
        "openai", "http://127.0.0.1:1234/v1", "unsloth/gemma-4-26B-A4B-it-GGUF"
    )

    assert compat.supports_developer_role is False
    assert compat.supports_reasoning_effort is False
    assert compat.thinking_format == "llama_gemma"


def test_openai_completions_prefixes_think_token_for_local_gemma() -> None:
    provider = OpenAICompletionsProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="http://127.0.0.1:1234/v1",
            model="unsloth/gemma-4-26B-A4B-it-GGUF",
            provider="openai",
            thinking_level="medium",
        )
    )

    messages = provider._convert_messages([], "You are helpful", provider._compat)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "<|think|>You are helpful"


def test_openai_completions_does_not_prefix_think_token_when_thinking_disabled() -> None:
    provider = OpenAICompletionsProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="http://127.0.0.1:1234/v1",
            model="unsloth/gemma-4-26B-A4B-it-GGUF",
            provider="openai",
            thinking_level="none",
        )
    )

    messages = provider._convert_messages([], "You are helpful", provider._compat)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"


def test_openai_completions_uses_developer_without_think_prefix_for_openai_api() -> None:
    provider = OpenAICompletionsProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5",
            provider="openai",
            thinking_level="medium",
        )
    )

    messages = provider._convert_messages([], "You are helpful", provider._compat)

    assert messages[0]["role"] == "developer"
    assert messages[0]["content"] == "You are helpful"


class _EmptyAsyncIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _DummyChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _EmptyAsyncIterator()


@pytest.mark.asyncio
async def test_openai_completions_sends_enable_thinking_for_local_gemma() -> None:
    provider = OpenAICompletionsProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="http://127.0.0.1:1234/v1",
            model="unsloth/gemma-4-26B-A4B-it-GGUF",
            provider="openai",
            thinking_level="medium",
        )
    )
    dummy_chat = _DummyChatCompletions()
    provider._client = cast(
        Any,
        type("DummyClient", (), {"chat": type("DummyChat", (), {"completions": dummy_chat})()})(),
    )

    stream = await provider._stream_impl(messages=[], system_prompt="You are helpful")
    async for _ in stream:
        pass

    kwargs = dummy_chat.calls[0]
    assert kwargs["extra_body"] == {"enable_thinking": True}
    assert "reasoning_effort" not in kwargs
    assert kwargs["messages"][0]["content"] == "<|think|>You are helpful"


def test_openai_responses_uses_system_for_local_api() -> None:
    provider = OpenAIResponsesProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="http://127.0.0.1:1234/v1",
            model="qwen/qwen3.5-35b-a3b",
            provider="openai",
        )
    )

    messages = provider._convert_messages([], "You are helpful")

    assert messages[0]["role"] == "system"


def test_openai_responses_uses_developer_for_openai_api() -> None:
    provider = OpenAIResponsesProvider(
        ProviderConfig(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5",
            provider="openai",
        )
    )

    messages = provider._convert_messages([], "You are helpful")

    assert messages[0]["role"] == "developer"


def test_openai_codex_request_uses_session_for_prompt_caching() -> None:
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4",
            provider="openai-codex",
            session_id="session-123",
        )
    )

    body = provider._build_request_body([], "You are helpful", None, None)
    headers = provider._build_headers("token", "account")

    assert body["prompt_cache_key"] == "session-123"
    assert headers["session_id"] == "session-123"
    assert headers["conversation_id"] == "session-123"


def test_openai_codex_request_omits_prompt_cache_fields_without_session() -> None:
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(
            base_url="https://chatgpt.com/backend-api", model="gpt-5.4", provider="openai-codex"
        )
    )

    body = provider._build_request_body([], "You are helpful", None, None)
    headers = provider._build_headers("token", "account")

    assert "prompt_cache_key" not in body
    assert "session_id" not in headers
    assert "conversation_id" not in headers


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://localhost:8080/v1", True),
        ("http://192.168.1.10:8080/v1", True),
        ("https://api.openai.com/v1", False),
        (None, False),
    ],
)
def test_is_local_base_url(base_url: str | None, expected: bool) -> None:
    assert is_local_base_url(base_url) is expected


def test_resolve_api_key_uses_placeholder_for_local_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        resolve_api_key(
            None,
            env_vars=("OPENAI_API_KEY",),
            base_url="http://127.0.0.1:8080/v1",
            auth_mode="auto",
        )
        == "kon-local"
    )


def test_resolve_api_key_requires_key_for_remote_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        resolve_api_key(
            None,
            env_vars=("OPENAI_API_KEY",),
            base_url="https://api.openai.com/v1",
            auth_mode="auto",
        )
        is None
    )


def test_resolve_api_key_uses_placeholder_for_none_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key(None, env_vars=("OPENAI_API_KEY",), auth_mode="none") == "kon-local"
