from typing import Any

import pytest

from kon.core.types import UserMessage
from kon.llm.base import ProviderConfig
from kon.llm.oauth.xai import XaiCredentials
from kon.llm.providers.xai_responses import XaiResponsesProvider


def test_xai_responses_uses_xai_request_fields() -> None:
    provider = XaiResponsesProvider(
        ProviderConfig(
            model="grok-4.5",
            provider="xai",
            base_url="https://api.x.ai/v1",
            thinking_level="medium",
            session_id="session-123",
        )
    )

    params = provider._build_params(
        [UserMessage(content="hello")], "You are helpful", None, None, "session-123"
    )

    assert params["reasoning"] == {"effort": "medium"}
    assert params["include"] == ["reasoning.encrypted_content"]
    assert params["prompt_cache_key"] == "session-123"
    assert params["input"][0] == {"role": "developer", "content": "You are helpful"}


@pytest.mark.asyncio
async def test_xai_provider_uses_oauth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = XaiCredentials(access="oauth-token", refresh="refresh", expires=9999999999999)

    async def get_credentials() -> XaiCredentials:
        return credentials

    captured: dict[str, Any] = {}

    class FakeResponses:
        async def create(self, **params: Any) -> list[Any]:
            captured["params"] = params
            return []

    class FakeClient:
        def __init__(self, **kwargs: Any):
            captured["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(
        "kon.llm.providers.xai_responses.get_valid_xai_credentials", get_credentials
    )
    monkeypatch.setattr("kon.llm.providers.xai_responses.AsyncOpenAI", FakeClient)
    provider = XaiResponsesProvider(
        ProviderConfig(
            model="grok-4.5",
            provider="xai",
            base_url="https://api.x.ai/v1",
            thinking_level="low",
            session_id="session-123",
        )
    )

    await provider._stream_impl([UserMessage(content="hello")])

    assert captured["client"]["api_key"] == "oauth-token"
    assert captured["client"]["default_headers"] == {"session_id": "session-123"}
    assert captured["params"]["model"] == "grok-4.5"
