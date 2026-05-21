from unittest.mock import patch

import pytest

from kon.llm.oauth import openai as openai_oauth
from kon.llm.oauth.openai import (
    OpenAICredentials,
    get_valid_openai_credentials,
    get_valid_openai_token,
    login,
    save_openai_credentials,
)


@pytest.mark.asyncio
async def test_login_raises_runtime_error_when_server_fails_and_no_manual_input():
    with (
        patch("kon.llm.oauth.openai._start_callback_server", side_effect=OSError("port in use")),
        pytest.raises(RuntimeError, match="could not start callback server"),
    ):
        await login(on_auth_url=None, on_manual_input=None)


@pytest.mark.asyncio
async def test_get_valid_openai_credentials_returns_unexpired_credentials(tmp_path, monkeypatch):
    auth_path = tmp_path / "openai_auth.json"
    monkeypatch.setattr(openai_oauth, "get_openai_auth_path", lambda: auth_path)
    creds = OpenAICredentials(
        refresh="refresh", access="access", expires=9_999_999_999_999, account_id="account"
    )
    save_openai_credentials(creds)

    assert await get_valid_openai_credentials() == creds


@pytest.mark.asyncio
async def test_get_valid_openai_credentials_refreshes_expired_credentials(tmp_path, monkeypatch):
    auth_path = tmp_path / "openai_auth.json"
    monkeypatch.setattr(openai_oauth, "get_openai_auth_path", lambda: auth_path)
    expired = OpenAICredentials(
        refresh="old-refresh", access="old-access", expires=0, account_id="old"
    )
    refreshed = OpenAICredentials(
        refresh="new-refresh", access="new-access", expires=9_999_999_999_999, account_id="new"
    )
    save_openai_credentials(expired)
    seen: list[OpenAICredentials] = []

    async def fake_refresh(creds: OpenAICredentials) -> OpenAICredentials:
        seen.append(creds)
        return refreshed

    monkeypatch.setattr(openai_oauth, "refresh_openai_token", fake_refresh)

    assert await get_valid_openai_credentials() == refreshed
    assert seen == [expired]


@pytest.mark.asyncio
async def test_get_valid_openai_credentials_returns_none_when_refresh_fails(tmp_path, monkeypatch):
    auth_path = tmp_path / "openai_auth.json"
    monkeypatch.setattr(openai_oauth, "get_openai_auth_path", lambda: auth_path)
    save_openai_credentials(
        OpenAICredentials(refresh="refresh", access="access", expires=0, account_id="account")
    )

    async def fail_refresh(creds: OpenAICredentials) -> OpenAICredentials:
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(openai_oauth, "refresh_openai_token", fail_refresh)

    assert await get_valid_openai_credentials() is None


@pytest.mark.asyncio
async def test_get_valid_openai_token_returns_access_from_valid_credentials(monkeypatch):
    creds = OpenAICredentials(
        refresh="refresh", access="access-token", expires=9_999_999_999_999, account_id="account"
    )

    async def fake_get_credentials() -> OpenAICredentials:
        return creds

    monkeypatch.setattr(openai_oauth, "get_valid_openai_credentials", fake_get_credentials)

    assert await get_valid_openai_token() == "access-token"
