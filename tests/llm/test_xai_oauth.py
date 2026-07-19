import json
import time
from pathlib import Path
from typing import Any

import pytest

from kon.llm.oauth import xai as xai_oauth


class MockResponse:
    def __init__(self, data: dict[str, Any], status: int = 200):
        self.data = data
        self.status = status

    async def __aenter__(self) -> "MockResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self.data


class MockSession:
    def __init__(self, responses: list[MockResponse], requests: list[dict[str, Any]]):
        self.responses = responses
        self.requests = requests

    async def __aenter__(self) -> "MockSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, **kwargs: Any) -> MockResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def response_data(**overrides: Any) -> dict[str, Any]:
    return {
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 1,
        **overrides,
    }


def token_data(**overrides: Any) -> dict[str, Any]:
    return {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21600,
        **overrides,
    }


def mock_session(
    monkeypatch: pytest.MonkeyPatch, responses: list[MockResponse]
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        xai_oauth.aiohttp, "ClientSession", lambda: MockSession(responses, requests)
    )
    return requests


@pytest.mark.asyncio
async def test_xai_login_uses_device_flow_and_saves_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "xai_auth.json"
    monkeypatch.setattr(xai_oauth, "get_xai_auth_path", lambda: auth_path)
    monkeypatch.setattr(xai_oauth.asyncio, "sleep", lambda _: _noop())
    requests = mock_session(
        monkeypatch,
        [
            MockResponse(
                response_data(
                    verification_uri_complete=(
                        "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
                    )
                )
            ),
            MockResponse({"error": "authorization_pending"}, 400),
            MockResponse(token_data()),
        ],
    )
    notifications: list[tuple[str, str]] = []

    credentials = await xai_oauth.xai_login(
        on_user_code=lambda url, code: notifications.append((url, code))
    )

    assert notifications == [
        ("https://accounts.x.ai/oauth2/device?user_code=ABCD-1234", "ABCD-1234")
    ]
    assert credentials.access == "access-token"
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(auth_path.read_text())["refresh"] == "refresh-token"
    assert requests[0]["data"] == {
        "client_id": "b1a00492-073a-47ea-816f-4c329264a828",
        "scope": "openid profile email offline_access grok-cli:access api:access",
        "referrer": "kon",
    }
    assert requests[1]["data"]["grant_type"] == ("urn:ietf:params:oauth:grant-type:device_code")


async def _noop() -> None:
    pass


@pytest.mark.asyncio
async def test_xai_refresh_preserves_unrotated_refresh_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(xai_oauth, "get_xai_auth_path", lambda: tmp_path / "xai_auth.json")
    requests = mock_session(
        monkeypatch, [MockResponse(token_data(access_token="new-access", refresh_token=None))]
    )
    credentials = xai_oauth.XaiCredentials(access="old", refresh="keep-me", expires=0)

    refreshed = await xai_oauth.refresh_xai_token(credentials)

    assert refreshed.access == "new-access"
    assert refreshed.refresh == "keep-me"
    assert requests[0]["data"]["refresh_token"] == "keep-me"


@pytest.mark.asyncio
async def test_get_valid_xai_credentials_refreshes_expired_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "xai_auth.json"
    monkeypatch.setattr(xai_oauth, "get_xai_auth_path", lambda: auth_path)
    auth_path.write_text(json.dumps({"access": "old", "refresh": "refresh", "expires": 0}))
    mock_session(monkeypatch, [MockResponse(token_data(access_token="new"))])

    credentials = await xai_oauth.get_valid_xai_credentials()

    assert credentials is not None
    assert credentials.access == "new"
    assert credentials.expires > int(time.time() * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ["http://accounts.x.ai/device", "file:///etc/passwd", "invalid"])
async def test_xai_login_rejects_untrusted_verification_uri(
    monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    mock_session(monkeypatch, [MockResponse(response_data(verification_uri=uri))])

    with pytest.raises(RuntimeError, match="Untrusted verification URI"):
        await xai_oauth.start_xai_device_flow()
