import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from kon import get_config_dir

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_SCOPE = "openid profile email offline_access grok-cli:access api:access"
_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
_TOKEN_URL = "https://auth.x.ai/oauth2/token"
_REFRESH_SKEW_MS = 5 * 60 * 1000
_DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


@dataclass
class XaiCredentials:
    access: str
    refresh: str
    expires: int


@dataclass
class XaiDeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    interval: int
    expires_in: int


def get_xai_auth_path() -> Path:
    return get_config_dir() / "xai_auth.json"


def load_xai_credentials() -> XaiCredentials | None:
    path = get_xai_auth_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return XaiCredentials(
            access=data["access"], refresh=data["refresh"], expires=data["expires"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_xai_credentials(credentials: XaiCredentials) -> None:
    path = get_xai_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access": credentials.access,
                "refresh": credentials.refresh,
                "expires": credentials.expires,
            },
            indent=2,
        )
    )
    path.chmod(0o600)


def clear_xai_credentials() -> None:
    path = get_xai_auth_path()
    if path.exists():
        path.unlink()


def is_xai_logged_in() -> bool:
    return load_xai_credentials() is not None


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return int(value)


def _validate_verification_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    return value


def _request_error(action: str, status: int, data: dict[str, Any]) -> RuntimeError:
    details = [data.get("error"), data.get("error_description")]
    detail = ": ".join(value for value in details if isinstance(value, str) and value)
    suffix = f": {detail}" if detail else ""
    return RuntimeError(f"xAI OAuth {action} failed (HTTP {status}){suffix}")


async def _post_form(url: str, fields: dict[str, str]) -> tuple[int, dict[str, Any]]:
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=fields,
        ) as response,
    ):
        try:
            data = await response.json()
        except (aiohttp.ContentTypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"xAI OAuth returned invalid JSON (HTTP {response.status})"
            ) from error

    if not isinstance(data, dict):
        raise RuntimeError(f"xAI OAuth returned invalid JSON (HTTP {response.status})")
    return response.status, data


def _credentials_from_response(
    data: dict[str, Any], previous_refresh_token: str | None = None
) -> XaiCredentials:
    access = _required_string(data, "access_token")
    if data.get("refresh_token") is None and previous_refresh_token:
        refresh = previous_refresh_token
    else:
        refresh = _required_string(data, "refresh_token")
    expires_in = (
        _DEFAULT_TOKEN_LIFETIME_SECONDS
        if data.get("expires_in") is None
        else _positive_number(data, "expires_in")
    )
    return XaiCredentials(
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000) + expires_in * 1000 - _REFRESH_SKEW_MS,
    )


async def start_xai_device_flow() -> XaiDeviceCode:
    status, data = await _post_form(
        _DEVICE_CODE_URL, {"client_id": _CLIENT_ID, "scope": _SCOPE, "referrer": "kon"}
    )
    if status >= 400:
        raise _request_error("device authorization", status, data)

    complete = data.get("verification_uri_complete")
    if complete is not None:
        complete = _validate_verification_uri(_required_string(data, "verification_uri_complete"))
    interval = data.get("interval")
    return XaiDeviceCode(
        device_code=_required_string(data, "device_code"),
        user_code=_required_string(data, "user_code"),
        verification_uri=_validate_verification_uri(_required_string(data, "verification_uri")),
        verification_uri_complete=complete,
        interval=(
            int(interval)
            if isinstance(interval, (int, float))
            and not isinstance(interval, bool)
            and math.isfinite(interval)
            and interval > 0
            else 5
        ),
        expires_in=_positive_number(data, "expires_in"),
    )


async def poll_for_xai_token(device: XaiDeviceCode) -> XaiCredentials:
    deadline = time.monotonic() + device.expires_in
    interval = device.interval

    while time.monotonic() < deadline:
        await asyncio.sleep(interval)
        status, data = await _post_form(
            _TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": _CLIENT_ID,
                "device_code": device.device_code,
            },
        )
        if status < 400:
            return _credentials_from_response(data)

        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            reported_interval = data.get("interval")
            interval = (
                int(reported_interval)
                if isinstance(reported_interval, (int, float))
                and not isinstance(reported_interval, bool)
                and math.isfinite(reported_interval)
                and reported_interval > 0
                else interval + 5
            )
            continue
        if error in {"access_denied", "authorization_denied"}:
            raise RuntimeError("xAI device authorization was denied")
        if error == "expired_token":
            raise TimeoutError("xAI device code expired")
        raise _request_error("device token polling", status, data)

    raise TimeoutError("xAI device code flow timed out")


async def refresh_xai_token(credentials: XaiCredentials) -> XaiCredentials:
    status, data = await _post_form(
        _TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": _CLIENT_ID,
            "refresh_token": credentials.refresh,
        },
    )
    if status >= 400:
        raise _request_error("token refresh", status, data)

    refreshed = _credentials_from_response(data, credentials.refresh)
    save_xai_credentials(refreshed)
    return refreshed


async def get_valid_xai_credentials() -> XaiCredentials | None:
    credentials = load_xai_credentials()
    if not credentials:
        return None
    if int(time.time() * 1000) >= credentials.expires - 60_000:
        try:
            credentials = await refresh_xai_token(credentials)
        except Exception:
            return None
    return credentials


async def get_valid_xai_token() -> str | None:
    credentials = await get_valid_xai_credentials()
    return credentials.access if credentials else None


async def login(on_user_code: Any | None = None) -> XaiCredentials:
    device = await start_xai_device_flow()
    if on_user_code:
        on_user_code(device.verification_uri_complete or device.verification_uri, device.user_code)
    credentials = await poll_for_xai_token(device)
    save_xai_credentials(credentials)
    return credentials


xai_login = login
