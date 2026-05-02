import pytest
from curl_cffi import CurlOpt

from kon.tools import web_fetch
from kon.tools.web_fetch import WebFetchParams, WebFetchTool


def _fail_session(*args, **kwargs):
    raise AssertionError("fetch should be refused before creating a session")


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("127.0.0.1", False),
        ("10.0.0.1", False),
        ("192.168.0.1", False),
        ("169.254.169.254", False),
        ("100.100.100.200", False),
        ("::1", False),
        ("fc00::1", False),
        ("224.0.0.1", False),
        ("64:ff9b::169.254.169.254", False),
        ("93.184.216.34", True),
        ("2606:2800:220:1:248:1893:25c8:1946", True),
    ],
)
def test_public_ip_policy(ip, expected):
    assert web_fetch._is_public_ip(ip) is expected


@pytest.mark.asyncio
async def test_web_fetch_refuses_direct_loopback_before_fetch(monkeypatch):
    monkeypatch.setattr(web_fetch, "AsyncSession", _fail_session)

    result = await WebFetchTool().execute(WebFetchParams(url="http://127.0.0.1:8000/"))

    assert result.success is False
    assert result.result == "Refused: non-public address (127.0.0.1)"


@pytest.mark.asyncio
async def test_web_fetch_refuses_hostname_that_resolves_to_loopback(monkeypatch):
    async def resolve_host_addresses(host: str, port: int):
        assert host == "local.test"
        assert port == 80
        return ["127.0.0.1"]

    monkeypatch.setattr(web_fetch, "_resolve_host_addresses", resolve_host_addresses)
    monkeypatch.setattr(web_fetch, "AsyncSession", _fail_session)

    result = await WebFetchTool().execute(WebFetchParams(url="http://local.test/"))

    assert result.success is False
    assert result.result == "Refused: host did not resolve to a public address"


@pytest.mark.asyncio
async def test_web_fetch_pins_public_resolution_and_checks_connected_ip(monkeypatch):
    async def resolve_host_addresses(host: str, port: int):
        assert host == "example.test"
        assert port == 443
        return ["93.184.216.34", "127.0.0.1", "2606:2800:220:1:248:1893:25c8:1946"]

    class FakeResponse:
        primary_ip = "127.0.0.1"
        status_code = 200
        text = "<html><body>secret</body></html>"
        content = b"<html><body>secret</body></html>"

    class FakeSession:
        def __init__(self, *, allow_redirects, curl_options, **kwargs):
            assert allow_redirects == "safe"
            assert curl_options[CurlOpt.PROTOCOLS_STR] == "http,https"
            assert curl_options[CurlOpt.REDIR_PROTOCOLS_STR] == "http,https"
            assert curl_options[CurlOpt.RESOLVE] == [
                "example.test:443:93.184.216.34,[2606:2800:220:1:248:1893:25c8:1946]"
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, timeout):
            return FakeResponse()

    monkeypatch.setattr(web_fetch, "_resolve_host_addresses", resolve_host_addresses)
    monkeypatch.setattr(web_fetch, "AsyncSession", FakeSession)

    result = await WebFetchTool().execute(WebFetchParams(url="https://example.test/"))

    assert result.success is False
    assert result.result == "Refused: non-public address (127.0.0.1)"
