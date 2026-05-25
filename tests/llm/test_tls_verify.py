import asyncio

import httpx
import pytest

from kon.config import Config, reset_config, set_config
from kon.llm.base import make_http_client


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    reset_config()


def test_default_insecure_skip_verify_is_false():
    set_config(Config({}))

    assert make_http_client() is None


def test_make_http_client_skips_verify_when_enabled():
    set_config(Config({"llm": {"tls": {"insecure_skip_verify": True}}}))

    client = make_http_client()
    try:
        assert client is not None
    finally:
        if client is not None:
            asyncio.run(client.aclose())


def test_make_http_client_honors_request_timeout():
    set_config(
        Config({"llm": {"tls": {"insecure_skip_verify": True}, "request_timeout_seconds": 1234}})
    )

    client = make_http_client()
    try:
        assert client is not None
        assert client.timeout == httpx.Timeout(1234)
    finally:
        if client is not None:
            asyncio.run(client.aclose())
