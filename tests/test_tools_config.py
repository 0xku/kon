import pytest

from kon import Config, reset_config, set_config
from kon.tools import resolve_tools


@pytest.fixture
def configured():
    def _set(data):
        set_config(Config(data))

    yield _set
    reset_config()


def test_default_enabled_tools(configured):
    configured({})
    tools, unknown = resolve_tools()

    assert [tool.name for tool in tools] == [
        "read",
        "edit",
        "write",
        "bash",
        "grep",
        "find",
        "web_search",
        "web_fetch",
    ]
    assert unknown == []


def test_enabled_can_be_trimmed_to_bash_only(configured):
    configured({"tools": {"enabled": ["bash"], "extra": []}})
    tools, _ = resolve_tools()

    assert [tool.name for tool in tools] == ["bash"]


def test_extra_and_cli_tools_extend_enabled(configured):
    configured({"tools": {"enabled": ["bash"], "extra": ["web_search"]}})
    tools, unknown = resolve_tools(["web_fetch"])

    assert [tool.name for tool in tools] == ["bash", "web_search", "web_fetch"]
    assert unknown == []


def test_unknown_names_are_reported(configured):
    configured({"tools": {"enabled": ["bash", "nope"], "extra": []}})
    tools, unknown = resolve_tools()

    assert [tool.name for tool in tools] == ["bash"]
    assert unknown == ["nope"]
