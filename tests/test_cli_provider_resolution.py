from kon.llm import resolve_provider_api_type
from kon.llm.models import ApiType
from kon.runtime import default_base_url_for_api


def test_resolve_provider_api_type_known_provider():
    assert resolve_provider_api_type("github-copilot") == ApiType.GITHUB_COPILOT
    assert resolve_provider_api_type("openai") == ApiType.OPENAI_COMPLETIONS


def test_resolve_provider_api_type_unknown_provider():
    try:
        resolve_provider_api_type("invalid-provider")
    except ValueError as exc:
        assert "Unknown provider 'invalid-provider'" in str(exc)
        assert "Valid providers:" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid provider")


def testdefault_base_url_for_api_openai_completions(monkeypatch):
    monkeypatch.setenv("KON_BASE_URL", "http://localhost:1234/v1")
    assert default_base_url_for_api(ApiType.OPENAI_COMPLETIONS) == "http://localhost:1234/v1"


def testdefault_base_url_for_api_non_openai_completions():
    assert default_base_url_for_api(ApiType.ANTHROPIC_COPILOT) is None
