import pytest

from kon.core.types import Message, ToolDefinition
from kon.llm.base import BaseProvider, LLMStream, ProviderConfig
from kon.llm.models import ApiType, get_model
from kon.runtime import ConversationRuntime, create_provider
from kon.session import Session


class _FakeProvider(BaseProvider):
    name = "fake"

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        return LLMStream()

    def should_retry_for_error(self, error: Exception) -> bool:
        return False


def _runtime_with_provider(provider: BaseProvider) -> ConversationRuntime:
    runtime = ConversationRuntime(
        cwd="/test/project",
        model=provider.config.model,
        model_provider=provider.config.provider,
        api_key="test-key",
        base_url=None,
        thinking_level="high",
        tools=[],
    )
    runtime.provider = provider
    return runtime


@pytest.mark.parametrize(
    ("api_type", "provider_name", "expected_name"),
    [
        (ApiType.OPENAI_CODEX_RESPONSES, "openai-codex", "openai-codex"),
        (ApiType.GITHUB_COPILOT, "github-copilot", "github-copilot"),
        (ApiType.GITHUB_COPILOT_RESPONSES, "github-copilot", "github-copilot"),
        (ApiType.ANTHROPIC_COPILOT, "github-copilot", "github-copilot-anthropic"),
    ],
)
def test_create_provider_does_not_require_oauth_credentials(
    api_type: ApiType, provider_name: str, expected_name: str, tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))

    provider = create_provider(api_type, ProviderConfig(model="gpt-5.5", provider=provider_name))

    assert provider.name == expected_name


def test_initialize_creates_openai_codex_agent_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    runtime = ConversationRuntime(
        cwd=str(tmp_path),
        model="gpt-5.5",
        model_provider="openai-codex",
        api_key=None,
        base_url=None,
        thinking_level="high",
        tools=[],
    )

    result = runtime.initialize()

    assert result.provider_error is None
    assert runtime.provider is not None
    assert runtime.session is not None
    assert runtime.agent is not None


def test_switch_model_recreates_provider_when_openai_compatible_base_url_changes(monkeypatch):
    initial_provider = _FakeProvider(
        ProviderConfig(
            provider="zhipu", base_url="https://api.z.ai/api/coding/paas/v4", model="glm-4.7"
        )
    )
    runtime = _runtime_with_provider(initial_provider)
    target = get_model("deepseek-v4-flash", "deepseek")
    assert target is not None

    created_configs: list[ProviderConfig] = []

    def fake_create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        assert api_type == ApiType.OPENAI_COMPLETIONS
        created_configs.append(config)
        return _FakeProvider(config)

    monkeypatch.setattr("kon.runtime.create_provider", fake_create_provider)

    runtime.switch_model(target)

    assert len(created_configs) == 1
    assert runtime.provider is not initial_provider
    assert runtime.provider is not None
    assert runtime.provider.config.provider == "deepseek"
    assert runtime.provider.config.base_url == "https://api.deepseek.com"
    assert runtime.provider.config.model == "deepseek-v4-flash"
    assert runtime.model == "deepseek-v4-flash"
    assert runtime.model_provider == "deepseek"


def test_switch_model_reuses_provider_when_openai_compatible_base_url_is_unchanged(monkeypatch):
    initial_provider = _FakeProvider(
        ProviderConfig(
            provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-v4-pro"
        )
    )
    runtime = _runtime_with_provider(initial_provider)
    target = get_model("deepseek-v4-flash", "deepseek")
    assert target is not None

    def fail_create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        raise AssertionError("provider should not be recreated")

    monkeypatch.setattr("kon.runtime.create_provider", fail_create_provider)

    runtime.switch_model(target)

    assert runtime.provider is initial_provider
    provider = runtime.provider
    assert provider is not None
    assert provider.config.provider == "deepseek"
    assert provider.config.base_url == "https://api.deepseek.com"
    assert provider.config.model == "deepseek-v4-flash"


def test_switch_model_creates_provider_when_provider_is_none(monkeypatch):
    runtime = ConversationRuntime(
        cwd="/test/project",
        model="glm-4.7",
        model_provider="zhipu",
        api_key="test-key",
        base_url=None,
        thinking_level="high",
        tools=[],
    )
    assert runtime.provider is None

    target = get_model("deepseek-v4-flash", "deepseek")
    assert target is not None

    created: list[ProviderConfig] = []

    def fake_create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        assert api_type == ApiType.OPENAI_COMPLETIONS
        created.append(config)
        return _FakeProvider(config)

    monkeypatch.setattr("kon.runtime.create_provider", fake_create_provider)

    runtime.switch_model(target)

    assert len(created) == 1
    assert runtime.provider is not None
    assert runtime.provider.config.model == "deepseek-v4-flash"
    assert runtime.provider.config.provider == "deepseek"
    assert runtime.model == "deepseek-v4-flash"
    assert runtime.model_provider == "deepseek"


def test_load_session_reuses_existing_provider_for_unknown_session_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("kon.session.Session.get_sessions_dir", lambda cwd: tmp_path)

    saved = Session.create(
        "/test/project",
        provider="custom-unknown",
        model_id="custom-unknown-model",
        thinking_level="high",
        system_prompt="prompt",
    )
    saved.append_model_change("custom-unknown", "custom-unknown-model", None)
    assert saved.session_file is not None
    saved.ensure_persisted()

    initial_provider = _FakeProvider(
        ProviderConfig(
            provider="custom-unknown",
            base_url="https://example.invalid",
            model="custom-unknown-model",
        )
    )
    runtime = _runtime_with_provider(initial_provider)

    def fail_create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        raise AssertionError("provider should not be recreated")

    monkeypatch.setattr("kon.runtime.create_provider", fail_create_provider)

    loaded = runtime.load_session(saved.session_file)

    assert loaded.id == saved.id
    assert runtime.provider is initial_provider
    assert runtime.model == "custom-unknown-model"
    assert runtime.model_provider == "custom-unknown"


def test_load_session_reuses_provider_when_runtime_provider_unknown_and_apis_match(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("kon.session.Session.get_sessions_dir", lambda cwd: tmp_path)

    target = get_model("deepseek-v4-flash", "deepseek")
    assert target is not None
    assert target.api == ApiType.OPENAI_COMPLETIONS

    saved = Session.create(
        "/test/project",
        provider=target.provider,
        model_id=target.id,
        thinking_level="high",
        system_prompt="prompt",
    )
    saved.append_model_change(target.provider, target.id, None)
    assert saved.session_file is not None
    saved.ensure_persisted()

    runtime = ConversationRuntime(
        cwd="/test/project",
        model="custom-unknown-model",
        model_provider="custom-unknown",
        api_key="test-key",
        base_url=None,
        thinking_level="high",
        tools=[],
    )
    initial_provider = _FakeProvider(
        ProviderConfig(
            provider="custom-unknown",
            base_url="https://example.invalid",
            model="custom-unknown-model",
        )
    )
    runtime.provider = initial_provider

    def fail_create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        raise AssertionError("provider should not be recreated")

    monkeypatch.setattr("kon.runtime.create_provider", fail_create_provider)

    runtime.load_session(saved.session_file)

    assert runtime.provider is initial_provider
    assert runtime.model == target.id
    assert runtime.model_provider == target.provider
