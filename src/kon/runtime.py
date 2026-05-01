from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .core.compaction import generate_summary
from .core.handoff import generate_handoff_prompt
from .core.types import AssistantMessage
from .llm import (
    API_TYPE_TO_PROVIDER_CLASS,
    ApiType,
    BaseProvider,
    Model,
    ProviderConfig,
    get_max_tokens,
    get_model,
    is_copilot_logged_in,
    is_openai_logged_in,
    resolve_provider_api_type,
)
from .llm.base import AuthMode
from .loop import Agent, build_system_prompt
from .session import MessageEntry, Session
from .tools import BaseTool

_COPILOT_API_TYPES: frozenset[ApiType] = frozenset(
    {ApiType.GITHUB_COPILOT, ApiType.GITHUB_COPILOT_RESPONSES, ApiType.ANTHROPIC_COPILOT}
)
_OPENAI_OAUTH_API_TYPES: frozenset[ApiType] = frozenset({ApiType.OPENAI_CODEX_RESPONSES})
_API_TYPE_BY_PROVIDER: dict[type[BaseProvider], ApiType] = {
    v: k for k, v in API_TYPE_TO_PROVIDER_CLASS.items()
}


def default_base_url_for_api(api_type: ApiType) -> str | None:
    if api_type == ApiType.OPENAI_COMPLETIONS:
        return os.environ.get("KON_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    return None


def create_provider(api_type: ApiType, config: ProviderConfig) -> BaseProvider:
    if api_type in _COPILOT_API_TYPES and not is_copilot_logged_in():
        raise ValueError("Not logged in to GitHub Copilot. Use /login to authenticate.")
    if api_type in _OPENAI_OAUTH_API_TYPES and not is_openai_logged_in():
        raise ValueError("Not logged in to OpenAI. Use /login to authenticate.")
    cls = API_TYPE_TO_PROVIDER_CLASS.get(api_type)
    if cls is None:
        raise ValueError(f"Unsupported API type: {api_type.value}")
    return cls(config)


def get_provider_api_type(provider: BaseProvider) -> ApiType:
    return _API_TYPE_BY_PROVIDER.get(type(provider), ApiType.OPENAI_COMPLETIONS)


@dataclass
class RuntimeInitResult:
    provider_error: str | None = None


@dataclass
class CompactionResult:
    tokens_before: int


@dataclass
class HandoffResult:
    prompt: str
    source_session: Session
    new_session: Session


class ConversationRuntime:
    def __init__(
        self,
        *,
        cwd: str,
        model: str,
        model_provider: str | None,
        api_key: str | None,
        base_url: str | None,
        thinking_level: str,
        tools: list[BaseTool],
        openai_compat_auth_mode: AuthMode = "auto",
        anthropic_compat_auth_mode: AuthMode = "auto",
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.base_url = base_url
        self.thinking_level = thinking_level
        self.tools = tools
        self.openai_compat_auth_mode: AuthMode = openai_compat_auth_mode
        self.anthropic_compat_auth_mode: AuthMode = anthropic_compat_auth_mode

        self.provider: BaseProvider | None = None
        self.session: Session | None = None
        self.agent: Agent | None = None

    def resolve_system_prompt(self, session: Session | None = None) -> str:
        return (session.system_prompt if session else None) or build_system_prompt(
            self.cwd, tools=self.tools
        )

    def _provider_config(
        self,
        *,
        model: str,
        provider: str | None,
        base_url: str | None,
        thinking_level: str | None = None,
        session_id: str | None = None,
    ) -> ProviderConfig:
        return ProviderConfig(
            api_key=self.api_key,
            base_url=base_url,
            model=model,
            max_tokens=get_max_tokens(model),
            thinking_level=thinking_level or self.thinking_level,
            provider=provider,
            session_id=session_id,
            openai_compat_auth_mode=self.openai_compat_auth_mode,
            anthropic_compat_auth_mode=self.anthropic_compat_auth_mode,
        )

    def _model_api_and_base_url(
        self, model: str, provider: str | None
    ) -> tuple[ApiType, str | None]:
        model_info = get_model(model, provider)
        if model_info:
            return model_info.api, self.base_url or model_info.base_url
        api_type = resolve_provider_api_type(provider)
        return api_type, self.base_url or default_base_url_for_api(api_type)

    def _new_agent(self, provider: BaseProvider, session: Session) -> Agent:
        return Agent(
            provider=provider,
            tools=self.tools,
            session=session,
            cwd=self.cwd,
            system_prompt=self.resolve_system_prompt(session),
        )

    def initialize(
        self, *, resume_session: str | None = None, continue_recent: bool = False
    ) -> RuntimeInitResult:
        session: Session | None = None
        model = self.model
        model_provider = self.model_provider
        base_url_override = self.base_url
        thinking_level = self.thinking_level

        if resume_session:
            session = Session.continue_by_id(self.cwd, resume_session)
            if session.entries:
                model_info = session.model
                if model_info:
                    model_provider, model, session_base_url = model_info
                    if base_url_override is None and session_base_url:
                        base_url_override = session_base_url
                thinking_level = session.thinking_level
        elif continue_recent:
            session = Session.continue_recent(
                self.cwd,
                provider=model_provider,
                model_id=model,
                thinking_level=thinking_level,
                system_prompt=self.resolve_system_prompt(None),
            )
            if session.entries:
                model_info = session.model
                if model_info:
                    model_provider, model, session_base_url = model_info
                    if base_url_override is None and session_base_url:
                        base_url_override = session_base_url
                thinking_level = session.thinking_level

        self.base_url = base_url_override
        api_type, effective_base_url = self._model_api_and_base_url(model, model_provider)
        provider_config = self._provider_config(
            model=model,
            provider=model_provider,
            base_url=effective_base_url,
            thinking_level=thinking_level,
            session_id=session.id if session else None,
        )

        provider: BaseProvider | None = None
        provider_error: str | None = None
        try:
            provider = create_provider(api_type, provider_config)
        except ValueError as e:
            provider_error = str(e)

        if provider:
            valid_levels = provider.thinking_levels
            if thinking_level not in valid_levels:
                thinking_level = valid_levels[0] if valid_levels else "high"
                provider.set_thinking_level(thinking_level)

        if not continue_recent and not resume_session:
            selected_model = get_model(model, model_provider)
            model_provider = (
                selected_model.provider
                if selected_model
                else (provider.name if provider else model_provider)
            )
            session = Session.create(
                self.cwd,
                provider=model_provider,
                model_id=model,
                thinking_level=thinking_level,
                system_prompt=self.resolve_system_prompt(None),
                tools=[t.name for t in self.tools],
            )
            if model_provider:
                session.append_model_change(model_provider, model, effective_base_url)

        self.model = model
        self.model_provider = model_provider
        self.thinking_level = thinking_level
        self.provider = provider
        self.session = session
        self.agent = self._new_agent(provider, session) if provider and session else None
        self._sync_provider_session_id()

        return RuntimeInitResult(provider_error=provider_error)

    def _sync_provider_session_id(self) -> None:
        if self.provider and self.session:
            self.provider.config.session_id = self.session.id

    def create_session(self) -> Session:
        selected_model = get_model(self.model, self.model_provider)
        model_provider = (
            selected_model.provider
            if selected_model
            else (self.provider.name if self.provider else self.model_provider or "openai")
        )
        model_base_url = selected_model.base_url if selected_model else None
        if model_base_url is None and self.provider:
            model_base_url = self.provider.config.base_url

        session = Session.create(
            self.cwd,
            provider=model_provider,
            model_id=self.model,
            thinking_level=self.thinking_level,
            system_prompt=self.resolve_system_prompt(),
            tools=[t.name for t in self.tools],
        )
        session.append_model_change(model_provider, self.model, model_base_url)
        return session

    def new_session(self, *, reload_context: bool = False) -> Session:
        session = self.create_session()
        self.session = session
        self.model_provider = session.model[0] if session.model else self.model_provider
        self._sync_provider_session_id()
        if self.agent is not None:
            self.agent.session = session
            if reload_context:
                self.agent.reload_context()
        elif self.provider is not None:
            self.agent = self._new_agent(self.provider, session)
        return session

    def switch_model(self, model: Model) -> None:
        current_api_type = get_provider_api_type(self.provider) if self.provider else None
        replacement_provider: BaseProvider | None = None

        if model.api != current_api_type:
            provider_config = self._provider_config(
                model=model.id,
                provider=model.provider,
                base_url=model.base_url,
                session_id=self.session.id if self.session else None,
            )
            replacement_provider = create_provider(model.api, provider_config)

        if replacement_provider is not None:
            self.provider = replacement_provider
        elif self.provider:
            self.provider.config.model = model.id
            self.provider.config.base_url = model.base_url
            self.provider.config.max_tokens = get_max_tokens(model.id)
            self.provider.config.provider = model.provider

        self.model = model.id
        self.model_provider = model.provider

        if self.session:
            self.session.set_model(model.provider, model.id, model.base_url)
        if self.agent and self.provider:
            self.agent.provider = self.provider

    def set_thinking_level(self, level: str) -> None:
        if self.provider is None:
            return
        self.provider.set_thinking_level(level)
        self.thinking_level = level
        if self.session:
            self.session.set_thinking_level(level)

    def load_session(self, session_path: str | Path) -> Session:
        session = Session.load(session_path)
        model = self.model
        model_provider = self.model_provider
        provider = self.provider
        thinking_level = session.thinking_level

        model_info = session.model
        if model_info:
            model_provider, model, session_base_url = model_info
            restored_model = get_model(model, model_provider)
            restored_base_url = session_base_url or (
                restored_model.base_url if restored_model else None
            )

            if restored_model:
                current_api_type = get_provider_api_type(provider) if provider else None
                if restored_model.api != current_api_type:
                    provider_config = self._provider_config(
                        model=model,
                        provider=model_provider,
                        base_url=restored_base_url,
                        thinking_level=thinking_level,
                        session_id=session.id,
                    )
                    provider = create_provider(restored_model.api, provider_config)
                elif provider:
                    pass
                else:
                    provider_config = self._provider_config(
                        model=model,
                        provider=model_provider,
                        base_url=restored_base_url,
                        thinking_level=thinking_level,
                        session_id=session.id,
                    )
                    provider = create_provider(restored_model.api, provider_config)
            elif provider is None:
                api_type = resolve_provider_api_type(model_provider)
                provider_config = self._provider_config(
                    model=model,
                    provider=model_provider,
                    base_url=restored_base_url or default_base_url_for_api(api_type),
                    thinking_level=thinking_level,
                    session_id=session.id,
                )
                provider = create_provider(api_type, provider_config)
        else:
            restored_base_url = None

        if provider:
            valid_levels = provider.thinking_levels
            if valid_levels and thinking_level not in valid_levels:
                thinking_level = valid_levels[0]

        # Commit only after all provider construction/validation above has succeeded.
        self.session = session
        self.model = model
        self.model_provider = model_provider
        self.thinking_level = thinking_level
        self.provider = provider

        if model_info and self.provider:
            self.provider.config.model = model
            if restored_base_url:
                self.provider.config.base_url = restored_base_url
            self.provider.config.max_tokens = get_max_tokens(model)
            self.provider.config.provider = model_provider
            self.provider.config.session_id = session.id

        if self.provider:
            self.provider.set_thinking_level(thinking_level)
            self.agent = self._new_agent(self.provider, session)
        elif self.agent is not None:
            self.agent.session = session

        return session

    def prepare_for_run(self) -> Agent | None:
        if self.provider is None or self.session is None:
            return None
        if self.agent is None:
            self.agent = self._new_agent(self.provider, self.session)

        model_info = get_model(self.model, self.model_provider)
        self.agent.provider = self.provider
        self.agent.session = self.session
        self.agent.tools = self.tools
        self.agent.config.context_window = model_info.context_window if model_info else None
        self.agent.config.max_output_tokens = model_info.max_tokens if model_info else None
        return self.agent

    def reload_context(self) -> None:
        if self.agent is not None:
            self.agent.reload_context()

    def latest_assistant_usage_tokens(self) -> int:
        if self.session is None:
            return 0
        for entry in reversed(self.session.entries):
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage is None:
                    continue
                return (
                    usage.input_tokens
                    + usage.output_tokens
                    + usage.cache_read_tokens
                    + usage.cache_write_tokens
                )
        return 0

    async def compact_now(self) -> CompactionResult:
        if self.provider is None or self.session is None or self.agent is None:
            raise RuntimeError("Agent not initialized")

        tokens_before = self.latest_assistant_usage_tokens()
        summary = await generate_summary(
            self.session.all_messages, self.provider, system_prompt=self.agent.system_prompt
        )
        self.session.append_compaction(
            summary=summary,
            first_kept_entry_id=self.session.leaf_id or "",
            tokens_before=tokens_before,
        )
        return CompactionResult(tokens_before=tokens_before)

    async def create_handoff(self, query: str) -> HandoffResult:
        if self.provider is None or self.session is None or self.agent is None:
            raise RuntimeError("Agent not initialized")

        source_session = self.session
        prompt = await generate_handoff_prompt(
            source_session.all_messages,
            self.provider,
            system_prompt=self.agent.system_prompt,
            query=query,
        )

        source_session_id = source_session.id
        new_session = self.create_session()
        new_session.append_custom_message(
            "handoff_backlink",
            f"Handoff from {source_session_id[:8]}",
            display=False,
            details={"target_session_id": source_session_id, "query": query},
        )
        source_session.append_custom_message(
            "handoff_forward_link",
            f"Handoff to {new_session.id[:8]}",
            display=False,
            details={"target_session_id": new_session.id, "query": query},
        )

        new_session.ensure_persisted()
        source_session.ensure_persisted()

        self.session = new_session
        self._sync_provider_session_id()
        if self.agent is not None:
            self.agent.session = new_session

        return HandoffResult(prompt=prompt, source_session=source_session, new_session=new_session)
