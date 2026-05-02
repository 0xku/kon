from ..base import BaseProvider
from ..models import ApiType
from .azure_ai_foundry import AzureAIFoundryProvider
from .copilot import CopilotProvider, CopilotResponsesProvider, is_copilot_logged_in
from .copilot_anthropic import CopilotAnthropicProvider
from .mock import MockProvider
from .openai_codex_responses import OpenAICodexResponsesProvider, is_openai_logged_in
from .openai_completions import OpenAICompletionsProvider
from .openai_responses import OpenAIResponsesProvider

API_TYPE_TO_PROVIDER_CLASS: dict[ApiType, type[BaseProvider]] = {
    ApiType.GITHUB_COPILOT: CopilotProvider,
    ApiType.GITHUB_COPILOT_RESPONSES: CopilotResponsesProvider,
    ApiType.OPENAI_RESPONSES: OpenAIResponsesProvider,
    ApiType.OPENAI_CODEX_RESPONSES: OpenAICodexResponsesProvider,
    ApiType.ANTHROPIC_COPILOT: CopilotAnthropicProvider,
    ApiType.AZURE_AI_FOUNDRY: AzureAIFoundryProvider,
    ApiType.OPENAI_COMPLETIONS: OpenAICompletionsProvider,
}

PROVIDER_API_BY_NAME: dict[str, ApiType] = {
    "openai": ApiType.OPENAI_COMPLETIONS,
    "zhipu": ApiType.OPENAI_COMPLETIONS,
    "deepseek": ApiType.OPENAI_COMPLETIONS,
    "github-copilot": ApiType.GITHUB_COPILOT,
    "openai-responses": ApiType.OPENAI_RESPONSES,
    "openai-codex": ApiType.OPENAI_CODEX_RESPONSES,
    "azure-ai-foundry": ApiType.AZURE_AI_FOUNDRY,
}


def resolve_provider_api_type(provider: str | None) -> ApiType:
    if provider is None:
        return ApiType.OPENAI_COMPLETIONS

    api_type = PROVIDER_API_BY_NAME.get(provider)
    if api_type is None:
        valid = ", ".join(sorted(PROVIDER_API_BY_NAME))
        raise ValueError(f"Unknown provider '{provider}'. Valid providers: {valid}")

    return api_type


__all__ = [
    "API_TYPE_TO_PROVIDER_CLASS",
    "PROVIDER_API_BY_NAME",
    "AzureAIFoundryProvider",
    "CopilotAnthropicProvider",
    "CopilotProvider",
    "CopilotResponsesProvider",
    "MockProvider",
    "OpenAICodexResponsesProvider",
    "OpenAICompletionsProvider",
    "OpenAIResponsesProvider",
    "is_copilot_logged_in",
    "is_openai_logged_in",
    "resolve_provider_api_type",
]
