from ..base import BaseProvider
from ..models import ApiType

PROVIDER_API_BY_NAME: dict[str, ApiType] = {
    "openai": ApiType.OPENAI_COMPLETIONS,
    "zhipu": ApiType.OPENAI_COMPLETIONS,
    "deepseek": ApiType.OPENAI_COMPLETIONS,
    "xai": ApiType.OPENAI_COMPLETIONS,
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


def get_provider_class(api_type: ApiType) -> type[BaseProvider]:
    match api_type:
        case ApiType.GITHUB_COPILOT:
            from .copilot import CopilotProvider

            return CopilotProvider
        case ApiType.GITHUB_COPILOT_RESPONSES:
            from .copilot import CopilotResponsesProvider

            return CopilotResponsesProvider
        case ApiType.OPENAI_RESPONSES:
            from .openai_responses import OpenAIResponsesProvider

            return OpenAIResponsesProvider
        case ApiType.OPENAI_CODEX_RESPONSES:
            from .openai_codex_responses import OpenAICodexResponsesProvider

            return OpenAICodexResponsesProvider
        case ApiType.ANTHROPIC_COPILOT:
            from .copilot_anthropic import CopilotAnthropicProvider

            return CopilotAnthropicProvider
        case ApiType.AZURE_AI_FOUNDRY:
            from .azure_ai_foundry import AzureAIFoundryProvider

            return AzureAIFoundryProvider
        case ApiType.OPENAI_COMPLETIONS:
            from .openai_completions import OpenAICompletionsProvider

            return OpenAICompletionsProvider

    raise ValueError(f"Unsupported API type: {api_type.value}")


__all__ = ["PROVIDER_API_BY_NAME", "get_provider_class", "resolve_provider_api_type"]
