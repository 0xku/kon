"""
Manually maintained model catalog.

Add models here as needed. Each model defines its capabilities,
API type, and any special handling (e.g., vision fallback model).
"""
# TODO: should use something like https://github.com/anomalyco/models.dev in future

from dataclasses import dataclass
from enum import Enum

DEFAULT_MAX_TOKENS = 16384


class ApiType(Enum):
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    OPENAI_CODEX_RESPONSES = "openai-codex-responses"
    ANTHROPIC_COPILOT = "anthropic-copilot"
    AZURE_AI_FOUNDRY = "azure-ai-foundry"
    GITHUB_COPILOT = "github-copilot"
    GITHUB_COPILOT_RESPONSES = "github-copilot-responses"


@dataclass
class Model:
    id: str  # Model ID (e.g., "glm-4.7", "glm-5.1", "claude-opus-4.5")
    provider: str  # "openai", "zhipu", "github-copilot", "openai-codex"
    api: ApiType  # Which API format to use
    base_url: str  # API endpoint
    max_tokens: int  # Max output tokens
    supports_images: bool  # Native vision support
    supports_thinking: bool  # Reasoning/thinking support
    context_window: int | None = None  # Max context (None = use config default)
    vision_model: str | None = None  # Fallback vision model if no native support


MODELS: dict[str, Model] = {
    # ZhiPu models
    "glm-4.7": Model(
        id="glm-4.7",
        provider="zhipu",
        api=ApiType.OPENAI_COMPLETIONS,
        base_url="https://api.z.ai/api/coding/paas/v4",
        max_tokens=8192,
        supports_images=False,
        supports_thinking=True,
        vision_model="glm-4v-flash",
    ),
    "glm-5.1": Model(
        id="glm-5.1",
        provider="zhipu",
        api=ApiType.OPENAI_COMPLETIONS,
        base_url="https://api.z.ai/api/coding/paas/v4",
        max_tokens=8192,
        supports_images=False,
        supports_thinking=True,
        vision_model="glm-4v-flash",
    ),
    # DeepSeek models (OpenAI-compatible Chat Completions API)
    "deepseek-v4-flash": Model(
        id="deepseek-v4-flash",
        provider="deepseek",
        api=ApiType.OPENAI_COMPLETIONS,
        base_url="https://api.deepseek.com",
        max_tokens=8192,
        supports_images=False,
        supports_thinking=True,
    ),
    "deepseek-v4-pro": Model(
        id="deepseek-v4-pro",
        provider="deepseek",
        api=ApiType.OPENAI_COMPLETIONS,
        base_url="https://api.deepseek.com",
        max_tokens=8192,
        supports_images=False,
        supports_thinking=True,
    ),
    # GitHub Copilot models - Claude (uses Anthropic Messages API for thinking support)
    "claude-sonnet-4.6-copilot": Model(
        id="claude-sonnet-4.6",
        provider="github-copilot",
        api=ApiType.ANTHROPIC_COPILOT,
        base_url="https://api.individual.githubcopilot.com",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "claude-opus-4.6-copilot": Model(
        id="claude-opus-4.6",
        provider="github-copilot",
        api=ApiType.ANTHROPIC_COPILOT,
        base_url="https://api.individual.githubcopilot.com",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    # GitHub Copilot models - GPT/Codex (uses OpenAI Responses API)
    "gpt-5.3-codex-copilot": Model(
        id="gpt-5.3-codex",
        provider="github-copilot",
        api=ApiType.GITHUB_COPILOT_RESPONSES,
        base_url="https://api.individual.githubcopilot.com",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "gpt-5.4-copilot": Model(
        id="gpt-5.4",
        provider="github-copilot",
        api=ApiType.GITHUB_COPILOT_RESPONSES,
        base_url="https://api.individual.githubcopilot.com",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "gpt-5.5-copilot": Model(
        id="gpt-5.5",
        provider="github-copilot",
        api=ApiType.GITHUB_COPILOT_RESPONSES,
        base_url="https://api.individual.githubcopilot.com",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    # OpenAI Codex OAuth models (ChatGPT Plus/Pro subscription)
    "gpt-5.3-codex": Model(
        id="gpt-5.3-codex",
        provider="openai-codex",
        api=ApiType.OPENAI_CODEX_RESPONSES,
        base_url="https://chatgpt.com/backend-api",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "gpt-5.4": Model(
        id="gpt-5.4",
        provider="openai-codex",
        api=ApiType.OPENAI_CODEX_RESPONSES,
        base_url="https://chatgpt.com/backend-api",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "gpt-5.5": Model(
        id="gpt-5.5",
        provider="openai-codex",
        api=ApiType.OPENAI_CODEX_RESPONSES,
        base_url="https://chatgpt.com/backend-api",
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    # Azure AI Foundry models (Anthropic via Azure)
    "claude-sonnet-4.6-azure": Model(
        id="claude-sonnet-4.6",
        provider="azure-ai-foundry",
        api=ApiType.AZURE_AI_FOUNDRY,
        base_url="",  # resolved from AZURE_AI_FOUNDRY_BASE_URL env var
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
    "claude-opus-4.6-azure": Model(
        id="claude-opus-4.6",
        provider="azure-ai-foundry",
        api=ApiType.AZURE_AI_FOUNDRY,
        base_url="",  # resolved from AZURE_AI_FOUNDRY_BASE_URL env var
        max_tokens=8192 * 2,
        supports_images=True,
        supports_thinking=True,
    ),
}


def get_model(model_id: str, provider: str | None = None) -> Model | None:
    if provider:
        for model in MODELS.values():
            if model.id == model_id and model.provider == provider:
                return model

    direct = MODELS.get(model_id)
    if direct:
        return direct

    for model in MODELS.values():
        if model.id == model_id:
            return model

    return None


def get_all_models() -> list[Model]:
    return list(MODELS.values())


def get_models_by_provider(provider: str) -> list[Model]:
    return [m for m in MODELS.values() if m.provider == provider]


def get_max_tokens(model_id: str) -> int:
    model = MODELS.get(model_id)
    return model.max_tokens if model else DEFAULT_MAX_TOKENS
