import os

from anthropic import AsyncAnthropic

from ..base import ProviderConfig, get_env_api_key
from .anthropic import AnthropicProvider


class AzureAIFoundryProvider(AnthropicProvider):
    name = "azure-ai-foundry"

    def __init__(self, config: ProviderConfig):
        # Skip AnthropicProvider.__init__ — we resolve key + base_url ourselves
        super(AnthropicProvider, self).__init__(config)

        api_key = config.api_key or get_env_api_key(self.name)
        if not api_key:
            raise ValueError(
                f"No API key found for {self.name}. "
                "Set AZURE_AI_FOUNDRY_API_KEY environment variable or pass api_key in config."
            )

        base_url = config.base_url or os.environ.get("AZURE_AI_FOUNDRY_BASE_URL")
        if not base_url:
            raise ValueError(
                "No base URL found for azure-ai-foundry. "
                "Set AZURE_AI_FOUNDRY_BASE_URL environment variable or pass base_url in config."
            )

        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
