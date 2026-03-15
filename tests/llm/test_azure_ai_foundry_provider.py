import os
from unittest.mock import patch

import pytest

from kon.llm.base import ProviderConfig
from kon.llm.providers.azure_ai_foundry import AzureAIFoundryProvider


def test_azure_ai_foundry_inherits_anthropic_conversion():
    provider = AzureAIFoundryProvider.__new__(AzureAIFoundryProvider)
    assert provider.name == "azure-ai-foundry"
    # Should have anthropic thinking levels
    assert "xhigh" in provider.thinking_levels


def test_azure_ai_foundry_requires_api_key():
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="No API key found"),
    ):
        AzureAIFoundryProvider(ProviderConfig(model="claude-sonnet-4.6"))


def test_azure_ai_foundry_requires_base_url():
    with (
        patch.dict(os.environ, {"AZURE_AI_FOUNDRY_API_KEY": "test-key"}, clear=True),
        pytest.raises(ValueError, match="No base URL found"),
    ):
        AzureAIFoundryProvider(ProviderConfig(model="claude-sonnet-4.6"))


def test_azure_ai_foundry_init_from_env():
    env = {
        "AZURE_AI_FOUNDRY_API_KEY": "test-key",
        "AZURE_AI_FOUNDRY_BASE_URL": "https://example.services.ai.azure.com/anthropic/",
    }
    with patch.dict(os.environ, env, clear=True):
        provider = AzureAIFoundryProvider(ProviderConfig(model="claude-sonnet-4.6"))
        assert provider._client.api_key == "test-key"
        assert "example.services.ai.azure.com" in str(provider._client.base_url)


def test_azure_ai_foundry_init_from_config():
    with patch.dict(os.environ, {}, clear=True):
        provider = AzureAIFoundryProvider(
            ProviderConfig(
                model="claude-sonnet-4.6",
                api_key="config-key",
                base_url="https://myresource.services.ai.azure.com/anthropic/",
            )
        )
        assert provider._client.api_key == "config-key"
        assert "myresource.services.ai.azure.com" in str(provider._client.base_url)
