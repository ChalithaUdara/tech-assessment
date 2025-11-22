"""Unit tests for LLM client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_openai import AzureChatOpenAI

from datacom_ai.clients.llm_client import create_llm_client
from datacom_ai.config.settings import Settings


class TestLLMClient:
    """Tests for LLM client creation."""

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object."""
        settings = Mock(spec=Settings)
        settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
        settings.OPENAI_API_KEY = "test-key"
        settings.AZURE_OPENAI_DEPLOYMENT = "test-deployment"
        settings.AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
        settings.MODEL_NAME = "gpt-4o"
        settings.LLM_TEMPERATURE = 0.7
        settings.validate = Mock()
        return settings

    def test_create_llm_client_success(self, mock_settings):
        """Test successful LLM client creation."""
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                client = create_llm_client()

                assert client == mock_client
                mock_settings.validate.assert_called_once()
                mock_client_class.assert_called_once()
                
                # Verify call arguments
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
                assert call_kwargs["azure_deployment"] == "test-deployment"
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["api_version"] == "2024-02-15-preview"
                assert call_kwargs["streaming"] is True
                assert call_kwargs["temperature"] == 0.7

    def test_create_llm_client_endpoint_normalization(self, mock_settings):
        """Test that endpoint URL is normalized with trailing slash."""
        # Test with endpoint that already has trailing slash
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
        
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"

        # Test with endpoint without trailing slash
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
        
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"

    def test_create_llm_client_validation_error(self, mock_settings):
        """Test that validation errors are raised."""
        mock_settings.validate.side_effect = ValueError("Missing endpoint")

        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with pytest.raises(ValueError, match="Missing endpoint"):
                create_llm_client()

    def test_create_llm_client_streaming_enabled(self, mock_settings):
        """Test that streaming is always enabled."""
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["streaming"] is True

    def test_create_llm_client_temperature_setting(self, mock_settings):
        """Test that temperature is set correctly."""
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["temperature"] == 0.7

    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("https://test.openai.azure.com", "https://test.openai.azure.com/"),
            ("https://test.openai.azure.com/", "https://test.openai.azure.com/"),
            ("https://test.openai.azure.com//", "https://test.openai.azure.com/"),
        ],
    )
    def test_endpoint_normalization_variations(self, endpoint, expected, mock_settings):
        """Test endpoint normalization with various formats."""
        mock_settings.AZURE_OPENAI_ENDPOINT = endpoint
        
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["azure_endpoint"] == expected

    def test_create_llm_client_with_different_api_version(self, mock_settings):
        """Test client creation with different API version."""
        mock_settings.AZURE_OPENAI_API_VERSION = "2024-01-01"
        
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["api_version"] == "2024-01-01"

    def test_create_llm_client_with_different_deployment(self, mock_settings):
        """Test client creation with different deployment name."""
        mock_settings.AZURE_OPENAI_DEPLOYMENT = "gpt-4o-deployment"
        
        with patch("datacom_ai.clients.llm_client.settings", mock_settings):
            with patch("datacom_ai.clients.llm_client.AzureChatOpenAI") as mock_client_class:
                mock_client = Mock(spec=AzureChatOpenAI)
                mock_client_class.return_value = mock_client

                create_llm_client()
                
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["azure_deployment"] == "gpt-4o-deployment"

