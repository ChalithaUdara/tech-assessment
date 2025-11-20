"""Configuration settings for the Datacom AI platform."""

import os
from typing import Optional

from dotenv import load_dotenv
from datacom_ai.utils.logger import logger

# Load environment variables from .env file
load_dotenv()
logger.debug("Environment variables loaded from .env file")


class Settings:
    """Application settings loaded from environment variables."""

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")  # For display/telemetry purposes

    # GPT-4o Pricing Constants (per 1M tokens)
    # Source: OpenAI pricing as of 2024
    GPT4O_INPUT_PRICE_PER_1M: float = 2.50  # USD per 1M input tokens
    GPT4O_OUTPUT_PRICE_PER_1M: float = 10.00  # USD per 1M output tokens

    # Message Persistence
    MAX_MESSAGES: int = 10  # Keep last N messages

    @classmethod
    def validate(cls) -> None:
        """Validate that required environment variables are set."""
        logger.debug("Validating configuration settings")
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            logger.error("AZURE_OPENAI_ENDPOINT is not set")
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable is required. "
                "Set it in your .env file or environment. "
                "Format: https://<your-resource-name>.openai.azure.com/"
            )
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set")
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )
        if not cls.AZURE_OPENAI_DEPLOYMENT:
            logger.error("AZURE_OPENAI_DEPLOYMENT is not set")
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT environment variable is required. "
                "Set it in your .env file or environment. "
                "This should be the name of your Azure OpenAI deployment."
            )
        
        logger.info("Configuration validation successful")
        logger.debug(
            f"Configuration: endpoint={cls.AZURE_OPENAI_ENDPOINT[:50]}..., "
            f"deployment={cls.AZURE_OPENAI_DEPLOYMENT}, "
            f"api_version={cls.AZURE_OPENAI_API_VERSION}, "
            f"model={cls.MODEL_NAME}"
        )

    @classmethod
    def calculate_cost(
        cls, input_tokens: int, output_tokens: int, model_name: Optional[str] = None
    ) -> float:
        """
        Calculate the cost in USD for a given token usage.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model_name: Optional model name (currently only GPT-4o supported)

        Returns:
            Total cost in USD
        """
        # For now, we only support GPT-4o pricing
        # Can be extended for other models
        input_cost = (input_tokens / 1_000_000) * cls.GPT4O_INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * cls.GPT4O_OUTPUT_PRICE_PER_1M
        return input_cost + output_cost


# Global settings instance
settings = Settings()

