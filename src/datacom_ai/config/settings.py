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
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # GPT-4o Pricing Constants (per 1M tokens)
    # Source: OpenAI pricing as of 2024
    GPT4O_INPUT_PRICE_PER_1M: float = 2.50  # USD per 1M input tokens
    GPT4O_OUTPUT_PRICE_PER_1M: float = 10.00  # USD per 1M output tokens

    # Message Persistence
    MAX_MESSAGES: int = 10  # Keep last N messages

    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "datacom_rag")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Text Splitter Configuration
    # Options: "recursive", "character"
    TEXT_SPLITTER_TYPE: str = os.getenv("TEXT_SPLITTER_TYPE", "recursive")

    # Vector Store Configuration
    # Options: "qdrant"
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "qdrant")
    
    # Embedding Configuration
    # Options: "azure_openai", "fastembed"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    
    # For Azure OpenAI
    EMBEDDING_DEPLOYMENT: str = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    # For FastEmbed
    # Default: BAAI/bge-small-en-v1.5
    FASTEMBED_MODEL_NAME: str = os.getenv("FASTEMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    
    # Vector size configuration (mapping from embedding provider to vector dimension)
    EMBEDDING_VECTOR_SIZES: dict[str, int] = {
        "fastembed": 384,
        "azure_openai": 1536,
    }
    
    @classmethod
    def get_vector_size(cls, embedding_provider: str) -> int:
        """
        Get vector size for a given embedding provider.
        
        Args:
            embedding_provider: The embedding provider name
            
        Returns:
            Vector dimension size
        """
        return cls.EMBEDDING_VECTOR_SIZES.get(embedding_provider, 1536)

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

