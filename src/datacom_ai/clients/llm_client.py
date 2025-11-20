"""LangChain LLM client initialization for Azure OpenAI."""

from langchain_openai import AzureChatOpenAI

from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger


def create_llm_client() -> AzureChatOpenAI:
    """
    Create and configure an AzureChatOpenAI client for Azure OpenAI.

    Returns:
        Configured AzureChatOpenAI instance with streaming enabled

    Raises:
        ValueError: If required configuration is missing
    """
    logger.info("Creating AzureChatOpenAI client")
    settings.validate()

    # Normalize Azure endpoint URL (ensure it ends with a trailing slash)
    endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/") + "/"
    logger.debug(
        f"Client configuration: endpoint={endpoint[:50]}..., "
        f"deployment={settings.AZURE_OPENAI_DEPLOYMENT}, "
        f"api_version={settings.AZURE_OPENAI_API_VERSION}, "
        f"model={settings.MODEL_NAME}"
    )

    client = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_key=settings.OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        streaming=True,
        temperature=0.7,
    )

    logger.success("AzureChatOpenAI client created successfully")
    return client

