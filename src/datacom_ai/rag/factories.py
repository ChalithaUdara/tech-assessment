from typing import Optional, Any
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from datacom_ai.rag.embeddings import CustomFastEmbedEmbeddings
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from datacom_ai.config.settings import settings
from datacom_ai.rag.adapters import QdrantAdapter, VectorStoreAdapter
from datacom_ai.utils.logger import logger

class EmbeddingFactory:
    @staticmethod
    def create() -> Embeddings:
        """Create an embedding model based on configuration."""
        if settings.EMBEDDING_PROVIDER == "fastembed":
            logger.info(f"Using FastEmbed with model: {settings.FASTEMBED_MODEL_NAME}")
            return CustomFastEmbedEmbeddings(model_name=settings.FASTEMBED_MODEL_NAME)
        elif settings.EMBEDDING_PROVIDER == "azure_openai":
            logger.info("Using Azure OpenAI Embeddings")
            return AzureOpenAIEmbeddings(
                azure_deployment=settings.EMBEDDING_DEPLOYMENT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.OPENAI_API_KEY,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")

class TextSplitterFactory:
    @staticmethod
    def create() -> TextSplitter:
        """Create a text splitter based on configuration."""
        if settings.TEXT_SPLITTER_TYPE == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )
        elif settings.TEXT_SPLITTER_TYPE == "character":
            return CharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separator="\n\n"
            )
        else:
            raise ValueError(f"Unsupported text splitter type: {settings.TEXT_SPLITTER_TYPE}")

class LLMFactory:
    @staticmethod
    def create() -> BaseChatModel:
        """Create an LLM based on configuration."""
        from datacom_ai.clients.llm_client import create_llm_client
        return create_llm_client()

class VectorStoreFactory:
    """Factory for creating vector store clients and adapters."""
    
    @staticmethod
    def create_client() -> Any:
        """Create a vector store client."""
        if settings.VECTOR_STORE_TYPE == "qdrant":
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        else:
            raise ValueError(f"Unsupported vector store type: {settings.VECTOR_STORE_TYPE}")

    @staticmethod
    def create_adapter(client: Any, embedding_provider: str, collection_name: str) -> VectorStoreAdapter:
        """
        Create a vector store adapter.
        
        Args:
            client: Vector store client
            embedding_provider: Embedding provider name
            collection_name: Collection name
            
        Returns:
            VectorStoreAdapter instance
        """
        if settings.VECTOR_STORE_TYPE == "qdrant":
            return QdrantAdapter(client, embedding_provider, collection_name)
        else:
            raise ValueError(f"Unsupported vector store type: {settings.VECTOR_STORE_TYPE}")

    @staticmethod
    def create_store(embeddings: Embeddings, client: Any) -> QdrantVectorStore:
        """
        Create a vector store instance.
        
        This method uses the adapter pattern internally to ensure collection exists.
        """
        adapter = VectorStoreFactory.create_adapter(
            client=client,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        return adapter.create_store(embeddings)
    
    @staticmethod
    def ensure_collection_exists(
        client: Any,
        collection_name: str,
        embedding_provider: str
    ) -> None:
        """
        Ensure collection exists with correct configuration.
        
        This centralizes collection creation logic that was duplicated.
        
        Args:
            client: Vector store client
            collection_name: Collection name
            embedding_provider: Embedding provider name
        """
        adapter = VectorStoreFactory.create_adapter(
            client=client,
            embedding_provider=embedding_provider,
            collection_name=collection_name
        )
        adapter.ensure_collection_exists()
