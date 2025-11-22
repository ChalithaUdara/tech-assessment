from typing import Optional, Any
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from datacom_ai.rag.embeddings import CustomFastEmbedEmbeddings
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from datacom_ai.config.settings import settings
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
    def create_store(embeddings: Embeddings, client: Any) -> QdrantVectorStore:
        """Create a vector store instance."""
        if settings.VECTOR_STORE_TYPE == "qdrant":
            # Ensure collection exists logic could be moved here or kept in indexing
            # For now, let's keep it simple and just return the store wrapper
            if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                 vector_size = 384 if settings.EMBEDDING_PROVIDER == "fastembed" else 1536
                 logger.info(f"Creating collection with vector size: {vector_size}")
                 
                 client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )

            return QdrantVectorStore(
                client=client,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embedding=embeddings,
            )
        else:
            raise ValueError(f"Unsupported vector store type: {settings.VECTOR_STORE_TYPE}")
