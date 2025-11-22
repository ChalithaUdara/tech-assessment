"""Vector store adapters for abstracting vector store operations."""

from typing import Protocol, Any
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger


class VectorStoreAdapter(Protocol):
    """
    Protocol for vector store adapters that abstract vector store operations.
    """

    def ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure the collection exists with correct configuration."""
        ...

    def get_vector_size(self) -> int:
        """Get the vector size for this adapter."""
        ...

    def create_store(self, embeddings: Embeddings) -> VectorStore:
        """Create a vector store instance."""
        ...


class QdrantAdapter:
    """
    Adapter for Qdrant vector store operations.
    Encapsulates all Qdrant-specific logic.
    """

    def __init__(self, client: Any, embedding_provider: str, collection_name: str):
        """
        Initialize the Qdrant adapter.

        Args:
            client: QdrantClient instance
            embedding_provider: Name of the embedding provider
            collection_name: Name of the collection
        """
        self.client = client
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self._vector_size = settings.get_vector_size(embedding_provider)

    def get_vector_size(self) -> int:
        """Get the vector size for this adapter."""
        return self._vector_size

    def ensure_collection_exists(self, collection_name: str | None = None) -> None:
        """
        Ensure Qdrant collection exists with correct configuration.

        Args:
            collection_name: Optional collection name (uses instance default if not provided)
        """
        from qdrant_client.http import models

        target_collection = collection_name or self.collection_name

        if not self.client.collection_exists(target_collection):
            logger.info(f"Creating collection '{target_collection}' with vector size: {self._vector_size}")

            self.client.create_collection(
                collection_name=target_collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE
                ),
            )
        else:
            logger.debug(f"Collection '{target_collection}' already exists")

    def create_store(self, embeddings: Embeddings) -> VectorStore:
        """
        Create a QdrantVectorStore instance.

        Args:
            embeddings: Embeddings instance

        Returns:
            QdrantVectorStore instance
        """
        from langchain_qdrant import QdrantVectorStore

        # Ensure collection exists before creating store
        self.ensure_collection_exists()

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

