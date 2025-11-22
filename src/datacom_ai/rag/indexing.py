from typing import List, Any
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models

from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger

class IndexPipeline:
    """Handles document splitting, embedding, and indexing."""

    def __init__(
        self, 
        text_splitter: TextSplitter, 
        embeddings: Embeddings, 
        vector_store_client: Any
    ):
        self.text_splitter = text_splitter
        self.embeddings = embeddings
        self.client = vector_store_client

    def run(self, documents: List[Document]):
        """
        Run the indexing pipeline: split -> embed -> index.
        """
        if not documents:
            logger.warning("No documents to index.")
            return

        # 1. Split documents
        logger.info(f"Splitting documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

        # 2. Index into Vector Store
        # Note: We are tightly coupled to Qdrant here for collection creation. 
        # Ideally this should be abstracted further, but for now this is better than before.
        if settings.VECTOR_STORE_TYPE == "qdrant":
            logger.info(f"Indexing into Qdrant collection: {settings.QDRANT_COLLECTION_NAME}...")
            
            # Ensure collection exists
            if not self.client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                 vector_size = 384 if settings.EMBEDDING_PROVIDER == "fastembed" else 1536
                 logger.info(f"Creating collection with vector size: {vector_size}")
                 
                 self.client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )

            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                force_recreate=False 
            )
        else:
             raise ValueError(f"Unsupported vector store type for indexing: {settings.VECTOR_STORE_TYPE}")
        
        logger.info("Indexing complete.")
