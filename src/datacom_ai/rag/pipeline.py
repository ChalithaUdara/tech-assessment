from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from datacom_ai.rag.loader import DocumentLoader
from datacom_ai.rag.indexing import IndexPipeline
from datacom_ai.rag.retrieval import RetrievalPipeline
from datacom_ai.rag.factories import (
    EmbeddingFactory,
    TextSplitterFactory,
    VectorStoreFactory,
    LLMFactory
)
from datacom_ai.utils.logger import logger

class RAGPipeline:
    """Facade for the RAG system."""

    def __init__(self):
        # Create components using factories
        self.embeddings = EmbeddingFactory.create()
        self.text_splitter = TextSplitterFactory.create()
        self.llm = LLMFactory.create()
        
        # Vector store needs client and embeddings
        self.vector_store_client = VectorStoreFactory.create_client()
        self.vector_store = VectorStoreFactory.create_store(
            self.embeddings, 
            self.vector_store_client
        )

        # Inject dependencies
        self.indexer = IndexPipeline(
            text_splitter=self.text_splitter,
            embeddings=self.embeddings,
            vector_store_client=self.vector_store_client
        )
        
        self.retriever = RetrievalPipeline(
            vector_store=self.vector_store,
            llm=self.llm
        )

    def index_directory(self, directory_path: str):
        """Load and index documents from a directory."""
        loader = DocumentLoader(directory_path)
        documents = loader.load_documents()
        self.indexer.run(documents)

    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG system."""
        return self.retriever.run(query)

    def stream(self, query: str, conversation_history: Optional[List[BaseMessage]] = None):
        """
        Stream the RAG system response.
        
        Args:
            query: The current query/question
            conversation_history: Optional conversation history to include in context
        """
        return self.retriever.stream(query, conversation_history)
