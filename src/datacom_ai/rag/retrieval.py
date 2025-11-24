from typing import Dict, Any, List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger
from datacom_ai.utils.structured_logging import log_rag_query, log_rag_retrieval
import time

class RetrievalPipeline:
    """Handles retrieval and generation using RAG."""
    
    # System prompt for RAG - extracted to avoid duplication
    SYSTEM_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for RAG."""
        return ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

    def __init__(
        self, 
        vector_store: VectorStore, 
        llm: BaseChatModel,
        k: Optional[int] = None
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            vector_store: Vector store for document retrieval
            llm: Language model for generation
            k: Number of documents to retrieve (defaults to settings.RETRIEVER_K or 4)
        """
        self.vector_store = vector_store
        self.llm = llm
        # Use settings value if available, otherwise default to 4
        self.k = k or getattr(settings, 'RETRIEVER_K', None) or 4

    def get_retriever(self, k: Optional[int] = None):
        """
        Get the retriever from the vector store.

        Args:
            k: Optional override for number of documents to retrieve.
               If not provided, uses the instance's k value.

        Returns:
            Configured retriever
        """
        k = k or self.k
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the RAG pipeline for a given query.
        """
        logger.info(f"Running RAG for query: {query}")
        
        retriever = self.get_retriever()
        prompt = self._create_prompt()
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response

    def stream(self, query: str, conversation_history: Optional[List[BaseMessage]] = None):
        """
        Stream the RAG pipeline response.
        
        Args:
            query: The current query/question
            conversation_history: Optional conversation history to include in context
        """
        logger.info(f"Streaming RAG for query: {query}")
        if conversation_history:
            logger.debug(f"Including {len(conversation_history)} messages from conversation history")
        
        # Log RAG query
        log_rag_query(query=query, k_value=self.k, conversation_history_length=len(conversation_history) if conversation_history else 0)
        
        retriever = self.get_retriever()
        prompt = self._create_prompt()
        
        # Stream the response
        # First, retrieve documents with timing
        retrieval_start = time.time()
        docs = retriever.invoke(query)
        retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
        
        # Extract similarity scores if available
        retrieval_scores = []
        if docs:
            for doc in docs:
                # Try to extract score from metadata
                if hasattr(doc, "metadata") and doc.metadata:
                    score = doc.metadata.get("score") or doc.metadata.get("similarity_score")
                    if score is not None:
                        retrieval_scores.append(float(score))
                # If no score in metadata, try to get from the document object itself
                elif hasattr(doc, "score"):
                    retrieval_scores.append(float(doc.score))
        
        # Log retrieval metrics
        log_rag_retrieval(
            query=query,
            k_value=self.k,
            retrieved_doc_count=len(docs),
            retrieval_scores=retrieval_scores if retrieval_scores else None,
            retrieval_latency_ms=retrieval_latency_ms,
            success=True
        )
        
        yield {"context": docs}
        
        # Then generate answer using the retrieved documents
        # We manually construct the chain to ensure we get the raw LLM output (with usage metadata)
        # instead of just the string content which create_stuff_documents_chain might return
        
        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)
            
        formatted_context = format_docs(docs)
        
        # Build messages with conversation history if provided
        messages = []
        
        # Add system prompt with context
        # Format the system prompt with the retrieved context
        system_content = self.SYSTEM_PROMPT.format(context=formatted_context)
        messages.append(SystemMessage(content=system_content))
        
        # Add conversation history (excluding the last message which is the current query)
        if conversation_history:
            # Exclude the last message as it's the current query
            history_messages = conversation_history[:-1] if len(conversation_history) > 1 else []
            for msg in history_messages:
                messages.append(msg)
        
        # Add the current query
        messages.append(HumanMessage(content=query))
        
        # Stream directly from LLM
        for chunk in self.llm.stream(messages):
            # Yield content
            if chunk.content:
                yield {"answer": chunk.content}
                
            # Yield usage metadata if present
            if chunk.usage_metadata:
                yield {"usage": chunk.usage_metadata}
