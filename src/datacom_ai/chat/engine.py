"""Chat engine abstraction for different generation strategies."""

from typing import Generator, List, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from datacom_ai.chat.models import StreamUpdate
from datacom_ai.telemetry.metrics import TelemetryMetrics
from datacom_ai.utils.logger import logger


class ChatEngine(Protocol):
    """
    Protocol for chat engines that generate responses.
    """

    def stream(
        self, messages: List[BaseMessage]
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response based on the conversation history.

        Args:
            messages: List of conversation messages

        Yields:
            StreamUpdate objects
        """
        ...


class SimpleChatEngine:
    """
    Standard LLM chat engine using Azure OpenAI.
    """

    def __init__(self, llm_client: AzureChatOpenAI):
        """
        Initialize the simple chat engine.

        Args:
            llm_client: Configured AzureChatOpenAI client
        """
        self.llm_client = llm_client

    def stream(
        self, messages: List[BaseMessage]
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response using the LLM directly.
        """
        logger.info("Starting SimpleChatEngine stream")
        
        # Initialize metrics
        metrics = TelemetryMetrics()
        metrics.start_timer()
        
        accumulated_content = ""
        chunk_count = 0
        
        try:
            # Stream tokens from the LLM
            for chunk in self.llm_client.stream(messages):
                chunk_count += 1
                
                # Yield content tokens
                if hasattr(chunk, "content") and chunk.content:
                    accumulated_content += chunk.content
                    yield StreamUpdate(content=chunk.content)

                # Check for usage metadata
                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
                    prompt_tokens = usage_metadata.get("input_tokens", 0)
                    completion_tokens = usage_metadata.get("output_tokens", 0)
                    total_tokens = usage_metadata.get("total_tokens", 0)
                    metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)

            # Stop the timer
            metrics.stop_timer()
            
            logger.info(
                f"Engine stream completed: {chunk_count} chunks, "
                f"{len(accumulated_content)} chars"
            )

            # Yield stats
            stats_dict = metrics.to_dict()
            yield StreamUpdate(metadata=stats_dict)

        except Exception as e:
            metrics.stop_timer()
            logger.error(f"Engine failed to generate response: {e}")
            raise


class RAGChatEngine:
    """
    Chat engine using RAG pipeline.
    """

    def __init__(self, rag_pipeline):
        """
        Initialize the RAG chat engine.

        Args:
            rag_pipeline: Configured RAGPipeline instance
        """
        self.rag_pipeline = rag_pipeline

    def stream(
        self, messages: List[BaseMessage]
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response using the RAG pipeline.
        """
        logger.info("Starting RAGChatEngine stream")
        
        # Initialize metrics
        metrics = TelemetryMetrics()
        metrics.start_timer()
        
        accumulated_content = ""
        chunk_count = 0
        
        try:
            # Get the last user message as query
            # Ideally we should condense history into a standalone query, but for baseline we use last msg
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                query = last_message.content
            else:
                # Fallback or handle error
                query = str(last_message.content)

            # Stream tokens from the RAG pipeline
            for chunk in self.rag_pipeline.stream(query):
                if isinstance(chunk, dict):
                    if "context" in chunk:
                        # Process citations
                        docs = chunk["context"]
                        citations = []
                        for doc in docs:
                            # Extract relevant metadata
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            citations.append(f"Source: {source}, Page: {page}")
                        
                        # Yield citations as metadata
                        yield StreamUpdate(metadata={"citations": citations})
                    
                    elif "answer" in chunk:
                        chunk_count += 1
                        accumulated_content += chunk["answer"]
                        yield StreamUpdate(content=chunk["answer"])
                    
                    elif "usage" in chunk:
                        usage_metadata = chunk["usage"]
                        prompt_tokens = usage_metadata.get("input_tokens", 0)
                        completion_tokens = usage_metadata.get("output_tokens", 0)
                        total_tokens = usage_metadata.get("total_tokens", 0)
                        metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)
                else:
                    # Fallback if pipeline yields strings directly (though we changed it)
                    chunk_count += 1
                    accumulated_content += str(chunk)
                    yield StreamUpdate(content=str(chunk))

            # Stop the timer
            metrics.stop_timer()
            
            logger.info(
                f"RAG Engine stream completed: {chunk_count} chunks, "
                f"{len(accumulated_content)} chars"
            )

            # Yield stats (approximate or placeholder as RAG chain might not return token usage easily in stream)
            # For now, we can just track latency
            stats_dict = metrics.to_dict()
            yield StreamUpdate(metadata=stats_dict)

        except Exception as e:
            metrics.stop_timer()
            logger.error(f"RAG Engine failed to generate response: {e}")
            raise
