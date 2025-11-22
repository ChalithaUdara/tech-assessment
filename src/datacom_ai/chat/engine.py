"""Chat engine abstraction for different generation strategies."""

from abc import ABC, abstractmethod
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


class BaseChatEngine(ABC):
    """
    Base class for chat engines with common streaming and metrics logic.
    """

    def _stream_with_metrics(
        self,
        generator: Generator,
        engine_name: str
    ) -> Generator[StreamUpdate, None, None]:
        """
        Common streaming logic with metrics tracking.

        Args:
            generator: Generator that yields chunks to process
            engine_name: Name of the engine for logging

        Yields:
            StreamUpdate objects
        """
        logger.info(f"Starting {engine_name} stream")

        # Initialize metrics
        metrics = TelemetryMetrics()
        metrics.start_timer()

        accumulated_content = ""
        chunk_count = 0

        try:
            for chunk in generator:
                # Process chunk using subclass-specific logic
                update = self._process_chunk(chunk, metrics)
                if update:
                    if update.content:
                        accumulated_content += update.content
                        chunk_count += 1
                    yield update

            # Stop the timer
            metrics.stop_timer()

            logger.info(
                f"{engine_name} stream completed: {chunk_count} chunks, "
                f"{len(accumulated_content)} chars"
            )

            # Yield stats
            stats_dict = metrics.to_dict()
            yield StreamUpdate(metadata=stats_dict)

        except Exception as e:
            metrics.stop_timer()
            logger.error(f"{engine_name} failed to generate response: {e}")
            raise

    @abstractmethod
    def _process_chunk(self, chunk, metrics: TelemetryMetrics) -> StreamUpdate | None:
        """
        Process a single chunk from the generator.

        Args:
            chunk: The chunk to process
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate if chunk should be yielded, None otherwise
        """
        pass


class SimpleChatEngine(BaseChatEngine):
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
        generator = self.llm_client.stream(messages)
        yield from self._stream_with_metrics(generator, "SimpleChatEngine")

    def _process_chunk(self, chunk, metrics: TelemetryMetrics) -> StreamUpdate | None:
        """
        Process a chunk from the LLM stream.

        Args:
            chunk: LLM chunk with content and optional usage_metadata
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate with content, or None if chunk should be skipped
        """
        update = None

        # Yield content tokens
        if hasattr(chunk, "content") and chunk.content:
            update = StreamUpdate(content=chunk.content)

        # Check for usage metadata
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage_metadata = chunk.usage_metadata
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)
            metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)

        return update


class RAGChatEngine(BaseChatEngine):
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
        # Get the last user message as query
        # Ideally we should condense history into a standalone query, but for baseline we use last msg
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            # Fallback or handle error
            query = str(last_message.content)

        # Stream tokens from the RAG pipeline
        generator = self.rag_pipeline.stream(query)
        yield from self._stream_with_metrics(generator, "RAGChatEngine")

    def _process_chunk(self, chunk, metrics: TelemetryMetrics) -> StreamUpdate | None:
        """
        Process a chunk from the RAG pipeline.

        Args:
            chunk: RAG pipeline chunk (dict with "context", "answer", or "usage")
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate with content or metadata, or None if chunk should be skipped
        """
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
                return StreamUpdate(metadata={"citations": citations})

            elif "answer" in chunk:
                return StreamUpdate(content=chunk["answer"])

            elif "usage" in chunk:
                usage_metadata = chunk["usage"]
                prompt_tokens = usage_metadata.get("input_tokens", 0)
                completion_tokens = usage_metadata.get("output_tokens", 0)
                total_tokens = usage_metadata.get("total_tokens", 0)
                metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)
                return None  # Usage metadata is tracked, no need to yield
        else:
            # Fallback if pipeline yields strings directly (legacy support)
            return StreamUpdate(content=str(chunk))

        return None
