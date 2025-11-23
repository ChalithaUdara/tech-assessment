"""Chat engine abstraction for different generation strategies."""

from abc import ABC, abstractmethod
from typing import Generator, List, Protocol, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from datacom_ai.chat.models import StreamUpdate
from datacom_ai.chat.chunk_parsers import RAGChunkParser
from datacom_ai.telemetry.metrics import TelemetryMetrics
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger
from datacom_ai.utils.structured_logging import log_llm_call


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
            
            # Log LLM call metrics in structured format
            log_llm_call(
                model_name=stats_dict.get("model_name", "unknown"),
                prompt_tokens=stats_dict["prompt_tokens"],
                completion_tokens=stats_dict["completion_tokens"],
                total_tokens=stats_dict["total_tokens"],
                latency_ms=stats_dict["latency_ms"],
                cost_usd=stats_dict["cost_usd"],
                deployment=stats_dict.get("deployment"),
                engine_type=engine_name,
                chunk_count=chunk_count,
                response_length=len(accumulated_content)
            )
            
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
        # Track data for evaluation
        self._current_query = None
        self._retrieved_docs = []
        self._accumulated_answer = ""

    def stream(
        self, messages: List[BaseMessage]
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response using the RAG pipeline.
        
        Maintains conversation history by passing all messages to the RAG pipeline,
        which includes them in the context for generation while using the last message
        as the retrieval query.
        """
        # Get the last user message as query for retrieval
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            # Fallback or handle error
            query = str(last_message.content)

        # Store query and reset tracking for this request
        self._current_query = query
        self._retrieved_docs = []
        self._accumulated_answer = ""

        # Pass full conversation history to RAG pipeline for context
        # The pipeline will use the last message for retrieval but include
        # all previous messages in the generation context
        conversation_history = messages if len(messages) > 1 else None
        
        # Stream tokens from the RAG pipeline with conversation history
        generator = self.rag_pipeline.stream(query, conversation_history)
        
        # Use custom streaming that tracks data and calculates metrics
        yield from self._stream_with_rag_evaluation(generator, "RAGChatEngine")

    def _stream_with_rag_evaluation(
        self,
        generator: Generator,
        engine_name: str
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream with metrics tracking and RAG evaluation.
        
        Tracks query, context, and answer, then calculates DeepEval metrics after completion.
        
        Args:
            generator: Generator that yields chunks to process
            engine_name: Name of the engine for logging
            
        Yields:
            StreamUpdate objects
        """
        logger.info(f"Starting {engine_name} stream with RAG evaluation")
        
        # Initialize metrics
        metrics = TelemetryMetrics()
        metrics.start_timer()
        
        accumulated_content = ""
        chunk_count = 0
        
        try:
            for chunk in generator:
                # Track context and answer for evaluation
                if isinstance(chunk, dict):
                    if "context" in chunk:
                        self._retrieved_docs = chunk["context"]
                    if "answer" in chunk:
                        self._accumulated_answer += chunk["answer"]
                
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
            
            # Log LLM call metrics in structured format
            log_llm_call(
                model_name=stats_dict.get("model_name", "unknown"),
                prompt_tokens=stats_dict["prompt_tokens"],
                completion_tokens=stats_dict["completion_tokens"],
                total_tokens=stats_dict["total_tokens"],
                latency_ms=stats_dict["latency_ms"],
                cost_usd=stats_dict["cost_usd"],
                deployment=stats_dict.get("deployment"),
                engine_type=engine_name,
                chunk_count=chunk_count,
                response_length=len(accumulated_content)
            )
            
            yield StreamUpdate(metadata=stats_dict)
            
            # Calculate and log RAG evaluation metrics after streaming completes
            if settings.ENABLE_RAG_EVALUATION and self._accumulated_answer and self._retrieved_docs:
                try:
                    evaluation_metrics = self._calculate_rag_metrics(
                        query=self._current_query,
                        answer=self._accumulated_answer,
                        retrieval_context=self._retrieved_docs
                    )
                    
                    # Log evaluation metrics as a separate event to update the retrieval log
                    # This adds evaluation metrics to the existing rag_retrieval event via correlation_id
                    if evaluation_metrics:
                        from datacom_ai.utils.structured_logging import log_rag_retrieval
                        # Get k value from retriever
                        k_value = getattr(self.rag_pipeline.retriever, 'k', settings.RETRIEVER_K)
                        # Log with minimal info since retrieval was already logged
                        # The correlation_id will link this to the original retrieval event
                        log_rag_retrieval(
                            query=self._current_query or "",
                            k_value=k_value,
                            retrieved_doc_count=len(self._retrieved_docs),
                            retrieval_scores=None,  # Already logged in retrieval step
                            retrieval_latency_ms=None,  # Already logged in retrieval step
                            success=True,
                            # Only include evaluation metrics
                            faithfulness=evaluation_metrics.get("faithfulness"),
                            answer_relevancy=evaluation_metrics.get("answer_relevancy")
                        )
                        logger.debug(f"RAG evaluation metrics logged: {evaluation_metrics}")
                except Exception as e:
                    logger.warning(f"Failed to calculate RAG evaluation metrics: {e}")
                    logger.debug(f"Evaluation error details: {e}", exc_info=True)

        except Exception as e:
            metrics.stop_timer()
            logger.error(f"{engine_name} failed to generate response: {e}")
            raise

    def _calculate_rag_metrics(
        self,
        query: str,
        answer: str,
        retrieval_context: List
    ) -> Dict[str, Any]:
        """
        Calculate DeepEval metrics for RAG evaluation.
        
        Calculates FaithfulnessMetric and AnswerRelevancyMetric which don't require ground truth.
        Based on DeepEval RAG evaluation guide: https://deepeval.com/guides/guides-rag-evaluation
        
        Args:
            query: The user query
            answer: The generated answer
            retrieval_context: List of retrieved documents
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase
            from deepeval.models import AzureOpenAIModel
            
            # Format retrieval context as list of strings
            retrieval_context_str = []
            for doc in retrieval_context:
                if hasattr(doc, "page_content"):
                    retrieval_context_str.append(doc.page_content)
                elif isinstance(doc, str):
                    retrieval_context_str.append(doc)
                else:
                    retrieval_context_str.append(str(doc))
            
            # Create test case (no expected_output needed for these metrics)
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output="",  # Not needed for faithfulness/answer_relevancy
                retrieval_context=retrieval_context_str
            )
            
            # Configure metrics with Azure OpenAI if available
            metric_kwargs = {
                "threshold": settings.RAG_EVALUATION_THRESHOLD,
                "include_reason": False,  # Don't include reason to reduce token usage
            }
            
            # Try to use Azure OpenAI for evaluation if configured
            try:
                endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
                if not endpoint.endswith("/"):
                    endpoint += "/"
                
                evaluation_model = AzureOpenAIModel(
                    model_name=settings.MODEL_NAME,
                    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
                    azure_openai_api_key=settings.OPENAI_API_KEY,
                    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=endpoint,
                    temperature=0,
                )
                metric_kwargs["model"] = evaluation_model
            except Exception as e:
                logger.debug(f"Could not configure Azure OpenAI for evaluation, using default: {e}")
                # Will use default OpenAI API if available
            
            # Calculate FaithfulnessMetric (checks if answer contradicts retrieval context)
            faithfulness_metric = FaithfulnessMetric(**metric_kwargs)
            faithfulness_metric.measure(test_case)
            
            # Calculate AnswerRelevancyMetric (checks if answer is relevant to query)
            answer_relevancy_metric = AnswerRelevancyMetric(**metric_kwargs)
            answer_relevancy_metric.measure(test_case)
            
            # Return metrics
            metrics = {}
            if faithfulness_metric.score is not None:
                metrics["faithfulness"] = faithfulness_metric.score
            if answer_relevancy_metric.score is not None:
                metrics["answer_relevancy"] = answer_relevancy_metric.score
            
            logger.debug(
                f"RAG evaluation metrics calculated: "
                f"faithfulness={metrics.get('faithfulness')}, "
                f"answer_relevancy={metrics.get('answer_relevancy')}"
            )
            
            return metrics
            
        except ImportError:
            logger.warning(
                "DeepEval not installed. RAG evaluation metrics will not be calculated. "
                "Install with: pip install deepeval"
            )
            return {}
        except Exception as e:
            logger.warning(f"Error calculating RAG evaluation metrics: {e}")
            logger.debug(f"Evaluation error details: {e}", exc_info=True)
            return {}

    def _process_chunk(self, chunk, metrics: TelemetryMetrics) -> StreamUpdate | None:
        """
        Process a chunk from the RAG pipeline.

        Args:
            chunk: RAG pipeline chunk (dict with "context", "answer", or "usage")
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate with content or metadata, or None if chunk should be skipped
        """
        return RAGChunkParser.parse(chunk, metrics)


class PlanningAgentChatEngine(BaseChatEngine):
    """
    Chat engine wrapper for PlanningAgent to follow ChatEngine protocol.
    
    This adapter wraps PlanningAgent to make it compatible with the ChatEngine
    interface, allowing it to use the shared metrics tracking from BaseChatEngine.
    """

    def __init__(self, planning_agent):
        """
        Initialize the PlanningAgent chat engine.

        Args:
            planning_agent: Configured PlanningAgent instance
        """
        self.planning_agent = planning_agent

    def stream(
        self, messages: List[BaseMessage]
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response using the PlanningAgent.
        
        Converts PlanningAgent's stream format to our standard StreamUpdate format.
        """
        # Convert PlanningAgent stream format to a generator that BaseChatEngine can process
        def agent_generator():
            """Generator that converts PlanningAgent chunks to a format _process_chunk can handle."""
            for chunk in self.planning_agent.stream(messages):
                if isinstance(chunk, dict):
                    # Yield content if present
                    if "content" in chunk:
                        yield chunk["content"]
                    # Yield usage metadata separately for processing
                    if "metadata" in chunk and "usage" in chunk["metadata"]:
                        yield {"usage": chunk["metadata"]["usage"]}
                else:
                    # Fallback for string chunks
                    yield str(chunk)
        
        yield from self._stream_with_metrics(agent_generator(), "PlanningAgentChatEngine")

    def _process_chunk(self, chunk, metrics: TelemetryMetrics) -> StreamUpdate | None:
        """
        Process a chunk from the PlanningAgent stream.

        Args:
            chunk: PlanningAgent chunk (string content or dict with usage metadata)
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate with content, or None if chunk should be skipped
        """
        # Handle usage metadata
        if isinstance(chunk, dict) and "usage" in chunk:
            usage = chunk["usage"]
            metrics.set_token_usage(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0)
            )
            return None  # Usage metadata is tracked, no need to yield
        
        # Handle content (string)
        if isinstance(chunk, str) and chunk:
            return StreamUpdate(content=chunk)
        
        # Fallback for other types
        if chunk:
            return StreamUpdate(content=str(chunk))
        
        return None
