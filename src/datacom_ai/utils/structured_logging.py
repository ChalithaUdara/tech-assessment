"""Structured logging helpers for analytics dashboard."""

import contextvars
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID for request tracking.
    
    Returns:
        Unique correlation ID string (e.g., "req-abc123")
    """
    return f"req-{uuid.uuid4().hex[:8]}"


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from context.
    
    Returns:
        Current correlation ID or None if not set
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID in the current context.
    
    Args:
        correlation_id: Correlation ID to set
    """
    _correlation_id.set(correlation_id)


class CorrelationContext:
    """Context manager for correlation ID tracking."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize correlation context.
        
        Args:
            correlation_id: Optional correlation ID. If None, generates a new one.
        """
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token = None
        self._previous_value = None
    
    def __enter__(self):
        """Enter context and set correlation ID."""
        try:
            # Try to get current value before setting
            self._previous_value = _correlation_id.get()
        except (LookupError, RuntimeError):
            self._previous_value = None
        
        try:
            self._token = _correlation_id.set(self.correlation_id)
        except (RuntimeError, ValueError):
            # If setting fails (e.g., in different context), just store the ID
            # We'll still use it for logging even if context var doesn't work
            self._token = None
        
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous correlation ID."""
        if self._token:
            try:
                _correlation_id.reset(self._token)
            except (ValueError, RuntimeError):
                # Token was created in a different context (e.g., different thread)
                # This is okay - the context will be cleaned up when the thread/context ends
                # We can't reset it safely, so we just ignore the error
                pass


def _log_structured_event(
    event_type: str,
    level: str = "INFO",
    message: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log a structured event with consistent format.
    
    Args:
        event_type: Type of event (e.g., "chat_request", "rag_retrieval")
        level: Log level (INFO, DEBUG, WARNING, ERROR)
        message: Optional message to log
        **kwargs: Additional fields to include in the log
    """
    # Get correlation ID from context if available
    correlation_id = get_correlation_id()
    
    # Generate timestamp fields for time-series analysis
    now = datetime.now()
    timestamp_iso = now.isoformat()
    timestamp_unix = now.timestamp()
    date = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    
    # Build log message with structured data
    log_data = {
        "event_type": event_type,
        "timestamp_iso": timestamp_iso,
        "timestamp_unix": timestamp_unix,
        "date": date,
        "hour": hour,
        **kwargs
    }
    
    if correlation_id and settings.ENABLE_CORRELATION_IDS:
        log_data["correlation_id"] = correlation_id
    
    # Use loguru's bind to add structured data
    log_message = message or f"{event_type} event"
    
    # Bind extra data for JSON serialization and call appropriate log level
    bound_logger = logger.bind(**log_data)
    log_func = getattr(bound_logger, level.lower())
    log_func(log_message)


def log_chat_request(
    user_message: str,
    mode: str,
    message_length: Optional[int] = None,
    **kwargs: Any
) -> str:
    """
    Log a chat request event.
    
    Args:
        user_message: The user's message
        mode: Chat mode (default, rag, planning_agent)
        message_length: Optional message length (auto-calculated if not provided)
        **kwargs: Additional fields to include
        
    Returns:
        Correlation ID for this request
    """
    correlation_id = get_correlation_id() or generate_correlation_id()
    if not get_correlation_id():
        set_correlation_id(correlation_id)
    
    _log_structured_event(
        event_type="chat_request",
        correlation_id=correlation_id,
        user_message=user_message[:200] if len(user_message) > 200 else user_message,  # Truncate long messages
        message_length=message_length or len(user_message),
        mode=mode,
        **kwargs
    )
    
    return correlation_id


def log_chat_response(
    latency_ms: int,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    success: bool = True,
    error_message: Optional[str] = None,
    engine_type: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log a chat response with metrics.
    
    Args:
        latency_ms: Response latency in milliseconds
        cost_usd: Cost in USD
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        success: Whether the request was successful
        error_message: Error message if request failed
        engine_type: Type of engine used (SimpleChatEngine, RAGChatEngine, etc.)
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="chat_response",
        latency_ms=latency_ms,
        cost_usd=round(cost_usd, 8),  # Round to 8 decimal places
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        success=success,
        error_message=error_message,
        engine_type=engine_type,
        **kwargs
    )


def log_rag_query(
    query: str,
    k_value: int,
    **kwargs: Any
) -> None:
    """
    Log a RAG query event.
    
    Args:
        query: The query text
        k_value: Number of documents to retrieve (k)
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="rag_query",
        query=query[:200] if len(query) > 200 else query,  # Truncate long queries
        k_value=k_value,
        **kwargs
    )


def log_rag_retrieval(
    query: str,
    k_value: int,
    retrieved_doc_count: int,
    retrieval_scores: Optional[List[float]] = None,
    retrieval_latency_ms: Optional[int] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    # Accuracy metrics (optional, populated when evaluation metrics are available)
    precision_at_k: Optional[float] = None,
    recall_at_k: Optional[float] = None,
    contextual_precision: Optional[float] = None,
    contextual_recall: Optional[float] = None,
    contextual_relevancy: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
    faithfulness: Optional[float] = None,
    relevant_doc_count: Optional[int] = None,
    relevance_labels: Optional[List[bool]] = None,
    **kwargs: Any
) -> None:
    """
    Log RAG retrieval metrics with optional accuracy metrics.
    
    Args:
        query: The query text
        k_value: Number of documents requested (k)
        retrieved_doc_count: Number of documents actually retrieved
        retrieval_scores: List of similarity scores for retrieved documents
        retrieval_latency_ms: Time taken for retrieval in milliseconds
        success: Whether retrieval was successful
        error_message: Error message if retrieval failed
        precision_at_k: Precision at k metric (optional)
        recall_at_k: Recall at k metric (optional)
        contextual_precision: Contextual precision score from DeepEval (optional)
        contextual_recall: Contextual recall score from DeepEval (optional)
        contextual_relevancy: Contextual relevancy score from DeepEval (optional)
        answer_relevancy: Answer relevancy score from DeepEval (optional)
        faithfulness: Faithfulness score from DeepEval (optional)
        relevant_doc_count: Number of actually relevant documents (ground truth)
        relevance_labels: List of boolean labels indicating relevance per document
        **kwargs: Additional fields to include
    """
    # Build accuracy metrics dict, only including non-None values
    accuracy_metrics = {}
    if precision_at_k is not None:
        accuracy_metrics["precision_at_k"] = precision_at_k
    if recall_at_k is not None:
        accuracy_metrics["recall_at_k"] = recall_at_k
    if contextual_precision is not None:
        accuracy_metrics["contextual_precision"] = contextual_precision
    if contextual_recall is not None:
        accuracy_metrics["contextual_recall"] = contextual_recall
    if contextual_relevancy is not None:
        accuracy_metrics["contextual_relevancy"] = contextual_relevancy
    if answer_relevancy is not None:
        accuracy_metrics["answer_relevancy"] = answer_relevancy
    if faithfulness is not None:
        accuracy_metrics["faithfulness"] = faithfulness
    if relevant_doc_count is not None:
        accuracy_metrics["relevant_doc_count"] = relevant_doc_count
    if relevance_labels is not None:
        accuracy_metrics["relevance_labels"] = relevance_labels
    
    _log_structured_event(
        event_type="rag_retrieval",
        query=query[:200] if len(query) > 200 else query,  # Truncate long queries
        k_value=k_value,
        retrieved_doc_count=retrieved_doc_count,
        retrieval_scores=retrieval_scores,
        retrieval_latency_ms=retrieval_latency_ms,
        success=success,
        error_message=error_message,
        **accuracy_metrics,
        **kwargs
    )


def log_agent_step(
    step_name: str,
    status: str,  # "success" or "failure"
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    error_message: Optional[str] = None,
    execution_time_ms: Optional[int] = None,
    **kwargs: Any
) -> None:
    """
    Log an agent step event.
    
    Args:
        step_name: Name of the agent step (planner, tools, finalizer, formatter)
        status: Status of the step ("success" or "failure")
        tool_calls: List of tool calls made in this step
        error_message: Error message if step failed
        execution_time_ms: Execution time in milliseconds
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="agent_step",
        step_name=step_name,
        status=status,
        tool_calls=tool_calls,
        error_message=error_message,
        execution_time_ms=execution_time_ms,
        **kwargs
    )


def log_agent_execution(
    agent_type: str,
    overall_status: str,  # "success", "failure", or "partial"
    total_execution_time_ms: int,
    steps_executed: List[str],
    steps_succeeded: int,
    steps_failed: int,
    failure_step: Optional[str] = None,
    failure_type: Optional[str] = None,
    tool_calls_count: int = 0,
    tool_calls_succeeded: int = 0,
    tool_calls_failed: int = 0,
    has_itinerary: Optional[bool] = None,
    **kwargs: Any
) -> None:
    """
    Log an agent execution summary event with aggregated metrics.
    
    Args:
        agent_type: Type of agent (e.g., "PlanningAgent")
        overall_status: Overall execution status ("success", "failure", "partial")
        total_execution_time_ms: Total execution time in milliseconds
        steps_executed: List of step names that were executed
        steps_succeeded: Number of steps that succeeded
        steps_failed: Number of steps that failed
        failure_step: Name of the step that failed (if any)
        failure_type: Type of failure (e.g., "planner_error", "tool_error")
        tool_calls_count: Total number of tool calls made
        tool_calls_succeeded: Number of tool calls that succeeded
        tool_calls_failed: Number of tool calls that failed
        has_itinerary: Whether an itinerary was generated (for PlanningAgent)
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="agent_execution",
        agent_type=agent_type,
        overall_status=overall_status,
        total_execution_time_ms=total_execution_time_ms,
        steps_executed=steps_executed,
        steps_succeeded=steps_succeeded,
        steps_failed=steps_failed,
        failure_step=failure_step,
        failure_type=failure_type,
        tool_calls_count=tool_calls_count,
        tool_calls_succeeded=tool_calls_succeeded,
        tool_calls_failed=tool_calls_failed,
        has_itinerary=has_itinerary,
        **kwargs
    )


def log_error(
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> None:
    """
    Log a structured error event.
    
    Args:
        error_type: Type of error (e.g., "llm_error", "retrieval_error")
        error_message: Error message
        context: Additional context about the error
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="error",
        level="ERROR",
        error_type=error_type,
        error_message=error_message,
        context=context,
        **kwargs
    )


def log_llm_call(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: int,
    cost_usd: float,
    deployment: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log an LLM API call with metrics.
    
    Args:
        model_name: Name of the model used
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        latency_ms: Latency in milliseconds
        cost_usd: Cost in USD
        deployment: Deployment name (for Azure OpenAI)
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="llm_call",
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cost_usd=round(cost_usd, 8),
        deployment=deployment,
        **kwargs
    )


def log_evaluation_run(
    evaluation_type: str,
    metrics: Dict[str, float],
    test_cases_count: int,
    timestamp: Optional[datetime] = None,
    **kwargs: Any
) -> None:
    """
    Log an evaluation run with metrics for historical tracking.
    
    Args:
        evaluation_type: Type of evaluation (e.g., "rag_retrieval", "agent_performance")
        metrics: Dictionary of metric names and their values (e.g., {"precision": 0.75, "recall": 0.80})
        test_cases_count: Number of test cases evaluated
        timestamp: Optional timestamp for the evaluation (defaults to now)
        **kwargs: Additional fields to include
    """
    _log_structured_event(
        event_type="evaluation_run",
        evaluation_type=evaluation_type,
        metrics=metrics,
        test_cases_count=test_cases_count,
        evaluation_timestamp=timestamp.isoformat() if timestamp else None,
        **kwargs
    )

