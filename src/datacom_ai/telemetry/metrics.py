"""Telemetry metrics tracking: tokens, cost, and latency."""

import time
from typing import Optional, Dict, Any
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger


class TelemetryMetrics:
    """Track and format telemetry metrics for chat interactions."""

    def __init__(self):
        """Initialize metrics tracking."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        logger.debug("TelemetryMetrics initialized")

    def start_timer(self) -> None:
        """Start the latency timer."""
        self.start_time = time.time()
        logger.debug("Telemetry timer started")

    def stop_timer(self) -> None:
        """Stop the latency timer."""
        self.end_time = time.time()
        if self.start_time:
            latency_ms = self.get_latency_ms()
            logger.debug(f"Telemetry timer stopped: {latency_ms}ms latency")

    def get_latency_ms(self) -> int:
        """
        Get the latency in milliseconds.

        Returns:
            Latency in milliseconds, or 0 if timer wasn't started/stopped
        """
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time) * 1000)

    def set_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> None:
        """
        Set token usage metrics.

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens
            total_tokens: Total tokens used
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        logger.debug(
            f"Token usage set: prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={total_tokens}"
        )

    def get_cost(self) -> float:
        """
        Calculate the cost in USD based on token usage.

        Returns:
            Total cost in USD
        """
        cost = settings.calculate_cost(self.prompt_tokens, self.completion_tokens)
        logger.debug(f"Calculated cost: ${cost:.6f} USD")
        return cost

    def format_stats(self) -> str:
        """
        Format metrics as a stats string.

        Returns:
            Formatted string: [stats] prompt=X completion=Y cost=$Z.ZZZZZZ latency=W ms
        """
        cost = self.get_cost()
        latency = self.get_latency_ms()
        return (
            f"[stats] prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            f"cost=${cost:.6f} "
            f"latency={latency} ms"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.get_cost(),
            "latency_ms": self.get_latency_ms(),
        }

    @staticmethod
    def extract_usage_from_response_metadata(
        response_metadata: Dict[str, Any]
    ) -> tuple[int, int, int]:
        """
        Extract token usage from LangChain response metadata.

        Args:
            response_metadata: Response metadata from LangChain AIMessage

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens)
        """
        # Try usage_metadata first (newer format)
        usage_metadata = response_metadata.get("usage_metadata", {})
        if usage_metadata:
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)
            if prompt_tokens > 0 or completion_tokens > 0:
                return prompt_tokens, completion_tokens, total_tokens

        # Fallback to usage (older format)
        usage = response_metadata.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Also check token_usage in response_metadata (some providers use this)
        if prompt_tokens == 0 and completion_tokens == 0:
            token_usage = response_metadata.get("token_usage", {})
            if token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)

        return prompt_tokens, completion_tokens, total_tokens

