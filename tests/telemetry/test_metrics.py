"""Unit tests for TelemetryMetrics."""

import time
import pytest
from datacom_ai.telemetry.metrics import TelemetryMetrics


class TestTelemetryMetrics:
    """Tests for TelemetryMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = TelemetryMetrics()
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0

    def test_latency_tracking(self):
        """Test latency measurement."""
        metrics = TelemetryMetrics()
        metrics.start_timer()
        time.sleep(0.1)
        metrics.stop_timer()

        latency = metrics.get_latency_ms()
        assert latency >= 100  # At least 100ms

    def test_latency_without_start(self):
        """Test latency when timer wasn't started."""
        metrics = TelemetryMetrics()
        latency = metrics.get_latency_ms()
        assert latency == 0

    def test_latency_without_stop(self):
        """Test latency when timer wasn't stopped."""
        metrics = TelemetryMetrics()
        metrics.start_timer()
        latency = metrics.get_latency_ms()
        assert latency == 0

    def test_latency_precision(self):
        """Test that latency is measured in milliseconds."""
        metrics = TelemetryMetrics()
        metrics.start_timer()
        time.sleep(0.001)  # 1ms
        metrics.stop_timer()

        latency = metrics.get_latency_ms()
        assert latency >= 1
        assert isinstance(latency, int)

    def test_token_usage(self):
        """Test setting and getting token usage."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20
        assert metrics.total_tokens == 30

    def test_token_usage_zero(self):
        """Test setting zero token usage."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0

    @pytest.mark.parametrize(
        "prompt_tokens,completion_tokens,total_tokens",
        [
            (100, 50, 150),
            (1000, 500, 1500),
            (10000, 5000, 15000),
            (0, 0, 0),
            (1, 1, 2),
        ],
    )
    def test_token_usage_variations(self, prompt_tokens, completion_tokens, total_tokens):
        """Test setting various token usage values."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)

        assert metrics.prompt_tokens == prompt_tokens
        assert metrics.completion_tokens == completion_tokens
        assert metrics.total_tokens == total_tokens

    def test_cost_calculation(self):
        """Test cost calculation with GPT-4o pricing."""
        metrics = TelemetryMetrics()
        # 1000 input tokens, 500 output tokens
        metrics.set_token_usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        cost = metrics.get_cost()
        # Expected: (1000/1M * 2.50) + (500/1M * 10.00) = 0.0025 + 0.005 = 0.0075
        expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected_cost) < 0.000001

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        cost = metrics.get_cost()
        assert cost == 0.0

    @pytest.mark.parametrize(
        "prompt_tokens,completion_tokens,expected_cost",
        [
            (1000, 500, (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00),
            (100, 50, (100 / 1_000_000) * 2.50 + (50 / 1_000_000) * 10.00),
            (0, 0, 0.0),
            (1_000_000, 1_000_000, 2.50 + 10.00),  # 1M tokens each
        ],
    )
    def test_cost_calculation_variations(
        self, prompt_tokens, completion_tokens, expected_cost
    ):
        """Test cost calculation with various token amounts."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        cost = metrics.get_cost()
        assert abs(cost - expected_cost) < 0.000001

    def test_format_stats(self):
        """Test stats formatting."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=8, completion_tokens=23, total_tokens=31)
        metrics.start_timer()
        time.sleep(0.001)
        metrics.stop_timer()

        stats = metrics.format_stats()
        assert "prompt=8" in stats
        assert "completion=23" in stats
        assert "cost=" in stats
        assert "latency=" in stats
        assert "ms" in stats

    def test_format_stats_zero_values(self):
        """Test stats formatting with zero values."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        stats = metrics.format_stats()
        assert "prompt=0" in stats
        assert "completion=0" in stats
        assert "cost=$0.000000" in stats

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = TelemetryMetrics()
        metrics.set_token_usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        metrics.start_timer()
        time.sleep(0.001)
        metrics.stop_timer()

        stats_dict = metrics.to_dict()
        assert stats_dict["prompt_tokens"] == 10
        assert stats_dict["completion_tokens"] == 20
        assert stats_dict["total_tokens"] == 30
        assert "cost_usd" in stats_dict
        assert "latency_ms" in stats_dict
        assert isinstance(stats_dict["cost_usd"], float)
        assert isinstance(stats_dict["latency_ms"], int)

    def test_to_dict_zero_values(self):
        """Test to_dict with zero values."""
        metrics = TelemetryMetrics()
        stats_dict = metrics.to_dict()
        assert stats_dict["prompt_tokens"] == 0
        assert stats_dict["completion_tokens"] == 0
        assert stats_dict["total_tokens"] == 0
        assert stats_dict["cost_usd"] == 0.0
        assert stats_dict["latency_ms"] == 0

    def test_extract_usage_from_response_metadata_usage_key(self):
        """Test extracting usage from 'usage' key."""
        metadata = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 10
        assert completion == 20
        assert total == 30

    def test_extract_usage_from_response_metadata_usage_metadata_key(self):
        """Test extracting usage from 'usage_metadata' key."""
        metadata = {
            "usage_metadata": {
                "input_tokens": 15,
                "output_tokens": 25,
                "total_tokens": 40,
            }
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 15
        assert completion == 25
        assert total == 40

    def test_extract_usage_from_response_metadata_token_usage_key(self):
        """Test extracting usage from 'token_usage' key (fallback)."""
        metadata = {
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50,
            }
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 20
        assert completion == 30
        assert total == 50

    def test_extract_usage_from_response_metadata_token_usage_input_output(self):
        """Test extracting usage from token_usage with input_tokens/output_tokens."""
        metadata = {
            "token_usage": {
                "input_tokens": 25,
                "output_tokens": 35,
                "total_tokens": 60,
            }
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 25
        assert completion == 35
        assert total == 60

    def test_extract_usage_from_response_metadata_empty(self):
        """Test extracting usage from empty metadata."""
        metadata = {}
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 0
        assert completion == 0
        assert total == 0

    def test_extract_usage_from_response_metadata_prefers_usage_metadata(self):
        """Test that usage_metadata is preferred over usage."""
        metadata = {
            "usage_metadata": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            },
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        # Should prefer usage_metadata
        assert prompt == 10
        assert completion == 20
        assert total == 30

    def test_extract_usage_from_response_metadata_missing_fields(self):
        """Test extracting usage with missing fields."""
        metadata = {
            "usage": {
                "prompt_tokens": 10,
                # Missing completion_tokens and total_tokens
            }
        }
        prompt, completion, total = TelemetryMetrics.extract_usage_from_response_metadata(
            metadata
        )
        assert prompt == 10
        assert completion == 0
        assert total == 0

    def test_multiple_timer_starts(self):
        """Test that multiple timer starts reset the timer."""
        metrics = TelemetryMetrics()
        metrics.start_timer()
        time.sleep(0.01)
        metrics.start_timer()  # Reset
        time.sleep(0.01)
        metrics.stop_timer()

        latency = metrics.get_latency_ms()
        # Should be approximately 10ms, not 20ms
        assert latency < 20

    def test_multiple_timer_stops(self):
        """Test that multiple timer stops don't cause issues."""
        metrics = TelemetryMetrics()
        metrics.start_timer()
        time.sleep(0.01)
        metrics.stop_timer()
        latency1 = metrics.get_latency_ms()
        
        metrics.stop_timer()  # Second stop
        latency2 = metrics.get_latency_ms()
        
        # Should be the same (or very close)
        assert abs(latency1 - latency2) < 5

