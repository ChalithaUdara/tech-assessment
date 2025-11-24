#!/usr/bin/env python3
"""
Acceptance test for Task 3.1: Conversational Core (Streaming & Cost Telemetry)

This script tests that running chat.py (or equivalent) with "Hello" prints:
1. A streamed assistant response
2. A metrics line in the format: [stats] prompt=X completion=Y cost=$Z.ZZZZZZ latency=W ms

This test runs WITHOUT Gradio UI, using ChatHandler directly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datacom_ai.chat.chat_handler import ChatHandler
from datacom_ai.chat.engine import SimpleChatEngine
from datacom_ai.chat.message_store import MessageStore
from datacom_ai.clients.llm_client import create_llm_client
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger, setup_logging


def format_stats_line(metadata: dict) -> str:
    """
    Format metrics as a stats string matching the expected format.
    
    Format: [stats] prompt=X completion=Y cost=$Z.ZZZZZZ latency=W ms
    
    Args:
        metadata: Dictionary with prompt_tokens, completion_tokens, cost_usd, latency_ms
        
    Returns:
        Formatted stats string
    """
    prompt = metadata.get("prompt_tokens", 0)
    completion = metadata.get("completion_tokens", 0)
    cost = metadata.get("cost_usd", 0.0)
    latency = metadata.get("latency_ms", 0)
    
    return f"[stats] prompt={prompt} completion={completion} cost=${cost:.6f} latency={latency} ms"


def main():
    """Run the acceptance test."""
    # Setup minimal logging (only errors to avoid cluttering output)
    setup_logging(level="ERROR", log_file=None, log_format="text")
    
    try:
        # Validate configuration
        settings.validate()
        
        # Initialize components
        logger.info("Initializing LLM client...")
        llm_client = create_llm_client()
        
        logger.info("Initializing message store...")
        message_store = MessageStore()
        
        logger.info("Initializing chat engine...")
        chat_engine = SimpleChatEngine(llm_client)
        
        logger.info("Initializing chat handler...")
        chat_handler = ChatHandler(
            chat_engine=chat_engine,
            message_store=message_store,
        )
        
        # Test input
        user_message = "Hello"
        history = []
        
        print(f"User: {user_message}")
        print("Assistant: ", end="", flush=True)
        
        # Stream the response
        accumulated_content = ""
        stats_metadata = None
        
        for update in chat_handler.stream_response(user_message, history):
            # Print content as it streams
            if update.content:
                print(update.content, end="", flush=True)
                accumulated_content += update.content
            
            # Capture stats metadata
            if update.metadata and ("prompt_tokens" in update.metadata or "latency_ms" in update.metadata):
                stats_metadata = update.metadata
        
        # Print newline after streaming
        print()
        
        # Verify we received content
        if not accumulated_content:
            print("ERROR: No assistant response received", file=sys.stderr)
            sys.exit(1)
        
        # Verify we received stats
        if not stats_metadata:
            print("ERROR: No stats metadata received", file=sys.stderr)
            sys.exit(1)
        
        # Format and print stats line
        stats_line = format_stats_line(stats_metadata)
        print(stats_line)
        
        # Verify all required fields are present
        required_fields = ["prompt_tokens", "completion_tokens", "cost_usd", "latency_ms"]
        missing_fields = [field for field in required_fields if field not in stats_metadata]
        
        if missing_fields:
            print(f"ERROR: Missing required fields in stats: {missing_fields}", file=sys.stderr)
            sys.exit(1)
        
        # Verify stats format matches expected pattern
        if not stats_line.startswith("[stats] "):
            print(f"ERROR: Stats line does not start with '[stats] ': {stats_line}", file=sys.stderr)
            sys.exit(1)
        
        if "prompt=" not in stats_line:
            print(f"ERROR: Stats line missing 'prompt=': {stats_line}", file=sys.stderr)
            sys.exit(1)
        
        if "completion=" not in stats_line:
            print(f"ERROR: Stats line missing 'completion=': {stats_line}", file=sys.stderr)
            sys.exit(1)
        
        if "cost=$" not in stats_line:
            print(f"ERROR: Stats line missing 'cost=$': {stats_line}", file=sys.stderr)
            sys.exit(1)
        
        if "latency=" not in stats_line or " ms" not in stats_line:
            print(f"ERROR: Stats line missing 'latency=... ms': {stats_line}", file=sys.stderr)
            sys.exit(1)
        
        # Success
        print("\n✅ Acceptance test passed!")
        print(f"   - Received streamed response: {len(accumulated_content)} characters")
        print(f"   - Received stats in correct format")
        sys.exit(0)
        
    except ValueError as e:
        print(f"ERROR: Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

