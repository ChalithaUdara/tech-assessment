"""Shared pytest fixtures for all tests."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from datacom_ai.chat.message_store import MessageStore
from datacom_ai.chat.chat_handler import ChatHandler


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock()
    return client


@pytest.fixture
def message_store():
    """Create a message store for testing."""
    return MessageStore(max_messages=10)


@pytest.fixture
def message_store_small():
    """Create a small message store for testing trimming."""
    return MessageStore(max_messages=3)


@pytest.fixture
def chat_handler(mock_llm_client, message_store):
    """Create a chat handler for testing."""
    from datacom_ai.chat.engine import SimpleChatEngine
    chat_engine = SimpleChatEngine(mock_llm_client)
    return ChatHandler(chat_engine, message_store)


@pytest.fixture
def sample_human_message():
    """Create a sample human message."""
    return HumanMessage(content="Hello, how are you?")


@pytest.fixture
def sample_ai_message():
    """Create a sample AI message."""
    return AIMessage(content="I'm doing well, thank you!")


@pytest.fixture
def sample_messages(sample_human_message, sample_ai_message):
    """Create a list of sample messages."""
    return [sample_human_message, sample_ai_message]


@pytest.fixture
def gradio_history():
    """Create sample Gradio history format."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.fixture
def mock_streaming_chunks():
    """Create mock streaming chunks for LLM responses."""
    return [
        Mock(content="Hello", response_metadata={}, usage_metadata={}),
        Mock(content=" there", response_metadata={}, usage_metadata={}),
        Mock(
            content="!",
            response_metadata={
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                }
            },
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
        ),
    ]


@pytest.fixture
def mock_streaming_chunks_with_usage_metadata():
    """Create mock streaming chunks with usage_metadata."""
    return [
        Mock(content="Response", response_metadata={}, usage_metadata={}),
        Mock(
            content=" complete",
            response_metadata={},
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        ),
    ]


@pytest.fixture
def mock_streaming_chunks_empty():
    """Create mock streaming chunks with no content."""
    return [
        Mock(content="", response_metadata={}, usage_metadata={}),
    ]

