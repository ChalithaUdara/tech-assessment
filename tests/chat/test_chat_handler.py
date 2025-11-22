"""Unit tests for ChatHandler."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage

from datacom_ai.chat.chat_handler import ChatHandler
from datacom_ai.chat.models import StreamUpdate


class TestChatHandler:
    """Tests for ChatHandler."""

    def test_stream_response_basic(
        self, chat_handler, mock_llm_client, mock_streaming_chunks
    ):
        """Test basic streaming response."""
        mock_llm_client.stream.return_value = iter(mock_streaming_chunks)

        history = []
        response = list(chat_handler.stream_response("Hi", history))

        # Should yield StreamUpdate objects
        content_updates = [r for r in response if r.content]
        stats_updates = [r for r in response if r.metadata]

        assert len(content_updates) == 3  # Three content chunks
        assert "".join(u.content for u in content_updates) == "Hello there!"
        assert len(stats_updates) == 1  # One stats update
        assert "latency_ms" in stats_updates[0].metadata

    def test_stream_response_with_usage_metadata(
        self, chat_handler, mock_llm_client, mock_streaming_chunks_with_usage_metadata
    ):
        """Test streaming response with usage_metadata in chunks."""
        mock_llm_client.stream.return_value = iter(
            mock_streaming_chunks_with_usage_metadata
        )

        history = []
        response = list(chat_handler.stream_response("Test", history))

        content_updates = [r for r in response if r.content]
        stats_updates = [r for r in response if r.metadata]

        assert len(content_updates) >= 1
        assert len(stats_updates) == 1
        stats_dict = stats_updates[0].metadata
        assert stats_dict["prompt_tokens"] == 10
        assert stats_dict["completion_tokens"] == 5

    def test_stream_response_empty_content(
        self, chat_handler, mock_llm_client, mock_streaming_chunks_empty
    ):
        """Test streaming response with empty content chunks."""
        mock_llm_client.stream.return_value = iter(mock_streaming_chunks_empty)

        history = []
        response = list(chat_handler.stream_response("Test", history))

        # Should still yield stats even with empty content
        stats_updates = [r for r in response if r.metadata]
        assert len(stats_updates) == 1

    def test_message_persistence(self, chat_handler, mock_llm_client, message_store):
        """Test that messages are persisted correctly."""
        mock_chunks = [
            Mock(
                content="Response",
                response_metadata={},
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
        mock_llm_client.stream.return_value = iter(mock_chunks)

        history = []
        list(chat_handler.stream_response("Test message", history))

        # Check that messages were stored
        stored_messages = message_store.get_messages()
        assert len(stored_messages) == 2  # User message + assistant response
        assert isinstance(stored_messages[0], HumanMessage)
        assert stored_messages[0].content == "Test message"
        assert isinstance(stored_messages[1], AIMessage)
        assert stored_messages[1].content == "Response"

    def test_stream_response_with_history(
        self, chat_handler, mock_llm_client, gradio_history
    ):
        """Test streaming response with existing history."""
        mock_chunks = [
            Mock(
                content="New response",
                response_metadata={},
                usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            ),
        ]
        mock_llm_client.stream.return_value = iter(mock_chunks)

        response = list(chat_handler.stream_response("New message", gradio_history))

        # Should process with history
        assert len(response) >= 1
        # Verify history was converted and used
        assert mock_llm_client.stream.called
        call_args = mock_llm_client.stream.call_args[0][0]
        assert len(call_args) == 3  # 2 from history + 1 new user message

    def test_stream_response_error_handling(
        self, chat_handler, mock_llm_client
    ):
        """Test error handling during streaming."""
        mock_llm_client.stream.side_effect = Exception("LLM error")

        history = []
        with pytest.raises(Exception, match="LLM error"):
            list(chat_handler.stream_response("Test", history))

    def test_get_history(self, chat_handler, message_store):
        """Test getting history in Gradio format."""
        message_store.add_message(HumanMessage(content="Hello"))
        message_store.add_message(AIMessage(content="Hi"))

        history = chat_handler.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi"}

    def test_get_history_empty(self, chat_handler):
        """Test getting empty history."""
        history = chat_handler.get_history()
        assert history == []

    def test_clear_history(self, chat_handler, message_store):
        """Test clearing history."""
        message_store.add_message(HumanMessage(content="Test"))
        message_store.add_message(AIMessage(content="Response"))
        chat_handler.clear_history()
        assert len(message_store.get_messages()) == 0

    @pytest.mark.parametrize(
        "user_message,expected_content",
        [
            ("Hello", "Hello"),
            ("", ""),
            ("A" * 1000, "A" * 1000),
            ("Special chars: !@#$%^&*()", "Special chars: !@#$%^&*()"),
        ],
    )
    def test_stream_response_various_messages(
        self, chat_handler, mock_llm_client, user_message, expected_content
    ):
        """Test streaming with various message types."""
        mock_chunks = [
            Mock(
                content="Response",
                response_metadata={},
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        ]
        mock_llm_client.stream.return_value = iter(mock_chunks)

        history = []
        list(chat_handler.stream_response(user_message, history))

        stored_messages = chat_handler.message_store.get_messages()
        assert stored_messages[0].content == expected_content

    def test_stream_response_accumulates_content(
        self, chat_handler, mock_llm_client
    ):
        """Test that content is properly accumulated across chunks."""
        mock_chunks = [
            Mock(content="Part", response_metadata={}, usage_metadata={}),
            Mock(content=" one", response_metadata={}, usage_metadata={}),
            Mock(content=" and", response_metadata={}, usage_metadata={}),
            Mock(
                content=" two",
                response_metadata={},
                usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            ),
        ]
        mock_llm_client.stream.return_value = iter(mock_chunks)

        history = []
        response = list(chat_handler.stream_response("Test", history))

        content_updates = [r for r in response if r.content]
        full_content = "".join(u.content for u in content_updates)
        assert full_content == "Part one and two"

        # Verify stored message has full content
        stored_messages = chat_handler.message_store.get_messages()
        assert stored_messages[1].content == "Part one and two"

