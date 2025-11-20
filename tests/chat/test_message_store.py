"""Unit tests for MessageStore."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from datacom_ai.chat.message_store import MessageStore


class TestMessageStore:
    """Tests for MessageStore."""

    def test_add_and_get_messages(self, sample_human_message, sample_ai_message):
        """Test adding and retrieving messages."""
        store = MessageStore(max_messages=5)
        store.add_message(sample_human_message)
        store.add_message(sample_ai_message)

        messages = store.get_messages()
        assert len(messages) == 2
        assert messages[0] == sample_human_message
        assert messages[1] == sample_ai_message

    def test_add_multiple_messages(self, sample_messages):
        """Test adding multiple messages at once."""
        store = MessageStore(max_messages=10)
        store.add_messages(sample_messages)

        messages = store.get_messages()
        assert len(messages) == 2
        assert messages == sample_messages

    def test_message_trimming(self, message_store_small):
        """Test that messages are trimmed when exceeding max."""
        store = message_store_small
        for i in range(5):
            store.add_message(HumanMessage(content=f"Message {i}"))

        messages = store.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"

    def test_message_trimming_with_add_messages(self):
        """Test trimming when adding multiple messages at once."""
        store = MessageStore(max_messages=3)
        # Add 5 messages at once
        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        store.add_messages(messages)

        stored = store.get_messages()
        assert len(stored) == 3
        assert stored[0].content == "Message 2"
        assert stored[1].content == "Message 3"
        assert stored[2].content == "Message 4"

    def test_clear_messages(self, message_store):
        """Test clearing all messages."""
        store = message_store
        store.add_message(HumanMessage(content="Test"))
        store.add_message(AIMessage(content="Response"))
        store.clear()
        assert len(store.get_messages()) == 0

    def test_get_messages_returns_copy(self, message_store):
        """Test that get_messages returns a copy, not the original list."""
        store = message_store
        store.add_message(HumanMessage(content="Test"))
        messages1 = store.get_messages()
        messages2 = store.get_messages()
        
        # Modifying one shouldn't affect the other
        messages1.append(HumanMessage(content="Extra"))
        assert len(store.get_messages()) == 1
        assert len(messages1) == 2
        assert len(messages2) == 1

    @pytest.mark.parametrize(
        "max_messages,expected_max",
        [
            (5, 5),
            (10, 10),
            (100, 100),
            (None, 10),  # Default from settings
        ],
    )
    def test_max_messages_initialization(self, max_messages, expected_max):
        """Test MessageStore initialization with different max_messages values."""
        if max_messages is None:
            store = MessageStore()
        else:
            store = MessageStore(max_messages=max_messages)
        assert store.max_messages == expected_max

    def test_from_gradio_history(self, gradio_history):
        """Test converting Gradio history to LangChain messages."""
        messages = MessageStore.from_gradio_history(gradio_history)

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there"

    def test_from_gradio_history_empty(self):
        """Test converting empty Gradio history."""
        messages = MessageStore.from_gradio_history([])
        assert messages == []

    def test_from_gradio_history_with_chatmessage_objects(self):
        """Test converting Gradio ChatMessage objects to LangChain messages."""
        # Simulate ChatMessage objects (like Gradio uses)
        class ChatMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        gradio_history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]
        messages = MessageStore.from_gradio_history(gradio_history)

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there"

    def test_from_gradio_history_ignores_unknown_roles(self):
        """Test that unknown roles are ignored."""
        gradio_history = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Hi"},
            {"role": "unknown", "content": "Unknown"},
        ]
        messages = MessageStore.from_gradio_history(gradio_history)

        # Only user and assistant messages should be included
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi"

    def test_from_gradio_history_handles_missing_content(self):
        """Test handling of messages with missing content."""
        gradio_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},  # Missing content
            {"role": "user", "content": "Another"},
        ]
        messages = MessageStore.from_gradio_history(gradio_history)

        assert len(messages) == 3
        assert messages[1].content == ""  # Empty content for missing field

    def test_to_gradio_format(self, sample_messages):
        """Test converting LangChain messages to Gradio format."""
        gradio_format = MessageStore.to_gradio_format(sample_messages)

        assert len(gradio_format) == 2
        assert gradio_format[0] == {"role": "user", "content": sample_messages[0].content}
        assert gradio_format[1] == {"role": "assistant", "content": sample_messages[1].content}

    def test_to_gradio_format_empty(self):
        """Test converting empty message list."""
        gradio_format = MessageStore.to_gradio_format([])
        assert gradio_format == []

    def test_to_gradio_format_only_human(self):
        """Test converting only human messages."""
        messages = [HumanMessage(content="Hello")]
        gradio_format = MessageStore.to_gradio_format(messages)
        assert len(gradio_format) == 1
        assert gradio_format[0] == {"role": "user", "content": "Hello"}

    def test_to_gradio_format_only_ai(self):
        """Test converting only AI messages."""
        messages = [AIMessage(content="Response")]
        gradio_format = MessageStore.to_gradio_format(messages)
        assert len(gradio_format) == 1
        assert gradio_format[0] == {"role": "assistant", "content": "Response"}

    def test_to_langchain_messages(self, message_store, sample_messages):
        """Test converting stored messages to LangChain format."""
        store = message_store
        store.add_messages(sample_messages)
        langchain_messages = store.to_langchain_messages()

        assert langchain_messages == sample_messages
        assert langchain_messages is not store.get_messages()  # Should be a copy

    def test_trimming_preserves_order(self):
        """Test that trimming preserves message order."""
        store = MessageStore(max_messages=3)
        messages = [HumanMessage(content=f"Msg {i}") for i in range(5)]
        for msg in messages:
            store.add_message(msg)

        stored = store.get_messages()
        assert len(stored) == 3
        # Should keep the last 3 messages in order
        assert stored[0].content == "Msg 2"
        assert stored[1].content == "Msg 3"
        assert stored[2].content == "Msg 4"

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "A",
            "A" * 1000,
            "Multi\nline\ncontent",
            "Special: !@#$%^&*()",
            "Unicode: 你好世界 🌍",
        ],
    )
    def test_message_content_variations(self, content):
        """Test storing messages with various content types."""
        store = MessageStore()
        store.add_message(HumanMessage(content=content))
        messages = store.get_messages()
        assert messages[0].content == content

