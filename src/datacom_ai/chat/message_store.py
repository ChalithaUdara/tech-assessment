"""Message persistence and history management."""

from typing import List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger


class MessageStore:
    """In-memory message store maintaining the last N messages."""

    def __init__(self, max_messages: int = None):
        """
        Initialize the message store.

        Args:
            max_messages: Maximum number of messages to keep (defaults to settings.MAX_MESSAGES)
        """
        self.max_messages = max_messages or settings.MAX_MESSAGES
        self._messages: List[BaseMessage] = []
        logger.debug(f"MessageStore initialized with max_messages={self.max_messages}")

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the store and trim if necessary.

        Args:
            message: The message to add
        """
        message_type = type(message).__name__
        content_preview = (
            message.content[:50] + "..." if len(message.content) > 50 else message.content
        )
        logger.debug(
            f"Adding {message_type} message (preview: {content_preview}), "
            f"current count: {len(self._messages)}"
        )
        self._messages.append(message)
        self._trim_if_needed()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        Add multiple messages to the store and trim if necessary.

        Args:
            messages: List of messages to add
        """
        logger.debug(f"Adding {len(messages)} messages, current count: {len(self._messages)}")
        self._messages.extend(messages)
        self._trim_if_needed()

    def get_messages(self) -> List[BaseMessage]:
        """
        Get all stored messages.

        Returns:
            List of messages in chronological order
        """
        return self._messages.copy()

    def clear(self) -> None:
        """Clear all messages from the store."""
        message_count = len(self._messages)
        self._messages.clear()
        logger.info(f"Cleared {message_count} messages from store")

    def _trim_if_needed(self) -> None:
        """Trim messages if they exceed the maximum count."""
        if len(self._messages) > self.max_messages:
            trimmed_count = len(self._messages) - self.max_messages
            # Keep only the last N messages
            self._messages = self._messages[-self.max_messages :]
            logger.debug(
                f"Trimmed {trimmed_count} messages, keeping last {self.max_messages} messages"
            )

    def to_langchain_messages(self) -> List[BaseMessage]:
        """
        Convert stored messages to LangChain message format.

        Returns:
            List of LangChain BaseMessage objects
        """
        return self.get_messages()

    @staticmethod
    def from_gradio_history(history: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Convert Gradio chat history to LangChain messages.
        
        Handles both dict format and ChatMessage objects. Only extracts content,
        ignoring metadata (stats) to keep history clean.

        Args:
            history: Gradio chat history (list of dicts or ChatMessage objects with 'role' and 'content')

        Returns:
            List of LangChain BaseMessage objects
        """
        messages = []
        for msg in history:
            # Handle both dict and ChatMessage formats
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                # ChatMessage object
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            
            # Only extract content (metadata is ignored, which is what we want)
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def to_gradio_format(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to Gradio chat format.

        Args:
            messages: List of LangChain BaseMessage objects

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        gradio_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                gradio_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gradio_messages.append({"role": "assistant", "content": msg.content})
        return gradio_messages

