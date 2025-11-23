"""Gradio chat history management."""

from typing import List, Optional
import gradio as gr

from datacom_ai.utils.logger import logger


class GradioHistoryManager:
    """Manages Gradio chat history state."""

    def __init__(self, history: List):
        """
        Initialize the history manager.

        Args:
            history: Gradio chat history list
        """
        self.history = history
        self.assistant_message_index: Optional[int] = None

    def add_user_message(self, content: str) -> None:
        """
        Add user message to history.

        Args:
            content: User message content
        """
        self.history.append(gr.ChatMessage(role="user", content=content))
        logger.debug(f"Added user message to history (length: {len(content)} chars)")

    def initialize_assistant_message(self) -> None:
        """Initialize assistant message and track its index."""
        assistant_msg = gr.ChatMessage(role="assistant", content="")
        self.history.append(assistant_msg)
        self.assistant_message_index = len(self.history) - 1
        logger.debug(f"Initialized assistant message at index {self.assistant_message_index}")

    def update_assistant_content(self, content: str) -> None:
        """
        Update assistant message content.

        Args:
            content: New content to append or set
        """
        if self.assistant_message_index is not None:
            msg = self.history[self.assistant_message_index]
            msg.content = content
            self.history[self.assistant_message_index] = msg
            logger.debug(f"Updated assistant message content (total length: {len(content)} chars)")

    def append_to_assistant_content(self, content: str) -> None:
        """
        Append content to existing assistant message.

        Args:
            content: Content to append
        """
        if self.assistant_message_index is not None:
            msg = self.history[self.assistant_message_index]
            msg.content += content
            self.history[self.assistant_message_index] = msg

    def add_metadata_message(self, title: str, content: str = "\u200b", log: Optional[str] = None) -> None:
        """
        Add a metadata message (stats, citations, etc.).

        Args:
            title: Title to display in metadata
            content: Message content (defaults to zero-width space)
            log: Optional log content for metadata
        """
        metadata = {"title": title}
        if log:
            metadata["log"] = log

        msg = gr.ChatMessage(
            role="assistant",
            content=content,
            metadata=metadata
        )
        self.history.append(msg)
        logger.debug(f"Added metadata message: {title}")

    def get_history(self) -> List:
        """Get the current history."""
        return self.history


