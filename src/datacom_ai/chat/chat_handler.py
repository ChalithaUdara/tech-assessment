"""Core chat handler with streaming support."""

from typing import Generator, List, Dict, Any, Union, Tuple, Optional

from datacom_ai.chat.models import StreamUpdate, ChatMode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from datacom_ai.chat.engine import ChatEngine
from datacom_ai.chat.message_store import MessageStore
from datacom_ai.utils.logger import logger


class ChatHandler:
    """Handles chat interactions with streaming and telemetry."""

    def __init__(
        self, 
        chat_engine: ChatEngine, 
        message_store: MessageStore, 
        rag_engine: Optional[ChatEngine] = None,
        planning_engine: Optional[ChatEngine] = None
    ):
        """
        Initialize the chat handler.

        Args:
            chat_engine: Configured ChatEngine (e.g., SimpleChatEngine)
            message_store: Message store for conversation history
            rag_engine: Optional RAG ChatEngine
            planning_engine: Optional PlanningAgent ChatEngine
        """
        self.chat_engine = chat_engine
        self.message_store = message_store
        self.rag_engine = rag_engine
        self.planning_engine = planning_engine

    def stream_response(
        self, 
        user_message: str, 
        history: List[Dict[str, Any]], 
        mode: str | ChatMode = ChatMode.DEFAULT
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response to a user message with telemetry.

        Args:
            user_message: The user's message
            history: Gradio chat history (list of dicts with 'role' and 'content')
            mode: Chat mode (ChatMode enum or string for backward compatibility)

        Yields:
            StreamUpdate objects containing content chunks or metadata
        """
        # Convert string mode to enum if needed (backward compatibility)
        if isinstance(mode, str):
            try:
                mode = ChatMode(mode)
            except ValueError:
                logger.warning(f"Invalid mode '{mode}', defaulting to {ChatMode.DEFAULT}")
                mode = ChatMode.DEFAULT

        logger.debug(f"Starting stream response for user message (length: {len(user_message)} chars), mode: {mode}")
        logger.debug(f"History contains {len(history)} previous messages")

        # Convert Gradio history to LangChain messages
        langchain_messages = MessageStore.from_gradio_history(history)
        logger.debug(f"Converted {len(langchain_messages)} messages from history")

        # Add the new user message
        user_msg = HumanMessage(content=user_message)
        langchain_messages.append(user_msg)

        # Store the user message
        self.message_store.add_message(user_msg)
        logger.debug("User message stored")

        # Select engine based on mode
        if mode == ChatMode.RAG and self.rag_engine:
            logger.info("Using RAG Engine")
            engine = self.rag_engine
        elif mode == ChatMode.PLANNING_AGENT and self.planning_engine:
            logger.info("Using Planning Agent Engine")
            engine = self.planning_engine
        else:
            logger.info("Using Default Chat Engine")
            engine = self.chat_engine

        # Stream the response via the engine
        accumulated_content = ""
        
        try:
            logger.info(f"Delegating to {mode.value} engine")
            
            # All engines now follow the ChatEngine protocol, so we can use them uniformly
            for update in engine.stream(langchain_messages):
                # Accumulate content for storage
                if update.content:
                    accumulated_content += update.content
                
                # Yield the update to the UI
                yield update

            # Store the assistant's response
            assistant_msg = AIMessage(content=accumulated_content)
            self.message_store.add_message(assistant_msg)
            logger.debug("Assistant message stored")

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            logger.exception("Exception during stream response")
            error_msg = f"Failed to generate response: {str(e)}"
            yield StreamUpdate(error=error_msg)
            raise

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history in Gradio format.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = self.message_store.get_messages()
        logger.debug(f"Retrieving history: {len(messages)} messages")
        return MessageStore.to_gradio_format(messages)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        logger.info("Clearing conversation history")
        self.message_store.clear()
        logger.debug("History cleared")

