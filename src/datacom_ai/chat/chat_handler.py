"""Core chat handler with streaming support."""

from typing import Generator, List, Dict, Any, Union, Tuple

from datacom_ai.chat.models import StreamUpdate

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langchain_openai import AzureChatOpenAI  # Removed as it's now in the engine

from datacom_ai.chat.engine import ChatEngine
from datacom_ai.chat.message_store import MessageStore
# from datacom_ai.telemetry.metrics import TelemetryMetrics # Removed as it's now in the engine
from datacom_ai.utils.logger import logger


class ChatHandler:
    """Handles chat interactions with streaming and telemetry."""

    def __init__(self, chat_engine: ChatEngine, message_store: MessageStore, rag_engine: ChatEngine = None):
        """
        Initialize the chat handler.

        Args:
            chat_engine: Configured ChatEngine (e.g., SimpleChatEngine)
            message_store: Message store for conversation history
            rag_engine: Optional RAG ChatEngine
        """
        self.chat_engine = chat_engine
        self.message_store = message_store
        self.rag_engine = rag_engine
        
        # Initialize Planning Agent
        # In a real app, this should be injected or configured properly
        try:
            from datacom_ai.agent.planning_agent import PlanningAgent
            self.planning_agent = PlanningAgent()
        except Exception as e:
            logger.error(f"Failed to initialize PlanningAgent: {e}")
            self.planning_agent = None

    def stream_response(
        self, user_message: str, history: List[Dict[str, Any]], mode: str = "Default Chat"
    ) -> Generator[StreamUpdate, None, None]:
        """
        Stream a response to a user message with telemetry.

        Args:
            user_message: The user's message
            history: Gradio chat history (list of dicts with 'role' and 'content')
            mode: Chat mode ("Default Chat", "RAG", or "Planning Agent")

        Yields:
            StreamUpdate objects containing content chunks or metadata
        """
        from datacom_ai.chat.models import StreamUpdate

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
        if mode == "RAG" and self.rag_engine:
            logger.info("Using RAG Engine")
            engine = self.rag_engine
        elif mode == "Planning Agent" and self.planning_agent:
            logger.info("Using Planning Agent")
            # PlanningAgent doesn't strictly follow ChatEngine protocol yet, but we'll adapt
            engine = self.planning_agent
        else:
            logger.info("Using Default Chat Engine")
            engine = self.chat_engine

        # Stream the response via the engine
        accumulated_content = ""
        
        try:
            logger.info(f"Delegating to {mode} engine")
            
            if mode == "Planning Agent":
                # Special handling for Planning Agent stream
                # Initialize metrics for this stream
                from datacom_ai.telemetry.metrics import TelemetryMetrics
                metrics = TelemetryMetrics()
                metrics.start_timer()
                
                for chunk in engine.stream(langchain_messages):
                    # Chunk is a dict with "content" or "metadata"
                    if isinstance(chunk, dict):
                        if "content" in chunk:
                            content = chunk["content"]
                            update = StreamUpdate(content=content)
                            accumulated_content += content
                            yield update
                        
                        if "metadata" in chunk and "usage" in chunk["metadata"]:
                            usage = chunk["metadata"]["usage"]
                            prompt_tokens = usage.get("input_tokens", 0)
                            completion_tokens = usage.get("output_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                            metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)
                    else:
                        # Fallback for string chunks
                        update = StreamUpdate(content=chunk)
                        accumulated_content += chunk
                        yield update
                
                # Stop timer and yield final stats
                metrics.stop_timer()
                stats_dict = metrics.to_dict()
                yield StreamUpdate(metadata=stats_dict)
            else:
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

