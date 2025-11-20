"""Core chat handler with streaming support."""

from typing import Generator, List, Dict, Any, Union, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from datacom_ai.chat.message_store import MessageStore
from datacom_ai.telemetry.metrics import TelemetryMetrics
from datacom_ai.utils.logger import logger


class ChatHandler:
    """Handles chat interactions with streaming and telemetry."""

    def __init__(self, llm_client: AzureChatOpenAI, message_store: MessageStore):
        """
        Initialize the chat handler.

        Args:
            llm_client: Configured LangChain AzureChatOpenAI client
            message_store: Message store for conversation history
        """
        self.llm_client = llm_client
        self.message_store = message_store

    def stream_response(
        self, user_message: str, history: List[Dict[str, Any]]
    ) -> Generator[Union[str, Tuple[str, Dict[str, Any]]], None, None]:
        """
        Stream a response to a user message with telemetry.

        Args:
            user_message: The user's message
            history: Gradio chat history (list of dicts with 'role' and 'content')

        Yields:
            Either:
            - str: Incremental content tokens from the streaming response
            - Tuple[str, Dict]: ("stats", stats_dict) for metadata to display separately
        """
        logger.debug(f"Starting stream response for user message (length: {len(user_message)} chars)")
        logger.debug(f"History contains {len(history)} previous messages")

        # Initialize metrics
        metrics = TelemetryMetrics()
        metrics.start_timer()

        # Convert Gradio history to LangChain messages
        langchain_messages = MessageStore.from_gradio_history(history)
        logger.debug(f"Converted {len(langchain_messages)} messages from history")

        # Add the new user message
        user_msg = HumanMessage(content=user_message)
        langchain_messages.append(user_msg)

        # Store the user message
        self.message_store.add_message(user_msg)
        logger.debug("User message stored")

        # Stream the response
        accumulated_content = ""
        final_chunk = None
        chunk_count = 0

        try:
            logger.info("Starting LLM stream")
            # Stream tokens from the LLM
            for chunk in self.llm_client.stream(langchain_messages):
                chunk_count += 1
                # Keep track of the final chunk (it may contain usage metadata)
                final_chunk = chunk
                
                # Yield content tokens (only actual content, not stats)
                if hasattr(chunk, "content") and chunk.content:
                    accumulated_content += chunk.content
                    yield chunk.content

                # Check for usage metadata in each chunk (may appear in any chunk)
                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
                    prompt_tokens = usage_metadata.get("input_tokens", 0)
                    completion_tokens = usage_metadata.get("output_tokens", 0)
                    total_tokens = usage_metadata.get("total_tokens", 0)
                    metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)
                    logger.debug(
                        f"Token usage updated: prompt={prompt_tokens}, "
                        f"completion={completion_tokens}, total={total_tokens}"
                    )

            # Stop the timer
            metrics.stop_timer()

            logger.info(
                f"Stream completed: {chunk_count} chunks, "
                f"{len(accumulated_content)} chars, "
                f"{metrics.get_latency_ms()}ms latency"
            )

            # Store the assistant's response (clean content without stats)
            # Use the final chunk if it's an AIMessage, otherwise create a new one
            if isinstance(final_chunk, AIMessage):
                assistant_msg = final_chunk
            else:
                assistant_msg = AIMessage(content=accumulated_content)
            self.message_store.add_message(assistant_msg)
            logger.debug("Assistant message stored")

            # Yield stats as metadata (separate from content)
            stats_dict = metrics.to_dict()
            logger.info(
                f"Response metrics: {stats_dict['prompt_tokens']} prompt tokens, "
                f"{stats_dict['completion_tokens']} completion tokens, "
                f"${stats_dict['cost_usd']:.6f} cost, "
                f"{stats_dict['latency_ms']}ms latency"
            )
            yield ("stats", stats_dict)

        except Exception as e:
            metrics.stop_timer()
            logger.error(f"Failed to generate response: {e}")
            logger.exception("Exception during stream response")
            error_msg = f"\n\n[error] Failed to generate response: {str(e)}"
            yield error_msg
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

