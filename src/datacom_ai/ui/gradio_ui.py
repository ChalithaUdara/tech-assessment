"""Gradio UI for the chat interface."""

from typing import Generator, Union, Tuple, Dict, Any

import gradio as gr

from datacom_ai.chat.chat_handler import ChatHandler
from datacom_ai.utils.logger import logger

from gradio import Blocks


def create_chat_interface(chat_handler: ChatHandler) -> "gr.Blocks":
    """
    Create a Gradio chat interface with streaming support.

    Args:
        chat_handler: ChatHandler instance for processing messages

    Returns:
        Configured Gradio Blocks interface
    """
    logger.info("Creating Gradio chat interface")

    def chat_fn(message: str, history: list) -> Generator[list, None, None]:
        """
        Chat function that streams responses.

        Args:
            message: User's message
            history: Gradio chat history (may contain dicts or ChatMessage objects)

        Yields:
            Updated history with streaming tokens and metadata
        """
        logger.info(f"Received user message (length: {len(message)} chars)")
        logger.debug(f"Processing with {len(history)} messages in history")

        # Convert history to dict format for chat_handler (it only needs role/content)
        history_dicts = []
        for msg in history:
            if isinstance(msg, dict):
                history_dicts.append(msg)
            else:
                # ChatMessage object - convert to dict (only role and content, not metadata)
                history_dicts.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        # Add user message to history
        history.append(gr.ChatMessage(role="user", content=message))

        # Initialize assistant message (main response - always visible)
        assistant_message = gr.ChatMessage(role="assistant", content="")
        history.append(assistant_message)

        # Stream the response (pass dict format to handler)
        try:
            for item in chat_handler.stream_response(message, history_dicts):
                # Check if this is stats metadata (tuple) or content (string)
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "stats":
                    # This is stats metadata - create a separate message with metadata only
                    # This will appear as a collapsible section below the main message
                    stats_dict = item[1]
                    stats_title = (
                        f"📊 Stats: prompt={stats_dict['prompt_tokens']} "
                        f"completion={stats_dict['completion_tokens']} "
                        f"cost=${stats_dict['cost_usd']:.6f} "
                        f"latency={stats_dict['latency_ms']} ms"
                    )
                    logger.debug(f"Received stats metadata: {stats_dict}")
                    # Create a separate message with metadata for stats
                    # This will display as a collapsible section below the main message
                    # Using minimal invisible content to ensure proper rendering
                    stats_message = gr.ChatMessage(
                        role="assistant",
                        content="\u200b",  # Zero-width space - invisible but ensures message renders
                        metadata={"title": stats_title}
                    )
                    # Add stats message after the main response message
                    history.append(stats_message)
                elif isinstance(item, str):
                    # This is content - accumulate it in the main assistant message
                    assistant_message.content += item
                    # Update history with the modified message
                    history[-1] = assistant_message
                else:
                    # Fallback for any other format (e.g., error messages)
                    logger.warning(f"Unexpected item type in stream: {type(item)}")
                    assistant_message.content += str(item)
                    history[-1] = assistant_message
                
                yield history
        except Exception as e:
            logger.error(f"Error in chat_fn: {e}")
            logger.exception("Exception during chat function execution")
            raise

    def clear_fn() -> list:
        """Clear the chat history."""
        logger.info("User requested to clear chat history")
        chat_handler.clear_history()
        return []

    # Create the chat interface
    with gr.Blocks(title="Datacom AI Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Datacom AI Chat Interface
            
            Chat with GPT-4o with real-time token streaming and telemetry metrics.
            """
        )

        chatbot = gr.Chatbot(
            type="messages",
            label="Conversation",
            height=500,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                scale=4,
                container=False,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        # Event handlers
        msg.submit(chat_fn, [msg, chatbot], [chatbot], queue=True).then(
            lambda: "", None, [msg], queue=False
        )
        submit_btn.click(chat_fn, [msg, chatbot], [chatbot], queue=True).then(
            lambda: "", None, [msg], queue=False
        )
        clear_btn.click(clear_fn, None, [chatbot], queue=False)

        gr.Markdown(
            """
            ### Features
            - **Token-level streaming**: See responses appear in real-time
            - **Telemetry metrics**: View token usage, cost, and latency after each response
            - **Message persistence**: Last 10 messages are maintained in conversation history
            """
        )

    return demo

