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

    def chat_fn(message: str, history: list, mode: str) -> Generator[list, None, None]:
        """
        Chat function that streams responses.

        Args:
            message: User's message
            history: Gradio chat history (may contain dicts or ChatMessage objects)
            mode: Chat mode ("Default Chat" or "RAG")

        Yields:
            Updated history with streaming tokens and metadata
        """
        logger.info(f"Received user message (length: {len(message)} chars), mode: {mode}")
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
        assistant_message_index = len(history) - 1  # Track the index of the assistant message

        # Stream the response (pass dict format to handler)
        try:
            # Update chat handler engine based on mode
            # This is a bit of a hack, ideally handler should manage engines or we pass engine to stream_response
            # For now, we'll assume chat_handler has a method to switch or we pass it
            
            # NOTE: We need to update ChatHandler to support switching engines or pass the mode
            # Let's assume we update ChatHandler.stream_response to accept mode or we set it here
            
            for update in chat_handler.stream_response(message, history_dicts, mode=mode):
                # Handle metadata (stats or citations)
                if update.metadata:
                    metadata = update.metadata
                    
                    # Handle citations
                    if "citations" in metadata:
                        citations = metadata["citations"]
                        if citations:
                            # Format citations for metadata
                            citation_str = "\n".join([f"• {c}" for c in citations])
                            
                            # Create a separate message for citations, similar to stats
                            citations_message = gr.ChatMessage(
                                role="assistant",
                                content="\u200b",  # Zero-width space
                                metadata={"title": "📚 Citations", "log": citation_str}
                            )
                            history.append(citations_message)
                    
                    # Handle stats
                    if "prompt_tokens" in metadata or "latency_ms" in metadata:
                        stats_dict = metadata
                        stats_title = (
                            f"📊 Stats: prompt={stats_dict.get('prompt_tokens', 0)} "
                            f"completion={stats_dict.get('completion_tokens', 0)} "
                            f"cost=${stats_dict.get('cost_usd', 0.0):.6f} "
                            f"latency={stats_dict.get('latency_ms', 0)} ms"
                        )
                        logger.debug(f"Received stats metadata: {stats_dict}")
                        
                        stats_message = gr.ChatMessage(
                            role="assistant",
                            content="\u200b",  # Zero-width space
                            metadata={"title": stats_title}
                        )
                        history.append(stats_message)
                        # Note: assistant_message_index remains unchanged
                
                # Handle error
                elif update.error:
                    logger.warning(f"Received error in stream: {update.error}")
                    assistant_message.content += f"\n\n[Error] {update.error}"
                    # Update the assistant message at its tracked index, not history[-1]
                    # This preserves the stats message if it was added
                    history[assistant_message_index] = assistant_message
                
                # Handle content
                elif update.content:
                    assistant_message.content += update.content
                    # Update the assistant message at its tracked index, not history[-1]
                    # This preserves the stats message if it was added
                    history[assistant_message_index] = assistant_message
                
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
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    container=False,
                )
            with gr.Column(scale=1):
                mode_dropdown = gr.Dropdown(
                    choices=["Default Chat", "RAG"],
                    value="Default Chat",
                    label="Chat Mode",
                    interactive=True
                )
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        # Event handlers
        msg.submit(chat_fn, [msg, chatbot, mode_dropdown], [chatbot], queue=True).then(
            lambda: "", None, [msg], queue=False
        )
        submit_btn.click(chat_fn, [msg, chatbot, mode_dropdown], [chatbot], queue=True).then(
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

