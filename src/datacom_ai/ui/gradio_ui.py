"""Gradio UI for the chat interface."""

from typing import Generator, Union, Tuple, Dict, Any

import gradio as gr

from datacom_ai.chat.chat_handler import ChatHandler
from datacom_ai.chat.models import ChatMode
from datacom_ai.ui.history_manager import GradioHistoryManager
from datacom_ai.utils.logger import logger




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

        # Use history manager for cleaner state management
        history_manager = GradioHistoryManager(history)
        history_manager.add_user_message(message)
        history_manager.initialize_assistant_message()

        # Stream the response (pass dict format to handler)
        try:
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
                            history_manager.add_metadata_message(
                                title="📚 Citations",
                                log=citation_str
                            )
                    
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
                        history_manager.add_metadata_message(title=stats_title)
                
                # Handle error
                elif update.error:
                    logger.warning(f"Received error in stream: {update.error}")
                    current_content = history_manager.history[history_manager.assistant_message_index].content
                    history_manager.update_assistant_content(
                        f"{current_content}\n\n[Error] {update.error}"
                    )
                
                # Handle content
                elif update.content:
                    history_manager.append_to_assistant_content(update.content)
                
                yield history_manager.get_history()
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
    with gr.Blocks(title="Datacom AI Chat") as demo:
        gr.Markdown(
            """
            # Datacom AI Chat Interface
            
            Chat with GPT-4o with real-time token streaming and telemetry metrics.
            """
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
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
                            choices=[mode.value for mode in ChatMode],
                            value=ChatMode.DEFAULT.value,
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
                
                # Clear chat history when mode changes
                def clear_on_mode_change(mode: str) -> list:
                    """Clear the chat history when switching modes."""
                    logger.info(f"Mode changed to {mode}, clearing chat history")
                    chat_handler.clear_history()
                    return []
                
                mode_dropdown.change(clear_on_mode_change, [mode_dropdown], [chatbot], queue=False)

                gr.Markdown(
                    """
                    ### Features
                    - **Token-level streaming**: See responses appear in real-time
                    - **Telemetry metrics**: View token usage, cost, and latency after each response
                    - **Message persistence**: Last 10 messages are maintained in conversation history
                    """
                )

            with gr.Tab("Coding Agent"):
                gr.Markdown("### Self-Healing Code Assistant")
                gr.Markdown("Enter a natural language coding task (e.g., 'Write quicksort in Python'). The agent will generate code, run tests, and self-heal if errors occur.")
                
                with gr.Row():
                    agent_input = gr.Textbox(label="Task Prompt", placeholder="e.g. Write a Python function to calculate fibonacci sequence")
                    agent_run_btn = gr.Button("Run Agent", variant="primary")
                
                agent_output = gr.Textbox(label="Agent Logs", interactive=False, lines=20)

                def run_coding_agent(task: str):
                    from datacom_ai.self_heal.agent import SelfHealingAgent
                    agent = SelfHealingAgent()
                    logs = ""
                    yield logs
                    for update in agent.stream(task):
                        logs += update + "\n"
                        yield logs

                agent_run_btn.click(run_coding_agent, [agent_input], [agent_output])

    return demo

