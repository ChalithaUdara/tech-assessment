#!/usr/bin/env python3
"""Main entry point for the chat application."""

import sys
import os

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datacom_ai.chat.chat_handler import ChatHandler
from datacom_ai.chat.engine import SimpleChatEngine
from datacom_ai.chat.message_store import MessageStore
from datacom_ai.clients.llm_client import create_llm_client
from datacom_ai.config.settings import settings
from datacom_ai.ui.gradio_ui import create_chat_interface
from datacom_ai.utils.logger import logger, setup_logging


def main():
    """Initialize and launch the chat interface."""
    # Setup logging - can be configured via environment variable
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/chat.log")
    log_format = os.getenv("LOG_FORMAT", "both")  # json, text, or both
    json_log_file = os.getenv("LOG_JSON_FILE", "logs/chat.jsonl")
    
    setup_logging(
        level=log_level, 
        log_file=log_file if log_file else None,
        log_format=log_format,
        json_log_file=json_log_file if log_format in ("json", "both") else None
    )

    try:
        logger.info("Starting chat application initialization")

        # Validate configuration
        logger.debug("Validating configuration")
        settings.validate()
        logger.info("Configuration validated successfully")

        # Initialize components
        logger.info("Initializing LLM client")
        llm_client = create_llm_client()
        logger.success("LLM client initialized")

        logger.info("Initializing message store")
        message_store = MessageStore()
        logger.success("Message store initialized")

        logger.info("Initializing chat engine")
        chat_engine = SimpleChatEngine(llm_client)
        logger.success("Chat engine initialized")

        # Initialize RAG components
        logger.info("Initializing RAG components")
        from datacom_ai.rag.pipeline import RAGPipeline
        from datacom_ai.chat.engine import RAGChatEngine
        
        rag_pipeline = RAGPipeline()
        rag_engine = RAGChatEngine(rag_pipeline)
        logger.success("RAG components initialized")

        # Initialize Planning Agent components
        logger.info("Initializing Planning Agent components")
        try:
            from datacom_ai.agent.planning_agent import PlanningAgent
            from datacom_ai.chat.engine import PlanningAgentChatEngine
            
            planning_agent = PlanningAgent(llm_client=llm_client)
            planning_engine = PlanningAgentChatEngine(planning_agent)
            logger.success("Planning Agent components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Planning Agent: {e}")
            logger.warning("Planning Agent mode will not be available")
            planning_engine = None

        logger.info("Initializing chat handler")
        chat_handler = ChatHandler(
            chat_engine=chat_engine,
            message_store=message_store,
            rag_engine=rag_engine,
            planning_engine=planning_engine
        )
        logger.success("Chat handler initialized")

        logger.info("Creating Gradio interface")
        demo = create_chat_interface(chat_handler)
        logger.success("Gradio interface created")

        logger.info("=" * 60)
        logger.info("Chat interface ready!")
        logger.info("=" * 60)
        logger.info(f"Model: {settings.MODEL_NAME}")
        logger.info(f"Azure Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
        logger.info(f"API Version: {settings.AZURE_OPENAI_API_VERSION}")
        logger.info("=" * 60)

        # Launch the interface
        logger.info("Launching Gradio interface on 0.0.0.0:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.exception("Configuration validation failed")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error starting chat application: {e}")
        logger.exception("Unexpected error during application startup")
        sys.exit(1)


if __name__ == "__main__":
    main()

