# Datacom AI Technical Assessment

AI platform implementation for retrieval-augmented pipelines, reasoning agents, and self-evaluation loops.

## Task 3.1: Conversational Core (Streaming & Cost Telemetry)

A streaming chat interface with token-level streaming, message persistence, and comprehensive telemetry using LangChain and Gradio.

## Prerequisites

- Python >= 3.10
- uv (recommended) or pip for package management
- Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, OPENAI_API_KEY)

## Installation

1. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e ".[ui]"
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```

   Required variables:
   - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint (e.g., `https://<your-resource-name>.openai.azure.com/`)
   - `AZURE_OPENAI_DEPLOYMENT`: Your Azure OpenAI deployment name
   - `OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_API_VERSION`: API version (default: `2024-02-15-preview`)
   - `MODEL_NAME`: Model name for display/telemetry (default: `gpt-4o`)

## Running the Chat Application

```bash
# Using uv (recommended)
uv run python chat.py

# Or using pip/standard Python
python chat.py
```

The Gradio interface will launch at `http://localhost:7860`

## Verification

### Quick Verification Script

Run the verification script to check all components:

```bash
# Using uv (recommended)
uv run python scripts/verify_chat.py

# Or using standard Python
python scripts/verify_chat.py
```

This will verify:
- ✅ All imports are working
- ✅ Configuration is set correctly
- ✅ Message store functionality
- ✅ Telemetry calculations
- ✅ LLM client initialization

### End-to-End Test

1. **Start the application:**
   ```bash
   uv run python chat.py
   # Or: python chat.py
   ```

2. **In the Gradio interface:**
   - Type "Hello" in the message box
   - Press Enter or click Send
   - Verify:
     - ✅ Response streams token-by-token in real-time
     - ✅ Metrics appear at the end: `[stats] prompt=X completion=Y cost=$Z.ZZZZZZ latency=W ms`
     - ✅ Multiple messages persist (try 2-3 exchanges)

3. **Expected output format:**
   ```
   [stats] prompt=8 completion=23 cost=$0.000146 latency=623 ms
   ```

### Running Tests

```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
pytest

# Run with coverage
pytest --cov=src/datacom_ai --cov-report=term-missing

# Run specific test file
pytest tests/chat/test_chat_handler.py -v
```

## Architecture

- **Configuration**: Environment variable management and pricing constants
- **LLM Client**: LangChain AzureChatOpenAI for Azure OpenAI integration
- **Message Store**: In-memory persistence (last 10 messages)
- **Telemetry**: Token counting, cost calculation, latency tracking
- **Chat Handler**: Streaming response generation with metrics
- **Gradio UI**: Web interface with real-time token streaming

## Features

- ✅ Token-level streaming responses
- ✅ Message persistence (last 10 messages)
- ✅ Telemetry: prompt tokens, completion tokens, cost (USD), latency
- ✅ Real-time metrics display after each response

