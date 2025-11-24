# Datacom AI Technical Assessment

AI platform implementation for retrieval-augmented pipelines, reasoning agents, and self-evaluation loops. Features a streaming chat interface with token-level streaming, message persistence, comprehensive telemetry, and autonomous planning capabilities.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Architecture](#architecture)
- [Running the Application](#running-the-application)
- [Running Automated Tests](#running-automated-tests)
- [Task-Specific Documentation](#task-specific-documentation)
- [Docker & Containerization](#docker--containerization)
- [Analytics Dashboard](#analytics-dashboard)
- [Troubleshooting](#troubleshooting)

## Overview

This platform implements four core tasks:

1. **Task 3.1: Conversational Core** - Streaming chat with token-level streaming, message persistence, and cost telemetry
2. **Task 3.2: High-Performance RAG** - Retrieval-augmented QA with sub-300ms retrieval times and automated evaluation
3. **Task 3.3: Autonomous Planning Agent** - Multi-tool agent for complex planning tasks (e.g., trip planning)
4. **Task 3.4: Self-Healing Code Assistant** - Iterative code generation with automatic test execution and error recovery

**Stretch Goal**: Analytics dashboard with latency, cost, retrieval accuracy, and agent performance metrics.

## Prerequisites

- **Python** >= 3.10
- **Docker Desktop** (or Docker Engine) - Required for Qdrant vector database
- **uv** (recommended) or **pip** for package management
- **Azure OpenAI** credentials:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `OPENAI_API_KEY`
  - `AZURE_OPENAI_API_VERSION` (default: `2024-02-15-preview`)

## Installation & Setup

### 1. Clone and Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[ui]"
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=Gpt4o
OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
MODEL_NAME=Gpt4o

# Optional: Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=datacom_rag

# Optional: RAG Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVER_K=5
EMBEDDING_PROVIDER=fastembed  # Options: fastembed, azure_openai

# Optional: Logging Configuration
LOG_FORMAT=both  # Options: json, text, both
LOG_JSON_FILE=logs/chat.jsonl

# Optional: Planning Agent API Keys
# See Task 3.3 section for detailed setup instructions
AMADEUS_CLIENT_ID=your-amadeus-client-id  # For real flight search
AMADEUS_CLIENT_SECRET=your-amadeus-client-secret
GPLACES_API_KEY=your-google-places-api-key  # For real attractions search
```

### 3. Start Qdrant Vector Database

```bash
docker-compose up -d
```

Verify Qdrant is running:
- Dashboard: http://localhost:6333/dashboard
- Health check: `curl http://localhost:6333/health`

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI Layer                        │
│  (Chat, RAG, Planning Agent, Coding Agent Tabs)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Chat Handler                              │
│  (Routes to appropriate engine based on mode)               │
└──────┬──────────────┬──────────────┬───────────────────────┘
       │              │              │
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────────┐
│ Simple Chat │ │ RAG Engine  │ │ Planning Agent     │
│   Engine    │ │             │ │   Engine           │
└─────────────┘ └─────┬───────┘ └───────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼────────┐      ┌─────────▼─────────┐
│ Retrieval       │      │  Vector Store      │
│ Pipeline        │      │  (Qdrant)          │
└─────────────────┘      └───────────────────┘
```

### Core Components

#### 1. **Chat System** (`src/datacom_ai/chat/`)
- **ChatHandler**: Main orchestrator, routes requests to appropriate engines
- **SimpleChatEngine**: Basic LLM chat without RAG
- **RAGChatEngine**: RAG-enabled chat with retrieval
- **PlanningAgentChatEngine**: Planning agent integration
- **MessageStore**: In-memory persistence (last 10 messages)

#### 2. **RAG Pipeline** (`src/datacom_ai/rag/`)
- **RAGPipeline**: Facade for the RAG system
- **IndexPipeline**: Document loading, chunking, embedding, indexing
- **RetrievalPipeline**: Query processing, retrieval, answer generation
- **DocumentLoader**: Loads documents from directory
- **Factories**: Embedding, text splitter, vector store factories

#### 3. **Planning Agent** (`src/datacom_ai/agent/`)
- **PlanningAgent**: LangGraph-based agent with tool calling
- **Tools**: Flight search, weather, attractions (mock/real APIs)
- **Scratch-pad reasoning**: Visible in logs
- **Constraint handling**: Budget, dates, etc.

#### 4. **Self-Healing Code Assistant** (`src/datacom_ai/self_heal/`)
- **SelfHealingAgent**: LangGraph-based iterative code generation
- **Test execution**: Automatic pytest/cargo test execution
- **Error recovery**: Feeds errors back to LLM, retries up to 3 times
- **Progress streaming**: Real-time updates to console/UI

#### 5. **Telemetry** (`src/datacom_ai/telemetry/`)
- **TelemetryMetrics**: Token counting, cost calculation, latency tracking
- **Logging**: Structured logging with JSON/text formats
- **Metrics display**: Real-time stats after each response

#### 6. **Configuration** (`src/datacom_ai/config/`)
- **Settings**: Centralized configuration management
- **Environment variable loading**: Via python-dotenv
- **Validation**: Ensures required variables are set

### Data Flow

1. **User Input** → Gradio UI
2. **Chat Handler** → Routes to appropriate engine
3. **Engine Processing**:
   - Simple: Direct LLM call
   - RAG: Retrieval → Context injection → LLM call
   - Planning: Tool calls → Reasoning → Response
   - Coding: Code generation → Test execution → Retry if needed
4. **Streaming Response** → Token-by-token to UI
5. **Telemetry** → Metrics logged and displayed
6. **Message Persistence** → Last 10 messages stored

## Running the Application

### Start the Chat Interface

```bash
# Using uv (recommended)
uv run python chat.py

# Or using pip/standard Python
python chat.py
```

The Gradio interface will launch at `http://localhost:7860`

### Chat Modes

The UI provides multiple tabs:

1. **Chat**: Basic conversational mode
2. **RAG**: Retrieval-augmented generation (requires indexed documents)
3. **Planning Agent**: Autonomous planning with tool calling
4. **Coding Agent**: Self-healing code generation

### Task 3.1 Acceptance Test

1. Start the application: `python chat.py`
2. In the Gradio interface:
   - Type "Hello" in the message box
   - Press Enter or click Send
   - Verify:
     - ✅ Response streams token-by-token in real-time
     - ✅ Metrics appear at the end: `[stats] prompt=X completion=Y cost=$Z.ZZZZZZ latency=W ms`
     - ✅ Multiple messages persist (try 2-3 exchanges)

Expected output format:
```
[stats] prompt=8 completion=23 cost=$0.000146 latency=623 ms
```

## Running Automated Tests

### Unit Tests

```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
pytest -q
```

### Acceptance Tests

#### Task 3.1: Streaming & Telemetry Acceptance Test

```bash
# Test streaming response and metrics display
uv run python scripts/acceptance_test_streaming_telemetry.py
```

**Expected Output:**
```
User: Hello
Assistant: [streamed response]
[stats] prompt=8 completion=23 cost=$0.000146 latency=623 ms
✅ Acceptance test PASSED
```

#### Task 3.2: RAG Retrieval Acceptance Test

**Prerequisites:**
1. Qdrant must be running: `docker-compose up -d`
2. Dataset must be downloaded: `python scripts/download_corpus.py`
3. Documents must be indexed: `python scripts/index_data.py`
4. Evaluation dataset must exist: `data/processed/synthetic_dataset_ragas.json`

**Run Test:**
```bash
# Run full RAG evaluation (retrieval + generation metrics)
uv run python scripts/acceptance_test_rag_retrieval.py
```

**Expected Output:**
```
=== Evaluation Summary ===
Total test cases evaluated: 30
Retrieval k value: 5

--- RETRIEVAL METRICS ---
  Contextual Precision@5: 0.8234 (threshold: 0.5)
  Contextual Recall@5: 0.7567 (threshold: 0.5)
  Contextual Relevancy@5: 0.6891 (threshold: 0.5)

--- GENERATION METRICS ---
  Answer Relevancy: 0.8543 (threshold: 0.5)
  Faithfulness: 0.9123 (threshold: 0.5)

✅ Acceptance test PASSED: Answer Relevancy meets threshold
```

**Metrics Explained:**
- **Contextual Precision**: Reranker quality and ranking order
- **Contextual Recall**: Embedding model accuracy
- **Contextual Relevancy**: Chunk size and top-K parameter tuning
- **Answer Relevancy**: Prompt template quality
- **Faithfulness**: LLM output quality (no hallucinations)

#### Task 3.3: Planning Agent Tests

```bash
# Run planning agent evaluation tests
pytest tests/test_planning_agent_deepeval.py -v
```

#### Task 3.4: Self-Healing Code Assistant Tests

```bash
# Run coding agent tests
pytest tests/test_coding_agent.py -v

# Run retry logic tests
pytest tests/test_retry_logic.py -v
```

### Test Coverage

View test coverage report:
```bash
pytest --cov=src/datacom_ai --cov-report=html
open htmlcov/index.html
```

## Task-Specific Documentation

### Task 3.1: Conversational Core

**Features:**
- ✅ Token-level streaming responses
- ✅ Message persistence (last 10 messages)
- ✅ Telemetry: prompt tokens, completion tokens, cost (USD), latency
- ✅ Real-time metrics display

**Files:**
- `src/datacom_ai/chat/` - Chat system implementation
- `src/datacom_ai/telemetry/` - Telemetry and metrics
- `scripts/acceptance_test_streaming_telemetry.py` - Acceptance test

### Task 3.2: High-Performance RAG

**Features:**
- ✅ Document ingestion (50+ MB supported)
- ✅ Chunking and embedding (FastEmbed or Azure OpenAI)
- ✅ Vector storage (Qdrant)
- ✅ QA with inline citations
- ✅ Sub-300ms median retrieval time
- ✅ Automated evaluation script (top-5 retrieval accuracy)

**Setup Steps:**

1. **Download Dataset:**
   ```bash
   # Download 50MB+ Project Gutenberg corpus to data/raw/
   python scripts/download_corpus.py
   ```
   
   This script downloads a curated collection of public-domain literature (50+ MB) from Project Gutenberg, including works by:
   - Charles Dickens, Mark Twain, Jane Austen, Arthur Conan Doyle
   - Leo Tolstoy, Victor Hugo, Fyodor Dostoevsky, and more
   
   The corpus is automatically cleaned (headers removed) and saved to `data/raw/`.

2. **Index Documents:**
   ```bash
   # Index documents from data/raw/ into Qdrant
   python scripts/index_data.py
   ```
   
   Alternatively, specify a custom directory:
   ```bash
   python scripts/index_data.py --data-dir /path/to/your/documents
   ```

2. **Generate Evaluation Dataset:**
   ```bash
   # Option 1: Fast generation (questions only)
   uv run python scripts/generate_dataset_basellm.py
   
   # Option 2: Comprehensive (multi-hop questions)
   uv run python scripts/generate_dataset_ragas.py
   ```

3. **Run Evaluation:**
   ```bash
   uv run python scripts/acceptance_test_rag_retrieval.py
   ```

**Configuration:**
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `RETRIEVER_K`: Number of documents to retrieve (default: 5)
- `EMBEDDING_PROVIDER`: `fastembed` or `azure_openai`

**Documentation:**
- `RAG_USER_GUIDE.md` - Detailed RAG setup guide
- `scripts/README_EVALUATION.md` - Evaluation guide
- `scripts/README_DATASET_GENERATION.md` - Dataset generation options

### Task 3.3: Autonomous Planning Agent

**Features:**
- ✅ Multi-tool agent (flights, weather, attractions)
- ✅ Scratch-pad reasoning (visible in logs)
- ✅ Constraint handling (budget, dates)
- ✅ JSON schema output (itinerary)

**Setup Steps:**

#### 1. AMADEUS API Setup (Flight Search)

The planning agent uses AMADEUS API for real-time flight search. To enable this:

1. **Create AMADEUS Account:**
   - Go to [https://developers.amadeus.com/](https://developers.amadeus.com/)
   - Sign up for a free account
   - Navigate to "My Self-Service" → "Create New App"

2. **Get API Credentials:**
   - Create a new app (choose "Test" environment for development)
   - Copy your `Client ID` and `Client Secret`
   - Note: The test environment has rate limits but is free

3. **Add to `.env` file:**
   ```bash
   AMADEUS_CLIENT_ID=your-amadeus-client-id
   AMADEUS_CLIENT_SECRET=your-amadeus-client-secret
   ```

4. **Verify Installation:**
   ```bash
   # The 'amadeus' package should already be installed via pyproject.toml
   # If not, install it:
   pip install amadeus
   ```

**Note:** If AMADEUS credentials are not provided, the agent will use a mock flight search tool.

#### 2. Google Places API Setup (Attractions)

The planning agent uses Google Places API for location-based attraction recommendations:

1. **Create Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable billing (required for Places API, but free tier available)

2. **Enable Places API:**
   - Navigate to "APIs & Services" → "Library"
   - Search for "Places API" (Legacy)
   - **Important:** Enable "Places API" (Legacy), NOT "Places API (New)"
   - Click "Enable"

3. **Create API Key:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the generated API key
   - (Optional) Restrict the key to "Places API" for security

4. **Add to `.env` file:**
   ```bash
   GPLACES_API_KEY=your-google-places-api-key
   ```

5. **Verify Installation:**
   ```bash
   # The 'langchain-google-community' package should already be installed via pyproject.toml
   # If not, install it:
   pip install langchain-google-community
   ```

**Note:** If Google Places API key is not provided, the agent will use a mock attractions tool.

#### 3. Complete Setup

After configuring both APIs, your `.env` file should include:

```bash
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=Gpt4o
OPENAI_API_KEY=your-azure-key

# AMADEUS (optional - for real flight search)
AMADEUS_CLIENT_ID=your-amadeus-client-id
AMADEUS_CLIENT_SECRET=your-amadeus-client-secret

# Google Places (optional - for real attractions)
GPLACES_API_KEY=your-google-places-api-key
```

**Usage:**
1. Start chat application: `python chat.py`
2. Select "Planning Agent" tab
3. Enter prompt: "Plan a 2-day trip to Auckland for under NZ$500"
4. Agent will call tools and generate itinerary

**Tools:**
- **Flight Search**: AMADEUS API (real) or mock tool
- **Weather**: Mock tool (can be extended with real API)
- **Attractions**: Google Places API (real) or mock tool

**Files:**
- `src/datacom_ai/agent/planning_agent.py` - Planning agent implementation
- `src/datacom_ai/agent/tools/` - Tool implementations
- `tests/test_planning_agent_deepeval.py` - Evaluation tests
- `README_TRAVEL_AGENT.md` - Additional travel agent documentation

### Task 3.4: Self-Healing Code Assistant

**Features:**
- ✅ Natural language coding task input
- ✅ Code generation via LLM
- ✅ Automatic test execution (pytest/cargo test)
- ✅ Error feedback and retry (up to 3 attempts)
- ✅ Progress streaming

**Usage:**

**Via UI:**
1. Start chat application: `python chat.py`
2. Select "Coding Agent" tab
3. Enter task: "write quicksort in Rust"
4. Watch progress and final result

**Via CLI:**
```bash
python -m datacom_ai.self_heal.cli "write quicksort in Rust"
```

**Files:**
- `src/datacom_ai/self_heal/agent.py` - Self-healing agent
- `src/datacom_ai/self_heal/cli.py` - CLI interface
- `tests/test_coding_agent.py` - Evaluation tests

## Docker & Containerization

### Start Services

```bash
# Start Qdrant vector database
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f qdrant

# Stop services
docker-compose down
```

### Docker Compose Services

- **Qdrant**: Vector database (ports 6333, 6334)
  - Dashboard: http://localhost:6333/dashboard
  - Storage: `./qdrant_storage/` (persisted locally)

## Analytics Dashboard

### Start Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard.py
```

The dashboard will launch at `http://localhost:8501`

### Dashboard Features

1. **Latency & Cost Metrics**
   - Average latency over time
   - Total cost over time
   - Request counts
   - Average cost per request

2. **Retrieval Accuracy**
   - Contextual Precision, Recall, Relevancy metrics
   - Time series visualization
   - Summary statistics

3. **Agent Performance**
   - Success/failure breakdown
   - Success rate over time
   - Failure type analysis
   - Execution time metrics

### Data Source

The dashboard reads from `logs/chat.jsonl` (JSONL format). Ensure logging is enabled:

```bash
# In .env file
LOG_FORMAT=both  # or "json"
LOG_JSON_FILE=logs/chat.jsonl
```

## Troubleshooting

### Common Issues

#### 1. "Configuration error: AZURE_OPENAI_ENDPOINT is not set"
- **Solution**: Create `.env` file with required variables (see [Installation & Setup](#installation--setup))

#### 2. "Qdrant connection error"
- **Solution**: 
  ```bash
  docker-compose up -d
  # Verify: curl http://localhost:6333/health
  ```

#### 3. "No documents retrieved" (RAG)
- **Solution**: 
  ```bash
  # Download dataset first (if not already done)
  python scripts/download_corpus.py
  
  # Then index documents
  python scripts/index_data.py
  ```

#### 4. "Dataset not found" (Evaluation)
- **Solution**: 
  ```bash
  # Generate evaluation dataset
  uv run python scripts/generate_dataset_ragas.py
  ```

#### 5. "Tokenizers parallelism deadlock"
- **Solution**: Already handled in code (`TOKENIZERS_PARALLELISM=false`)

#### 6. "Import errors" or "Module not found"
- **Solution**: 
  ```bash
  # Reinstall dependencies
  uv sync
  # Or
  pip install -e ".[ui]"
  ```

#### 7. "Amadeus Error (401)" or "Failed to initialize Amadeus toolkit"
- **Solution**: 
  - Ensure `AMADEUS_CLIENT_ID` and `AMADEUS_CLIENT_SECRET` are set in `.env`
  - Verify credentials are correct (not your OpenAI key)
  - Check that you're using test environment credentials (not production)
  - If issues persist, the agent will fall back to mock flight search

#### 8. "Google Places Error (Request Denied)" or "Failed to initialize Google Places tool"
- **Solution**: 
  - Ensure `GPLACES_API_KEY` is set in `.env`
  - Verify "Places API" (Legacy) is enabled in Google Cloud Console (not "Places API New")
  - Check API key restrictions (should allow Places API)
  - Verify billing is enabled (required even for free tier)
  - If issues persist, the agent will fall back to mock attractions tool

### Logging

Logs are written to:
- `logs/chat.log` - Text format logs
- `logs/chat.jsonl` - JSON format logs (for dashboard)

Configure logging in `.env`:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=both  # json, text, both
LOG_FILE=logs/chat.log
LOG_JSON_FILE=logs/chat.jsonl
```

### Performance Tuning

**RAG Retrieval Speed:**
- Reduce `RETRIEVER_K` (fewer documents retrieved)
- Use `fastembed` instead of `azure_openai` (faster embeddings)
- Optimize chunk size (`CHUNK_SIZE`)

**LLM Response Speed:**
- Use faster model (if available)
- Reduce `LLM_TEMPERATURE` (may improve speed slightly)

**Evaluation Speed:**
- Use `generate_dataset_basellm.py` instead of `generate_dataset_ragas.py`
- Set `GENERATE_ANSWERS=False` in dataset generation
- Reduce dataset size

## Project Structure

```
tech-assesment/
├── src/datacom_ai/          # Main source code
│   ├── agent/               # Planning agent
│   ├── chat/                # Chat system
│   ├── clients/             # LLM client
│   ├── config/              # Configuration
│   ├── rag/                 # RAG pipeline
│   ├── self_heal/           # Self-healing code assistant
│   ├── telemetry/           # Metrics and logging
│   └── ui/                  # Gradio UI
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
│   ├── index_data.py        # Document indexing
│   ├── evaluate_retrieval.py # RAG evaluation
│   ├── generate_dataset_*.py # Dataset generation
│   └── acceptance_test_*.py # Acceptance tests
├── data/                    # Data directory
│   ├── raw/                 # Raw documents
│   └── processed/           # Processed datasets
├── logs/                    # Log files
├── dashboard.py             # Streamlit dashboard
├── chat.py                  # Main chat application
├── docker-compose.yml       # Docker services
└── pyproject.toml          # Project dependencies
```

## License

This project is part of a technical assessment.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [DeepEval RAG Evaluation](https://deepeval.com/guides/guides-rag-evaluation)
- [Ragas Framework](https://docs.ragas.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Gradio Documentation](https://www.gradio.app/docs/)
