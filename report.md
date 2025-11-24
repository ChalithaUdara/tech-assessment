# Technical Assessment Report: Design Decisions & Trade-offs

## Architecture Overview

Modular architecture with clear separation between chat orchestration, RAG pipelines, reasoning agents, and telemetry. Prioritizes observability, maintainability, and production-readiness while balancing performance and cost.

## Key Design Decisions

### 1. Modular Architecture with Factory Patterns

**Decision**: Separated concerns into distinct modules with factory-based dependency injection.

**Rationale**: Enables independent testing, easy extension (swapping vector stores/embeddings), and clear ownership. Facade pattern in `RAGPipeline` simplifies interactions.

**Trade-off**: Slight abstraction overhead for significant maintainability gains.

### 2. Telemetry-First Design

**Decision**: Built comprehensive telemetry (token usage, latency, cost) into chat engine base class with structured JSON logging.

**Rationale**: Directly addresses assessment requirement to "audit token usage, latency and error modes." Real-time metrics enable immediate monitoring.

**Trade-off**: ~1s overhead per request for critical observability. With streaming chat delay happens at the end, so this wouldn't effect initial response time.

### 3. Embedding Provider Flexibility

**Decision**: Used FastEmbed with BAAI/bge-small-en-v1.5 model for faster embeddings, Supports Azure OpenAI embeddings via factory pattern, easily extendible to support other embeddings.

**Rationale**: FastEmbed achieves sub-300ms retrieval with zero API costs and runs on CPU;

**Trade-off**: FastEmbed (BAAI/bge-small-en-v1.5) (384-dim) faster but potentially less accurate than Azure OpenAI (1536-dim). Chose speed/cost for assessment, extensibility for production.

### 4. Dual Evaluation Strategy

**Decision**: DeepEval metrics (Faithfulness, Answer Relevancy) per-query without ground truth, plus batch RAGAS evaluation for retrieval metrics.

**Rationale**: Continuous quality monitoring in production plus rigorous offline evaluation. Per-query metrics catch hallucinations immediately.

**Trade-off**: Adds ~200-500ms latency but provides actionable feedback. Configurable via `ENABLE_RAG_EVALUATION` flag.

### 5. Streaming Architecture

**Decision**: Token-level streaming throughout stack (LLM → Chat Engine → UI) with metrics computed post-stream.

**Rationale**: Essential for responsive UX. Post-completion metrics ensure accurate token counts without blocking streaming.

**Trade-off**: More complex state management for critical user experience.

### 6. LangGraph for Agent Orchestration

**Decision**: LangGraph for planning and self-healing agents with explicit state management.

**Rationale**: Structured reasoning loops, error recovery, observable execution traces. Scratch-pad reasoning visible in logs supports debugging.

**Trade-off**: Framework overhead vs. direct LLM calls, but enables complex multi-step workflows.

### 7. In-Memory Message Store

**Decision**: Simple in-memory store (last 10 messages) rather than persistent database.

**Rationale**: Sufficient for assessment, minimal dependencies, fast iteration. Easy to extend to Redis/PostgreSQL.

**Trade-off**: No persistence across restarts, but acceptable for MVP.

### 8. Local Qdrant Vector Store

**Decision**: Local Qdrant via Docker Compose with adapter pattern for extensibility.

**Rationale**: Zero cost, full control, sufficient for assessment. Adapter pattern enables cloud migration.

**Trade-off**: Not production-scalable, but demonstrates vector store understanding.

## Error Handling & Observability

Error modes captured via: structured logging with correlation IDs, exception handling at boundaries, retry logic (3 attempts with error feedback), and rate limit handling with exponential backoff.

## Development Approach

Development was conducted using AI-assisted coding with Antigravity, enabling rapid iteration.
