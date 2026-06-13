# AI GenAI Context Memory

A human brain-like memory graph system that acts as a transparent LLM proxy to build, store, and inject cognitive memories (thoughts, beliefs, preferences, aversions, emotions, goals, experiences) into LLM conversations. The system intercepts conversations, automatically extracts memory nodes using LLM analysis, and injects relevant memories into subsequent interactions via semantic similarity search and graph traversal.

## Architecture

```
Client App  -->  Proxy (FastAPI)  -->  LLM Provider (OpenAI/Anthropic/any)
                     |                         |
                     | inject memories         | response
                     | from graph              |
                     v                         v
              Context Builder <---- Memory Graph (NetworkX + SQLite)
                                          ^
                                          | extract (async)
                                   Extraction Pipeline
                                          ^
                                          |
                                   Conversation logs
```

**Core Flow:**
1. Client sends a request through the proxy
2. Proxy retrieves relevant memories via semantic search + graph traversal
3. Memories are injected into the system prompt before forwarding
4. Response is returned to the client immediately
5. Conversation is queued for async background memory extraction
6. Extraction LLM analyzes the conversation and creates new memory nodes/edges

## Project Structure

```
ai-genai-context-memory/
├── config/                         # Environment-specific YAML configuration
│   ├── application.yaml            # Default settings (all configurable values)
│   ├── application-dev.yaml        # Development overrides
│   ├── application-prod.yaml       # Production overrides
│   └── application-test.yaml       # Test overrides
├── src/memory_graph/               # Main application source code
│   ├── main.py                     # FastAPI app factory + lifespan events
│   ├── config/                     # Settings and logging configuration
│   │   ├── settings.py             # Pydantic Settings with YAML loader
│   │   └── logging_config.py      # Structured logging (structlog)
│   ├── models/                     # Pydantic data models and enums
│   │   ├── memory_types.py        # NodeType, EdgeType, EmotionValence enums
│   │   ├── memory_node.py         # MemoryNode model
│   │   ├── memory_edge.py         # MemoryEdge model
│   │   ├── api_requests.py        # Request DTOs
│   │   ├── api_responses.py       # Response DTOs
│   │   └── proxy_models.py        # LLM proxy data models
│   ├── persistence/                # SQLite database layer
│   │   ├── database.py            # Connection manager + schema
│   │   ├── node_repository.py     # Node CRUD operations
│   │   ├── edge_repository.py     # Edge CRUD operations
│   │   └── embedding_repository.py # Vector blob storage
│   ├── graph/                      # NetworkX in-memory graph
│   │   ├── graph_manager.py       # Graph lifecycle + mutations
│   │   ├── graph_query.py         # BFS traversal + filtering
│   │   └── graph_sync.py          # NetworkX <-> SQLite sync
│   ├── embeddings/                 # Vector similarity search
│   │   ├── embedding_service.py   # sentence-transformers wrapper
│   │   ├── similarity_search.py   # Cosine similarity top-k
│   │   └── embedding_cache.py     # LRU cache for embeddings
│   ├── extraction/                 # Async memory extraction pipeline
│   │   ├── extraction_service.py  # Extraction orchestrator
│   │   ├── extraction_prompts.py  # LLM prompts for extraction
│   │   ├── extraction_parser.py   # Structured output parser
│   │   └── extraction_queue.py    # asyncio background queue
│   ├── context/                    # Context building for injection
│   │   ├── context_builder.py     # Main orchestrator
│   │   ├── context_ranker.py      # Multi-signal scoring
│   │   ├── context_formatter.py   # Markdown/XML/plain formatting
│   │   └── token_budget.py        # tiktoken-based budget
│   ├── proxy/                      # LLM proxy layer
│   │   ├── proxy_router.py        # FastAPI proxy endpoints
│   │   ├── proxy_service.py       # Proxy orchestration
│   │   ├── provider_registry.py   # Provider adapter registry
│   │   ├── interceptor.py         # Pre/post processing hooks
│   │   └── providers/             # LLM provider adapters
│   │       ├── base_provider.py   # Abstract interface
│   │       ├── openai_provider.py # OpenAI adapter
│   │       ├── anthropic_provider.py # Anthropic adapter
│   │       └── generic_provider.py   # Generic/custom adapter
│   ├── api/                        # REST API endpoints
│   │   ├── memory_router.py      # Memory CRUD
│   │   ├── graph_router.py       # Graph queries
│   │   ├── context_router.py     # Context retrieval
│   │   └── health_router.py      # Health checks
│   ├── security/                   # Security components
│   │   ├── rate_limiter.py       # Request throttling
│   │   ├── input_validator.py    # Input sanitization
│   │   └── secrets_manager.py    # API key management
│   └── middleware/                 # Request middleware
│       ├── error_handler.py      # Global exception handling
│       └── request_logging.py    # Request/response logging
├── tests/                          # Test suite
│   ├── conftest.py               # Shared fixtures
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── scripts/                        # Utility scripts
│   ├── run_dev.py                # Dev server launcher
│   └── seed_data.py             # Sample data seeder
├── pyproject.toml                  # Project metadata + dependencies
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.115.0 | REST API framework |
| uvicorn | >=0.34.0 | ASGI server |
| pydantic | >=2.10.0 | Data validation |
| pydantic-settings | >=2.7.0 | Configuration management |
| networkx | >=3.4.0 | In-memory graph operations |
| sentence-transformers | >=3.4.0 | Text embeddings |
| numpy | >=2.2.0 | Vector operations |
| httpx | >=0.28.0 | Async HTTP client |
| pyyaml | >=6.0.2 | YAML config parsing |
| tiktoken | >=0.9.0 | Token counting |
| slowapi | >=0.1.9 | Rate limiting |
| structlog | >=24.4.0 | Structured logging |

**Dev dependencies:** pytest, pytest-asyncio, pytest-cov, ruff, mypy

## Deployment

### Prerequisites

- Python 3.12 or higher
- pip (package installer)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-genai-context-memory

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

1. Copy and customize the configuration:
   ```bash
   # All settings are in config/application.yaml (defaults)
   # Override per environment in config/application-{env}.yaml
   ```

2. Set required environment variables:
   ```bash
   export APP_ENV=dev                    # Environment: dev, prod, test
   export OPENAI_API_KEY=sk-...         # OpenAI API key (for proxy/extraction)
   export ANTHROPIC_API_KEY=sk-ant-...  # Anthropic API key (optional)
   ```

3. Key configuration options (in `config/application.yaml`):
   - `server.port` - HTTP port (default: 8000)
   - `database.path` - SQLite file path
   - `embeddings.model_name` - Sentence transformer model
   - `extraction.enabled` - Toggle memory extraction
   - `extraction.provider` - LLM for extraction (openai/anthropic)
   - `context.max_token_budget` - Max tokens for context injection
   - `logging.level` - Log level (DEBUG/INFO/WARNING/ERROR)

### Running

```bash
# Development (with hot reload)
python scripts/run_dev.py

# Production
uvicorn memory_graph.main:app --host 0.0.0.0 --port 8000 --workers 4

# Seed sample data
python scripts/seed_data.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=memory_graph --cov-report=html

# Run only unit tests
pytest tests/unit/

# Lint check
ruff check src/ tests/

# Type check
mypy src/
```

## API Usage

### Proxy (transparent LLM forwarding with memory injection)

```bash
# OpenAI-compatible proxy
curl -X POST http://localhost:8000/v1/proxy/openai/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What programming language should I use?"}]
  }'

# Disable memory injection for a request
curl -X POST http://localhost:8000/v1/proxy/openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Memory-Context: disabled" \
  -d '{"model": "gpt-4o", "messages": [...]}'
```

### Memory CRUD

```bash
# Create a memory
curl -X POST http://localhost:8000/api/v1/memories/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "node_type": "preference",
    "content": "User prefers dark mode in all applications",
    "summary": "Prefers dark mode",
    "confidence": 0.9,
    "tags": ["ui", "preferences"]
  }'

# List memories
curl http://localhost:8000/api/v1/memories/nodes?page=0&page_size=20

# Get context for a query
curl -X POST http://localhost:8000/api/v1/context/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What UI theme should I recommend?"}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Memory Types

| Type | Description |
|------|-------------|
| thought | General thoughts or observations |
| belief | Held beliefs or convictions |
| aversion | Things the person dislikes or avoids |
| preference | Things the person prefers or likes |
| emotion | Emotional states or feelings |
| goal | Desired outcomes or aspirations |
| experience | Past events or lived experiences |
| fact | Factual information about the person |
| habit | Recurring behavioral patterns |
| value | Core values or principles |
| relationship | Information about relationships |
| skill | Known skills or competencies |
| custom | User-defined types |

## Edge Types (Relationships)

contradicts, supports, caused_by, leads_to, related_to, evolved_from, conflicts_with, depends_on, temporal_before, temporal_after, part_of, generalizes, specifies

## License

MIT
