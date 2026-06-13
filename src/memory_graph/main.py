"""FastAPI application factory and lifespan management."""

from contextlib import asynccontextmanager  # Async context manager for lifespan
from collections.abc import AsyncGenerator  # Async generator type

import structlog  # Structured logging
from fastapi import FastAPI  # Web application framework

from memory_graph.api.context_router import create_context_router  # Context endpoints
from memory_graph.api.graph_router import create_graph_router  # Graph endpoints
from memory_graph.api.health_router import create_health_router  # Health endpoints
from memory_graph.api.memory_router import create_memory_router  # Memory CRUD endpoints
from memory_graph.config.logging_config import configure_logging  # Logging setup
from memory_graph.config.settings import get_settings  # Application settings
from memory_graph.context.context_builder import ContextBuilder  # Context orchestrator
from memory_graph.context.context_formatter import ContextFormatter  # Text formatting
from memory_graph.context.context_ranker import ContextRanker  # Memory scoring
from memory_graph.context.token_budget import TokenBudget  # Token management
from memory_graph.embeddings.embedding_cache import EmbeddingCache  # Vector cache
from memory_graph.embeddings.embedding_service import EmbeddingService  # Embeddings
from memory_graph.embeddings.similarity_search import SimilaritySearch  # Semantic search
from memory_graph.extraction.extraction_parser import ExtractionParser  # Output parser
from memory_graph.extraction.extraction_queue import ExtractionQueue  # Background queue
from memory_graph.extraction.extraction_service import ExtractionService  # Extraction logic
from memory_graph.graph.graph_manager import GraphManager  # Graph lifecycle
from memory_graph.graph.graph_query import GraphQuery  # Graph traversal
from memory_graph.graph.graph_sync import GraphSync  # Sync layer
from memory_graph.middleware.error_handler import register_error_handlers  # Error handling
from memory_graph.middleware.request_logging import RequestLoggingMiddleware  # Request logs
from memory_graph.persistence.database import DatabaseManager  # SQLite management
from memory_graph.persistence.edge_repository import EdgeRepository  # Edge CRUD
from memory_graph.persistence.embedding_repository import EmbeddingRepository  # Embedding storage
from memory_graph.persistence.node_repository import NodeRepository  # Node CRUD
from memory_graph.proxy.interceptor import ProxyInterceptor  # Pre/post hooks
from memory_graph.proxy.provider_registry import ProviderRegistry  # Provider lookup
from memory_graph.proxy.proxy_router import create_proxy_router  # Proxy endpoints
from memory_graph.proxy.proxy_service import ProxyService  # Proxy logic
from memory_graph.security.input_validator import InputValidator  # Input sanitization
from memory_graph.security.rate_limiter import create_rate_limiter, register_rate_limiter  # Throttle
from memory_graph.security.secrets_manager import SecretsManager  # Secrets access

# Module logger (configured during startup)
logger = structlog.get_logger(__name__)  # Named logger for this module


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.

    Initializes all services on startup and cleans up on shutdown.
    """
    # === STARTUP ===
    settings = get_settings()  # Load application settings

    # Configure structured logging
    configure_logging(settings.logging)  # Apply log settings
    logger.info("application_starting", version="0.1.0")  # Log startup

    # Initialize persistence layer
    db_manager = DatabaseManager(settings.database)  # Create DB manager
    db_manager.initialize_schema()  # Create tables if needed

    # Initialize repositories
    node_repo = NodeRepository(db_manager)  # Node CRUD
    edge_repo = EdgeRepository(db_manager)  # Edge CRUD
    embedding_repo = EmbeddingRepository(db_manager)  # Embedding storage

    # Initialize graph layer
    graph_manager = GraphManager(settings.graph)  # Create graph manager
    graph_query = GraphQuery(graph_manager)  # Graph traversal service
    graph_sync = GraphSync(graph_manager, node_repo, edge_repo)  # Sync service

    # Load the full graph from persistence
    await graph_sync.load_full_graph()  # Hydrate NetworkX from SQLite

    # Initialize embedding layer
    embedding_service = EmbeddingService(settings.embeddings)  # Sentence transformers
    embedding_cache = EmbeddingCache(settings.embeddings)  # LRU cache
    similarity_search = SimilaritySearch(  # Search index
        embedding_service, embedding_repo, embedding_cache
    )
    similarity_search.load_index()  # Load all embeddings into memory

    # Initialize context layer
    token_budget = TokenBudget(settings.context)  # Token counting
    context_ranker = ContextRanker(settings.context.ranking_weights, graph_query)  # Ranking
    context_formatter = ContextFormatter(settings.context.format)  # Formatting
    context_builder = ContextBuilder(  # Full context pipeline
        settings.context, similarity_search, graph_query,
        node_repo, context_ranker, token_budget, context_formatter
    )

    # Initialize security layer
    secrets_manager = SecretsManager()  # Secrets access
    input_validator = InputValidator(settings.security.input)  # Input validation

    # Initialize proxy layer
    provider_registry = ProviderRegistry(settings.proxy, secrets_manager)  # Providers
    interceptor = ProxyInterceptor(context_builder)  # Pre/post hooks

    # Initialize extraction pipeline
    extraction_queue = ExtractionQueue(settings.extraction)  # Background queue
    extraction_parser = ExtractionParser()  # Output parser
    extraction_service = ExtractionService(  # Extraction orchestrator
        settings.extraction, settings.embeddings, provider_registry,
        extraction_parser, embedding_service, embedding_repo,
        similarity_search, graph_sync
    )
    extraction_queue.set_processor(extraction_service.process_conversation)  # Wire callback

    # Initialize proxy service
    proxy_service = ProxyService(provider_registry, interceptor, extraction_queue)  # Proxy

    # Start extraction workers
    await extraction_queue.start_workers()  # Launch background workers

    # Register routers
    health_router = create_health_router(graph_manager)  # Health endpoints
    memory_router = create_memory_router(  # Memory CRUD endpoints
        node_repo, edge_repo, embedding_repo,
        embedding_service, similarity_search, graph_sync, input_validator
    )
    graph_router = create_graph_router(graph_query, node_repo, edge_repo, input_validator)  # Graph
    context_router = create_context_router(context_builder, input_validator)  # Context
    proxy_router = create_proxy_router(proxy_service)  # Proxy endpoints

    app.include_router(health_router)  # Mount health routes
    app.include_router(memory_router)  # Mount memory routes
    app.include_router(graph_router)  # Mount graph routes
    app.include_router(context_router)  # Mount context routes
    app.include_router(proxy_router)  # Mount proxy routes

    logger.info("application_started")  # Log successful startup

    yield  # Application is running

    # === SHUTDOWN ===
    logger.info("application_shutting_down")  # Log shutdown start
    await extraction_queue.stop_workers()  # Stop extraction workers
    db_manager.close()  # Close database connection
    logger.info("application_shutdown_complete")  # Log shutdown complete


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()  # Load settings for initial configuration

    # Create the FastAPI application
    app = FastAPI(  # Build app instance
        title="Memory Graph - Cognitive Memory for LLMs",  # API title
        description="Human brain-like memory graph system for LLM context injection",  # Description
        version="0.1.0",  # API version
        lifespan=lifespan,  # Async lifespan handler
    )

    # Register global error handlers
    register_error_handlers(app)  # Error handling middleware

    # Register rate limiter
    limiter = create_rate_limiter(settings.security.rate_limit)  # Create limiter
    register_rate_limiter(app, limiter)  # Register on app

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)  # Log all requests

    return app  # Return configured application


# Create the app instance for uvicorn
app = create_app()  # Module-level app instance
