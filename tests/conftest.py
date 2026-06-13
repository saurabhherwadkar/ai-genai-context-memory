"""Shared test fixtures for the memory graph test suite."""

import os  # Environment variable manipulation
from unittest.mock import MagicMock, patch  # Mocking utilities

import numpy as np  # Numerical arrays for test embeddings
import pytest  # Test framework

from memory_graph.config.settings import (  # Settings classes
    AppSettings,
    ContextSettings,
    DatabaseSettings,
    EmbeddingSettings,
    ExtractionSettings,
    GraphSettings,
    RankingWeights,
    reset_settings,
)
from memory_graph.embeddings.embedding_cache import EmbeddingCache  # Cache
from memory_graph.embeddings.embedding_service import EmbeddingService  # Embeddings
from memory_graph.embeddings.similarity_search import SimilaritySearch  # Search
from memory_graph.graph.graph_manager import GraphManager  # Graph lifecycle
from memory_graph.graph.graph_query import GraphQuery  # Graph traversal
from memory_graph.graph.graph_sync import GraphSync  # Sync layer
from memory_graph.models.memory_edge import MemoryEdge  # Edge model
from memory_graph.models.memory_node import MemoryNode  # Node model
from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums
from memory_graph.persistence.database import DatabaseManager  # Database
from memory_graph.persistence.edge_repository import EdgeRepository  # Edge CRUD
from memory_graph.persistence.embedding_repository import EmbeddingRepository  # Embedding storage
from memory_graph.persistence.node_repository import NodeRepository  # Node CRUD
from memory_graph.security.input_validator import InputValidator  # Validation


@pytest.fixture(autouse=True)
def reset_settings_fixture():
    """Reset settings singleton between tests to prevent state leakage."""
    reset_settings()  # Clear cached settings
    os.environ["APP_ENV"] = "test"  # Force test environment
    yield  # Run the test
    reset_settings()  # Clean up after test


@pytest.fixture
def db_settings() -> DatabaseSettings:
    """Create in-memory database settings for testing."""
    return DatabaseSettings(path=":memory:")  # Use SQLite in-memory


@pytest.fixture
def db_manager(db_settings: DatabaseSettings) -> DatabaseManager:
    """Create and initialize an in-memory database manager."""
    manager = DatabaseManager(db_settings)  # Create manager
    manager.initialize_schema()  # Create tables
    return manager  # Return initialized manager


@pytest.fixture
def node_repository(db_manager: DatabaseManager) -> NodeRepository:
    """Create a node repository backed by in-memory database."""
    return NodeRepository(db_manager)  # Return repository


@pytest.fixture
def edge_repository(db_manager: DatabaseManager) -> EdgeRepository:
    """Create an edge repository backed by in-memory database."""
    return EdgeRepository(db_manager)  # Return repository


@pytest.fixture
def embedding_repository(db_manager: DatabaseManager) -> EmbeddingRepository:
    """Create an embedding repository backed by in-memory database."""
    return EmbeddingRepository(db_manager)  # Return repository


@pytest.fixture
def graph_settings() -> GraphSettings:
    """Create graph settings for testing with small limits."""
    return GraphSettings(max_nodes=100, max_edges_per_node=10)  # Small test limits


@pytest.fixture
def graph_manager(graph_settings: GraphSettings) -> GraphManager:
    """Create a fresh graph manager for testing."""
    return GraphManager(graph_settings)  # Return new manager


@pytest.fixture
def graph_query(graph_manager: GraphManager) -> GraphQuery:
    """Create a graph query service for testing."""
    return GraphQuery(graph_manager)  # Return query service


@pytest.fixture
def graph_sync(
    graph_manager: GraphManager,
    node_repository: NodeRepository,
    edge_repository: EdgeRepository,
) -> GraphSync:
    """Create a graph sync service for testing."""
    return GraphSync(graph_manager, node_repository, edge_repository)  # Return sync


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    """Create embedding settings for testing."""
    return EmbeddingSettings(
        model_name="all-MiniLM-L6-v2",
        batch_size=4,
        cache_size=50,
        similarity_threshold=0.92,
    )


@pytest.fixture
def embedding_cache(embedding_settings: EmbeddingSettings) -> EmbeddingCache:
    """Create an embedding cache for testing."""
    return EmbeddingCache(embedding_settings)  # Return cache


@pytest.fixture
def input_validator() -> InputValidator:
    """Create an input validator for testing."""
    from memory_graph.config.settings import InputSettings  # Import
    settings = InputSettings(max_content_length=50000, max_tags=20)  # Test settings
    return InputValidator(settings)  # Return validator


@pytest.fixture
def sample_node() -> MemoryNode:
    """Create a sample memory node for testing."""
    return MemoryNode(
        id="test-node-001",
        node_type=NodeType.PREFERENCE,
        content="User prefers Python over Java for backend development",
        summary="Prefers Python for backend",
        confidence=0.85,
        emotional_valence=EmotionValence.POSITIVE,
        intensity=0.7,
        tags=["programming", "languages"],
    )


@pytest.fixture
def sample_node_2() -> MemoryNode:
    """Create a second sample memory node for testing."""
    return MemoryNode(
        id="test-node-002",
        node_type=NodeType.BELIEF,
        content="User believes test-driven development leads to better code quality",
        summary="Believes TDD improves code quality",
        confidence=0.75,
        emotional_valence=EmotionValence.POSITIVE,
        intensity=0.6,
        tags=["programming", "testing"],
    )


@pytest.fixture
def sample_edge(sample_node: MemoryNode, sample_node_2: MemoryNode) -> MemoryEdge:
    """Create a sample memory edge connecting the two sample nodes."""
    return MemoryEdge(
        id="test-edge-001",
        source_node_id=sample_node.id,
        target_node_id=sample_node_2.id,
        edge_type=EdgeType.SUPPORTS,
        weight=0.7,
        description="Python preference supports TDD belief",
    )


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Create a sample embedding vector for testing."""
    rng = np.random.default_rng(42)  # Deterministic random generator
    embedding = rng.random(384).astype(np.float32)  # 384-dim vector (MiniLM size)
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
    return embedding  # Return normalized vector
