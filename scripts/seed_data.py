"""Sample data seeder for development and demonstration."""

import os  # Environment setup
import sys  # Path manipulation
from pathlib import Path  # Path handling

# Add src to path for imports
project_root = Path(__file__).resolve().parent.parent  # Project root
sys.path.insert(0, str(project_root / "src"))  # Add src to path

os.environ.setdefault("APP_ENV", "dev")  # Use dev environment


def seed_sample_data() -> None:
    """Seed the database with sample memory nodes and edges for testing."""
    from memory_graph.config.settings import get_settings  # Settings
    from memory_graph.models.memory_edge import MemoryEdge  # Edge model
    from memory_graph.models.memory_node import MemoryNode  # Node model
    from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums
    from memory_graph.persistence.database import DatabaseManager  # Database
    from memory_graph.persistence.edge_repository import EdgeRepository  # Edge CRUD
    from memory_graph.persistence.node_repository import NodeRepository  # Node CRUD

    settings = get_settings()  # Load configuration
    db_manager = DatabaseManager(settings.database)  # Create DB connection
    db_manager.initialize_schema()  # Ensure tables exist

    node_repo = NodeRepository(db_manager)  # Node repository
    edge_repo = EdgeRepository(db_manager)  # Edge repository

    # Create sample memory nodes
    nodes = [
        MemoryNode(
            id="seed-001",
            node_type=NodeType.PREFERENCE,
            content="User strongly prefers Python for backend development due to its readability",
            summary="Prefers Python for backend",
            confidence=0.9,
            emotional_valence=EmotionValence.POSITIVE,
            intensity=0.8,
            tags=["programming", "python", "backend"],
        ),
        MemoryNode(
            id="seed-002",
            node_type=NodeType.AVERSION,
            content="User dislikes verbose boilerplate code and prefers concise solutions",
            summary="Dislikes verbose boilerplate",
            confidence=0.85,
            emotional_valence=EmotionValence.NEGATIVE,
            intensity=0.7,
            tags=["programming", "code-style"],
        ),
        MemoryNode(
            id="seed-003",
            node_type=NodeType.BELIEF,
            content="User believes automated testing is essential for production code quality",
            summary="Believes in automated testing",
            confidence=0.8,
            emotional_valence=EmotionValence.POSITIVE,
            intensity=0.6,
            tags=["testing", "quality"],
        ),
        MemoryNode(
            id="seed-004",
            node_type=NodeType.GOAL,
            content="User wants to learn Rust for systems programming this year",
            summary="Goal: Learn Rust this year",
            confidence=0.7,
            emotional_valence=EmotionValence.POSITIVE,
            intensity=0.5,
            tags=["learning", "rust", "goals"],
        ),
        MemoryNode(
            id="seed-005",
            node_type=NodeType.EXPERIENCE,
            content="User had a bad experience with MongoDB scaling issues in a previous project",
            summary="Bad MongoDB scaling experience",
            confidence=0.75,
            emotional_valence=EmotionValence.NEGATIVE,
            intensity=0.6,
            tags=["databases", "mongodb", "scaling"],
        ),
    ]

    # Create sample edges
    edges = [
        MemoryEdge(
            id="seed-edge-001",
            source_node_id="seed-001",
            target_node_id="seed-002",
            edge_type=EdgeType.SUPPORTS,
            weight=0.7,
            description="Python preference aligns with dislike of boilerplate",
        ),
        MemoryEdge(
            id="seed-edge-002",
            source_node_id="seed-003",
            target_node_id="seed-001",
            edge_type=EdgeType.RELATED_TO,
            weight=0.5,
            description="Testing belief relates to Python preference",
        ),
        MemoryEdge(
            id="seed-edge-003",
            source_node_id="seed-004",
            target_node_id="seed-001",
            edge_type=EdgeType.EVOLVED_FROM,
            weight=0.4,
            description="Rust goal evolved from Python experience",
        ),
    ]

    # Persist nodes
    print("Seeding memory nodes...")
    for node in nodes:
        node_repo.create(node)
        print(f"  Created: {node.summary}")

    # Persist edges
    print("\nSeeding memory edges...")
    for edge in edges:
        edge_repo.create(edge)
        print(f"  Created: {edge.source_node_id} --[{edge.edge_type.value}]--> {edge.target_node_id}")

    db_manager.close()  # Close connection
    print("\nSeed data complete!")


if __name__ == "__main__":
    seed_sample_data()  # Run when executed directly
