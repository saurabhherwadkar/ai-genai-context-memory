"""Unit tests for the MemoryEdge data model."""

import pytest  # Test framework

from memory_graph.models.memory_edge import MemoryEdge  # Model under test
from memory_graph.models.memory_types import EdgeType  # Edge type enum


class TestMemoryEdgeCreation:
    """Tests for MemoryEdge instantiation and defaults."""

    def test_create_minimal_edge(self):
        """Test creating an edge with only required fields."""
        edge = MemoryEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.RELATED_TO,
        )
        assert edge.id is not None  # UUID auto-generated
        assert edge.source_node_id == "node-001"  # Source set
        assert edge.target_node_id == "node-002"  # Target set
        assert edge.edge_type == EdgeType.RELATED_TO  # Type set
        assert edge.weight == 0.5  # Default weight
        assert edge.description is None  # Optional, defaults to None

    def test_create_full_edge(self):
        """Test creating an edge with all fields specified."""
        edge = MemoryEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.CONTRADICTS,
            weight=0.9,
            description="Direct contradiction between beliefs",
            metadata={"auto_detected": True},
        )
        assert edge.weight == 0.9  # Custom weight
        assert edge.description == "Direct contradiction between beliefs"  # Description set
        assert edge.metadata == {"auto_detected": True}  # Metadata set


class TestMemoryEdgeValidation:
    """Tests for MemoryEdge validation rules."""

    def test_self_loop_rejected(self):
        """Test that an edge cannot connect a node to itself."""
        with pytest.raises(ValueError, match="cannot connect a node to itself"):
            MemoryEdge(
                source_node_id="node-001",
                target_node_id="node-001",  # Same as source
                edge_type=EdgeType.RELATED_TO,
            )

    def test_weight_must_be_in_range(self):
        """Test that weight rejects values outside 0-1 range."""
        with pytest.raises(ValueError):
            MemoryEdge(
                source_node_id="node-001",
                target_node_id="node-002",
                edge_type=EdgeType.SUPPORTS,
                weight=1.5,  # Above maximum
            )

    def test_weight_minimum_boundary(self):
        """Test weight at minimum boundary."""
        edge = MemoryEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.SUPPORTS,
            weight=0.0,
        )
        assert edge.weight == 0.0  # Minimum accepted

    def test_source_must_not_be_empty(self):
        """Test that empty source_node_id is rejected."""
        with pytest.raises(ValueError):
            MemoryEdge(
                source_node_id="",  # Empty
                target_node_id="node-002",
                edge_type=EdgeType.RELATED_TO,
            )

    def test_target_must_not_be_empty(self):
        """Test that empty target_node_id is rejected."""
        with pytest.raises(ValueError):
            MemoryEdge(
                source_node_id="node-001",
                target_node_id="",  # Empty
                edge_type=EdgeType.RELATED_TO,
            )

    def test_all_edge_types_are_valid(self):
        """Test that all defined EdgeType values create valid edges."""
        for edge_type in EdgeType:
            edge = MemoryEdge(
                source_node_id="node-001",
                target_node_id="node-002",
                edge_type=edge_type,
            )
            assert edge.edge_type == edge_type  # Type preserved

    def test_timestamp_auto_generated(self):
        """Test that created_at is auto-populated."""
        edge = MemoryEdge(
            source_node_id="node-001",
            target_node_id="node-002",
            edge_type=EdgeType.RELATED_TO,
        )
        assert edge.created_at is not None  # Auto-generated
