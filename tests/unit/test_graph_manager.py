"""Unit tests for the GraphManager service."""

import pytest  # Test framework

from memory_graph.graph.graph_manager import GraphManager  # Manager under test
from memory_graph.models.memory_edge import MemoryEdge  # Edge model
from memory_graph.models.memory_node import MemoryNode  # Node model
from memory_graph.models.memory_types import EdgeType, NodeType  # Enums


class TestGraphManagerNodes:
    """Tests for graph node operations."""

    @pytest.mark.asyncio
    async def test_add_node(self, graph_manager: GraphManager, sample_node: MemoryNode):
        """Test adding a node to the graph."""
        result = await graph_manager.add_node(sample_node)  # Add node
        assert result is True  # Addition succeeded
        assert graph_manager.node_count == 1  # One node in graph
        assert graph_manager.has_node(sample_node.id)  # Node exists

    @pytest.mark.asyncio
    async def test_add_node_respects_capacity(self, graph_manager: GraphManager):
        """Test that node addition is rejected when capacity is reached."""
        # Fill to capacity (max_nodes=100 in test fixture)
        for i in range(100):
            node = MemoryNode(
                id=f"node-{i}",
                node_type=NodeType.THOUGHT,
                content=f"Thought {i}",
                summary=f"T{i}",
            )
            await graph_manager.add_node(node)

        # Next addition should fail
        overflow_node = MemoryNode(
            id="overflow",
            node_type=NodeType.THOUGHT,
            content="Overflow",
            summary="Over",
        )
        result = await graph_manager.add_node(overflow_node)
        assert result is False  # Rejected due to capacity

    @pytest.mark.asyncio
    async def test_remove_node(self, graph_manager: GraphManager, sample_node: MemoryNode):
        """Test removing a node from the graph."""
        await graph_manager.add_node(sample_node)  # Add first
        result = await graph_manager.remove_node(sample_node.id)  # Remove

        assert result is True  # Removal succeeded
        assert graph_manager.node_count == 0  # Graph empty
        assert not graph_manager.has_node(sample_node.id)  # Node gone

    @pytest.mark.asyncio
    async def test_remove_nonexistent_node(self, graph_manager: GraphManager):
        """Test removing a non-existent node returns False."""
        result = await graph_manager.remove_node("nonexistent")  # Remove missing
        assert result is False  # Nothing to remove

    @pytest.mark.asyncio
    async def test_get_node_attributes(self, graph_manager: GraphManager, sample_node: MemoryNode):
        """Test retrieving node attributes from the graph."""
        await graph_manager.add_node(sample_node)  # Add node
        attrs = graph_manager.get_node_attributes(sample_node.id)  # Get attributes

        assert attrs is not None  # Attributes found
        assert attrs["node_type"] == "preference"  # Type stored
        assert attrs["confidence"] == 0.85  # Confidence stored


class TestGraphManagerEdges:
    """Tests for graph edge operations."""

    @pytest.mark.asyncio
    async def test_add_edge(
        self,
        graph_manager: GraphManager,
        sample_node: MemoryNode,
        sample_node_2: MemoryNode,
        sample_edge: MemoryEdge,
    ):
        """Test adding an edge between two nodes."""
        await graph_manager.add_node(sample_node)  # Add source
        await graph_manager.add_node(sample_node_2)  # Add target
        result = await graph_manager.add_edge(sample_edge)  # Add edge

        assert result is True  # Edge added
        assert graph_manager.edge_count == 1  # One edge in graph

    @pytest.mark.asyncio
    async def test_add_edge_missing_source(
        self,
        graph_manager: GraphManager,
        sample_node_2: MemoryNode,
        sample_edge: MemoryEdge,
    ):
        """Test that adding an edge with missing source fails."""
        await graph_manager.add_node(sample_node_2)  # Only add target
        result = await graph_manager.add_edge(sample_edge)  # Try adding edge

        assert result is False  # Rejected - source missing

    @pytest.mark.asyncio
    async def test_get_neighbors(
        self,
        graph_manager: GraphManager,
        sample_node: MemoryNode,
        sample_node_2: MemoryNode,
        sample_edge: MemoryEdge,
    ):
        """Test getting successor neighbors of a node."""
        await graph_manager.add_node(sample_node)
        await graph_manager.add_node(sample_node_2)
        await graph_manager.add_edge(sample_edge)

        neighbors = graph_manager.get_neighbors(sample_node.id)  # Get successors
        assert sample_node_2.id in neighbors  # Target is a neighbor

    @pytest.mark.asyncio
    async def test_get_predecessors(
        self,
        graph_manager: GraphManager,
        sample_node: MemoryNode,
        sample_node_2: MemoryNode,
        sample_edge: MemoryEdge,
    ):
        """Test getting predecessor nodes."""
        await graph_manager.add_node(sample_node)
        await graph_manager.add_node(sample_node_2)
        await graph_manager.add_edge(sample_edge)

        predecessors = graph_manager.get_predecessors(sample_node_2.id)  # Get predecessors
        assert sample_node.id in predecessors  # Source is a predecessor

    @pytest.mark.asyncio
    async def test_clear_graph(
        self,
        graph_manager: GraphManager,
        sample_node: MemoryNode,
    ):
        """Test clearing all nodes and edges from the graph."""
        await graph_manager.add_node(sample_node)  # Add a node
        await graph_manager.clear()  # Clear everything

        assert graph_manager.node_count == 0  # Empty
        assert graph_manager.edge_count == 0  # Empty
