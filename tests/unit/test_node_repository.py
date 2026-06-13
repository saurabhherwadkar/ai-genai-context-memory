"""Unit tests for the NodeRepository persistence layer."""

import pytest  # Test framework

from memory_graph.models.memory_node import MemoryNode  # Node model
from memory_graph.models.memory_types import EmotionValence, NodeType  # Enums
from memory_graph.persistence.node_repository import NodeRepository  # Repository under test


class TestNodeRepositoryCreate:
    """Tests for node creation operations."""

    def test_create_and_retrieve_node(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test creating a node and retrieving it by ID."""
        node_repository.create(sample_node)  # Persist the node
        retrieved = node_repository.get_by_id(sample_node.id)  # Fetch back

        assert retrieved is not None  # Found
        assert retrieved.id == sample_node.id  # Same ID
        assert retrieved.content == sample_node.content  # Same content
        assert retrieved.node_type == sample_node.node_type  # Same type
        assert retrieved.confidence == sample_node.confidence  # Same confidence

    def test_create_preserves_tags(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test that tags survive serialization/deserialization."""
        node_repository.create(sample_node)  # Persist
        retrieved = node_repository.get_by_id(sample_node.id)  # Fetch

        assert retrieved.tags == ["programming", "languages"]  # Tags preserved

    def test_create_preserves_metadata(self, node_repository: NodeRepository):
        """Test that metadata dict survives round-trip."""
        node = MemoryNode(
            node_type=NodeType.FACT,
            content="Test metadata",
            summary="Test",
            metadata={"key": "value", "nested": {"a": 1}},
        )
        node_repository.create(node)  # Persist
        retrieved = node_repository.get_by_id(node.id)  # Fetch

        assert retrieved.metadata == {"key": "value", "nested": {"a": 1}}  # Metadata preserved


class TestNodeRepositoryQuery:
    """Tests for node query operations."""

    def test_get_nonexistent_returns_none(self, node_repository: NodeRepository):
        """Test that querying a non-existent ID returns None."""
        result = node_repository.get_by_id("nonexistent-id")  # Query missing node
        assert result is None  # Not found

    def test_get_all_active_pagination(self, node_repository: NodeRepository):
        """Test pagination of active nodes."""
        # Create 5 nodes
        for i in range(5):
            node = MemoryNode(
                node_type=NodeType.THOUGHT,
                content=f"Thought number {i}",
                summary=f"Thought {i}",
            )
            node_repository.create(node)

        # Get first page
        page_0 = node_repository.get_all_active(page=0, page_size=3)
        assert len(page_0) == 3  # First page has 3

        # Get second page
        page_1 = node_repository.get_all_active(page=1, page_size=3)
        assert len(page_1) == 2  # Second page has remaining 2

    def test_get_by_type(self, node_repository: NodeRepository):
        """Test filtering nodes by type."""
        # Create nodes of different types
        node_repository.create(MemoryNode(
            node_type=NodeType.BELIEF, content="A belief", summary="Belief"
        ))
        node_repository.create(MemoryNode(
            node_type=NodeType.PREFERENCE, content="A preference", summary="Pref"
        ))
        node_repository.create(MemoryNode(
            node_type=NodeType.BELIEF, content="Another belief", summary="Belief 2"
        ))

        beliefs = node_repository.get_by_type(NodeType.BELIEF)  # Query beliefs
        assert len(beliefs) == 2  # Two beliefs found

    def test_count_active(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test counting active nodes."""
        assert node_repository.count_active() == 0  # Initially empty
        node_repository.create(sample_node)  # Add one
        assert node_repository.count_active() == 1  # One active


class TestNodeRepositoryUpdate:
    """Tests for node update operations."""

    def test_update_content(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test updating a node's content."""
        node_repository.create(sample_node)  # Create

        sample_node.content = "Updated content"  # Modify
        node_repository.update(sample_node)  # Persist update

        retrieved = node_repository.get_by_id(sample_node.id)  # Fetch
        assert retrieved.content == "Updated content"  # Updated

    def test_soft_delete_via_update(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test soft deleting by setting is_active to False."""
        node_repository.create(sample_node)  # Create
        sample_node.is_active = False  # Soft delete
        node_repository.update(sample_node)  # Persist

        # Should not appear in active queries
        active = node_repository.get_all_active()
        assert len(active) == 0  # Not in active list

    def test_record_access(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test recording memory access updates count and timestamp."""
        node_repository.create(sample_node)  # Create
        node_repository.record_access(sample_node.id)  # Record access

        retrieved = node_repository.get_by_id(sample_node.id)  # Fetch
        assert retrieved.access_count == 1  # Incremented
        assert retrieved.last_accessed_at is not None  # Timestamp set


class TestNodeRepositoryDelete:
    """Tests for node deletion operations."""

    def test_delete_existing_node(self, node_repository: NodeRepository, sample_node: MemoryNode):
        """Test hard deleting an existing node."""
        node_repository.create(sample_node)  # Create
        result = node_repository.delete(sample_node.id)  # Delete

        assert result is True  # Deletion succeeded
        assert node_repository.get_by_id(sample_node.id) is None  # Gone

    def test_delete_nonexistent_returns_false(self, node_repository: NodeRepository):
        """Test deleting a non-existent node returns False."""
        result = node_repository.delete("nonexistent-id")  # Delete missing
        assert result is False  # Nothing to delete
