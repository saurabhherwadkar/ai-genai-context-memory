"""Unit tests for the MemoryNode data model."""

import pytest  # Test framework

from memory_graph.models.memory_node import MemoryNode  # Model under test
from memory_graph.models.memory_types import EmotionValence, NodeType  # Enums


class TestMemoryNodeCreation:
    """Tests for MemoryNode instantiation and default values."""

    def test_create_minimal_node(self):
        """Test creating a node with only required fields."""
        node = MemoryNode(
            node_type=NodeType.THOUGHT,
            content="A simple thought",
            summary="Simple thought",
        )
        assert node.id is not None  # UUID auto-generated
        assert node.node_type == NodeType.THOUGHT  # Type set correctly
        assert node.confidence == 0.8  # Default confidence
        assert node.emotional_valence == EmotionValence.NEUTRAL  # Default valence
        assert node.intensity == 0.5  # Default intensity
        assert node.is_active is True  # Default active
        assert node.access_count == 0  # Default zero access

    def test_create_full_node(self):
        """Test creating a node with all fields specified."""
        node = MemoryNode(
            node_type=NodeType.PREFERENCE,
            content="User prefers dark mode in all applications",
            summary="Prefers dark mode",
            confidence=0.95,
            emotional_valence=EmotionValence.POSITIVE,
            intensity=0.8,
            tags=["ui", "preferences"],
            metadata={"source": "direct_statement"},
        )
        assert node.confidence == 0.95  # Custom confidence
        assert node.emotional_valence == EmotionValence.POSITIVE  # Custom valence
        assert node.intensity == 0.8  # Custom intensity
        assert node.tags == ["ui", "preferences"]  # Tags set
        assert node.metadata == {"source": "direct_statement"}  # Metadata set

    def test_custom_type_requires_name(self):
        """Test that CUSTOM node type requires custom_type_name."""
        with pytest.raises(ValueError, match="custom_type_name is required"):
            MemoryNode(
                node_type=NodeType.CUSTOM,
                content="Something custom",
                summary="Custom memory",
                custom_type_name=None,  # Missing required name
            )

    def test_custom_type_with_name_succeeds(self):
        """Test that CUSTOM type with name creates successfully."""
        node = MemoryNode(
            node_type=NodeType.CUSTOM,
            content="A cultural preference",
            summary="Cultural pref",
            custom_type_name="cultural_preference",
        )
        assert node.custom_type_name == "cultural_preference"  # Name set


class TestMemoryNodeValidation:
    """Tests for MemoryNode field validation rules."""

    def test_confidence_must_be_in_range(self):
        """Test that confidence rejects values outside 0-1 range."""
        with pytest.raises(ValueError):
            MemoryNode(
                node_type=NodeType.THOUGHT,
                content="Test",
                summary="Test",
                confidence=1.5,  # Above maximum
            )

    def test_confidence_minimum_boundary(self):
        """Test confidence at minimum boundary."""
        node = MemoryNode(
            node_type=NodeType.THOUGHT,
            content="Test content",
            summary="Test",
            confidence=0.0,
        )
        assert node.confidence == 0.0  # Minimum accepted

    def test_intensity_must_be_in_range(self):
        """Test that intensity rejects values outside 0-1 range."""
        with pytest.raises(ValueError):
            MemoryNode(
                node_type=NodeType.THOUGHT,
                content="Test",
                summary="Test",
                intensity=-0.1,  # Below minimum
            )

    def test_content_must_not_be_empty(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError):
            MemoryNode(
                node_type=NodeType.THOUGHT,
                content="",  # Empty content
                summary="Test",
            )

    def test_tags_are_cleaned(self):
        """Test that whitespace-only tags are removed."""
        node = MemoryNode(
            node_type=NodeType.THOUGHT,
            content="Test content",
            summary="Test",
            tags=["valid", "  ", "", "also_valid"],
        )
        assert node.tags == ["valid", "also_valid"]  # Empty tags removed

    def test_timestamps_are_auto_generated(self):
        """Test that created_at and updated_at are auto-populated."""
        node = MemoryNode(
            node_type=NodeType.THOUGHT,
            content="Test content",
            summary="Test",
        )
        assert node.created_at is not None  # Auto-generated
        assert node.updated_at is not None  # Auto-generated

    def test_all_node_types_are_valid(self):
        """Test that all defined NodeType values create valid nodes."""
        for node_type in NodeType:
            kwargs = {
                "node_type": node_type,
                "content": f"Test {node_type.value}",
                "summary": f"Test {node_type.value}",
            }
            if node_type == NodeType.CUSTOM:
                kwargs["custom_type_name"] = "test_custom"
            node = MemoryNode(**kwargs)
            assert node.node_type == node_type  # Type preserved
