"""Unit tests for the ContextRanker scoring algorithm."""

from datetime import datetime, timedelta, timezone  # Timestamp manipulation
from unittest.mock import MagicMock  # Mocking utilities

import pytest  # Test framework

from memory_graph.config.settings import RankingWeights  # Weight config
from memory_graph.context.context_ranker import ContextRanker  # Ranker under test
from memory_graph.models.memory_node import MemoryNode  # Node model
from memory_graph.models.memory_types import EmotionValence, NodeType  # Enums


class TestContextRankerScoring:
    """Tests for the memory ranking algorithm."""

    def setup_method(self):
        """Create a ranker with known weights and mocked graph query."""
        self.weights = RankingWeights(
            semantic_similarity=0.35,
            recency=0.20,
            access_frequency=0.10,
            confidence=0.15,
            intensity=0.10,
            graph_centrality=0.10,
        )
        self.mock_graph_query = MagicMock()
        self.mock_graph_query.get_all_centralities.return_value = {}  # Empty centralities
        self.ranker = ContextRanker(self.weights, self.mock_graph_query)

    def test_higher_similarity_ranks_higher(self):
        """Test that nodes with higher similarity scores rank higher."""
        node_a = self._make_node("a", confidence=0.5, intensity=0.5)
        node_b = self._make_node("b", confidence=0.5, intensity=0.5)

        similarity_scores = {"a": 0.9, "b": 0.3}  # A much more similar
        results = self.ranker.rank_memories([node_a, node_b], similarity_scores)

        assert results[0][0].id == "a"  # Higher similarity ranks first
        assert results[0][1] > results[1][1]  # Score is higher

    def test_higher_confidence_ranks_higher(self):
        """Test that higher confidence improves ranking."""
        node_a = self._make_node("a", confidence=0.95, intensity=0.5)
        node_b = self._make_node("b", confidence=0.3, intensity=0.5)

        similarity_scores = {"a": 0.5, "b": 0.5}  # Equal similarity
        results = self.ranker.rank_memories([node_a, node_b], similarity_scores)

        assert results[0][0].id == "a"  # Higher confidence ranks first

    def test_recent_node_ranks_higher(self):
        """Test that more recently created nodes rank higher."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=90)  # 90 days ago

        node_a = self._make_node("a", confidence=0.5, intensity=0.5)
        node_a.created_at = now  # Very recent

        node_b = self._make_node("b", confidence=0.5, intensity=0.5)
        node_b.created_at = old_time  # Old

        similarity_scores = {"a": 0.5, "b": 0.5}  # Equal similarity
        results = self.ranker.rank_memories([node_a, node_b], similarity_scores)

        assert results[0][0].id == "a"  # More recent ranks first

    def test_empty_candidates_returns_empty(self):
        """Test that empty candidate list returns empty results."""
        results = self.ranker.rank_memories([], {})
        assert results == []  # Empty in, empty out

    def test_all_signals_contribute(self):
        """Test that a node with all high signals scores highest."""
        now = datetime.now(timezone.utc)

        # Node with all positive signals
        strong_node = self._make_node("strong", confidence=0.95, intensity=0.9)
        strong_node.created_at = now
        strong_node.access_count = 50

        # Node with all weak signals
        weak_node = self._make_node("weak", confidence=0.2, intensity=0.1)
        weak_node.created_at = now - timedelta(days=180)
        weak_node.access_count = 0

        similarity_scores = {"strong": 0.8, "weak": 0.3}
        results = self.ranker.rank_memories([strong_node, weak_node], similarity_scores)

        assert results[0][0].id == "strong"  # All-strong beats all-weak
        # Score difference should be substantial
        assert results[0][1] - results[1][1] > 0.3

    def _make_node(self, node_id: str, confidence: float, intensity: float) -> MemoryNode:
        """Helper to create a test node with specified parameters."""
        return MemoryNode(
            id=node_id,
            node_type=NodeType.THOUGHT,
            content=f"Test content for {node_id}",
            summary=f"Test {node_id}",
            confidence=confidence,
            intensity=intensity,
            emotional_valence=EmotionValence.NEUTRAL,
        )
