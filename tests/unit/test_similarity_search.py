"""Unit tests for the SimilaritySearch service."""

from unittest.mock import MagicMock  # Mocking utilities

import numpy as np  # Numerical arrays
import pytest  # Test framework

from memory_graph.embeddings.similarity_search import SimilaritySearch  # Service under test


class TestSimilaritySearchIndex:
    """Tests for the similarity search index operations."""

    def setup_method(self):
        """Create a similarity search with mocked dependencies."""
        self.mock_embedding_service = MagicMock()  # Mock embedding service
        self.mock_embedding_repo = MagicMock()  # Mock embedding repository
        self.mock_cache = MagicMock()  # Mock embedding cache

        self.search = SimilaritySearch(
            embedding_service=self.mock_embedding_service,
            embedding_repository=self.mock_embedding_repo,
            embedding_cache=self.mock_cache,
        )

    def test_empty_index_returns_no_results(self):
        """Test that searching an empty index returns empty list."""
        query = np.random.rand(384).astype(np.float32)  # Random query vector
        results = self.search.find_similar(query, top_k=5)  # Search
        assert results == []  # No results from empty index

    def test_add_to_index_and_search(self):
        """Test adding embeddings and finding similar ones."""
        # Create normalized test vectors
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Vector 1
        vec2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # Similar to vec1
        vec2 = vec2 / np.linalg.norm(vec2)  # Normalize
        vec3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Dissimilar

        # Add to index
        self.search.add_to_index("node-1", vec1)
        self.search.add_to_index("node-2", vec2)
        self.search.add_to_index("node-3", vec3)

        # Search with a vector similar to vec1
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Same as vec1
        results = self.search.find_similar(query, top_k=2)  # Find top 2

        assert len(results) == 2  # Got 2 results
        assert results[0].node_id == "node-1"  # Most similar first
        assert results[0].score > 0.9  # High similarity score

    def test_threshold_filters_results(self):
        """Test that similarity threshold filters out low-scoring results."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Vector 1
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal (score ~0)

        self.search.add_to_index("node-1", vec1)
        self.search.add_to_index("node-2", vec2)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.search.find_similar(query, top_k=10, threshold=0.5)  # High threshold

        assert len(results) == 1  # Only vec1 above threshold
        assert results[0].node_id == "node-1"

    def test_exclude_ids_filters_results(self):
        """Test that excluded IDs are removed from results."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        self.search.add_to_index("node-1", vec1)
        self.search.add_to_index("node-2", vec2)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.search.find_similar(
            query, top_k=10, exclude_ids={"node-1"}
        )  # Exclude node-1

        assert len(results) == 1  # Only node-2 remains
        assert results[0].node_id == "node-2"

    def test_remove_from_index(self):
        """Test removing an embedding from the index."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.search.add_to_index("node-1", vec1)
        assert self.search.index_size == 1  # One in index

        self.search.remove_from_index("node-1")  # Remove
        assert self.search.index_size == 0  # Empty

    def test_index_size_property(self):
        """Test the index_size property reflects actual count."""
        assert self.search.index_size == 0  # Initially empty

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.search.add_to_index("node-1", vec)
        assert self.search.index_size == 1  # One added

        self.search.add_to_index("node-2", vec)
        assert self.search.index_size == 2  # Two total

    def test_find_similar_to_text_uses_cache(self):
        """Test that find_similar_to_text checks cache first."""
        cached_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.mock_cache.get.return_value = cached_embedding  # Cache hit

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.search.add_to_index("node-1", vec)

        results = self.search.find_similar_to_text("test query", top_k=5)

        self.mock_cache.get.assert_called_once_with("test query")  # Cache checked
        self.mock_embedding_service.encode.assert_not_called()  # No encode needed
