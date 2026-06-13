"""Memory ranking algorithm for scoring relevance of candidate memories."""

import math  # Mathematical functions for decay calculation
from datetime import datetime, timezone  # Timestamp comparison

import structlog  # Structured logging

from memory_graph.config.settings import RankingWeights  # Weight configuration
from memory_graph.graph.graph_query import GraphQuery  # Centrality calculations
from memory_graph.models.memory_node import MemoryNode  # Node data model

# Module logger for ranking operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# Decay half-life in days for recency scoring
RECENCY_HALF_LIFE_DAYS = 30.0  # Memories lose half their recency score every 30 days


class ContextRanker:
    """Scores and ranks memory nodes for context injection relevance.

    Combines multiple signals (semantic similarity, recency, access frequency,
    confidence, intensity, graph centrality) into a single ranking score.
    """

    def __init__(self, weights: RankingWeights, graph_query: GraphQuery) -> None:
        """Initialize the ranker with scoring weights and graph query service."""
        self._weights = weights  # Configured weight for each signal
        self._graph_query = graph_query  # For centrality calculations
        self._centrality_cache: dict[str, float] = {}  # Cache centrality values
        logger.info("context_ranker_initialized")  # Log initialization

    def rank_memories(
        self,
        candidates: list[MemoryNode],
        similarity_scores: dict[str, float],
    ) -> list[tuple[MemoryNode, float]]:
        """Rank candidate memories by combined relevance score.

        Returns a list of (node, score) tuples sorted by score descending.
        """
        if not candidates:  # No candidates to rank
            return []  # Return empty list

        # Refresh centrality cache for scoring
        self._refresh_centrality_cache()  # Update cached centrality values

        # Score each candidate memory
        scored_candidates: list[tuple[MemoryNode, float]] = []  # Accumulate scored results
        for node in candidates:  # Process each candidate
            score = self._compute_score(node, similarity_scores)  # Calculate combined score
            scored_candidates.append((node, score))  # Store node with its score

        # Sort by score descending (highest relevance first)
        scored_candidates.sort(key=lambda item: item[1], reverse=True)  # Sort descending

        logger.debug(  # Log ranking results
            "memories_ranked",
            candidate_count=len(candidates),
            top_score=scored_candidates[0][1] if scored_candidates else 0.0,
        )
        return scored_candidates  # Return sorted results

    def _compute_score(self, node: MemoryNode, similarity_scores: dict[str, float]) -> float:
        """Compute the weighted relevance score for a single memory node.

        Combines all scoring signals using configured weights.
        """
        # Get semantic similarity score (from embedding search)
        semantic_score = similarity_scores.get(node.id, 0.0)  # Default 0 if not in search results

        # Calculate recency score using exponential decay
        recency_score = self._compute_recency_score(node)  # Time-based decay

        # Calculate access frequency score (normalized)
        frequency_score = self._compute_frequency_score(node)  # Usage-based

        # Get confidence directly from the node
        confidence_score = node.confidence  # Already in 0-1 range

        # Get intensity directly from the node
        intensity_score = node.intensity  # Already in 0-1 range

        # Get graph centrality for this node
        centrality_score = self._centrality_cache.get(node.id, 0.0)  # From cache

        # Compute weighted sum of all signals
        total_score = (
            self._weights.semantic_similarity * semantic_score  # Semantic relevance
            + self._weights.recency * recency_score  # Time-based importance
            + self._weights.access_frequency * frequency_score  # Usage popularity
            + self._weights.confidence * confidence_score  # Certainty level
            + self._weights.intensity * intensity_score  # Memory strength
            + self._weights.graph_centrality * centrality_score  # Network position
        )

        return total_score  # Return the combined score

    def _compute_recency_score(self, node: MemoryNode) -> float:
        """Calculate recency score using exponential decay from creation time.

        Returns a value between 0 and 1, where 1 means just created.
        """
        now = datetime.now(timezone.utc)  # Current UTC time

        # Use last_accessed_at if available, otherwise created_at
        reference_time = node.last_accessed_at or node.created_at  # Choose most recent interaction

        # Calculate age in days
        age_delta = now - reference_time  # Time difference
        age_days = age_delta.total_seconds() / 86400.0  # Convert to days

        # Exponential decay: score = 2^(-age/half_life)
        decay_score = math.pow(2.0, -age_days / RECENCY_HALF_LIFE_DAYS)  # Apply decay function

        return max(0.0, min(1.0, decay_score))  # Clamp to 0-1 range

    def _compute_frequency_score(self, node: MemoryNode) -> float:
        """Calculate normalized frequency score based on access count.

        Uses logarithmic scaling to prevent high-access nodes from dominating.
        """
        if node.access_count == 0:  # Never accessed
            return 0.0  # Zero frequency score

        # Logarithmic normalization: log(1 + count) / log(1 + max_expected)
        max_expected_accesses = 100.0  # Normalization ceiling
        raw_score = math.log1p(node.access_count) / math.log1p(max_expected_accesses)  # Log scale

        return max(0.0, min(1.0, raw_score))  # Clamp to 0-1 range

    def _refresh_centrality_cache(self) -> None:
        """Refresh the cached centrality values for all nodes."""
        self._centrality_cache = self._graph_query.get_all_centralities()  # Recompute all
