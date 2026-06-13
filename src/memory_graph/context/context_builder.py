"""Orchestrator for assembling memory context for LLM prompt injection."""

import structlog  # Structured logging

from memory_graph.config.settings import ContextSettings  # Context configuration
from memory_graph.context.context_formatter import ContextFormatter  # Text formatting
from memory_graph.context.context_ranker import ContextRanker  # Memory scoring
from memory_graph.context.token_budget import TokenBudget  # Token management
from memory_graph.embeddings.similarity_search import SimilaritySearch  # Semantic search
from memory_graph.graph.graph_query import GraphQuery  # Graph traversal
from memory_graph.models.memory_node import MemoryNode  # Node data model
from memory_graph.persistence.node_repository import NodeRepository  # Node data access

# Module logger for context building operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ContextBuilder:
    """Orchestrates the full context building pipeline for LLM injection.

    Combines embedding similarity search, graph expansion, ranking,
    token budget management, and formatting into a single retrieval flow.
    """

    def __init__(
        self,
        settings: ContextSettings,
        similarity_search: SimilaritySearch,
        graph_query: GraphQuery,
        node_repository: NodeRepository,
        context_ranker: ContextRanker,
        token_budget: TokenBudget,
        context_formatter: ContextFormatter,
    ) -> None:
        """Initialize the context builder with all required services."""
        self._settings = settings  # Context configuration
        self._similarity_search = similarity_search  # For semantic retrieval
        self._graph_query = graph_query  # For graph expansion
        self._node_repo = node_repository  # For fetching full node data
        self._ranker = context_ranker  # For scoring candidates
        self._token_budget = token_budget  # For budget enforcement
        self._formatter = context_formatter  # For text formatting
        logger.info("context_builder_initialized")  # Log initialization

    def build_context(
        self,
        query_text: str,
        max_tokens: int | None = None,
        output_format: str | None = None,
    ) -> tuple[str, int, int]:
        """Build a formatted memory context block for LLM injection.

        Returns a tuple of (formatted_text, token_count, memories_included).
        """
        logger.info("building_context", query_length=len(query_text))  # Log context build start

        # Step 1: Find semantically similar memories via embedding search
        similar_results = self._similarity_search.find_similar_to_text(  # Semantic search
            text=query_text,  # Use query text as search input
            top_k=self._settings.max_memories_injected * 2,  # Fetch extra for filtering
            threshold=0.1,  # Low threshold to get broad candidates
        )

        # Build similarity score mapping for the ranker
        similarity_scores: dict[str, float] = {}  # Map node_id -> similarity score
        candidate_ids: set[str] = set()  # Track all candidate node IDs

        for result in similar_results:  # Process similarity results
            similarity_scores[result.node_id] = result.score  # Store the score
            candidate_ids.add(result.node_id)  # Add to candidates

        # Step 2: Expand candidates via graph traversal (1-2 hops)
        expanded_ids: set[str] = set()  # Track expansion results
        for node_id in list(candidate_ids)[:10]:  # Expand from top similar nodes only
            related = self._graph_query.find_related_memories(  # Traverse outward
                node_id=node_id,  # Start from each similar node
                max_depth=2,  # Go 1-2 hops out
            )
            for related_id in related:  # Add discovered nodes
                if related_id not in candidate_ids:  # Not already a candidate
                    expanded_ids.add(related_id)  # Mark as expansion candidate
                    similarity_scores.setdefault(related_id, 0.05)  # Low default score

        candidate_ids.update(expanded_ids)  # Merge expansion into candidates

        logger.debug(  # Log candidate discovery
            "candidates_discovered",
            from_similarity=len(similar_results),
            from_expansion=len(expanded_ids),
            total=len(candidate_ids),
        )

        # Step 3: Fetch full node data for all candidates
        candidates: list[MemoryNode] = []  # Accumulate full node objects
        for node_id in candidate_ids:  # Fetch each candidate
            node = self._node_repo.get_by_id(node_id)  # Get full data from DB
            if node and node.is_active:  # Node exists and is active
                candidates.append(node)  # Add to candidates list

        if not candidates:  # No valid candidates found
            logger.info("no_candidates_found")  # Log empty result
            return ("", 0, 0)  # Return empty context

        # Step 4: Rank all candidates by combined relevance score
        ranked_memories = self._ranker.rank_memories(candidates, similarity_scores)  # Score and sort

        # Step 5: Apply token budget to select final memories
        effective_budget = max_tokens or self._settings.max_token_budget  # Use override or default
        selected_memories = self._apply_token_budget(ranked_memories, effective_budget)  # Budget filter

        # Step 6: Format selected memories into injectable text
        fmt = output_format or self._settings.format  # Resolve format
        formatted_text = self._formatter.format_memories(selected_memories, fmt)  # Generate text

        # Calculate final token count
        token_count = self._token_budget.count_tokens(formatted_text)  # Count tokens in output
        memories_included = len(selected_memories)  # Count of included memories

        # Record access for all included memories
        for node, _ in selected_memories:  # Update access stats
            self._node_repo.record_access(node.id)  # Increment access counter

        logger.info(  # Log context build completion
            "context_built",
            memories_included=memories_included,
            token_count=token_count,
            format=fmt,
        )

        return (formatted_text, token_count, memories_included)  # Return the triple

    def _apply_token_budget(
        self,
        ranked_memories: list[tuple[MemoryNode, float]],
        max_tokens: int,
    ) -> list[tuple[MemoryNode, float]]:
        """Select memories greedily within the token budget.

        Processes memories in ranked order, adding until budget is exhausted.
        """
        selected: list[tuple[MemoryNode, float]] = []  # Accumulate selected memories
        used_tokens = 50  # Reserve tokens for formatting overhead (headers, tags)

        for node, score in ranked_memories:  # Process in rank order
            # Estimate tokens for this memory's summary
            memory_tokens = self._token_budget.count_tokens(node.summary)  # Count summary tokens
            overhead_per_item = 10  # Account for bullet, confidence label, etc.

            if used_tokens + memory_tokens + overhead_per_item > max_tokens:  # Would exceed budget
                break  # Stop adding memories

            selected.append((node, score))  # Add to selection
            used_tokens += memory_tokens + overhead_per_item  # Update token usage

            # Also respect the maximum memories limit
            if len(selected) >= self._settings.max_memories_injected:  # Hit memory count limit
                break  # Stop adding

        logger.debug(  # Log selection results
            "memories_selected_within_budget",
            selected_count=len(selected),
            tokens_used=used_tokens,
            budget=max_tokens,
        )
        return selected  # Return the budget-constrained selection
