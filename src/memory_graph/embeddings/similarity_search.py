"""Cosine similarity search over memory node embeddings."""

from dataclasses import dataclass  # Lightweight data container for search results

import numpy as np  # Numerical array operations
import structlog  # Structured logging

from memory_graph.embeddings.embedding_cache import EmbeddingCache  # LRU cache
from memory_graph.embeddings.embedding_service import EmbeddingService  # Encoding service
from memory_graph.persistence.embedding_repository import EmbeddingRepository  # Persistence

# Module logger for similarity search operations
logger = structlog.get_logger(__name__)  # Named logger for this module


@dataclass
class SimilarityResult:
    """Container for a single similarity search result."""

    node_id: str  # ID of the matching memory node
    score: float  # Cosine similarity score (0.0 to 1.0)


class SimilaritySearch:
    """Performs cosine similarity search over stored memory embeddings.

    Maintains an in-memory numpy matrix of all embeddings for fast
    bulk similarity computation without an external vector database.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        embedding_repository: EmbeddingRepository,
        embedding_cache: EmbeddingCache,
    ) -> None:
        """Initialize the similarity search with required services."""
        self._embedding_service = embedding_service  # For encoding query text
        self._embedding_repo = embedding_repository  # For loading stored embeddings
        self._cache = embedding_cache  # For caching frequently searched texts
        self._node_ids: list[str] = []  # Ordered list of node IDs matching matrix rows
        self._embedding_matrix: np.ndarray | None = None  # 2D matrix of all embeddings
        logger.info("similarity_search_initialized")  # Log initialization

    def load_index(self) -> None:
        """Load all stored embeddings into the in-memory search index.

        Builds a numpy matrix from all persisted embeddings for fast
        batch similarity computation.
        """
        logger.info("loading_similarity_index")  # Log index load start

        # Load all embeddings from the database
        all_embeddings = self._embedding_repo.get_all()  # Dict of node_id -> vector

        if not all_embeddings:  # No embeddings stored yet
            self._node_ids = []  # Empty node ID list
            self._embedding_matrix = None  # No matrix to build
            logger.info("similarity_index_empty")  # Log empty state
            return  # Nothing to load

        # Build ordered arrays for matrix construction
        self._node_ids = list(all_embeddings.keys())  # Ordered node IDs
        vectors = list(all_embeddings.values())  # Corresponding vectors

        # Stack vectors into a 2D matrix (rows = nodes, columns = dimensions)
        self._embedding_matrix = np.vstack(vectors)  # Create the search matrix
        logger.info(  # Log successful index load
            "similarity_index_loaded",
            node_count=len(self._node_ids),
            dimension=self._embedding_matrix.shape[1],
        )

    def add_to_index(self, node_id: str, embedding: np.ndarray) -> None:
        """Incrementally add a new embedding to the search index.

        Appends to the existing matrix without requiring full reload.
        """
        if node_id in self._node_ids:  # Node already exists in index
            # Update existing entry in place
            idx = self._node_ids.index(node_id)  # Find position in matrix
            if self._embedding_matrix is not None:  # Matrix exists
                self._embedding_matrix[idx] = embedding  # Update the row
            return  # Done updating

        # Append new entry to the index
        self._node_ids.append(node_id)  # Add node ID to tracking list

        if self._embedding_matrix is None:  # First entry in the index
            self._embedding_matrix = embedding.reshape(1, -1)  # Create matrix with one row
        else:  # Append to existing matrix
            new_row = embedding.reshape(1, -1)  # Reshape to 2D row
            self._embedding_matrix = np.vstack([self._embedding_matrix, new_row])  # Stack

        logger.debug("embedding_added_to_index", node_id=node_id)  # Log addition

    def remove_from_index(self, node_id: str) -> None:
        """Remove an embedding from the search index."""
        if node_id not in self._node_ids:  # Node not in index
            return  # Nothing to remove

        idx = self._node_ids.index(node_id)  # Find position
        self._node_ids.pop(idx)  # Remove from ID list

        if self._embedding_matrix is not None:  # Matrix exists
            self._embedding_matrix = np.delete(self._embedding_matrix, idx, axis=0)  # Remove row
            if self._embedding_matrix.shape[0] == 0:  # Matrix now empty
                self._embedding_matrix = None  # Reset to None

        logger.debug("embedding_removed_from_index", node_id=node_id)  # Log removal

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        exclude_ids: set[str] | None = None,
    ) -> list[SimilarityResult]:
        """Find the top-k most similar nodes to a query embedding.

        Uses cosine similarity (embeddings are pre-normalized).
        Filters results below the threshold score.
        """
        if self._embedding_matrix is None or len(self._node_ids) == 0:  # Empty index
            return []  # No results possible

        # Compute cosine similarity (dot product since vectors are L2-normalized)
        query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)  # Normalize
        similarities = self._embedding_matrix @ query_normalized  # Batch dot product

        # Build exclusion set for filtering
        exclude = exclude_ids or set()  # Default to empty set

        # Create scored results with filtering
        scored_results: list[SimilarityResult] = []  # Accumulate valid results
        for idx, score in enumerate(similarities):  # Iterate all similarity scores
            node_id = self._node_ids[idx]  # Get corresponding node ID

            if node_id in exclude:  # Skip excluded nodes
                continue  # Move to next

            if score < threshold:  # Below minimum threshold
                continue  # Skip low-similarity results

            scored_results.append(SimilarityResult(node_id=node_id, score=float(score)))  # Add

        # Sort by score descending and take top-k
        scored_results.sort(key=lambda r: r.score, reverse=True)  # Sort highest first
        top_results = scored_results[:top_k]  # Limit to top-k results

        logger.debug(  # Log search results
            "similarity_search_complete",
            total_candidates=len(self._node_ids),
            results_returned=len(top_results),
        )
        return top_results  # Return the top-k similar nodes

    def find_similar_to_text(
        self,
        text: str,
        top_k: int = 10,
        threshold: float = 0.0,
        exclude_ids: set[str] | None = None,
    ) -> list[SimilarityResult]:
        """Find the top-k most similar nodes to a text query.

        Encodes the text to an embedding, using cache when available,
        then delegates to find_similar for the actual search.
        """
        # Check cache first for the query text embedding
        cached_embedding = self._cache.get(text)  # Look up in LRU cache
        if cached_embedding is not None:  # Cache hit
            query_embedding = cached_embedding  # Use cached vector
        else:  # Cache miss, need to compute
            query_embedding = self._embedding_service.encode(text)  # Encode the text
            self._cache.put(text, query_embedding)  # Store in cache for future use

        return self.find_similar(  # Delegate to vector-based search
            query_embedding=query_embedding,  # Encoded query vector
            top_k=top_k,  # Maximum results
            threshold=threshold,  # Minimum score
            exclude_ids=exclude_ids,  # Exclusions
        )

    @property
    def index_size(self) -> int:
        """Get the current number of embeddings in the search index."""
        return len(self._node_ids)  # Return count of indexed nodes
