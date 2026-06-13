"""LRU cache for embedding vectors to reduce redundant computation."""

from collections import OrderedDict  # Ordered dictionary for LRU eviction

import numpy as np  # Numerical arrays for embedding vectors
import structlog  # Structured logging

from memory_graph.config.settings import EmbeddingSettings  # Cache size configuration

# Module logger for cache operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors keyed by text content.

    Reduces redundant calls to the embedding model by caching recently
    computed vectors. Uses an OrderedDict for O(1) LRU eviction.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Initialize the cache with the configured maximum size."""
        self._max_size = settings.cache_size  # Maximum entries before eviction
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()  # LRU storage
        self._hits = 0  # Counter for cache hits
        self._misses = 0  # Counter for cache misses
        logger.info("embedding_cache_initialized", max_size=self._max_size)  # Log init

    def get(self, text: str) -> np.ndarray | None:
        """Retrieve a cached embedding for the given text.

        Returns None if the text is not in cache. Moves accessed entry
        to the end of the LRU order on hit.
        """
        if text in self._cache:  # Check if text exists in cache
            self._cache.move_to_end(text)  # Move to end (most recently used)
            self._hits += 1  # Increment hit counter
            return self._cache[text]  # Return the cached embedding

        self._misses += 1  # Increment miss counter
        return None  # Not found in cache

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache, evicting LRU entry if full.

        If the text already exists, updates the value and moves to end.
        """
        if text in self._cache:  # Text already cached
            self._cache.move_to_end(text)  # Update LRU position
            self._cache[text] = embedding  # Update the stored value
            return  # Done

        # Check if eviction is needed before adding
        if len(self._cache) >= self._max_size:  # Cache is at capacity
            evicted_key, _ = self._cache.popitem(last=False)  # Remove oldest entry
            logger.debug("cache_entry_evicted", evicted_key_length=len(evicted_key))  # Log eviction

        self._cache[text] = embedding  # Store the new entry

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()  # Remove all cached entries
        logger.info("embedding_cache_cleared")  # Log cache clear

    @property
    def size(self) -> int:
        """Get the current number of entries in the cache."""
        return len(self._cache)  # Return current entry count

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate as a percentage.

        Returns 0.0 if no accesses have been made.
        """
        total_accesses = self._hits + self._misses  # Total access attempts
        if total_accesses == 0:  # No accesses yet
            return 0.0  # Cannot calculate rate

        return self._hits / total_accesses  # Return hit ratio
