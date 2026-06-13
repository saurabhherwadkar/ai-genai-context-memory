"""Sentence transformer wrapper for generating text embeddings."""

import numpy as np  # Numerical arrays for embedding vectors
import structlog  # Structured logging
from sentence_transformers import SentenceTransformer  # Pre-trained embedding models

from memory_graph.config.settings import EmbeddingSettings  # Embedding configuration

# Module logger for embedding operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class EmbeddingService:
    """Wraps sentence-transformers to generate dense vector embeddings.

    Provides lazy model initialization and both single and batch
    encoding operations for memory content text.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Initialize the embedding service with configuration settings."""
        self._model_name = settings.model_name  # HuggingFace model identifier
        self._batch_size = settings.batch_size  # Texts per encoding batch
        self._model: SentenceTransformer | None = None  # Lazy-loaded model instance
        self._dimension: int | None = None  # Embedding vector dimension
        logger.info("embedding_service_initialized", model=self._model_name)  # Log init

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension for the loaded model."""
        self._ensure_model_loaded()  # Ensure model is available
        assert self._dimension is not None  # Dimension set after model load
        return self._dimension  # Return the vector dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense vector embedding.

        Returns a 1D numpy array of float32 values.
        """
        self._ensure_model_loaded()  # Ensure model is available
        assert self._model is not None  # Model guaranteed loaded

        # Encode the text and get the embedding vector
        embedding = self._model.encode(  # Generate embedding
            text,  # Input text to encode
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,  # Suppress progress bar for single texts
        )

        # Ensure output is float32 numpy array
        result = np.asarray(embedding, dtype=np.float32)  # Cast to float32

        logger.debug("text_encoded", text_length=len(text), dimension=result.shape[0])  # Log encoding
        return result  # Return the embedding vector

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts into embedding vectors in one batch.

        More efficient than encoding one at a time for bulk operations.
        Returns a list of 1D numpy arrays.
        """
        if not texts:  # Empty input returns empty output
            return []  # Return empty list

        self._ensure_model_loaded()  # Ensure model is available
        assert self._model is not None  # Model guaranteed loaded

        logger.debug("batch_encoding_started", count=len(texts))  # Log batch start

        # Encode all texts in optimized batches
        embeddings = self._model.encode(  # Batch encode
            texts,  # List of texts to encode
            batch_size=self._batch_size,  # Process in configured batch size
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,  # Suppress progress bar
        )

        # Convert to list of individual float32 vectors
        result = [np.asarray(emb, dtype=np.float32) for emb in embeddings]  # Cast each vector

        logger.debug("batch_encoding_complete", count=len(result))  # Log batch completion
        return result  # Return list of embedding vectors

    def _ensure_model_loaded(self) -> None:
        """Lazily load the sentence transformer model on first use.

        Avoids loading the potentially large model until actually needed.
        """
        if self._model is not None:  # Model already loaded
            return  # Skip loading

        logger.info("loading_embedding_model", model=self._model_name)  # Log model load
        self._model = SentenceTransformer(self._model_name)  # Download and load model
        # Determine vector dimension from a test encoding
        test_embedding = self._model.encode("test", normalize_embeddings=True)  # Encode test text
        self._dimension = len(test_embedding)  # Extract dimension from test output
        logger.info(  # Log successful load with dimension info
            "embedding_model_loaded",
            model=self._model_name,
            dimension=self._dimension,
        )
