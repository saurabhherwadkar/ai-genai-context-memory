"""Repository for vector embedding storage and retrieval in SQLite."""

from datetime import datetime, timezone  # Timestamp handling

import numpy as np  # Numerical array operations for embedding vectors
import structlog  # Structured logging

from memory_graph.persistence.database import DatabaseManager  # Database connection manager

# Module logger for embedding repository operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class EmbeddingRepository:
    """Handles persistence of vector embeddings as binary blobs in SQLite.

    Stores numpy arrays serialized to bytes and provides bulk loading
    for efficient startup initialization of the similarity search index.
    """

    def __init__(self, database_manager: DatabaseManager) -> None:
        """Initialize the embedding repository with a database manager."""
        self._db = database_manager  # Store reference to database manager

    def store(self, node_id: str, embedding: np.ndarray, model_name: str) -> None:
        """Store a vector embedding for a memory node.

        Serializes the numpy array to bytes and persists to SQLite.
        Replaces any existing embedding for the same node.
        """
        logger.debug("storing_embedding", node_id=node_id, dimension=embedding.shape[0])  # Log store

        # Serialize numpy array to raw bytes for compact storage
        embedding_bytes = embedding.astype(np.float32).tobytes()  # Convert to float32 bytes
        dimension = embedding.shape[0]  # Extract vector dimension
        created_at = datetime.now(timezone.utc).isoformat()  # Current UTC timestamp

        # Use INSERT OR REPLACE to handle updates to existing embeddings
        query = """
            INSERT OR REPLACE INTO embeddings (
                node_id, embedding, model_name, dimension, created_at
            ) VALUES (?, ?, ?, ?, ?)
        """  # Upsert query for embeddings
        params = (
            node_id,  # Reference to the memory node
            embedding_bytes,  # Binary blob of the vector
            model_name,  # Name of the embedding model used
            dimension,  # Vector dimensionality
            created_at,  # Storage timestamp
        )

        self._db.execute_write(query, params)  # Execute the upsert
        logger.debug("embedding_stored", node_id=node_id)  # Log success

    def get(self, node_id: str) -> np.ndarray | None:
        """Retrieve the embedding vector for a specific node.

        Returns None if no embedding exists for the node.
        """
        query = "SELECT embedding, dimension FROM embeddings WHERE node_id = ?"  # Select embedding
        rows = self._db.execute_query(query, (node_id,))  # Execute the query

        if not rows:  # No embedding found for this node
            return None  # Return None to indicate missing

        row = rows[0]  # Get the first (only) result
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)  # Deserialize bytes to numpy
        return embedding  # Return the reconstructed vector

    def get_all(self) -> dict[str, np.ndarray]:
        """Load all embeddings from the database into memory.

        Returns a dictionary mapping node_id to embedding vector.
        Used for initializing the similarity search index at startup.
        """
        logger.info("loading_all_embeddings")  # Log bulk load start
        query = "SELECT node_id, embedding, dimension FROM embeddings"  # Select all embeddings
        rows = self._db.execute_query(query)  # Execute the bulk query

        embeddings: dict[str, np.ndarray] = {}  # Initialize the result dictionary
        for row in rows:  # Iterate over all stored embeddings
            node_id = row["node_id"]  # Extract the node identifier
            vector = np.frombuffer(row["embedding"], dtype=np.float32)  # Deserialize to numpy
            embeddings[node_id] = vector  # Store in the result dictionary

        logger.info("all_embeddings_loaded", count=len(embeddings))  # Log completion with count
        return embeddings  # Return the complete embedding map

    def delete(self, node_id: str) -> bool:
        """Delete the embedding for a specific node.

        Returns True if an embedding was deleted, False if not found.
        """
        query = "DELETE FROM embeddings WHERE node_id = ?"  # Parameterized DELETE
        rows_affected = self._db.execute_write(query, (node_id,))  # Execute the delete
        return rows_affected > 0  # Return whether deletion occurred

    def count(self) -> int:
        """Count the total number of stored embeddings."""
        query = "SELECT COUNT(*) as count FROM embeddings"  # Count query
        rows = self._db.execute_query(query)  # Execute the count
        return rows[0]["count"]  # Extract the count value

    def exists(self, node_id: str) -> bool:
        """Check whether an embedding exists for a specific node."""
        query = "SELECT 1 FROM embeddings WHERE node_id = ? LIMIT 1"  # Existence check query
        rows = self._db.execute_query(query, (node_id,))  # Execute the check
        return len(rows) > 0  # Return True if embedding exists
