"""Repository for memory node CRUD operations against SQLite."""

import json  # JSON serialization for tags and metadata fields
from datetime import datetime, timezone  # Timestamp handling

import structlog  # Structured logging

from memory_graph.models.memory_node import MemoryNode  # Memory node data model
from memory_graph.models.memory_types import EmotionValence, NodeType  # Classification enums
from memory_graph.persistence.database import DatabaseManager  # Database connection manager

# Module logger for node repository operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class NodeRepository:
    """Handles persistence operations for memory nodes in SQLite.

    Provides CRUD operations with automatic serialization/deserialization
    between Pydantic models and SQLite row format.
    """

    def __init__(self, database_manager: DatabaseManager) -> None:
        """Initialize the node repository with a database manager."""
        self._db = database_manager  # Store reference to database manager

    def create(self, node: MemoryNode) -> MemoryNode:
        """Insert a new memory node into the database.

        Returns the persisted node with all fields populated.
        """
        logger.debug("creating_memory_node", node_id=node.id, node_type=node.node_type.value)  # Log creation

        # Serialize complex fields to JSON strings for SQLite storage
        tags_json = json.dumps(node.tags)  # Convert tag list to JSON string
        metadata_json = json.dumps(node.metadata)  # Convert metadata dict to JSON string

        # Prepare the INSERT query with all node fields
        query = """
            INSERT INTO memory_nodes (
                id, node_type, content, summary, confidence,
                emotional_valence, intensity, source_conversation_id,
                created_at, updated_at, last_accessed_at, access_count,
                tags, metadata, is_active, custom_type_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """  # Parameterized INSERT statement

        # Build the parameter tuple matching the query placeholders
        params = (
            node.id,  # Unique node identifier
            node.node_type.value,  # Enum value as string
            node.content,  # Full text content
            node.summary,  # Brief summary
            node.confidence,  # Confidence score
            node.emotional_valence.value,  # Enum value as string
            node.intensity,  # Intensity score
            node.source_conversation_id,  # Optional conversation reference
            node.created_at.isoformat(),  # ISO format timestamp
            node.updated_at.isoformat(),  # ISO format timestamp
            node.last_accessed_at.isoformat() if node.last_accessed_at else None,  # Optional timestamp
            node.access_count,  # Access counter
            tags_json,  # JSON-serialized tags
            metadata_json,  # JSON-serialized metadata
            1 if node.is_active else 0,  # Boolean as integer for SQLite
            node.custom_type_name,  # Optional custom type name
        )

        self._db.execute_write(query, params)  # Execute the insert operation
        logger.info("memory_node_created", node_id=node.id)  # Log successful creation
        return node  # Return the persisted node

    def get_by_id(self, node_id: str) -> MemoryNode | None:
        """Retrieve a memory node by its unique identifier.

        Returns None if no node exists with the given ID.
        """
        logger.debug("fetching_node_by_id", node_id=node_id)  # Log fetch attempt
        query = "SELECT * FROM memory_nodes WHERE id = ?"  # Parameterized SELECT
        rows = self._db.execute_query(query, (node_id,))  # Execute the query

        if not rows:  # No matching node found
            logger.debug("node_not_found", node_id=node_id)  # Log miss
            return None  # Return None to indicate not found

        return self._row_to_node(rows[0])  # Convert the first row to a MemoryNode

    def get_all_active(self, page: int = 0, page_size: int = 50) -> list[MemoryNode]:
        """Retrieve all active memory nodes with pagination.

        Returns nodes ordered by creation time descending.
        """
        offset = page * page_size  # Calculate the row offset for pagination
        query = """
            SELECT * FROM memory_nodes
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """  # Paginated query for active nodes
        rows = self._db.execute_query(query, (page_size, offset))  # Execute paginated query
        return [self._row_to_node(row) for row in rows]  # Convert all rows to MemoryNode objects

    def get_by_type(self, node_type: NodeType, active_only: bool = True) -> list[MemoryNode]:
        """Retrieve all memory nodes of a specific type.

        Optionally filters to only active nodes.
        """
        if active_only:  # Filter to active nodes only
            query = "SELECT * FROM memory_nodes WHERE node_type = ? AND is_active = 1"  # Active filter
        else:  # Include all nodes regardless of active status
            query = "SELECT * FROM memory_nodes WHERE node_type = ?"  # No active filter

        rows = self._db.execute_query(query, (node_type.value,))  # Execute type-filtered query
        return [self._row_to_node(row) for row in rows]  # Convert rows to model objects

    def update(self, node: MemoryNode) -> MemoryNode:
        """Update an existing memory node in the database.

        Overwrites all mutable fields with current model values.
        """
        logger.debug("updating_memory_node", node_id=node.id)  # Log update attempt

        # Update the modification timestamp
        node.updated_at = datetime.now(timezone.utc)  # Refresh the updated_at field

        # Serialize complex fields
        tags_json = json.dumps(node.tags)  # Serialize tags to JSON
        metadata_json = json.dumps(node.metadata)  # Serialize metadata to JSON

        # Prepare the UPDATE query
        query = """
            UPDATE memory_nodes SET
                content = ?, summary = ?, confidence = ?,
                emotional_valence = ?, intensity = ?,
                updated_at = ?, last_accessed_at = ?,
                access_count = ?, tags = ?, metadata = ?,
                is_active = ?, custom_type_name = ?
            WHERE id = ?
        """  # Parameterized UPDATE statement

        # Build the parameter tuple
        params = (
            node.content,  # Updated content text
            node.summary,  # Updated summary
            node.confidence,  # Updated confidence score
            node.emotional_valence.value,  # Updated emotional valence
            node.intensity,  # Updated intensity score
            node.updated_at.isoformat(),  # Updated timestamp
            node.last_accessed_at.isoformat() if node.last_accessed_at else None,  # Access timestamp
            node.access_count,  # Updated access count
            tags_json,  # Updated tags JSON
            metadata_json,  # Updated metadata JSON
            1 if node.is_active else 0,  # Active flag as integer
            node.custom_type_name,  # Custom type name
            node.id,  # WHERE clause: target node ID
        )

        self._db.execute_write(query, params)  # Execute the update
        logger.info("memory_node_updated", node_id=node.id)  # Log successful update
        return node  # Return the updated node

    def delete(self, node_id: str) -> bool:
        """Hard delete a memory node from the database.

        Also cascades to delete associated edges and embeddings.
        Returns True if a node was deleted, False if not found.
        """
        logger.warning("deleting_memory_node", node_id=node_id)  # Log deletion (warning level)
        query = "DELETE FROM memory_nodes WHERE id = ?"  # Parameterized DELETE
        rows_affected = self._db.execute_write(query, (node_id,))  # Execute the delete
        deleted = rows_affected > 0  # True if a row was actually deleted

        if deleted:  # Log outcome
            logger.info("memory_node_deleted", node_id=node_id)  # Successful deletion
        else:  # Node was not found
            logger.debug("node_not_found_for_deletion", node_id=node_id)  # Not found

        return deleted  # Return whether deletion occurred

    def count_active(self) -> int:
        """Count the total number of active memory nodes."""
        query = "SELECT COUNT(*) as count FROM memory_nodes WHERE is_active = 1"  # Count query
        rows = self._db.execute_query(query)  # Execute the count
        return rows[0]["count"]  # Extract count from first row

    def search_by_tags(self, tags: list[str]) -> list[MemoryNode]:
        """Search for active nodes that contain any of the specified tags.

        Uses JSON string matching against the serialized tags field.
        """
        if not tags:  # Empty tag list returns no results
            return []  # Return empty list immediately

        # Build LIKE conditions for each tag (searches within JSON array string)
        conditions = " OR ".join(["tags LIKE ?" for _ in tags])  # One LIKE per tag
        query = f"SELECT * FROM memory_nodes WHERE is_active = 1 AND ({conditions})"  # Combined query
        params = tuple(f'%"{tag}"%' for tag in tags)  # Wrap each tag in JSON string pattern

        rows = self._db.execute_query(query, params)  # Execute the search
        return [self._row_to_node(row) for row in rows]  # Convert results to models

    def record_access(self, node_id: str) -> None:
        """Record that a memory node was accessed for context retrieval.

        Increments access_count and updates last_accessed_at timestamp.
        """
        now_iso = datetime.now(timezone.utc).isoformat()  # Current UTC timestamp
        query = """
            UPDATE memory_nodes
            SET access_count = access_count + 1, last_accessed_at = ?
            WHERE id = ?
        """  # Atomic increment and timestamp update
        self._db.execute_write(query, (now_iso, node_id))  # Execute the access recording

    def _row_to_node(self, row: "sqlite3.Row") -> MemoryNode:
        """Convert a SQLite row to a MemoryNode model instance.

        Handles deserialization of JSON fields and type conversions.
        """
        return MemoryNode(
            id=row["id"],  # Unique identifier
            node_type=NodeType(row["node_type"]),  # Convert string to enum
            content=row["content"],  # Text content
            summary=row["summary"],  # Summary text
            confidence=row["confidence"],  # Float confidence score
            emotional_valence=EmotionValence(row["emotional_valence"]),  # Convert to enum
            intensity=row["intensity"],  # Float intensity score
            source_conversation_id=row["source_conversation_id"],  # Optional conversation ID
            created_at=datetime.fromisoformat(row["created_at"]),  # Parse ISO timestamp
            updated_at=datetime.fromisoformat(row["updated_at"]),  # Parse ISO timestamp
            last_accessed_at=(  # Parse optional timestamp
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
            access_count=row["access_count"],  # Integer counter
            tags=json.loads(row["tags"]),  # Parse JSON array to list
            metadata=json.loads(row["metadata"]),  # Parse JSON object to dict
            is_active=bool(row["is_active"]),  # Convert integer to boolean
            custom_type_name=row["custom_type_name"],  # Optional string
        )
