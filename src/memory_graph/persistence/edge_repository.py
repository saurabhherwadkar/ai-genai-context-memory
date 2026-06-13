"""Repository for memory edge CRUD operations against SQLite."""

import json  # JSON serialization for metadata field
from datetime import datetime  # Timestamp parsing

import structlog  # Structured logging

from memory_graph.models.memory_edge import MemoryEdge  # Memory edge data model
from memory_graph.models.memory_types import EdgeType  # Edge classification enum
from memory_graph.persistence.database import DatabaseManager  # Database connection manager

# Module logger for edge repository operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class EdgeRepository:
    """Handles persistence operations for memory edges in SQLite.

    Provides CRUD operations for directed relationships between memory nodes.
    """

    def __init__(self, database_manager: DatabaseManager) -> None:
        """Initialize the edge repository with a database manager."""
        self._db = database_manager  # Store reference to database manager

    def create(self, edge: MemoryEdge) -> MemoryEdge:
        """Insert a new memory edge into the database.

        Returns the persisted edge with all fields populated.
        """
        logger.debug(  # Log creation with relationship details
            "creating_memory_edge",
            edge_id=edge.id,
            source=edge.source_node_id,
            target=edge.target_node_id,
            edge_type=edge.edge_type.value,
        )

        # Serialize metadata to JSON string for storage
        metadata_json = json.dumps(edge.metadata)  # Convert dict to JSON string

        # Prepare the INSERT query
        query = """
            INSERT INTO memory_edges (
                id, source_node_id, target_node_id, edge_type,
                weight, description, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """  # Parameterized INSERT statement

        # Build the parameter tuple
        params = (
            edge.id,  # Unique edge identifier
            edge.source_node_id,  # Source node reference
            edge.target_node_id,  # Target node reference
            edge.edge_type.value,  # Enum value as string
            edge.weight,  # Relationship strength
            edge.description,  # Optional description
            edge.created_at.isoformat(),  # ISO format timestamp
            metadata_json,  # JSON-serialized metadata
        )

        self._db.execute_write(query, params)  # Execute the insert
        logger.info("memory_edge_created", edge_id=edge.id)  # Log success
        return edge  # Return the persisted edge

    def get_by_id(self, edge_id: str) -> MemoryEdge | None:
        """Retrieve a memory edge by its unique identifier.

        Returns None if no edge exists with the given ID.
        """
        logger.debug("fetching_edge_by_id", edge_id=edge_id)  # Log fetch attempt
        query = "SELECT * FROM memory_edges WHERE id = ?"  # Parameterized SELECT
        rows = self._db.execute_query(query, (edge_id,))  # Execute the query

        if not rows:  # No matching edge found
            return None  # Return None to indicate not found

        return self._row_to_edge(rows[0])  # Convert row to MemoryEdge

    def get_edges_from_node(self, node_id: str) -> list[MemoryEdge]:
        """Retrieve all outgoing edges from a specific node.

        Returns edges where the given node is the source.
        """
        query = "SELECT * FROM memory_edges WHERE source_node_id = ?"  # Outgoing edges query
        rows = self._db.execute_query(query, (node_id,))  # Execute the query
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def get_edges_to_node(self, node_id: str) -> list[MemoryEdge]:
        """Retrieve all incoming edges to a specific node.

        Returns edges where the given node is the target.
        """
        query = "SELECT * FROM memory_edges WHERE target_node_id = ?"  # Incoming edges query
        rows = self._db.execute_query(query, (node_id,))  # Execute the query
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def get_edges_for_node(self, node_id: str) -> list[MemoryEdge]:
        """Retrieve all edges connected to a node (both incoming and outgoing).

        Returns the union of incoming and outgoing edges.
        """
        query = """
            SELECT * FROM memory_edges
            WHERE source_node_id = ? OR target_node_id = ?
        """  # Both directions query
        rows = self._db.execute_query(query, (node_id, node_id))  # Execute with node_id for both
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def get_edges_by_type(self, edge_type: EdgeType) -> list[MemoryEdge]:
        """Retrieve all edges of a specific relationship type."""
        query = "SELECT * FROM memory_edges WHERE edge_type = ?"  # Type-filtered query
        rows = self._db.execute_query(query, (edge_type.value,))  # Execute with enum value
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def get_edge_between(self, source_id: str, target_id: str) -> list[MemoryEdge]:
        """Retrieve all edges between two specific nodes (in one direction).

        Returns edges from source to target only, not reverse.
        """
        query = """
            SELECT * FROM memory_edges
            WHERE source_node_id = ? AND target_node_id = ?
        """  # Directed edge query
        rows = self._db.execute_query(query, (source_id, target_id))  # Execute directed query
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def delete(self, edge_id: str) -> bool:
        """Delete an edge from the database by its ID.

        Returns True if an edge was deleted, False if not found.
        """
        logger.debug("deleting_memory_edge", edge_id=edge_id)  # Log deletion attempt
        query = "DELETE FROM memory_edges WHERE id = ?"  # Parameterized DELETE
        rows_affected = self._db.execute_write(query, (edge_id,))  # Execute the delete
        deleted = rows_affected > 0  # Check if deletion occurred

        if deleted:  # Log the outcome
            logger.info("memory_edge_deleted", edge_id=edge_id)  # Successful
        return deleted  # Return deletion status

    def delete_edges_for_node(self, node_id: str) -> int:
        """Delete all edges connected to a specific node.

        Removes both incoming and outgoing edges. Returns count deleted.
        """
        query = """
            DELETE FROM memory_edges
            WHERE source_node_id = ? OR target_node_id = ?
        """  # Delete all connected edges
        rows_affected = self._db.execute_write(query, (node_id, node_id))  # Execute bulk delete
        logger.info("edges_deleted_for_node", node_id=node_id, count=rows_affected)  # Log count
        return rows_affected  # Return number of edges removed

    def count_all(self) -> int:
        """Count the total number of edges in the database."""
        query = "SELECT COUNT(*) as count FROM memory_edges"  # Count query
        rows = self._db.execute_query(query)  # Execute the count
        return rows[0]["count"]  # Extract count value

    def get_all(self) -> list[MemoryEdge]:
        """Retrieve all edges from the database.

        Used for initial graph loading at startup.
        """
        query = "SELECT * FROM memory_edges"  # Select all edges
        rows = self._db.execute_query(query)  # Execute the query
        return [self._row_to_edge(row) for row in rows]  # Convert all rows

    def _row_to_edge(self, row: "sqlite3.Row") -> MemoryEdge:
        """Convert a SQLite row to a MemoryEdge model instance.

        Handles deserialization of JSON metadata and type conversions.
        """
        return MemoryEdge(
            id=row["id"],  # Unique edge identifier
            source_node_id=row["source_node_id"],  # Source node reference
            target_node_id=row["target_node_id"],  # Target node reference
            edge_type=EdgeType(row["edge_type"]),  # Convert string to enum
            weight=row["weight"],  # Float weight value
            description=row["description"],  # Optional description text
            created_at=datetime.fromisoformat(row["created_at"]),  # Parse ISO timestamp
            metadata=json.loads(row["metadata"]),  # Parse JSON to dict
        )
