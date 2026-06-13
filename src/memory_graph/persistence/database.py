"""SQLite database connection manager and schema migration handler."""

import sqlite3  # SQLite database interface from standard library
from pathlib import Path  # Object-oriented filesystem paths

import structlog  # Structured logging

from memory_graph.config.settings import DatabaseSettings  # Database configuration model

# Module logger for database operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# SQL schema definition for the memory graph database
SCHEMA_SQL = """
-- Memory nodes table stores all cognitive memory items
CREATE TABLE IF NOT EXISTS memory_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.8,
    emotional_valence TEXT NOT NULL DEFAULT 'neutral',
    intensity REAL NOT NULL DEFAULT 0.5,
    source_conversation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    custom_type_name TEXT
);

-- Memory edges table stores directed relationships between nodes
CREATE TABLE IF NOT EXISTS memory_edges (
    id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    description TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
);

-- Embeddings table stores vector representations of memory content
CREATE TABLE IF NOT EXISTS embeddings (
    node_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
);

-- Index for filtering nodes by type
CREATE INDEX IF NOT EXISTS idx_nodes_type ON memory_nodes(node_type);

-- Index for filtering active nodes
CREATE INDEX IF NOT EXISTS idx_nodes_active ON memory_nodes(is_active);

-- Index for ordering nodes by creation time
CREATE INDEX IF NOT EXISTS idx_nodes_created ON memory_nodes(created_at);

-- Index for traversing edges from source nodes
CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_node_id);

-- Index for traversing edges to target nodes
CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_node_id);

-- Index for filtering edges by type
CREATE INDEX IF NOT EXISTS idx_edges_type ON memory_edges(edge_type);
"""


class DatabaseManager:
    """Manages SQLite database connections and schema lifecycle.

    Provides connection factory methods and ensures schema is initialized
    on first use. Supports both file-based and in-memory databases.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialize the database manager with configuration settings."""
        self._db_path = settings.path  # Store the configured database path
        self._journal_mode = settings.journal_mode  # WAL mode for concurrency
        self._busy_timeout_ms = settings.busy_timeout_ms  # Timeout for locked database
        self._connection: sqlite3.Connection | None = None  # Cached connection instance
        logger.info("database_manager_initialized", path=self._db_path)  # Log initialization

    def _ensure_directory_exists(self) -> None:
        """Create the database directory if it does not exist."""
        if self._db_path != ":memory:":  # Skip directory creation for in-memory databases
            db_directory = Path(self._db_path).parent  # Get parent directory of database file
            db_directory.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing

    def get_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite database connection.

        Returns a cached connection with WAL mode and foreign keys enabled.
        """
        if self._connection is not None:  # Return cached connection if available
            return self._connection  # Reuse existing connection

        self._ensure_directory_exists()  # Create database directory if needed
        logger.debug("opening_database_connection", path=self._db_path)  # Log connection attempt

        # Create new connection with row factory for dict-like access
        self._connection = sqlite3.connect(  # Open SQLite connection
            self._db_path,  # Database file path or :memory:
            check_same_thread=False,  # Allow multi-thread access (protected by app-level locks)
        )
        self._connection.row_factory = sqlite3.Row  # Enable column-name access on rows
        self._connection.execute(f"PRAGMA journal_mode={self._journal_mode}")  # Set journal mode
        self._connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")  # Set busy timeout
        self._connection.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints

        logger.info("database_connection_established", path=self._db_path)  # Log success
        return self._connection  # Return the new connection

    def initialize_schema(self) -> None:
        """Create all database tables and indexes if they do not exist."""
        connection = self.get_connection()  # Get or create the connection
        logger.info("initializing_database_schema")  # Log schema initialization start
        connection.executescript(SCHEMA_SQL)  # Execute the full schema DDL
        connection.commit()  # Commit the schema changes
        logger.info("database_schema_initialized")  # Log schema initialization complete

    def close(self) -> None:
        """Close the database connection and release resources."""
        if self._connection is not None:  # Only close if connection exists
            logger.info("closing_database_connection")  # Log connection closure
            self._connection.close()  # Close the SQLite connection
            self._connection = None  # Clear the cached reference

    def execute_query(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a SELECT query and return all result rows.

        Provides a safe wrapper around cursor execute with parameterized queries.
        """
        connection = self.get_connection()  # Get the database connection
        cursor = connection.execute(query, params)  # Execute parameterized query
        return cursor.fetchall()  # Return all result rows

    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return rows affected.

        Automatically commits the transaction after execution.
        """
        connection = self.get_connection()  # Get the database connection
        cursor = connection.execute(query, params)  # Execute parameterized write query
        connection.commit()  # Commit the transaction
        return cursor.rowcount  # Return number of affected rows

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute a batch write operation with multiple parameter sets.

        Commits all operations in a single transaction for efficiency.
        """
        connection = self.get_connection()  # Get the database connection
        cursor = connection.executemany(query, params_list)  # Execute batch operation
        connection.commit()  # Commit the batch transaction
        return cursor.rowcount  # Return total affected rows
