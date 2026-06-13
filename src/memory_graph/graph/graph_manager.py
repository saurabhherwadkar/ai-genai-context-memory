"""NetworkX graph lifecycle manager for the in-memory cognitive graph."""

import asyncio  # Async lock for thread-safe graph mutations
from typing import Any  # Generic type for node attribute dictionaries

import networkx as nx  # Graph library for in-memory graph operations
import structlog  # Structured logging

from memory_graph.config.settings import GraphSettings  # Graph configuration
from memory_graph.models.memory_edge import MemoryEdge  # Edge data model
from memory_graph.models.memory_node import MemoryNode  # Node data model

# Module logger for graph manager operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class GraphManager:
    """Manages the NetworkX directed graph lifecycle and mutation operations.

    Provides thread-safe operations for adding, removing, and querying
    nodes and edges in the in-memory graph representation.
    """

    def __init__(self, settings: GraphSettings) -> None:
        """Initialize the graph manager with configuration settings."""
        self._graph = nx.DiGraph()  # Create an empty directed graph
        self._settings = settings  # Store graph configuration
        self._lock = asyncio.Lock()  # Async lock for thread-safe mutations
        logger.info("graph_manager_initialized", max_nodes=settings.max_nodes)  # Log initialization

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX directed graph instance."""
        return self._graph  # Direct access for read operations

    @property
    def node_count(self) -> int:
        """Get the current number of nodes in the graph."""
        return self._graph.number_of_nodes()  # Delegate to NetworkX

    @property
    def edge_count(self) -> int:
        """Get the current number of edges in the graph."""
        return self._graph.number_of_edges()  # Delegate to NetworkX

    async def add_node(self, node: MemoryNode) -> bool:
        """Add a memory node to the graph with all its attributes.

        Returns False if the graph has reached its maximum node capacity.
        """
        async with self._lock:  # Acquire lock for thread-safe mutation
            if self._graph.number_of_nodes() >= self._settings.max_nodes:  # Check capacity
                logger.warning("graph_node_limit_reached", max=self._settings.max_nodes)  # Log limit
                return False  # Reject addition due to capacity

            # Build the attribute dictionary for the graph node
            node_attributes = self._node_to_attributes(node)  # Convert model to attribute dict
            self._graph.add_node(node.id, **node_attributes)  # Add node with attributes
            logger.debug("node_added_to_graph", node_id=node.id)  # Log addition
            return True  # Successful addition

    async def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its incident edges from the graph.

        Returns False if the node does not exist in the graph.
        """
        async with self._lock:  # Acquire lock for thread-safe mutation
            if not self._graph.has_node(node_id):  # Check node existence
                logger.debug("node_not_in_graph", node_id=node_id)  # Log miss
                return False  # Node not found

            self._graph.remove_node(node_id)  # Remove node and all incident edges
            logger.debug("node_removed_from_graph", node_id=node_id)  # Log removal
            return True  # Successful removal

    async def add_edge(self, edge: MemoryEdge) -> bool:
        """Add a directed edge between two nodes in the graph.

        Returns False if either endpoint node does not exist in the graph.
        """
        async with self._lock:  # Acquire lock for thread-safe mutation
            if not self._graph.has_node(edge.source_node_id):  # Verify source exists
                logger.warning("edge_source_not_found", source=edge.source_node_id)  # Log error
                return False  # Cannot create edge without source

            if not self._graph.has_node(edge.target_node_id):  # Verify target exists
                logger.warning("edge_target_not_found", target=edge.target_node_id)  # Log error
                return False  # Cannot create edge without target

            # Check edge fan-out limit for the source node
            out_degree = self._graph.out_degree(edge.source_node_id)  # Count existing outgoing edges
            if out_degree >= self._settings.max_edges_per_node:  # Check limit
                logger.warning("edge_limit_reached", node_id=edge.source_node_id)  # Log limit
                return False  # Reject due to fan-out limit

            # Build edge attribute dictionary
            edge_attributes = self._edge_to_attributes(edge)  # Convert model to attributes
            self._graph.add_edge(  # Add directed edge with attributes
                edge.source_node_id,  # Source node
                edge.target_node_id,  # Target node
                key=edge.id,  # Unique edge identifier
                **edge_attributes,  # Edge attributes
            )
            logger.debug("edge_added_to_graph", edge_id=edge.id)  # Log addition
            return True  # Successful addition

    async def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove a directed edge between two nodes.

        Returns False if the edge does not exist.
        """
        async with self._lock:  # Acquire lock for thread-safe mutation
            if not self._graph.has_edge(source_id, target_id):  # Check edge existence
                return False  # Edge not found

            self._graph.remove_edge(source_id, target_id)  # Remove the edge
            logger.debug("edge_removed_from_graph", source=source_id, target=target_id)  # Log
            return True  # Successful removal

    def has_node(self, node_id: str) -> bool:
        """Check whether a node exists in the graph."""
        return self._graph.has_node(node_id)  # Delegate to NetworkX

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check whether a directed edge exists between two nodes."""
        return self._graph.has_edge(source_id, target_id)  # Delegate to NetworkX

    def get_node_attributes(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve all attributes for a specific node.

        Returns None if the node does not exist.
        """
        if not self._graph.has_node(node_id):  # Check existence first
            return None  # Node not found

        return dict(self._graph.nodes[node_id])  # Return copy of attributes

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all directly connected successor node IDs."""
        if not self._graph.has_node(node_id):  # Check existence
            return []  # Return empty for non-existent nodes

        return list(self._graph.successors(node_id))  # Return outgoing neighbor IDs

    def get_predecessors(self, node_id: str) -> list[str]:
        """Get all directly connected predecessor node IDs."""
        if not self._graph.has_node(node_id):  # Check existence
            return []  # Return empty for non-existent nodes

        return list(self._graph.predecessors(node_id))  # Return incoming neighbor IDs

    def get_all_node_ids(self) -> list[str]:
        """Get a list of all node IDs currently in the graph."""
        return list(self._graph.nodes())  # Return all node identifiers

    async def update_node_attributes(self, node_id: str, attributes: dict[str, Any]) -> bool:
        """Update specific attributes on an existing node.

        Returns False if the node does not exist.
        """
        async with self._lock:  # Acquire lock for thread-safe mutation
            if not self._graph.has_node(node_id):  # Check existence
                return False  # Node not found

            for key, value in attributes.items():  # Update each attribute
                self._graph.nodes[node_id][key] = value  # Set attribute value
            return True  # Successful update

    async def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        async with self._lock:  # Acquire lock for thread-safe mutation
            self._graph.clear()  # Remove everything from the graph
            logger.info("graph_cleared")  # Log the clear operation

    def _node_to_attributes(self, node: MemoryNode) -> dict[str, Any]:
        """Convert a MemoryNode model to a flat attribute dictionary for NetworkX."""
        return {
            "node_type": node.node_type.value,  # Store enum as string value
            "content": node.content,  # Full text content
            "summary": node.summary,  # Brief summary
            "confidence": node.confidence,  # Confidence score
            "emotional_valence": node.emotional_valence.value,  # Enum as string
            "intensity": node.intensity,  # Intensity score
            "created_at": node.created_at.isoformat(),  # ISO timestamp string
            "updated_at": node.updated_at.isoformat(),  # ISO timestamp string
            "access_count": node.access_count,  # Access counter
            "is_active": node.is_active,  # Active flag
            "tags": node.tags,  # Tag list (stored as-is in NetworkX)
        }

    def _edge_to_attributes(self, edge: MemoryEdge) -> dict[str, Any]:
        """Convert a MemoryEdge model to a flat attribute dictionary for NetworkX."""
        return {
            "edge_id": edge.id,  # Unique edge identifier
            "edge_type": edge.edge_type.value,  # Store enum as string value
            "weight": edge.weight,  # Relationship strength
            "description": edge.description,  # Optional description
            "created_at": edge.created_at.isoformat(),  # ISO timestamp string
        }
