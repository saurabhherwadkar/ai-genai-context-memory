"""Synchronization between the NetworkX in-memory graph and SQLite persistence."""

import structlog  # Structured logging

from memory_graph.graph.graph_manager import GraphManager  # Graph lifecycle manager
from memory_graph.models.memory_edge import MemoryEdge  # Edge data model
from memory_graph.models.memory_node import MemoryNode  # Node data model
from memory_graph.persistence.edge_repository import EdgeRepository  # Edge persistence
from memory_graph.persistence.node_repository import NodeRepository  # Node persistence

# Module logger for sync operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class GraphSync:
    """Handles bidirectional synchronization between NetworkX and SQLite.

    Responsible for loading the full graph from persistence at startup
    and providing write-through operations that keep both stores consistent.
    """

    def __init__(
        self,
        graph_manager: GraphManager,
        node_repository: NodeRepository,
        edge_repository: EdgeRepository,
    ) -> None:
        """Initialize the sync service with graph and persistence references."""
        self._graph_manager = graph_manager  # In-memory graph reference
        self._node_repo = node_repository  # Node persistence layer
        self._edge_repo = edge_repository  # Edge persistence layer

    async def load_full_graph(self) -> None:
        """Load the entire graph from SQLite into NetworkX at startup.

        Reads all active nodes and all edges, reconstructing the full
        in-memory graph representation.
        """
        logger.info("loading_full_graph_from_persistence")  # Log load start

        # Load all active nodes page by page to handle large datasets
        page = 0  # Start at first page
        page_size = 500  # Process nodes in batches
        total_nodes_loaded = 0  # Counter for loaded nodes

        while True:  # Paginate through all active nodes
            nodes = self._node_repo.get_all_active(page=page, page_size=page_size)  # Fetch page
            if not nodes:  # No more nodes to process
                break  # Exit pagination loop

            for node in nodes:  # Add each node to the graph
                await self._graph_manager.add_node(node)  # Add node with attributes
                total_nodes_loaded += 1  # Increment counter

            page += 1  # Move to next page

        logger.info("nodes_loaded_to_graph", count=total_nodes_loaded)  # Log node count

        # Load all edges and add to the graph
        all_edges = self._edge_repo.get_all()  # Fetch all edges from SQLite
        edges_loaded = 0  # Counter for loaded edges

        for edge in all_edges:  # Add each edge to the graph
            success = await self._graph_manager.add_edge(edge)  # Attempt to add edge
            if success:  # Edge successfully added
                edges_loaded += 1  # Increment counter
            else:  # Edge could not be added (missing endpoint or limit reached)
                logger.warning(  # Log failed edge addition
                    "edge_load_failed",
                    edge_id=edge.id,
                    source=edge.source_node_id,
                    target=edge.target_node_id,
                )

        logger.info(  # Log final load statistics
            "full_graph_loaded",
            nodes=total_nodes_loaded,
            edges=edges_loaded,
        )

    async def sync_add_node(self, node: MemoryNode) -> bool:
        """Add a node to both SQLite and the in-memory graph atomically.

        Persists to SQLite first, then adds to NetworkX. Returns False
        if either operation fails.
        """
        logger.debug("sync_adding_node", node_id=node.id)  # Log sync operation

        # Persist to SQLite first (source of truth)
        self._node_repo.create(node)  # Write to database

        # Add to in-memory graph
        success = await self._graph_manager.add_node(node)  # Add to NetworkX
        if not success:  # Graph rejected the node (capacity limit)
            logger.warning("node_added_to_db_but_not_graph", node_id=node.id)  # Log inconsistency
            return False  # Indicate partial failure

        return True  # Both stores updated successfully

    async def sync_add_edge(self, edge: MemoryEdge) -> bool:
        """Add an edge to both SQLite and the in-memory graph atomically.

        Persists to SQLite first, then adds to NetworkX.
        """
        logger.debug("sync_adding_edge", edge_id=edge.id)  # Log sync operation

        # Persist to SQLite first
        self._edge_repo.create(edge)  # Write to database

        # Add to in-memory graph
        success = await self._graph_manager.add_edge(edge)  # Add to NetworkX
        if not success:  # Graph rejected the edge
            logger.warning("edge_added_to_db_but_not_graph", edge_id=edge.id)  # Log
            return False  # Indicate partial failure

        return True  # Both stores updated

    async def sync_remove_node(self, node_id: str) -> bool:
        """Remove a node from both SQLite and the in-memory graph.

        Cascading deletes handle associated edges in both stores.
        """
        logger.debug("sync_removing_node", node_id=node_id)  # Log removal

        # Remove from in-memory graph first (includes edge cleanup)
        graph_removed = await self._graph_manager.remove_node(node_id)  # Remove from NetworkX

        # Remove from SQLite (foreign key cascades handle edges)
        db_removed = self._node_repo.delete(node_id)  # Delete from database

        if not graph_removed and not db_removed:  # Neither store had the node
            return False  # Node did not exist

        return True  # Successfully removed from at least one store

    async def sync_update_node(self, node: MemoryNode) -> bool:
        """Update a node in both SQLite and the in-memory graph.

        Writes updated fields to both stores.
        """
        logger.debug("sync_updating_node", node_id=node.id)  # Log update

        # Update in SQLite
        self._node_repo.update(node)  # Persist changes to database

        # Update attributes in the graph
        attributes = self._graph_manager._node_to_attributes(node)  # Convert to attribute dict
        success = await self._graph_manager.update_node_attributes(node.id, attributes)  # Update graph

        if not success:  # Node not found in graph
            logger.warning("node_updated_in_db_but_not_graph", node_id=node.id)  # Log
            return False  # Partial update

        return True  # Both stores updated
