"""Graph traversal and query operations for the memory cognitive graph."""

from collections import deque  # Efficient FIFO queue for BFS traversal
from typing import Any  # Generic type for attribute values

import networkx as nx  # Graph library for algorithms
import structlog  # Structured logging

from memory_graph.graph.graph_manager import GraphManager  # Graph lifecycle manager
from memory_graph.models.memory_types import EdgeType, NodeType  # Classification enums

# Module logger for graph query operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class GraphQuery:
    """Provides complex traversal and query operations on the memory graph.

    Supports BFS traversal, filtering by type, path finding, and
    centrality calculations for memory ranking.
    """

    def __init__(self, graph_manager: GraphManager) -> None:
        """Initialize the query service with a reference to the graph manager."""
        self._manager = graph_manager  # Store reference to graph manager

    def find_related_memories(
        self,
        node_id: str,
        max_depth: int = 2,
        edge_types: list[EdgeType] | None = None,
        node_types: list[NodeType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[str]:
        """Find related memory node IDs via BFS traversal from a starting node.

        Traverses outward up to max_depth hops, optionally filtering by
        edge types and result node types.
        """
        graph = self._manager.graph  # Get the underlying NetworkX graph

        if not graph.has_node(node_id):  # Verify starting node exists
            logger.debug("start_node_not_found", node_id=node_id)  # Log miss
            return []  # Return empty for non-existent start node

        # Convert edge type filter to a set of string values for fast lookup
        allowed_edge_types: set[str] | None = None  # None means allow all
        if edge_types:  # If filter specified
            allowed_edge_types = {et.value for et in edge_types}  # Convert to value set

        # Convert node type filter to a set of string values
        allowed_node_types: set[str] | None = None  # None means allow all
        if node_types:  # If filter specified
            allowed_node_types = {nt.value for nt in node_types}  # Convert to value set

        # BFS traversal with depth tracking
        visited: set[str] = {node_id}  # Track visited nodes, include start
        result_ids: list[str] = []  # Accumulate matching node IDs
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])  # BFS queue with depth

        while queue:  # Continue until all reachable nodes processed
            current_id, depth = queue.popleft()  # Dequeue next node and its depth

            if depth >= max_depth:  # Stop expanding beyond max depth
                continue  # Skip this node's neighbors

            # Iterate over outgoing edges from current node
            for neighbor_id in graph.successors(current_id):  # Get successor nodes
                if neighbor_id in visited:  # Skip already-visited nodes
                    continue  # Avoid cycles and redundant processing

                # Check edge type filter
                edge_data = graph.edges[current_id, neighbor_id]  # Get edge attributes
                if allowed_edge_types:  # If edge type filter is active
                    edge_type_value = edge_data.get("edge_type", "")  # Get edge type
                    if edge_type_value not in allowed_edge_types:  # Not in allowed set
                        continue  # Skip this edge

                # Check node type filter on the neighbor
                neighbor_attrs = graph.nodes[neighbor_id]  # Get neighbor attributes
                if allowed_node_types:  # If node type filter is active
                    neighbor_type = neighbor_attrs.get("node_type", "")  # Get node type
                    if neighbor_type not in allowed_node_types:  # Not in allowed set
                        visited.add(neighbor_id)  # Mark as visited to avoid revisiting
                        continue  # Skip this neighbor

                # Check confidence threshold
                neighbor_confidence = neighbor_attrs.get("confidence", 0.0)  # Get confidence
                if neighbor_confidence < min_confidence:  # Below threshold
                    visited.add(neighbor_id)  # Mark visited
                    continue  # Skip low-confidence nodes

                # Check active status
                if not neighbor_attrs.get("is_active", True):  # Inactive node
                    visited.add(neighbor_id)  # Mark visited
                    continue  # Skip inactive nodes

                visited.add(neighbor_id)  # Mark as visited
                result_ids.append(neighbor_id)  # Add to results
                queue.append((neighbor_id, depth + 1))  # Enqueue for further traversal

            # Also check incoming edges (predecessors) for bidirectional discovery
            for predecessor_id in graph.predecessors(current_id):  # Get predecessor nodes
                if predecessor_id in visited:  # Skip already visited
                    continue  # Avoid redundant processing

                # Check edge type filter on incoming edge
                edge_data = graph.edges[predecessor_id, current_id]  # Get edge attributes
                if allowed_edge_types:  # If filter active
                    edge_type_value = edge_data.get("edge_type", "")  # Get type
                    if edge_type_value not in allowed_edge_types:  # Not allowed
                        continue  # Skip

                # Check node type filter
                predecessor_attrs = graph.nodes[predecessor_id]  # Get attributes
                if allowed_node_types:  # If filter active
                    predecessor_type = predecessor_attrs.get("node_type", "")  # Get type
                    if predecessor_type not in allowed_node_types:  # Not allowed
                        visited.add(predecessor_id)  # Mark visited
                        continue  # Skip

                # Check confidence
                predecessor_confidence = predecessor_attrs.get("confidence", 0.0)  # Get confidence
                if predecessor_confidence < min_confidence:  # Below threshold
                    visited.add(predecessor_id)  # Mark visited
                    continue  # Skip

                # Check active status
                if not predecessor_attrs.get("is_active", True):  # Inactive
                    visited.add(predecessor_id)  # Mark visited
                    continue  # Skip

                visited.add(predecessor_id)  # Mark visited
                result_ids.append(predecessor_id)  # Add to results
                queue.append((predecessor_id, depth + 1))  # Enqueue for traversal

        logger.debug("related_memories_found", start=node_id, count=len(result_ids))  # Log results
        return result_ids  # Return all discovered related node IDs

    def find_contradictions(self, node_id: str) -> list[str]:
        """Find all memory nodes that contradict the given node.

        Searches for nodes connected via CONTRADICTS or CONFLICTS_WITH edges.
        """
        contradiction_types = {EdgeType.CONTRADICTS.value, EdgeType.CONFLICTS_WITH.value}  # Target types
        return self._find_by_edge_type(node_id, contradiction_types)  # Delegate to helper

    def find_supporting_evidence(self, node_id: str) -> list[str]:
        """Find all memory nodes that support the given node.

        Searches for nodes connected via SUPPORTS or DEPENDS_ON edges.
        """
        support_types = {EdgeType.SUPPORTS.value, EdgeType.DEPENDS_ON.value}  # Target types
        return self._find_by_edge_type(node_id, support_types)  # Delegate to helper

    def get_node_centrality(self, node_id: str) -> float:
        """Calculate the degree centrality of a specific node.

        Returns a value between 0 and 1 indicating how well-connected the node is.
        """
        graph = self._manager.graph  # Get the graph instance

        if not graph.has_node(node_id):  # Verify node exists
            return 0.0  # Non-existent nodes have zero centrality

        if graph.number_of_nodes() <= 1:  # Singleton graph edge case
            return 0.0  # No meaningful centrality

        # Calculate degree centrality for all nodes
        centrality_map = nx.degree_centrality(graph)  # NetworkX centrality calculation
        return centrality_map.get(node_id, 0.0)  # Return this node's centrality

    def get_all_centralities(self) -> dict[str, float]:
        """Calculate degree centrality for all nodes in the graph.

        Returns a mapping of node_id to centrality value.
        """
        graph = self._manager.graph  # Get the graph instance

        if graph.number_of_nodes() == 0:  # Empty graph edge case
            return {}  # Return empty dict

        return nx.degree_centrality(graph)  # Return full centrality map

    def find_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Find the shortest path between two nodes in the graph.

        Returns the list of node IDs along the path, or None if no path exists.
        """
        graph = self._manager.graph  # Get the graph instance

        if not graph.has_node(source_id) or not graph.has_node(target_id):  # Verify both exist
            return None  # Cannot find path for non-existent nodes

        try:  # Attempt to find shortest path
            path = nx.shortest_path(graph, source_id, target_id)  # Compute shortest path
            return list(path)  # Return as list of node IDs
        except nx.NetworkXNoPath:  # No path exists between nodes
            return None  # Return None to indicate disconnected

    def get_connected_components(self) -> list[set[str]]:
        """Get all weakly connected components in the graph.

        Returns a list of sets, each containing node IDs in one component.
        """
        graph = self._manager.graph  # Get the graph instance
        # Use weakly connected components for directed graphs
        components = list(nx.weakly_connected_components(graph))  # Compute components
        return [set(component) for component in components]  # Convert to list of sets

    def get_subgraph_by_type(self, node_type: NodeType) -> list[str]:
        """Get all node IDs of a specific type in the graph."""
        graph = self._manager.graph  # Get the graph instance
        matching_ids: list[str] = []  # Accumulate matching node IDs

        for node_id, attrs in graph.nodes(data=True):  # Iterate all nodes with attributes
            if attrs.get("node_type") == node_type.value:  # Check type match
                matching_ids.append(node_id)  # Add to results

        return matching_ids  # Return all matching node IDs

    def _find_by_edge_type(self, node_id: str, edge_type_values: set[str]) -> list[str]:
        """Helper to find nodes connected via specific edge types.

        Checks both outgoing and incoming edges for the given types.
        """
        graph = self._manager.graph  # Get the graph instance

        if not graph.has_node(node_id):  # Verify node exists
            return []  # Return empty for non-existent nodes

        related_ids: list[str] = []  # Accumulate related node IDs

        # Check outgoing edges
        for neighbor_id in graph.successors(node_id):  # Iterate successors
            edge_data = graph.edges[node_id, neighbor_id]  # Get edge attributes
            if edge_data.get("edge_type") in edge_type_values:  # Check type match
                related_ids.append(neighbor_id)  # Add matching neighbor

        # Check incoming edges
        for predecessor_id in graph.predecessors(node_id):  # Iterate predecessors
            edge_data = graph.edges[predecessor_id, node_id]  # Get edge attributes
            if edge_data.get("edge_type") in edge_type_values:  # Check type match
                related_ids.append(predecessor_id)  # Add matching predecessor

        return related_ids  # Return all related node IDs
