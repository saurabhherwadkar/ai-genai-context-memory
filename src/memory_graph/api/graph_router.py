"""Graph query endpoints for traversal and relationship discovery."""

import structlog  # Structured logging
from fastapi import APIRouter  # Router for endpoint grouping

from memory_graph.graph.graph_query import GraphQuery  # Graph traversal operations
from memory_graph.middleware.error_handler import NotFoundError  # Error types
from memory_graph.models.api_requests import GraphQueryRequest  # Request schema
from memory_graph.models.api_responses import GraphQueryResponse, MemoryEdgeResponse, MemoryNodeResponse  # Responses
from memory_graph.persistence.edge_repository import EdgeRepository  # Edge data access
from memory_graph.persistence.node_repository import NodeRepository  # Node data access
from memory_graph.security.input_validator import InputValidator  # Input sanitization

# Module logger for graph API operations
logger = structlog.get_logger(__name__)  # Named logger for this module


def create_graph_router(
    graph_query: GraphQuery,
    node_repository: NodeRepository,
    edge_repository: EdgeRepository,
    input_validator: InputValidator,
) -> APIRouter:
    """Create and configure the graph query router with dependencies."""

    router = APIRouter(prefix="/api/v1/graph", tags=["graph"])  # Graph endpoints group

    @router.post("/query", response_model=GraphQueryResponse)  # Graph traversal query
    async def query_related_memories(request: GraphQueryRequest) -> GraphQueryResponse:
        """Traverse the memory graph to find related memories.

        Performs BFS from the starting node, optionally filtering by
        edge types, node types, and confidence threshold.
        """
        logger.info(  # Log the query
            "graph_query_requested",
            node_id=request.node_id,
            max_depth=request.max_depth,
        )

        # Validate the starting node ID
        validated_id = input_validator.validate_node_id(request.node_id)  # Sanitize

        # Verify the starting node exists
        start_node = node_repository.get_by_id(validated_id)  # Fetch start node
        if start_node is None:  # Start node not found
            raise NotFoundError("MemoryNode", validated_id)  # Return 404

        # Execute the graph traversal
        related_ids = graph_query.find_related_memories(  # Perform BFS
            node_id=validated_id,  # Starting node
            max_depth=request.max_depth,  # Traversal depth
            edge_types=request.edge_types,  # Optional edge filter
            node_types=request.node_types,  # Optional node filter
            min_confidence=request.min_confidence,  # Confidence threshold
        )

        # Fetch full node data for all discovered nodes
        nodes: list[MemoryNodeResponse] = []  # Accumulate node responses
        for related_id in related_ids:  # Process each discovered node
            node = node_repository.get_by_id(related_id)  # Fetch full data
            if node:  # Node exists in database
                nodes.append(_node_to_response(node))  # Add to results

        # Fetch edges connecting the discovered nodes
        all_node_ids = {validated_id} | set(related_ids)  # All relevant node IDs
        edges: list[MemoryEdgeResponse] = []  # Accumulate edge responses
        for node_id in all_node_ids:  # Check edges for each node
            node_edges = edge_repository.get_edges_from_node(node_id)  # Outgoing edges
            for edge in node_edges:  # Filter to edges within the result set
                if edge.target_node_id in all_node_ids:  # Target is in result set
                    edges.append(_edge_to_response(edge))  # Include this edge

        logger.info(  # Log query results
            "graph_query_complete",
            nodes_found=len(nodes),
            edges_found=len(edges),
        )

        return GraphQueryResponse(  # Build the response
            nodes=nodes,  # Discovered nodes
            edges=edges,  # Connecting edges
            total_nodes=len(nodes),  # Node count
            total_edges=len(edges),  # Edge count
        )

    @router.get("/contradictions/{node_id}")  # Find contradictions
    async def find_contradictions(node_id: str) -> list[MemoryNodeResponse]:
        """Find all memories that contradict the given node."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate ID
        contradiction_ids = graph_query.find_contradictions(validated_id)  # Find contradictions

        results: list[MemoryNodeResponse] = []  # Accumulate results
        for cid in contradiction_ids:  # Fetch each contradicting node
            node = node_repository.get_by_id(cid)  # Get full data
            if node:  # Node exists
                results.append(_node_to_response(node))  # Add to results

        return results  # Return contradicting nodes

    @router.get("/supporting/{node_id}")  # Find supporting evidence
    async def find_supporting_evidence(node_id: str) -> list[MemoryNodeResponse]:
        """Find all memories that support the given node."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate ID
        support_ids = graph_query.find_supporting_evidence(validated_id)  # Find supporters

        results: list[MemoryNodeResponse] = []  # Accumulate results
        for sid in support_ids:  # Fetch each supporting node
            node = node_repository.get_by_id(sid)  # Get full data
            if node:  # Node exists
                results.append(_node_to_response(node))  # Add to results

        return results  # Return supporting nodes

    @router.get("/stats")  # Graph statistics
    async def get_graph_stats() -> dict:
        """Get statistical overview of the memory graph."""
        from memory_graph.graph.graph_manager import GraphManager  # Avoid circular import

        centralities = graph_query.get_all_centralities()  # Compute centralities
        components = graph_query.get_connected_components()  # Find components

        return {  # Return statistics summary
            "connected_components": len(components),  # Number of disconnected groups
            "avg_centrality": (  # Average node connectivity
                sum(centralities.values()) / len(centralities)
                if centralities
                else 0.0
            ),
            "max_centrality_node": (  # Most connected node
                max(centralities, key=centralities.get)
                if centralities
                else None
            ),
        }

    return router  # Return configured router


def _node_to_response(node) -> MemoryNodeResponse:
    """Convert a MemoryNode to response DTO."""
    return MemoryNodeResponse(
        id=node.id,
        node_type=node.node_type,
        content=node.content,
        summary=node.summary,
        confidence=node.confidence,
        emotional_valence=node.emotional_valence,
        intensity=node.intensity,
        source_conversation_id=node.source_conversation_id,
        created_at=node.created_at,
        updated_at=node.updated_at,
        last_accessed_at=node.last_accessed_at,
        access_count=node.access_count,
        tags=node.tags,
        is_active=node.is_active,
        custom_type_name=node.custom_type_name,
    )


def _edge_to_response(edge) -> MemoryEdgeResponse:
    """Convert a MemoryEdge to response DTO."""
    return MemoryEdgeResponse(
        id=edge.id,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        edge_type=edge.edge_type,
        weight=edge.weight,
        description=edge.description,
        created_at=edge.created_at,
    )
