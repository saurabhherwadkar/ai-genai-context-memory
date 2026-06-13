"""CRUD endpoints for memory node and edge operations."""

from datetime import datetime, timezone  # Timestamp handling

import structlog  # Structured logging
from fastapi import APIRouter  # Router for endpoint grouping

from memory_graph.embeddings.embedding_service import EmbeddingService  # Embedding generation
from memory_graph.embeddings.similarity_search import SimilaritySearch  # Search index updates
from memory_graph.graph.graph_sync import GraphSync  # Synchronized persistence
from memory_graph.middleware.error_handler import NotFoundError, ValidationError  # Error types
from memory_graph.models.api_requests import (  # Request schemas
    CreateMemoryEdgeRequest,
    CreateMemoryNodeRequest,
    UpdateMemoryNodeRequest,
)
from memory_graph.models.api_responses import (  # Response schemas
    MemoryEdgeResponse,
    MemoryNodeResponse,
    PaginatedResponse,
)
from memory_graph.models.memory_edge import MemoryEdge  # Edge data model
from memory_graph.models.memory_node import MemoryNode  # Node data model
from memory_graph.persistence.edge_repository import EdgeRepository  # Edge persistence
from memory_graph.persistence.embedding_repository import EmbeddingRepository  # Embedding storage
from memory_graph.persistence.node_repository import NodeRepository  # Node persistence
from memory_graph.security.input_validator import InputValidator  # Input sanitization

# Module logger for memory API operations
logger = structlog.get_logger(__name__)  # Named logger for this module


def create_memory_router(
    node_repository: NodeRepository,
    edge_repository: EdgeRepository,
    embedding_repository: EmbeddingRepository,
    embedding_service: EmbeddingService,
    similarity_search: SimilaritySearch,
    graph_sync: GraphSync,
    input_validator: InputValidator,
) -> APIRouter:
    """Create and configure the memory CRUD router with all dependencies."""

    router = APIRouter(prefix="/api/v1/memories", tags=["memories"])  # Memory endpoints group

    @router.post("/nodes", response_model=MemoryNodeResponse, status_code=201)  # Create node
    async def create_memory_node(request: CreateMemoryNodeRequest) -> MemoryNodeResponse:
        """Create a new memory node in the cognitive graph.

        Validates input, persists to both SQLite and NetworkX,
        generates and stores the embedding vector.
        """
        logger.info("creating_memory_node", node_type=request.node_type.value)  # Log creation

        # Validate and sanitize input content
        validated_content = input_validator.validate_content(request.content)  # Sanitize content
        validated_tags = input_validator.validate_tags(request.tags)  # Validate tags

        # Construct the memory node model
        node = MemoryNode(  # Create new node with validated data
            node_type=request.node_type,  # Classification type
            content=validated_content,  # Sanitized content
            summary=request.summary,  # Brief summary
            confidence=request.confidence,  # Confidence score
            emotional_valence=request.emotional_valence,  # Emotional polarity
            intensity=request.intensity,  # Memory strength
            tags=validated_tags,  # Validated tags
            metadata=request.metadata,  # Extensible metadata
            custom_type_name=request.custom_type_name,  # Optional custom type name
        )

        # Persist node to both SQLite and graph
        success = await graph_sync.sync_add_node(node)  # Write-through to both stores
        if not success:  # Graph capacity reached
            raise ValidationError("Graph capacity limit reached")  # Return error

        # Generate and store the embedding vector
        embedding = embedding_service.encode(node.content)  # Encode content to vector
        embedding_repository.store(  # Persist embedding
            node_id=node.id,  # Reference to the node
            embedding=embedding,  # Vector data
            model_name=embedding_service._model_name,  # Model used
        )
        similarity_search.add_to_index(node.id, embedding)  # Update search index

        logger.info("memory_node_created_via_api", node_id=node.id)  # Log success
        return _node_to_response(node)  # Return response DTO

    @router.get("/nodes/{node_id}", response_model=MemoryNodeResponse)  # Get single node
    async def get_memory_node(node_id: str) -> MemoryNodeResponse:
        """Retrieve a specific memory node by its ID."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate the ID format
        node = node_repository.get_by_id(validated_id)  # Fetch from database

        if node is None:  # Node not found
            raise NotFoundError("MemoryNode", validated_id)  # Return 404

        # Record the access for ranking purposes
        node_repository.record_access(validated_id)  # Update access stats

        return _node_to_response(node)  # Return response DTO

    @router.get("/nodes", response_model=PaginatedResponse)  # List nodes
    async def list_memory_nodes(page: int = 0, page_size: int = 50) -> PaginatedResponse:
        """List all active memory nodes with pagination."""
        # Clamp page_size to reasonable limits
        page_size = min(max(page_size, 1), 100)  # Between 1 and 100

        nodes = node_repository.get_all_active(page=page, page_size=page_size)  # Fetch page
        total = node_repository.count_active()  # Get total count
        has_next = (page + 1) * page_size < total  # Check if more pages exist

        return PaginatedResponse(  # Build paginated response
            items=[_node_to_response(n) for n in nodes],  # Convert all nodes
            total=total,  # Total item count
            page=page,  # Current page number
            page_size=page_size,  # Page size used
            has_next=has_next,  # Whether more pages exist
        )

    @router.patch("/nodes/{node_id}", response_model=MemoryNodeResponse)  # Update node
    async def update_memory_node(
        node_id: str,
        request: UpdateMemoryNodeRequest,
    ) -> MemoryNodeResponse:
        """Update an existing memory node with partial data."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate ID
        node = node_repository.get_by_id(validated_id)  # Fetch existing node

        if node is None:  # Node not found
            raise NotFoundError("MemoryNode", validated_id)  # Return 404

        # Apply partial updates from the request
        if request.content is not None:  # Content update provided
            node.content = input_validator.validate_content(request.content)  # Validate and set
        if request.summary is not None:  # Summary update provided
            node.summary = request.summary  # Update summary
        if request.confidence is not None:  # Confidence update provided
            node.confidence = request.confidence  # Update confidence
        if request.emotional_valence is not None:  # Valence update provided
            node.emotional_valence = request.emotional_valence  # Update valence
        if request.intensity is not None:  # Intensity update provided
            node.intensity = request.intensity  # Update intensity
        if request.tags is not None:  # Tags update provided
            node.tags = input_validator.validate_tags(request.tags)  # Validate and set
        if request.metadata is not None:  # Metadata update provided
            node.metadata = request.metadata  # Update metadata
        if request.is_active is not None:  # Active status update provided
            node.is_active = request.is_active  # Update active flag

        # Persist the updates to both stores
        await graph_sync.sync_update_node(node)  # Write-through update

        # Re-generate embedding if content changed
        if request.content is not None:  # Content was updated
            embedding = embedding_service.encode(node.content)  # Re-encode
            embedding_repository.store(node.id, embedding, embedding_service._model_name)  # Update
            similarity_search.add_to_index(node.id, embedding)  # Update index

        logger.info("memory_node_updated_via_api", node_id=node.id)  # Log success
        return _node_to_response(node)  # Return updated response

    @router.delete("/nodes/{node_id}", status_code=204)  # Delete node
    async def delete_memory_node(node_id: str) -> None:
        """Delete a memory node and all its associated edges and embeddings."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate ID

        # Remove from all stores
        removed = await graph_sync.sync_remove_node(validated_id)  # Remove from graph + DB
        if not removed:  # Node did not exist
            raise NotFoundError("MemoryNode", validated_id)  # Return 404

        # Remove from embedding index
        similarity_search.remove_from_index(validated_id)  # Remove from search
        embedding_repository.delete(validated_id)  # Remove stored embedding

        logger.info("memory_node_deleted_via_api", node_id=validated_id)  # Log deletion

    @router.post("/edges", response_model=MemoryEdgeResponse, status_code=201)  # Create edge
    async def create_memory_edge(request: CreateMemoryEdgeRequest) -> MemoryEdgeResponse:
        """Create a new relationship edge between two memory nodes."""
        logger.info(  # Log edge creation
            "creating_memory_edge",
            source=request.source_node_id,
            target=request.target_node_id,
            edge_type=request.edge_type.value,
        )

        # Validate that both endpoint nodes exist
        source = node_repository.get_by_id(request.source_node_id)  # Check source
        if source is None:  # Source not found
            raise NotFoundError("MemoryNode", request.source_node_id)  # 404

        target = node_repository.get_by_id(request.target_node_id)  # Check target
        if target is None:  # Target not found
            raise NotFoundError("MemoryNode", request.target_node_id)  # 404

        # Construct the edge model
        edge = MemoryEdge(  # Create new edge
            source_node_id=request.source_node_id,  # Source reference
            target_node_id=request.target_node_id,  # Target reference
            edge_type=request.edge_type,  # Relationship type
            weight=request.weight,  # Relationship strength
            description=request.description,  # Optional description
            metadata=request.metadata,  # Extensible metadata
        )

        # Persist to both stores
        success = await graph_sync.sync_add_edge(edge)  # Write-through
        if not success:  # Edge limit reached
            raise ValidationError("Edge limit reached for source node")  # Error

        logger.info("memory_edge_created_via_api", edge_id=edge.id)  # Log success
        return _edge_to_response(edge)  # Return response DTO

    @router.get("/edges/{node_id}", response_model=list[MemoryEdgeResponse])  # Get edges
    async def get_node_edges(node_id: str) -> list[MemoryEdgeResponse]:
        """Get all edges connected to a specific node."""
        validated_id = input_validator.validate_node_id(node_id)  # Validate ID
        edges = edge_repository.get_edges_for_node(validated_id)  # Fetch all connected edges
        return [_edge_to_response(e) for e in edges]  # Convert to response DTOs

    @router.delete("/edges/{edge_id}", status_code=204)  # Delete edge
    async def delete_memory_edge(edge_id: str) -> None:
        """Delete a specific relationship edge by its ID."""
        deleted = edge_repository.delete(edge_id)  # Attempt deletion
        if not deleted:  # Edge not found
            raise NotFoundError("MemoryEdge", edge_id)  # Return 404

        logger.info("memory_edge_deleted_via_api", edge_id=edge_id)  # Log deletion

    return router  # Return the fully configured router


def _node_to_response(node: MemoryNode) -> MemoryNodeResponse:
    """Convert a MemoryNode model to an API response DTO."""
    return MemoryNodeResponse(  # Build response from model
        id=node.id,  # Node identifier
        node_type=node.node_type,  # Classification type
        content=node.content,  # Full content
        summary=node.summary,  # Summary text
        confidence=node.confidence,  # Confidence score
        emotional_valence=node.emotional_valence,  # Emotional polarity
        intensity=node.intensity,  # Memory strength
        source_conversation_id=node.source_conversation_id,  # Origin reference
        created_at=node.created_at,  # Creation timestamp
        updated_at=node.updated_at,  # Update timestamp
        last_accessed_at=node.last_accessed_at,  # Last access time
        access_count=node.access_count,  # Access counter
        tags=node.tags,  # Category tags
        is_active=node.is_active,  # Active status
        custom_type_name=node.custom_type_name,  # Custom type name
    )


def _edge_to_response(edge: MemoryEdge) -> MemoryEdgeResponse:
    """Convert a MemoryEdge model to an API response DTO."""
    return MemoryEdgeResponse(  # Build response from model
        id=edge.id,  # Edge identifier
        source_node_id=edge.source_node_id,  # Source reference
        target_node_id=edge.target_node_id,  # Target reference
        edge_type=edge.edge_type,  # Relationship type
        weight=edge.weight,  # Strength score
        description=edge.description,  # Optional description
        created_at=edge.created_at,  # Creation timestamp
    )
