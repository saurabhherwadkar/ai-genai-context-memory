"""Response data transfer objects for the REST API endpoints."""

from datetime import datetime  # Date/time type for timestamp fields

from pydantic import BaseModel, Field  # Validation framework for response schemas

from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums


class MemoryNodeResponse(BaseModel):
    """Response body representing a single memory node."""

    id: str = Field(description="Unique node identifier")  # UUID string
    node_type: NodeType = Field(description="Cognitive category")  # Memory classification
    content: str = Field(description="Full text content")  # Memory content
    summary: str = Field(description="One-line summary")  # Brief summary
    confidence: float = Field(description="Confidence score")  # Certainty level
    emotional_valence: EmotionValence = Field(description="Emotional polarity")  # Emotional tone
    intensity: float = Field(description="Memory intensity")  # Strength score
    source_conversation_id: str | None = Field(description="Source conversation")  # Origin reference
    created_at: datetime = Field(description="Creation timestamp")  # When created
    updated_at: datetime = Field(description="Last update timestamp")  # When modified
    last_accessed_at: datetime | None = Field(description="Last access timestamp")  # Last retrieval
    access_count: int = Field(description="Access count")  # Retrieval counter
    tags: list[str] = Field(description="Category tags")  # Filtering labels
    is_active: bool = Field(description="Active status")  # Soft delete flag
    custom_type_name: str | None = Field(description="Custom type name")  # For CUSTOM types


class MemoryEdgeResponse(BaseModel):
    """Response body representing a single relationship edge."""

    id: str = Field(description="Unique edge identifier")  # UUID string
    source_node_id: str = Field(description="Source node ID")  # Origin node reference
    target_node_id: str = Field(description="Target node ID")  # Destination node reference
    edge_type: EdgeType = Field(description="Relationship type")  # Edge classification
    weight: float = Field(description="Relationship strength")  # Weight score
    description: str | None = Field(description="Edge description")  # Optional explanation
    created_at: datetime = Field(description="Creation timestamp")  # When established


class GraphQueryResponse(BaseModel):
    """Response body for graph traversal queries."""

    nodes: list[MemoryNodeResponse] = Field(description="Discovered nodes")  # Traversal result nodes
    edges: list[MemoryEdgeResponse] = Field(description="Connecting edges")  # Traversal result edges
    total_nodes: int = Field(description="Total nodes found")  # Count of discovered nodes
    total_edges: int = Field(description="Total edges found")  # Count of connecting edges


class ContextResponse(BaseModel):
    """Response body containing assembled memory context for LLM injection."""

    context_text: str = Field(description="Formatted memory context")  # Assembled text block
    token_count: int = Field(description="Estimated token count")  # Token usage estimate
    memories_included: int = Field(description="Number of memories included")  # Memory count
    format: str = Field(description="Output format used")  # Format identifier


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str = Field(description="Service health status")  # healthy/degraded/unhealthy
    version: str = Field(description="Application version")  # Package version string
    node_count: int = Field(description="Total memory nodes")  # Graph size indicator
    edge_count: int = Field(description="Total memory edges")  # Relationship count


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""

    items: list[MemoryNodeResponse] = Field(description="Page items")  # Current page of results
    total: int = Field(description="Total item count")  # Total items across all pages
    page: int = Field(description="Current page number")  # Zero-indexed page number
    page_size: int = Field(description="Items per page")  # Page size limit
    has_next: bool = Field(description="Whether more pages exist")  # Pagination indicator


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(description="Error type identifier")  # Machine-readable error code
    message: str = Field(description="Human-readable error message")  # Descriptive message
    details: dict | None = Field(default=None, description="Additional error details")  # Extra context
