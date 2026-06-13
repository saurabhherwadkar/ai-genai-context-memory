"""Request data transfer objects for the REST API endpoints."""

from typing import Any  # Generic type for flexible metadata values

from pydantic import BaseModel, Field  # Validation framework for request schemas

from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums for classification


class CreateMemoryNodeRequest(BaseModel):
    """Request body for creating a new memory node."""

    node_type: NodeType = Field(  # Required classification for the memory
        description="The cognitive category for this memory",
    )
    content: str = Field(  # Full text content of the memory
        min_length=1,  # Must not be empty
        max_length=50000,  # Upper bound on content size
        description="The full text content of this memory",
    )
    summary: str = Field(  # Brief one-line summary
        min_length=1,  # Must not be empty
        max_length=500,  # Keep summaries concise
        description="One-line summary for context injection",
    )
    confidence: float = Field(  # Certainty score between 0 and 1
        default=0.8,  # Default moderate-high confidence
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Confidence score for this memory",
    )
    emotional_valence: EmotionValence = Field(  # Emotional polarity
        default=EmotionValence.NEUTRAL,  # Default neutral tone
        description="Emotional polarity of this memory",
    )
    intensity: float = Field(  # Strength of memory
        default=0.5,  # Default moderate intensity
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Strength or importance of this memory",
    )
    tags: list[str] = Field(  # Categorical labels
        default_factory=list,  # Empty by default
        description="Tags for categorization",
    )
    metadata: dict[str, Any] = Field(  # Extensible metadata
        default_factory=dict,  # Empty by default
        description="Optional extensible metadata",
    )
    custom_type_name: str | None = Field(  # Name for custom types
        default=None,  # Only needed for CUSTOM node_type
        description="Custom type name when node_type is CUSTOM",
    )


class UpdateMemoryNodeRequest(BaseModel):
    """Request body for updating an existing memory node (partial update)."""

    content: str | None = Field(  # Optional updated content
        default=None,  # Null means no change
        min_length=1,  # If provided, must not be empty
        max_length=50000,  # Upper bound
        description="Updated content text",
    )
    summary: str | None = Field(  # Optional updated summary
        default=None,  # Null means no change
        min_length=1,  # If provided, must not be empty
        max_length=500,  # Keep concise
        description="Updated summary text",
    )
    confidence: float | None = Field(  # Optional updated confidence
        default=None,  # Null means no change
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Updated confidence score",
    )
    emotional_valence: EmotionValence | None = Field(  # Optional updated valence
        default=None,  # Null means no change
        description="Updated emotional polarity",
    )
    intensity: float | None = Field(  # Optional updated intensity
        default=None,  # Null means no change
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Updated intensity score",
    )
    tags: list[str] | None = Field(  # Optional updated tags
        default=None,  # Null means no change
        description="Updated tag list (replaces existing tags)",
    )
    metadata: dict[str, Any] | None = Field(  # Optional updated metadata
        default=None,  # Null means no change
        description="Updated metadata (replaces existing metadata)",
    )
    is_active: bool | None = Field(  # Optional soft delete toggle
        default=None,  # Null means no change
        description="Set active status (False for soft delete)",
    )


class CreateMemoryEdgeRequest(BaseModel):
    """Request body for creating a new relationship edge between nodes."""

    source_node_id: str = Field(  # Origin node ID
        min_length=1,  # Must not be empty
        description="ID of the source memory node",
    )
    target_node_id: str = Field(  # Destination node ID
        min_length=1,  # Must not be empty
        description="ID of the target memory node",
    )
    edge_type: EdgeType = Field(  # Relationship classification
        description="The type of relationship between the nodes",
    )
    weight: float = Field(  # Relationship strength
        default=0.5,  # Default moderate weight
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Strength of this relationship",
    )
    description: str | None = Field(  # Optional explanation
        default=None,  # Not required
        max_length=1000,  # Keep descriptions short
        description="Optional description of why these nodes are linked",
    )
    metadata: dict[str, Any] = Field(  # Extensible metadata
        default_factory=dict,  # Empty by default
        description="Optional extensible metadata",
    )


class GraphQueryRequest(BaseModel):
    """Request body for querying related memories in the graph."""

    node_id: str = Field(  # Starting node for traversal
        min_length=1,  # Must not be empty
        description="The node ID to start traversal from",
    )
    max_depth: int = Field(  # Maximum traversal hops
        default=2,  # Default shallow traversal
        ge=1,  # Minimum one hop
        le=5,  # Maximum five hops to prevent explosion
        description="Maximum traversal depth",
    )
    edge_types: list[EdgeType] | None = Field(  # Filter by edge type
        default=None,  # None means all edge types
        description="Filter traversal to specific edge types",
    )
    node_types: list[NodeType] | None = Field(  # Filter results by node type
        default=None,  # None means all node types
        description="Filter results to specific node types",
    )
    min_confidence: float = Field(  # Minimum confidence threshold
        default=0.0,  # Default includes all confidence levels
        ge=0.0,  # Minimum allowed
        le=1.0,  # Maximum allowed
        description="Minimum confidence threshold for results",
    )


class ContextQueryRequest(BaseModel):
    """Request body for retrieving assembled memory context for LLM injection."""

    query_text: str = Field(  # The text to find relevant memories for
        min_length=1,  # Must not be empty
        max_length=10000,  # Reasonable upper bound
        description="The text to find relevant memories for",
    )
    max_tokens: int | None = Field(  # Override default token budget
        default=None,  # None uses configured default
        ge=100,  # Minimum useful token count
        le=10000,  # Maximum reasonable budget
        description="Override the max token budget for this query",
    )
    format: str | None = Field(  # Override output format
        default=None,  # None uses configured default
        description="Override output format (markdown, xml, plain)",
    )
