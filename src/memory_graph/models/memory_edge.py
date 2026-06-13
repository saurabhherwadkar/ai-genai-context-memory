"""Memory edge data model representing relationships between memory nodes."""

from datetime import datetime, timezone  # Date/time types with timezone support
from typing import Any  # Generic type for flexible metadata dictionary
from uuid import uuid4  # UUID generation for unique edge identifiers

from pydantic import BaseModel, Field, model_validator  # Validation framework

from memory_graph.models.memory_types import EdgeType  # Edge classification enum


def _utc_now() -> datetime:
    """Generate current UTC timestamp for default field values."""
    return datetime.now(timezone.utc)  # Timezone-aware UTC timestamp


class MemoryEdge(BaseModel):
    """Represents a directional relationship between two memory nodes.

    Edges encode the nature and strength of the relationship, enabling
    graph traversal to discover related, supporting, or conflicting memories.
    """

    id: str = Field(  # Globally unique edge identifier
        default_factory=lambda: str(uuid4()),  # Generate UUID string on creation
        description="Unique identifier for this relationship edge",
    )
    source_node_id: str = Field(  # Origin node of the directed relationship
        min_length=1,  # Must be non-empty
        description="The ID of the source memory node",
    )
    target_node_id: str = Field(  # Destination node of the directed relationship
        min_length=1,  # Must be non-empty
        description="The ID of the target memory node",
    )
    edge_type: EdgeType = Field(  # Classification of this relationship
        description="The nature of the relationship between source and target",
    )
    weight: float = Field(  # Strength of the relationship (0.0 to 1.0)
        default=0.5,  # Default moderate weight
        ge=0.0,  # Minimum weight value
        le=1.0,  # Maximum weight value
        description="The strength of this relationship",
    )
    description: str | None = Field(  # Optional explanation of why nodes are linked
        default=None,  # Not required for all edges
        max_length=1000,  # Keep descriptions reasonably short
        description="Optional textual explanation of this relationship",
    )
    created_at: datetime = Field(  # Timestamp when this edge was created
        default_factory=_utc_now,  # Default to current UTC time
        description="When this relationship was established",
    )
    metadata: dict[str, Any] = Field(  # Extensible key-value metadata
        default_factory=dict,  # Default to empty metadata dictionary
        description="Extensible metadata for custom edge attributes",
    )

    @model_validator(mode="after")  # Validate the complete model after all fields are set
    def validate_no_self_loop(self) -> "MemoryEdge":
        """Ensure an edge does not connect a node to itself."""
        if self.source_node_id == self.target_node_id:  # Self-loops are invalid in this domain
            msg = "An edge cannot connect a node to itself"  # Descriptive error message
            raise ValueError(msg)  # Raise validation error
        return self  # Return the validated edge instance
