"""Memory node data model representing a single memory item in the cognitive graph."""

from datetime import datetime, timezone  # Date/time types with timezone support
from typing import Any  # Generic type for flexible metadata dictionary
from uuid import uuid4  # UUID generation for unique node identifiers

from pydantic import BaseModel, Field, field_validator  # Validation framework

from memory_graph.models.memory_types import EmotionValence, NodeType  # Node classification enums


def _utc_now() -> datetime:
    """Generate current UTC timestamp for default field values."""
    return datetime.now(timezone.utc)  # Timezone-aware UTC timestamp


class MemoryNode(BaseModel):
    """Represents a single memory item in the cognitive graph.

    Each node captures a discrete piece of cognitive information such as
    a belief, preference, or experience with associated metadata for
    ranking and retrieval.
    """

    id: str = Field(  # Globally unique node identifier
        default_factory=lambda: str(uuid4()),  # Generate UUID string on creation
        description="Unique identifier for this memory node",
    )
    node_type: NodeType = Field(  # Classification category for this memory
        description="The cognitive category this memory belongs to",
    )
    content: str = Field(  # Full text content of the memory
        min_length=1,  # Enforce non-empty content
        max_length=50000,  # Upper bound to prevent oversized entries
        description="The full text content of this memory",
    )
    summary: str = Field(  # Brief one-line summary for context injection
        min_length=1,  # Enforce non-empty summary
        max_length=500,  # Keep summaries concise
        description="One-line summary for quick context injection",
    )
    confidence: float = Field(  # System certainty about this memory (0.0 to 1.0)
        default=0.8,  # Default moderate-high confidence
        ge=0.0,  # Minimum confidence value
        le=1.0,  # Maximum confidence value
        description="How certain the system is about this memory",
    )
    emotional_valence: EmotionValence = Field(  # Emotional polarity of this memory
        default=EmotionValence.NEUTRAL,  # Default to neutral emotional tone
        description="The emotional polarity associated with this memory",
    )
    intensity: float = Field(  # Strength or importance of this memory (0.0 to 1.0)
        default=0.5,  # Default moderate intensity
        ge=0.0,  # Minimum intensity value
        le=1.0,  # Maximum intensity value
        description="The strength or importance of this memory",
    )
    source_conversation_id: str | None = Field(  # Origin conversation reference
        default=None,  # Optional, not all memories come from conversations
        description="The conversation ID this memory was extracted from",
    )
    created_at: datetime = Field(  # Timestamp when this node was first created
        default_factory=_utc_now,  # Default to current UTC time
        description="When this memory was first created",
    )
    updated_at: datetime = Field(  # Timestamp of last modification
        default_factory=_utc_now,  # Default to current UTC time
        description="When this memory was last updated",
    )
    last_accessed_at: datetime | None = Field(  # Timestamp of last retrieval
        default=None,  # Null until first access
        description="When this memory was last retrieved for context",
    )
    access_count: int = Field(  # Number of times this memory has been retrieved
        default=0,  # Starts at zero for new memories
        ge=0,  # Cannot be negative
        description="How many times this memory has been accessed",
    )
    tags: list[str] = Field(  # Categorical labels for filtering
        default_factory=list,  # Default to empty tag list
        max_length=20,  # Maximum number of tags allowed
        description="Categorical tags for filtering and grouping",
    )
    metadata: dict[str, Any] = Field(  # Extensible key-value metadata
        default_factory=dict,  # Default to empty metadata dictionary
        description="Extensible metadata for custom attributes",
    )
    is_active: bool = Field(  # Soft delete flag to deactivate without removing
        default=True,  # Active by default
        description="Whether this memory is active or soft-deleted",
    )
    custom_type_name: str | None = Field(  # Name for custom node types
        default=None,  # Only required when node_type is CUSTOM
        description="Custom type name when node_type is CUSTOM",
    )

    @field_validator("custom_type_name")  # Validate custom type name constraint
    @classmethod
    def validate_custom_type_name(cls, value: str | None, info: Any) -> str | None:
        """Ensure custom_type_name is set when node_type is CUSTOM."""
        node_type = info.data.get("node_type")  # Get the node_type from the data being validated
        if node_type == NodeType.CUSTOM and not value:  # CUSTOM type requires a name
            msg = "custom_type_name is required when node_type is CUSTOM"  # Descriptive error message
            raise ValueError(msg)  # Raise validation error
        return value  # Return the validated value

    @field_validator("tags")  # Validate tags are non-empty strings
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        """Ensure all tags are non-empty trimmed strings."""
        cleaned_tags = [tag.strip() for tag in value if tag.strip()]  # Strip whitespace and remove empty
        return cleaned_tags  # Return cleaned tag list
