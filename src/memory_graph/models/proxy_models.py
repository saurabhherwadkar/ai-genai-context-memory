"""Data models for the LLM proxy request/response handling."""

from typing import Any  # Generic type for flexible message content

from pydantic import BaseModel, Field  # Validation framework


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(description="Message role (system, user, assistant)")  # Speaker role
    content: str | list[Any] = Field(description="Message content")  # Text or multimodal content


class ProxyRequest(BaseModel):
    """Normalized proxy request independent of provider format."""

    provider: str = Field(description="Target LLM provider name")  # Provider identifier
    model: str = Field(description="Model identifier to use")  # Specific model name
    messages: list[ChatMessage] = Field(description="Conversation messages")  # Message history
    stream: bool = Field(default=False, description="Enable streaming response")  # SSE streaming flag
    temperature: float | None = Field(default=None, description="Sampling temperature")  # Randomness control
    max_tokens: int | None = Field(default=None, description="Max tokens to generate")  # Output token limit
    extra_params: dict[str, Any] = Field(  # Provider-specific extra parameters
        default_factory=dict,  # Empty by default
        description="Additional provider-specific parameters",
    )
    conversation_id: str | None = Field(  # Conversation tracking identifier
        default=None,  # Optional for stateless requests
        description="Conversation ID for memory tracking",
    )


class ProxyResponse(BaseModel):
    """Normalized proxy response independent of provider format."""

    provider: str = Field(description="Source LLM provider name")  # Provider that generated this
    model: str = Field(description="Model that generated the response")  # Model identifier
    content: str = Field(description="Generated text content")  # Response text
    finish_reason: str | None = Field(default=None, description="Stop reason")  # Why generation stopped
    usage: dict[str, int] = Field(  # Token usage statistics
        default_factory=dict,  # Empty by default
        description="Token usage statistics",
    )
    raw_response: dict[str, Any] = Field(  # Original provider response preserved
        default_factory=dict,  # Empty by default
        description="Original provider response for passthrough",
    )


class ConversationRecord(BaseModel):
    """Record of a complete conversation exchange for extraction processing."""

    conversation_id: str = Field(description="Unique conversation identifier")  # Tracking ID
    provider: str = Field(description="LLM provider used")  # Provider name
    model: str = Field(description="Model used for generation")  # Model identifier
    messages: list[ChatMessage] = Field(description="Full conversation messages")  # Complete history
    response_content: str = Field(description="LLM response content")  # Generated response text
    timestamp: str = Field(description="ISO timestamp of the exchange")  # When this occurred
