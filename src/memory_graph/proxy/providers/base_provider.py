"""Abstract base class for LLM provider adapters."""

from abc import ABC, abstractmethod  # Abstract base class utilities
from collections.abc import AsyncGenerator  # Async generator type hint
from typing import Any  # Generic type for raw responses

from memory_graph.models.proxy_models import ProxyRequest, ProxyResponse  # Proxy data models


class BaseProvider(ABC):
    """Abstract interface for LLM provider adapters.

    Each provider adapter translates between the normalized proxy format
    and the specific API format of an LLM provider (OpenAI, Anthropic, etc.).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name identifier for this provider."""
        ...  # Must be implemented by subclass

    @abstractmethod
    async def forward_request(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a non-streaming request to the provider and return the response.

        Transforms the normalized request to provider format, sends it,
        and transforms the provider response back to normalized format.
        """
        ...  # Must be implemented by subclass

    @abstractmethod
    async def forward_streaming_request(
        self, request: ProxyRequest
    ) -> AsyncGenerator[bytes, None]:
        """Forward a streaming request and yield response chunks.

        Each yielded chunk is a raw SSE-formatted bytes block ready
        to be sent to the client as-is.
        """
        ...  # Must be implemented by subclass

    @abstractmethod
    def transform_request(self, request: ProxyRequest) -> dict[str, Any]:
        """Transform a normalized proxy request to provider-specific format."""
        ...  # Must be implemented by subclass

    @abstractmethod
    def transform_response(self, raw_response: dict[str, Any]) -> ProxyResponse:
        """Transform a provider-specific response to normalized proxy format."""
        ...  # Must be implemented by subclass

    @abstractmethod
    def extract_response_content(self, raw_response: dict[str, Any]) -> str:
        """Extract the generated text content from a raw provider response."""
        ...  # Must be implemented by subclass
