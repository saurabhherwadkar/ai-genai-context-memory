"""Anthropic API provider adapter for the LLM proxy."""

from collections.abc import AsyncGenerator  # Async generator type
from typing import Any  # Generic type for raw responses

import httpx  # Async HTTP client
import structlog  # Structured logging

from memory_graph.models.proxy_models import ChatMessage, ProxyRequest, ProxyResponse  # Models
from memory_graph.proxy.providers.base_provider import BaseProvider  # Abstract interface

# Module logger for Anthropic provider operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Messages API.

    Handles translation between normalized proxy format and Anthropic's
    /v1/messages endpoint format with its distinct system prompt handling.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
        """Initialize the Anthropic provider with connection settings."""
        self._base_url = base_url.rstrip("/")  # Remove trailing slash
        self._api_key = api_key  # API key for authentication
        self._timeout = timeout  # Request timeout in seconds
        logger.info("anthropic_provider_initialized", base_url=self._base_url)  # Log init

    @property
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        return "anthropic"  # Fixed provider name

    async def forward_request(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a non-streaming request to Anthropic and return the response."""
        # Transform to Anthropic format
        anthropic_payload = self.transform_request(request)  # Build payload

        # Make the HTTP request
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create client
            response = await client.post(  # Send POST request
                f"{self._base_url}/messages",  # Anthropic messages endpoint
                json=anthropic_payload,  # Request body
                headers=self._build_headers(),  # Auth headers
            )

        # Handle error responses
        if response.status_code != 200:  # Non-success response
            logger.error(  # Log the error
                "anthropic_request_failed",
                status_code=response.status_code,
                body=response.text[:500],
            )
            response.raise_for_status()  # Raise HTTP error

        # Transform to normalized format
        raw_response = response.json()  # Parse JSON response
        return self.transform_response(raw_response)  # Return normalized

    async def forward_streaming_request(
        self, request: ProxyRequest
    ) -> AsyncGenerator[bytes, None]:
        """Forward a streaming request to Anthropic and yield SSE chunks."""
        # Transform to Anthropic format with streaming
        anthropic_payload = self.transform_request(request)  # Build payload
        anthropic_payload["stream"] = True  # Force streaming on

        # Stream the response
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create client
            async with client.stream(  # Open streaming connection
                "POST",
                f"{self._base_url}/messages",
                json=anthropic_payload,
                headers=self._build_headers(),
            ) as response:
                async for chunk in response.aiter_bytes():  # Iterate chunks
                    yield chunk  # Yield each chunk as-is

    def transform_request(self, request: ProxyRequest) -> dict[str, Any]:
        """Transform normalized proxy request to Anthropic format.

        Anthropic separates system messages from the conversation history.
        """
        # Separate system messages from conversation messages
        system_text = ""  # Accumulate system message content
        conversation_messages: list[dict[str, Any]] = []  # Non-system messages

        for msg in request.messages:  # Process each message
            if msg.role == "system":  # System message
                system_text += str(msg.content) + "\n"  # Append to system text
            else:  # User or assistant message
                conversation_messages.append(  # Add to conversation
                    {"role": msg.role, "content": str(msg.content)}
                )

        # Build the Anthropic payload
        payload: dict[str, Any] = {  # Anthropic-specific format
            "model": request.model,  # Model identifier
            "messages": conversation_messages,  # Conversation history
            "max_tokens": request.max_tokens or 4096,  # Required by Anthropic
        }

        if system_text.strip():  # System message present
            payload["system"] = system_text.strip()  # Add as top-level field

        # Add optional parameters
        if request.temperature is not None:  # Temperature specified
            payload["temperature"] = request.temperature  # Add to payload

        # Merge extra parameters
        payload.update(request.extra_params)  # Add provider-specific params

        return payload  # Return complete payload

    def transform_response(self, raw_response: dict[str, Any]) -> ProxyResponse:
        """Transform Anthropic response to normalized proxy format."""
        # Extract content from Anthropic response structure
        content = self.extract_response_content(raw_response)  # Get text content

        # Extract usage statistics
        usage = raw_response.get("usage", {})  # Get usage object
        usage_dict = {  # Normalize usage fields
            "prompt_tokens": usage.get("input_tokens", 0),  # Anthropic uses input_tokens
            "completion_tokens": usage.get("output_tokens", 0),  # Anthropic uses output_tokens
            "total_tokens": (  # Calculate total
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
        }

        return ProxyResponse(  # Build normalized response
            provider="anthropic",  # Provider identifier
            model=raw_response.get("model", ""),  # Model used
            content=content,  # Generated text
            finish_reason=raw_response.get("stop_reason"),  # Anthropic uses stop_reason
            usage=usage_dict,  # Token usage stats
            raw_response=raw_response,  # Original response preserved
        )

    def extract_response_content(self, raw_response: dict[str, Any]) -> str:
        """Extract text content from Anthropic response structure."""
        content_blocks = raw_response.get("content", [])  # Get content array

        if not content_blocks:  # No content blocks
            return ""  # Return empty

        # Concatenate all text content blocks
        text_parts: list[str] = []  # Accumulate text parts
        for block in content_blocks:  # Process each content block
            if block.get("type") == "text":  # Text block
                text_parts.append(block.get("text", ""))  # Extract text

        return "".join(text_parts)  # Return concatenated text

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Anthropic API requests."""
        return {
            "x-api-key": self._api_key,  # Anthropic uses x-api-key header
            "content-type": "application/json",  # JSON content type
            "anthropic-version": "2023-06-01",  # API version header
        }
