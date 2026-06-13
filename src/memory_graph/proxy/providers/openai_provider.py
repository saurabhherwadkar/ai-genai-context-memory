"""OpenAI API provider adapter for the LLM proxy."""

from collections.abc import AsyncGenerator  # Async generator type
from typing import Any  # Generic type for raw responses

import httpx  # Async HTTP client
import structlog  # Structured logging

from memory_graph.models.proxy_models import ProxyRequest, ProxyResponse  # Proxy data models
from memory_graph.proxy.providers.base_provider import BaseProvider  # Abstract interface

# Module logger for OpenAI provider operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class OpenAIProvider(BaseProvider):
    """Adapter for the OpenAI chat completions API.

    Handles translation between normalized proxy format and OpenAI's
    /v1/chat/completions endpoint format.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
        """Initialize the OpenAI provider with connection settings."""
        self._base_url = base_url.rstrip("/")  # Remove trailing slash from base URL
        self._api_key = api_key  # API key for authentication
        self._timeout = timeout  # Request timeout in seconds
        logger.info("openai_provider_initialized", base_url=self._base_url)  # Log init

    @property
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        return "openai"  # Fixed provider name

    async def forward_request(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a non-streaming request to OpenAI and return the response."""
        # Transform to OpenAI format
        openai_payload = self.transform_request(request)  # Build provider-specific payload

        # Make the HTTP request
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create async client
            response = await client.post(  # Send POST request
                f"{self._base_url}/chat/completions",  # OpenAI endpoint
                json=openai_payload,  # Request body
                headers=self._build_headers(),  # Auth headers
            )

        # Handle error responses
        if response.status_code != 200:  # Non-success response
            logger.error(  # Log the error
                "openai_request_failed",
                status_code=response.status_code,
                body=response.text[:500],
            )
            response.raise_for_status()  # Raise HTTP error

        # Transform the response to normalized format
        raw_response = response.json()  # Parse JSON response
        return self.transform_response(raw_response)  # Return normalized response

    async def forward_streaming_request(
        self, request: ProxyRequest
    ) -> AsyncGenerator[bytes, None]:
        """Forward a streaming request to OpenAI and yield SSE chunks."""
        # Transform to OpenAI format with streaming enabled
        openai_payload = self.transform_request(request)  # Build payload
        openai_payload["stream"] = True  # Force streaming on

        # Stream the response
        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create client
            async with client.stream(  # Open streaming connection
                "POST",
                f"{self._base_url}/chat/completions",
                json=openai_payload,
                headers=self._build_headers(),
            ) as response:
                async for chunk in response.aiter_bytes():  # Iterate over response chunks
                    yield chunk  # Yield each chunk as-is

    def transform_request(self, request: ProxyRequest) -> dict[str, Any]:
        """Transform normalized proxy request to OpenAI format."""
        payload: dict[str, Any] = {  # Build OpenAI-compatible payload
            "model": request.model,  # Model identifier
            "messages": [  # Convert messages to OpenAI format
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
        }

        # Add optional parameters if specified
        if request.temperature is not None:  # Temperature specified
            payload["temperature"] = request.temperature  # Add to payload
        if request.max_tokens is not None:  # Max tokens specified
            payload["max_tokens"] = request.max_tokens  # Add to payload
        if request.stream:  # Streaming requested
            payload["stream"] = True  # Enable streaming

        # Merge any extra provider-specific parameters
        payload.update(request.extra_params)  # Add extra params

        return payload  # Return the complete payload

    def transform_response(self, raw_response: dict[str, Any]) -> ProxyResponse:
        """Transform OpenAI response to normalized proxy format."""
        # Extract the generated content from choices
        content = self.extract_response_content(raw_response)  # Get text content

        # Extract usage statistics
        usage = raw_response.get("usage", {})  # Get token usage
        usage_dict = {  # Normalize usage fields
            "prompt_tokens": usage.get("prompt_tokens", 0),  # Input tokens
            "completion_tokens": usage.get("completion_tokens", 0),  # Output tokens
            "total_tokens": usage.get("total_tokens", 0),  # Total tokens
        }

        return ProxyResponse(  # Build normalized response
            provider="openai",  # Provider identifier
            model=raw_response.get("model", ""),  # Model used
            content=content,  # Generated text
            finish_reason=raw_response.get("choices", [{}])[0].get("finish_reason"),  # Stop reason
            usage=usage_dict,  # Token usage stats
            raw_response=raw_response,  # Original response preserved
        )

    def extract_response_content(self, raw_response: dict[str, Any]) -> str:
        """Extract text content from OpenAI response structure."""
        choices = raw_response.get("choices", [])  # Get choices array
        if not choices:  # No choices in response
            return ""  # Return empty string

        message = choices[0].get("message", {})  # Get first choice message
        return message.get("content", "")  # Extract content text

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for OpenAI API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",  # Bearer token auth
            "Content-Type": "application/json",  # JSON content type
        }
