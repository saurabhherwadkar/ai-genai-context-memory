"""Generic LLM provider adapter for custom or OpenAI-compatible APIs."""

from collections.abc import AsyncGenerator  # Async generator type
from typing import Any  # Generic type for raw responses

import httpx  # Async HTTP client
import structlog  # Structured logging

from memory_graph.models.proxy_models import ProxyRequest, ProxyResponse  # Proxy data models
from memory_graph.proxy.providers.base_provider import BaseProvider  # Abstract interface

# Module logger for generic provider operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class GenericProvider(BaseProvider):
    """Adapter for generic OpenAI-compatible LLM APIs.

    Assumes the target API follows the OpenAI chat completions format.
    Used for local models (Ollama, vLLM) or other compatible services.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize the generic provider with connection settings."""
        self._name = name  # Custom provider name
        self._base_url = base_url.rstrip("/")  # Remove trailing slash
        self._api_key = api_key  # Optional API key
        self._timeout = timeout  # Request timeout
        logger.info("generic_provider_initialized", name=name, base_url=self._base_url)  # Log

    @property
    def provider_name(self) -> str:
        """Return the custom provider name identifier."""
        return self._name  # Return configured name

    async def forward_request(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a non-streaming request to the generic provider."""
        payload = self.transform_request(request)  # Build payload

        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create client
            response = await client.post(  # Send POST request
                f"{self._base_url}/chat/completions",  # OpenAI-compatible endpoint
                json=payload,  # Request body
                headers=self._build_headers(),  # Auth headers
            )

        if response.status_code != 200:  # Non-success
            logger.error(  # Log failure
                "generic_provider_request_failed",
                provider=self._name,
                status_code=response.status_code,
            )
            response.raise_for_status()  # Raise error

        raw_response = response.json()  # Parse response
        return self.transform_response(raw_response)  # Return normalized

    async def forward_streaming_request(
        self, request: ProxyRequest
    ) -> AsyncGenerator[bytes, None]:
        """Forward a streaming request and yield response chunks."""
        payload = self.transform_request(request)  # Build payload
        payload["stream"] = True  # Enable streaming

        async with httpx.AsyncClient(timeout=self._timeout) as client:  # Create client
            async with client.stream(  # Open stream
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
            ) as response:
                async for chunk in response.aiter_bytes():  # Iterate chunks
                    yield chunk  # Pass through

    def transform_request(self, request: ProxyRequest) -> dict[str, Any]:
        """Transform to OpenAI-compatible format (assumed by generic providers)."""
        payload: dict[str, Any] = {  # Build compatible payload
            "model": request.model,  # Model identifier
            "messages": [  # Message array
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
        }

        if request.temperature is not None:  # Temperature specified
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:  # Max tokens specified
            payload["max_tokens"] = request.max_tokens

        payload.update(request.extra_params)  # Merge extras
        return payload  # Return payload

    def transform_response(self, raw_response: dict[str, Any]) -> ProxyResponse:
        """Transform OpenAI-compatible response to normalized format."""
        content = self.extract_response_content(raw_response)  # Get content
        usage = raw_response.get("usage", {})  # Get usage

        return ProxyResponse(  # Build response
            provider=self._name,  # Provider name
            model=raw_response.get("model", ""),  # Model used
            content=content,  # Text content
            finish_reason=raw_response.get("choices", [{}])[0].get("finish_reason"),  # Stop reason
            usage=usage,  # Token usage
            raw_response=raw_response,  # Raw preserved
        )

    def extract_response_content(self, raw_response: dict[str, Any]) -> str:
        """Extract text from OpenAI-compatible response."""
        choices = raw_response.get("choices", [])  # Get choices
        if not choices:  # No choices
            return ""  # Empty
        return choices[0].get("message", {}).get("content", "")  # Get content

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the generic provider."""
        headers = {"Content-Type": "application/json"}  # Base headers
        if self._api_key:  # API key configured
            headers["Authorization"] = f"Bearer {self._api_key}"  # Add auth
        return headers  # Return headers
