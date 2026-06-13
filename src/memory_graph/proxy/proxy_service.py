"""Core proxy service orchestrating request forwarding and extraction triggering."""

import asyncio  # Async task creation for background extraction
from collections.abc import AsyncGenerator  # Async generator type

import structlog  # Structured logging

from memory_graph.extraction.extraction_queue import ExtractionQueue  # Background queue
from memory_graph.models.proxy_models import ProxyRequest, ProxyResponse  # Proxy models
from memory_graph.proxy.interceptor import ProxyInterceptor  # Pre/post processing
from memory_graph.proxy.provider_registry import ProviderRegistry  # Provider lookup

# Module logger for proxy service operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ProxyService:
    """Orchestrates the full proxy pipeline: intercept, forward, extract.

    Coordinates between the interceptor (context injection), provider
    registry (request forwarding), and extraction queue (background processing).
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        interceptor: ProxyInterceptor,
        extraction_queue: ExtractionQueue,
    ) -> None:
        """Initialize the proxy service with required dependencies."""
        self._registry = provider_registry  # Provider lookup
        self._interceptor = interceptor  # Pre/post processing
        self._extraction_queue = extraction_queue  # Background extraction
        logger.info("proxy_service_initialized")  # Log initialization

    async def handle_request(self, request: ProxyRequest) -> ProxyResponse:
        """Handle a non-streaming proxy request through the full pipeline.

        1. Pre-process: inject memory context
        2. Forward: send to upstream provider
        3. Post-process: queue for extraction (fire-and-forget)
        4. Return response to client
        """
        logger.info(  # Log incoming request
            "proxy_request_received",
            provider=request.provider,
            model=request.model,
            stream=False,
        )

        # Step 1: Pre-process - inject memory context into the request
        processed_request = self._interceptor.pre_process(request)  # Inject context

        # Step 2: Resolve the provider adapter
        provider = self._registry.get_provider(request.provider)  # Look up provider
        if provider is None:  # Provider not registered
            logger.error("provider_not_available", provider=request.provider)  # Log error
            msg = f"Provider '{request.provider}' is not available"
            raise ValueError(msg)  # Raise descriptive error

        # Step 3: Forward the request to the upstream provider
        response = await provider.forward_request(processed_request)  # Forward and await

        # Step 4: Post-process - queue conversation for extraction (fire-and-forget)
        conversation_record = self._interceptor.post_process(  # Create record
            request=request,  # Original request (without injected context)
            response_content=response.content,  # LLM response text
        )
        await self._extraction_queue.enqueue(conversation_record)  # Queue for background

        logger.info(  # Log successful completion
            "proxy_request_completed",
            provider=request.provider,
            model=response.model,
        )
        return response  # Return response to client

    async def handle_streaming_request(
        self, request: ProxyRequest
    ) -> AsyncGenerator[bytes, None]:
        """Handle a streaming proxy request.

        Injects context, forwards as stream, collects full response
        for extraction, and yields chunks to client simultaneously.
        """
        logger.info(  # Log streaming request
            "proxy_streaming_request_received",
            provider=request.provider,
            model=request.model,
        )

        # Pre-process: inject memory context
        processed_request = self._interceptor.pre_process(request)  # Inject context

        # Resolve provider
        provider = self._registry.get_provider(request.provider)  # Look up
        if provider is None:  # Not registered
            logger.error("provider_not_available", provider=request.provider)
            msg = f"Provider '{request.provider}' is not available"
            raise ValueError(msg)

        # Collect full response text while streaming for extraction
        collected_chunks: list[bytes] = []  # Accumulate all chunks

        # Stream from provider, yielding to client and collecting
        async for chunk in provider.forward_streaming_request(processed_request):  # Stream
            collected_chunks.append(chunk)  # Store for extraction
            yield chunk  # Yield to client immediately

        # After streaming completes, trigger extraction with collected response
        full_response_text = self._extract_text_from_chunks(collected_chunks)  # Parse chunks

        if full_response_text:  # Got response text
            conversation_record = self._interceptor.post_process(  # Create record
                request=request,
                response_content=full_response_text,
            )
            await self._extraction_queue.enqueue(conversation_record)  # Queue extraction

        logger.info("proxy_streaming_completed", provider=request.provider)  # Log completion

    def _extract_text_from_chunks(self, chunks: list[bytes]) -> str:
        """Extract the full response text from collected SSE stream chunks.

        Parses SSE data lines to reconstruct the complete response.
        """
        full_text_parts: list[str] = []  # Accumulate text parts
        raw_data = b"".join(chunks).decode("utf-8", errors="replace")  # Combine and decode

        for line in raw_data.split("\n"):  # Process each SSE line
            if line.startswith("data: "):  # SSE data line
                data_content = line[6:]  # Remove "data: " prefix
                if data_content.strip() == "[DONE]":  # End of stream marker
                    break  # Stop processing

                try:  # Attempt to parse JSON data
                    import json  # JSON parsing
                    chunk_data = json.loads(data_content)  # Parse the data
                    # Extract content delta (OpenAI format)
                    choices = chunk_data.get("choices", [])  # Get choices
                    if choices:  # Has choices
                        delta = choices[0].get("delta", {})  # Get delta
                        content = delta.get("content", "")  # Get content
                        if content:  # Non-empty content
                            full_text_parts.append(content)  # Accumulate
                except (ValueError, KeyError):  # Parse error
                    continue  # Skip unparseable chunks

        return "".join(full_text_parts)  # Return reconstructed text
