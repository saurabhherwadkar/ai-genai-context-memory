"""FastAPI router for LLM proxy endpoints."""

import structlog  # Structured logging
from fastapi import APIRouter, Request  # Router and request types
from fastapi.responses import JSONResponse, StreamingResponse  # Response types

from memory_graph.models.proxy_models import ChatMessage, ProxyRequest  # Proxy models
from memory_graph.proxy.proxy_service import ProxyService  # Core proxy logic

# Module logger for proxy router operations
logger = structlog.get_logger(__name__)  # Named logger for this module


def create_proxy_router(proxy_service: ProxyService) -> APIRouter:
    """Create and configure the proxy router with the proxy service."""

    router = APIRouter(prefix="/v1/proxy", tags=["proxy"])  # Proxy endpoints group

    @router.post("/{provider}/chat/completions")  # Main proxy endpoint
    async def proxy_chat_completions(provider: str, request: Request) -> JSONResponse | StreamingResponse:
        """Proxy a chat completion request to the specified LLM provider.

        Injects relevant memory context, forwards the request, returns
        the response, and triggers async memory extraction.
        """
        logger.info("proxy_endpoint_hit", provider=provider)  # Log endpoint access

        # Parse the raw request body
        body = await request.json()  # Parse incoming JSON

        # Extract conversation ID from custom header if provided
        conversation_id = request.headers.get("X-Conversation-ID")  # Custom header
        # Check if context injection is disabled
        memory_context_header = request.headers.get("X-Memory-Context", "enabled")  # Default enabled

        # Build normalized proxy request from the raw body
        messages = [  # Convert raw messages to ChatMessage objects
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in body.get("messages", [])
        ]

        proxy_request = ProxyRequest(  # Build normalized request
            provider=provider,  # From URL path
            model=body.get("model", ""),  # Model from body
            messages=messages,  # Parsed messages
            stream=body.get("stream", False),  # Streaming flag
            temperature=body.get("temperature"),  # Optional temperature
            max_tokens=body.get("max_tokens"),  # Optional max tokens
            extra_params={  # Pass through unknown params
                k: v for k, v in body.items()
                if k not in ("model", "messages", "stream", "temperature", "max_tokens")
            },
            conversation_id=(  # Set to "disabled" if opted out
                "disabled" if memory_context_header == "disabled" else conversation_id
            ),
        )

        # Handle streaming vs non-streaming
        if proxy_request.stream:  # Streaming requested
            return StreamingResponse(  # Return SSE stream
                proxy_service.handle_streaming_request(proxy_request),  # Async generator
                media_type="text/event-stream",  # SSE content type
            )

        # Non-streaming: forward and return complete response
        response = await proxy_service.handle_request(proxy_request)  # Forward and wait

        # Return the raw provider response to maintain full compatibility
        return JSONResponse(content=response.raw_response)  # Pass through raw response

    @router.post("/{provider}/messages")  # Alternative Anthropic-style endpoint
    async def proxy_messages(provider: str, request: Request) -> JSONResponse | StreamingResponse:
        """Proxy a messages request (Anthropic format) to the specified provider.

        Supports both OpenAI and Anthropic message formats transparently.
        """
        logger.info("proxy_messages_endpoint_hit", provider=provider)  # Log access

        body = await request.json()  # Parse incoming JSON
        conversation_id = request.headers.get("X-Conversation-ID")  # Custom header
        memory_context_header = request.headers.get("X-Memory-Context", "enabled")  # Default

        # Handle Anthropic format where system is a top-level field
        messages: list[ChatMessage] = []  # Build message list
        if "system" in body:  # Anthropic system prompt
            messages.append(ChatMessage(role="system", content=body["system"]))  # Add system

        for msg in body.get("messages", []):  # Process conversation messages
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))  # Add each

        proxy_request = ProxyRequest(  # Build normalized request
            provider=provider,
            model=body.get("model", ""),
            messages=messages,
            stream=body.get("stream", False),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            extra_params={
                k: v for k, v in body.items()
                if k not in ("model", "messages", "stream", "temperature", "max_tokens", "system")
            },
            conversation_id=(
                "disabled" if memory_context_header == "disabled" else conversation_id
            ),
        )

        if proxy_request.stream:  # Streaming
            return StreamingResponse(
                proxy_service.handle_streaming_request(proxy_request),
                media_type="text/event-stream",
            )

        response = await proxy_service.handle_request(proxy_request)  # Forward
        return JSONResponse(content=response.raw_response)  # Return raw

    return router  # Return configured router
