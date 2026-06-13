"""Context retrieval endpoint for getting assembled memory context."""

import structlog  # Structured logging
from fastapi import APIRouter  # Router for endpoint grouping

from memory_graph.context.context_builder import ContextBuilder  # Context orchestrator
from memory_graph.models.api_requests import ContextQueryRequest  # Request schema
from memory_graph.models.api_responses import ContextResponse  # Response schema
from memory_graph.security.input_validator import InputValidator  # Input sanitization

# Module logger for context API operations
logger = structlog.get_logger(__name__)  # Named logger for this module


def create_context_router(
    context_builder: ContextBuilder,
    input_validator: InputValidator,
) -> APIRouter:
    """Create and configure the context retrieval router with dependencies."""

    router = APIRouter(prefix="/api/v1/context", tags=["context"])  # Context endpoints group

    @router.post("/retrieve", response_model=ContextResponse)  # Retrieve context
    async def retrieve_context(request: ContextQueryRequest) -> ContextResponse:
        """Retrieve assembled memory context relevant to the query text.

        Performs semantic search, graph expansion, ranking, and formatting
        to produce a text block suitable for LLM prompt injection.
        """
        logger.info("context_retrieval_requested", query_length=len(request.query_text))  # Log

        # Validate the query text
        validated_query = input_validator.validate_query_text(request.query_text)  # Sanitize

        # Build the context using the full pipeline
        formatted_text, token_count, memories_included = context_builder.build_context(  # Build
            query_text=validated_query,  # Sanitized query
            max_tokens=request.max_tokens,  # Optional token budget override
            output_format=request.format,  # Optional format override
        )

        return ContextResponse(  # Build the response
            context_text=formatted_text,  # The assembled context block
            token_count=token_count,  # Estimated token usage
            memories_included=memories_included,  # Number of memories included
            format=request.format or "markdown",  # Format used
        )

    return router  # Return configured router
