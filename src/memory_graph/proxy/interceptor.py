"""Pre/post processing hooks for the LLM proxy pipeline."""

from datetime import datetime, timezone  # Timestamp generation
from uuid import uuid4  # Unique ID generation

import structlog  # Structured logging

from memory_graph.context.context_builder import ContextBuilder  # Context retrieval
from memory_graph.models.proxy_models import ChatMessage, ConversationRecord, ProxyRequest  # Models

# Module logger for interceptor operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ProxyInterceptor:
    """Handles pre-request context injection and post-response extraction triggering.

    Pre-processing: Retrieves relevant memories and injects them into the
    system prompt before the request is forwarded to the LLM provider.

    Post-processing: Creates a conversation record and queues it for
    asynchronous memory extraction.
    """

    def __init__(self, context_builder: ContextBuilder) -> None:
        """Initialize the interceptor with the context builder."""
        self._context_builder = context_builder  # For memory retrieval
        logger.info("proxy_interceptor_initialized")  # Log initialization

    def pre_process(self, request: ProxyRequest) -> ProxyRequest:
        """Inject relevant memory context into the request before forwarding.

        Extracts the latest user message, finds relevant memories,
        and prepends them to the system prompt.
        """
        # Check if context injection is disabled via header (conversation_id == "disabled")
        if request.conversation_id == "disabled":  # Client opted out
            logger.debug("context_injection_disabled_by_client")  # Log opt-out
            return request  # Return unmodified request

        # Extract the latest user message for context retrieval
        latest_user_message = self._extract_latest_user_message(request)  # Find user text
        if not latest_user_message:  # No user message found
            logger.debug("no_user_message_for_context")  # Log skip
            return request  # Return unmodified

        # Build memory context relevant to the user's message
        context_text, token_count, memory_count = self._context_builder.build_context(  # Build
            query_text=latest_user_message,  # Use user message as query
        )

        if not context_text:  # No relevant memories found
            logger.debug("no_relevant_memories_found")  # Log empty context
            return request  # Return unmodified

        # Inject the memory context into the system prompt
        injected_request = self._inject_context(request, context_text)  # Modify request
        logger.info(  # Log successful injection
            "context_injected",
            memories=memory_count,
            tokens=token_count,
        )
        return injected_request  # Return modified request

    def post_process(
        self,
        request: ProxyRequest,
        response_content: str,
    ) -> ConversationRecord:
        """Create a conversation record for asynchronous memory extraction.

        Packages the full conversation exchange into a record that can
        be queued for background processing.
        """
        # Generate a conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid4())  # Use existing or generate

        # Build the conversation record
        record = ConversationRecord(  # Package the exchange
            conversation_id=conversation_id,  # Tracking ID
            provider=request.provider,  # Provider used
            model=request.model,  # Model used
            messages=request.messages,  # Full conversation history
            response_content=response_content,  # LLM response text
            timestamp=datetime.now(timezone.utc).isoformat(),  # Current timestamp
        )

        logger.debug(  # Log record creation
            "conversation_record_created",
            conversation_id=conversation_id,
        )
        return record  # Return the record for queueing

    def _extract_latest_user_message(self, request: ProxyRequest) -> str:
        """Extract the text content of the most recent user message."""
        for message in reversed(request.messages):  # Search from end
            if message.role == "user":  # Found a user message
                content = message.content  # Get the content
                if isinstance(content, str):  # Simple string content
                    return content  # Return directly
                if isinstance(content, list):  # Multimodal content array
                    # Extract text parts from the content array
                    text_parts = [  # Filter to text items
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    return " ".join(text_parts)  # Join text parts
        return ""  # No user message found

    def _inject_context(self, request: ProxyRequest, context_text: str) -> ProxyRequest:
        """Inject memory context into the system prompt of the request.

        If a system message exists, prepends context to it.
        If no system message exists, creates one with the context.
        """
        modified_messages: list[ChatMessage] = []  # Build new message list
        system_found = False  # Track whether we found a system message

        for message in request.messages:  # Process each message
            if message.role == "system" and not system_found:  # First system message
                # Prepend memory context to existing system message
                augmented_content = f"{context_text}\n\n{message.content}"  # Combine
                modified_messages.append(  # Add modified system message
                    ChatMessage(role="system", content=augmented_content)
                )
                system_found = True  # Mark as handled
            else:  # Non-system or subsequent system messages
                modified_messages.append(message)  # Pass through unchanged

        if not system_found:  # No existing system message
            # Insert a new system message with context at the beginning
            modified_messages.insert(0, ChatMessage(role="system", content=context_text))  # Prepend

        # Return a new request with modified messages
        return ProxyRequest(  # Build new request preserving all other fields
            provider=request.provider,
            model=request.model,
            messages=modified_messages,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_params=request.extra_params,
            conversation_id=request.conversation_id,
        )
