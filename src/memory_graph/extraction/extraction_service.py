"""Orchestrates memory extraction from conversation records."""

import asyncio  # Async operations for retry delays
from typing import Any  # Generic type

import structlog  # Structured logging

from memory_graph.config.settings import EmbeddingSettings, ExtractionSettings  # Config
from memory_graph.embeddings.embedding_service import EmbeddingService  # Embedding generation
from memory_graph.embeddings.similarity_search import SimilaritySearch  # Duplicate detection
from memory_graph.extraction.extraction_parser import ExtractionParser  # Output parsing
from memory_graph.extraction.extraction_prompts import (  # Prompt templates
    build_extraction_prompt,
    format_conversation_for_extraction,
)
from memory_graph.graph.graph_sync import GraphSync  # Synchronized persistence
from memory_graph.models.memory_edge import MemoryEdge  # Edge model
from memory_graph.models.memory_node import MemoryNode  # Node model
from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums
from memory_graph.models.proxy_models import ConversationRecord  # Input record
from memory_graph.persistence.embedding_repository import EmbeddingRepository  # Embedding storage
from memory_graph.proxy.provider_registry import ProviderRegistry  # LLM provider access

# Module logger for extraction service operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ExtractionService:
    """Orchestrates the complete memory extraction pipeline.

    Receives conversation records, calls an LLM for structured extraction,
    parses the output, detects duplicates, and persists new memories.
    """

    def __init__(
        self,
        extraction_settings: ExtractionSettings,
        embedding_settings: EmbeddingSettings,
        provider_registry: ProviderRegistry,
        extraction_parser: ExtractionParser,
        embedding_service: EmbeddingService,
        embedding_repository: EmbeddingRepository,
        similarity_search: SimilaritySearch,
        graph_sync: GraphSync,
    ) -> None:
        """Initialize the extraction service with all required dependencies."""
        self._settings = extraction_settings  # Extraction configuration
        self._embedding_settings = embedding_settings  # Embedding config for thresholds
        self._provider_registry = provider_registry  # Access to LLM providers
        self._parser = extraction_parser  # Output parser
        self._embedding_service = embedding_service  # Vector generation
        self._embedding_repo = embedding_repository  # Vector storage
        self._similarity_search = similarity_search  # Duplicate detection
        self._graph_sync = graph_sync  # Synchronized persistence
        logger.info("extraction_service_initialized")  # Log initialization

    async def process_conversation(self, record: ConversationRecord) -> None:
        """Process a conversation record through the full extraction pipeline.

        This is the main entry point called by the extraction queue workers.
        """
        logger.info(  # Log processing start
            "extraction_processing_started",
            conversation_id=record.conversation_id,
        )

        # Step 1: Format the conversation for the extraction prompt
        messages_as_dicts = [  # Convert ChatMessage objects to dicts
            {"role": msg.role, "content": msg.content}
            for msg in record.messages
        ]
        conversation_text = format_conversation_for_extraction(  # Format transcript
            messages=messages_as_dicts,
            response_content=record.response_content,
        )

        if not conversation_text.strip():  # Empty conversation
            logger.debug("empty_conversation_skipping_extraction")  # Log skip
            return  # Nothing to extract

        # Step 2: Call the extraction LLM with retry
        extraction_response = await self._call_extraction_llm(conversation_text)  # LLM call

        if not extraction_response:  # LLM call failed after retries
            logger.warning(  # Log failure
                "extraction_llm_call_failed",
                conversation_id=record.conversation_id,
            )
            return  # Abort extraction

        # Step 3: Parse the LLM output into structured memories
        extracted_memories = self._parser.parse_extraction_response(extraction_response)  # Parse

        if not extracted_memories:  # No memories extracted
            logger.info("no_memories_extracted", conversation_id=record.conversation_id)  # Log
            return  # Nothing to persist

        # Step 4: Process each extracted memory (deduplicate, persist)
        for memory_data in extracted_memories:  # Process each extraction
            await self._process_extracted_memory(memory_data, record.conversation_id)  # Handle

        logger.info(  # Log completion
            "extraction_processing_complete",
            conversation_id=record.conversation_id,
            memories_extracted=len(extracted_memories),
        )

    async def _call_extraction_llm(self, conversation_text: str) -> str | None:
        """Call the extraction LLM with retry logic.

        Returns the raw LLM response text or None if all retries fail.
        """
        # Build the extraction prompts
        system_prompt, user_prompt = build_extraction_prompt(  # Generate prompts
            conversation_text=conversation_text,
            max_memories=self._settings.max_memories_per_conversation,
        )

        # Get the provider for extraction
        provider = self._provider_registry.get_provider(self._settings.provider)  # Look up
        if provider is None:  # Provider not available
            logger.error("extraction_provider_not_available", provider=self._settings.provider)
            return None  # Cannot proceed

        # Retry loop with configurable attempts
        for attempt in range(self._settings.retry_attempts):  # Try up to max retries
            try:  # Attempt LLM call
                from memory_graph.models.proxy_models import ChatMessage, ProxyRequest  # Import

                extraction_request = ProxyRequest(  # Build extraction request
                    provider=self._settings.provider,
                    model=self._settings.model,
                    messages=[
                        ChatMessage(role="system", content=system_prompt),  # System prompt
                        ChatMessage(role="user", content=user_prompt),  # User prompt
                    ],
                    stream=False,  # No streaming for extraction
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000,  # Enough for structured output
                    conversation_id="disabled",  # Don't inject context into extraction calls
                )

                response = await provider.forward_request(extraction_request)  # Forward
                return response.content  # Return the response text

            except Exception as exc:  # LLM call failed
                logger.warning(  # Log retry
                    "extraction_llm_attempt_failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < self._settings.retry_attempts - 1:  # More retries available
                    await asyncio.sleep(self._settings.retry_delay_seconds)  # Wait before retry

        return None  # All retries exhausted

    async def _process_extracted_memory(
        self,
        memory_data: dict[str, Any],
        conversation_id: str,
    ) -> None:
        """Process a single extracted memory: deduplicate and persist.

        Checks for duplicates via embedding similarity, creates or updates
        the memory node, and establishes suggested edges.
        """
        content = memory_data["content"]  # Memory content text

        # Generate embedding for the extracted content
        embedding = self._embedding_service.encode(content)  # Encode to vector

        # Check for duplicates using similarity search
        similar_results = self._similarity_search.find_similar(  # Search
            query_embedding=embedding,
            top_k=1,  # Only need closest match
            threshold=self._embedding_settings.similarity_threshold,  # Duplicate threshold
        )

        if similar_results:  # Found a very similar existing memory
            # Treat as duplicate - update existing node's confidence
            existing_id = similar_results[0].node_id  # Get the existing node ID
            logger.info(  # Log dedup
                "duplicate_memory_detected",
                existing_id=existing_id,
                similarity=similar_results[0].score,
            )
            from memory_graph.persistence.node_repository import NodeRepository  # Import
            # Just log for now - updating existing nodes handled by graph_sync
            return  # Skip creating duplicate

        # No duplicate found - create a new memory node
        node = MemoryNode(  # Construct new node
            node_type=NodeType(memory_data["node_type"]),  # Set type from extraction
            content=content,  # Extracted content
            summary=memory_data["summary"],  # Extracted summary
            confidence=memory_data["confidence"],  # Extracted confidence
            emotional_valence=EmotionValence(memory_data["emotional_valence"]),  # Valence
            intensity=memory_data["intensity"],  # Extracted intensity
            source_conversation_id=conversation_id,  # Reference to source conversation
            tags=memory_data["tags"],  # Extracted tags
        )

        # Persist the new node
        success = await self._graph_sync.sync_add_node(node)  # Write to both stores
        if not success:  # Graph capacity reached
            logger.warning("extraction_node_not_added_capacity", node_id=node.id)  # Log
            return  # Cannot persist

        # Store the embedding
        self._embedding_repo.store(  # Persist embedding vector
            node_id=node.id,
            embedding=embedding,
            model_name=self._embedding_service._model_name,
        )
        self._similarity_search.add_to_index(node.id, embedding)  # Update search index

        # Process suggested edges
        for edge_suggestion in memory_data.get("suggested_edges", []):  # Handle edges
            await self._process_suggested_edge(node.id, edge_suggestion)  # Create edge

        logger.info("extracted_memory_persisted", node_id=node.id)  # Log success

    async def _process_suggested_edge(
        self,
        source_node_id: str,
        edge_suggestion: dict[str, str],
    ) -> None:
        """Attempt to create a suggested edge by finding the target node.

        Uses semantic search to find the node matching the target content hint.
        """
        target_hint = edge_suggestion["target_content_hint"]  # Content hint for target
        edge_type_str = edge_suggestion["edge_type"]  # Relationship type string

        # Search for the target node using the content hint
        search_results = self._similarity_search.find_similar_to_text(  # Search
            text=target_hint,
            top_k=1,  # Only need best match
            threshold=0.5,  # Moderate threshold for edge creation
            exclude_ids={source_node_id},  # Don't link to self
        )

        if not search_results:  # No matching target found
            return  # Cannot create edge without target

        target_node_id = search_results[0].node_id  # Best matching node

        # Create the edge
        edge = MemoryEdge(  # Build edge model
            source_node_id=source_node_id,  # From new memory
            target_node_id=target_node_id,  # To existing memory
            edge_type=EdgeType(edge_type_str),  # Relationship type
            weight=0.6,  # Moderate initial weight for auto-created edges
            description=f"Auto-extracted: {target_hint[:100]}",  # Description from hint
        )

        await self._graph_sync.sync_add_edge(edge)  # Persist the edge
        logger.debug("suggested_edge_created", edge_id=edge.id)  # Log creation
