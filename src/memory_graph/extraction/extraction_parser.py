"""Parser for structured memory extraction LLM output."""

import json  # JSON parsing
import re  # Regular expressions for fallback extraction
from typing import Any  # Generic type

import structlog  # Structured logging

from memory_graph.models.memory_types import EdgeType, EmotionValence, NodeType  # Enums

# Module logger for parser operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# Regex pattern to find JSON in LLM output that may contain surrounding text
JSON_EXTRACTION_PATTERN = re.compile(  # Compiled regex for performance
    r"\{[\s\S]*\"extracted_memories\"[\s\S]*\}",  # Match JSON with extracted_memories key
)

# Valid node type values for validation
VALID_NODE_TYPES = {nt.value for nt in NodeType}  # Set of valid type strings

# Valid edge type values for validation
VALID_EDGE_TYPES = {et.value for et in EdgeType}  # Set of valid edge strings

# Valid emotional valence values
VALID_VALENCES = {ev.value for ev in EmotionValence}  # Set of valid valence strings


class ExtractionParser:
    """Parses and validates structured output from the extraction LLM.

    Handles well-formed JSON, JSON wrapped in markdown code fences,
    and attempts regex-based extraction for malformed responses.
    """

    def parse_extraction_response(self, raw_text: str) -> list[dict[str, Any]]:
        """Parse the extraction LLM response into validated memory dictionaries.

        Returns a list of validated memory extraction dictionaries.
        Returns empty list if parsing fails completely.
        """
        if not raw_text.strip():  # Empty response
            logger.warning("empty_extraction_response")  # Log empty
            return []  # Nothing to parse

        # Attempt to parse as direct JSON first
        parsed_data = self._try_parse_json(raw_text)  # Direct parse attempt

        if parsed_data is None:  # Direct parse failed
            # Try extracting JSON from markdown code fences
            parsed_data = self._try_extract_from_code_fence(raw_text)  # Fence extraction

        if parsed_data is None:  # Code fence extraction failed
            # Try regex-based extraction as last resort
            parsed_data = self._try_regex_extraction(raw_text)  # Regex fallback

        if parsed_data is None:  # All parsing attempts failed
            logger.error("extraction_parsing_failed", text_preview=raw_text[:200])  # Log failure
            return []  # Return empty

        # Extract the memories list from the parsed data
        memories_raw = parsed_data.get("extracted_memories", [])  # Get memories array

        if not isinstance(memories_raw, list):  # Invalid type
            logger.warning("extracted_memories_not_a_list")  # Log type error
            return []  # Return empty

        # Validate and normalize each extracted memory
        validated_memories: list[dict[str, Any]] = []  # Accumulate valid entries
        for raw_memory in memories_raw:  # Process each extraction
            validated = self._validate_memory(raw_memory)  # Validate
            if validated:  # Passed validation
                validated_memories.append(validated)  # Add to results

        logger.info("extraction_parsed", count=len(validated_memories))  # Log success
        return validated_memories  # Return validated memories

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt direct JSON parsing of the response text."""
        try:  # Try parsing the entire text as JSON
            result = json.loads(text.strip())  # Parse and strip whitespace
            if isinstance(result, dict):  # Valid JSON object
                return result  # Return parsed dict
        except json.JSONDecodeError:  # Not valid JSON
            pass  # Fall through to next method
        return None  # Parsing failed

    def _try_extract_from_code_fence(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from markdown code fences (```json ... ```)."""
        # Match content between code fence markers
        fence_pattern = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```")  # Fence regex
        match = fence_pattern.search(text)  # Find code fence

        if match:  # Found a code fence
            json_content = match.group(1).strip()  # Extract content
            return self._try_parse_json(json_content)  # Try parsing extracted content

        return None  # No code fence found

    def _try_regex_extraction(self, text: str) -> dict[str, Any] | None:
        """Attempt to extract JSON using regex pattern matching."""
        match = JSON_EXTRACTION_PATTERN.search(text)  # Search for JSON structure

        if match:  # Found potential JSON
            json_candidate = match.group(0)  # Get the matched text
            return self._try_parse_json(json_candidate)  # Try parsing it

        return None  # No match found

    def _validate_memory(self, raw: Any) -> dict[str, Any] | None:
        """Validate and normalize a single extracted memory dictionary.

        Returns None if the memory fails validation.
        """
        if not isinstance(raw, dict):  # Must be a dictionary
            return None  # Invalid type

        # Validate required fields
        content = raw.get("content", "").strip()  # Get content
        if not content:  # Content is required
            return None  # Missing content

        summary = raw.get("summary", "").strip()  # Get summary
        if not summary:  # Generate from content if missing
            summary = content[:100]  # Truncate content as summary

        # Validate and normalize node_type
        node_type = raw.get("node_type", "thought")  # Default to thought
        if node_type not in VALID_NODE_TYPES:  # Invalid type
            node_type = "thought"  # Fall back to thought

        # Validate and clamp confidence
        confidence = self._clamp_float(raw.get("confidence", 0.7), 0.0, 1.0)  # Clamp

        # Validate emotional_valence
        valence = raw.get("emotional_valence", "neutral")  # Default neutral
        if valence not in VALID_VALENCES:  # Invalid valence
            valence = "neutral"  # Fall back to neutral

        # Validate and clamp intensity
        intensity = self._clamp_float(raw.get("intensity", 0.5), 0.0, 1.0)  # Clamp

        # Validate tags (must be list of strings)
        tags = raw.get("tags", [])  # Get tags
        if not isinstance(tags, list):  # Invalid type
            tags = []  # Default to empty
        tags = [str(t).strip() for t in tags if t][:5]  # Clean and limit to 5

        # Validate suggested_edges
        suggested_edges = self._validate_edges(raw.get("suggested_edges", []))  # Validate edges

        return {  # Return the validated memory dictionary
            "content": content,
            "summary": summary[:500],  # Enforce summary length limit
            "node_type": node_type,
            "confidence": confidence,
            "emotional_valence": valence,
            "intensity": intensity,
            "tags": tags,
            "suggested_edges": suggested_edges,
        }

    def _validate_edges(self, raw_edges: Any) -> list[dict[str, str]]:
        """Validate and normalize suggested edge relationships."""
        if not isinstance(raw_edges, list):  # Must be a list
            return []  # Default to empty

        validated: list[dict[str, str]] = []  # Accumulate valid edges
        for edge in raw_edges:  # Process each edge
            if not isinstance(edge, dict):  # Must be dict
                continue  # Skip invalid

            target_hint = str(edge.get("target_content_hint", "")).strip()  # Get hint
            edge_type = str(edge.get("edge_type", "related_to")).strip()  # Get type

            if not target_hint:  # Missing hint
                continue  # Skip

            if edge_type not in VALID_EDGE_TYPES:  # Invalid type
                edge_type = "related_to"  # Default to related_to

            validated.append({  # Add validated edge
                "target_content_hint": target_hint,
                "edge_type": edge_type,
            })

        return validated  # Return validated edges

    def _clamp_float(self, value: Any, min_val: float, max_val: float) -> float:
        """Clamp a value to a float within the specified range."""
        try:  # Attempt conversion and clamping
            float_val = float(value)  # Convert to float
            return max(min_val, min(max_val, float_val))  # Clamp to range
        except (TypeError, ValueError):  # Conversion failed
            return (min_val + max_val) / 2  # Return midpoint as default
