"""Input sanitization and validation for protection against injection attacks."""

import re  # Regular expressions for pattern matching
from html import escape as html_escape  # HTML entity escaping

import structlog  # Structured logging

from memory_graph.config.settings import InputSettings  # Input validation configuration

# Module logger for input validation operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# Pattern to detect potential script injection attempts
SCRIPT_INJECTION_PATTERN = re.compile(  # Compiled regex for performance
    r"<\s*script|javascript\s*:|on\w+\s*=",  # Match script tags and event handlers
    re.IGNORECASE,  # Case-insensitive matching
)

# Pattern to detect SQL injection attempts
SQL_INJECTION_PATTERN = re.compile(  # Compiled regex for SQL injection detection
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b.*\b(FROM|INTO|SET|TABLE)\b)",  # SQL keywords
    re.IGNORECASE,  # Case-insensitive
)

# Pattern to detect null bytes that could truncate strings
NULL_BYTE_PATTERN = re.compile(r"\x00")  # Match null byte characters


class InputValidator:
    """Validates and sanitizes user input to prevent injection attacks.

    Provides content length validation, injection detection, and
    safe character filtering for all user-supplied text fields.
    """

    def __init__(self, settings: InputSettings) -> None:
        """Initialize the validator with configured limits."""
        self._max_content_length = settings.max_content_length  # Maximum text length
        self._max_tags = settings.max_tags  # Maximum number of tags

    def validate_content(self, content: str) -> str:
        """Validate and sanitize memory content text.

        Checks length limits and removes potentially dangerous patterns.
        Returns the sanitized content string.
        Raises ValueError if content violates safety rules.
        """
        # Check for null bytes that could cause truncation
        if NULL_BYTE_PATTERN.search(content):  # Null byte detected
            logger.warning("null_byte_in_content")  # Log the attempt
            content = NULL_BYTE_PATTERN.sub("", content)  # Remove null bytes

        # Enforce maximum content length
        if len(content) > self._max_content_length:  # Content exceeds limit
            msg = f"Content exceeds maximum length of {self._max_content_length} characters"
            raise ValueError(msg)  # Reject oversized content

        # Check for empty content after stripping
        stripped_content = content.strip()  # Remove leading/trailing whitespace
        if not stripped_content:  # Empty after stripping
            msg = "Content must not be empty or whitespace-only"
            raise ValueError(msg)  # Reject empty content

        # Detect potential script injection
        if SCRIPT_INJECTION_PATTERN.search(content):  # Script pattern found
            logger.warning("script_injection_detected")  # Log the attempt
            content = html_escape(content)  # Escape HTML entities

        return content  # Return the validated content

    def validate_tags(self, tags: list[str]) -> list[str]:
        """Validate a list of tags for content and count limits.

        Returns cleaned tags within configured limits.
        Raises ValueError if tag count exceeds maximum.
        """
        if len(tags) > self._max_tags:  # Too many tags
            msg = f"Maximum {self._max_tags} tags allowed, got {len(tags)}"
            raise ValueError(msg)  # Reject excess tags

        # Clean and validate each tag
        validated_tags: list[str] = []  # Accumulate valid tags
        for tag in tags:  # Process each tag
            cleaned_tag = tag.strip()  # Remove whitespace
            if not cleaned_tag:  # Empty after strip
                continue  # Skip empty tags

            if len(cleaned_tag) > 100:  # Individual tag too long
                cleaned_tag = cleaned_tag[:100]  # Truncate to limit

            # Remove null bytes from tags
            cleaned_tag = NULL_BYTE_PATTERN.sub("", cleaned_tag)  # Strip null bytes
            validated_tags.append(cleaned_tag)  # Add to result

        return validated_tags  # Return validated tag list

    def validate_node_id(self, node_id: str) -> str:
        """Validate a node ID for safe format.

        Node IDs should be UUID strings without dangerous characters.
        """
        stripped_id = node_id.strip()  # Remove whitespace

        if not stripped_id:  # Empty after stripping
            msg = "Node ID must not be empty"
            raise ValueError(msg)  # Reject empty ID

        if len(stripped_id) > 100:  # Unreasonably long ID
            msg = "Node ID exceeds maximum length"
            raise ValueError(msg)  # Reject oversized ID

        # Check for null bytes
        if NULL_BYTE_PATTERN.search(stripped_id):  # Null byte in ID
            msg = "Node ID contains invalid characters"
            raise ValueError(msg)  # Reject dangerous ID

        return stripped_id  # Return validated ID

    def validate_query_text(self, text: str) -> str:
        """Validate query text used for context retrieval.

        Applies basic sanitization without rejecting legitimate queries.
        """
        # Remove null bytes
        cleaned = NULL_BYTE_PATTERN.sub("", text)  # Strip null bytes

        # Enforce length limit
        if len(cleaned) > self._max_content_length:  # Too long
            cleaned = cleaned[: self._max_content_length]  # Truncate to limit

        return cleaned.strip()  # Return trimmed result
