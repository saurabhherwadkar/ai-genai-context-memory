"""Formats ranked memories into injectable text blocks for LLM context."""

import structlog  # Structured logging

from memory_graph.models.memory_node import MemoryNode  # Node data model
from memory_graph.models.memory_types import NodeType  # Node classification enum

# Module logger for formatting operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# Mapping of node types to human-readable section headers
NODE_TYPE_HEADERS: dict[str, str] = {  # Section header for each node type
    NodeType.THOUGHT.value: "Thoughts & Observations",
    NodeType.BELIEF.value: "Beliefs & Convictions",
    NodeType.AVERSION.value: "Aversions & Dislikes",
    NodeType.PREFERENCE.value: "Preferences & Likes",
    NodeType.EMOTION.value: "Emotional States",
    NodeType.GOAL.value: "Goals & Aspirations",
    NodeType.EXPERIENCE.value: "Past Experiences",
    NodeType.FACT.value: "Known Facts",
    NodeType.HABIT.value: "Habits & Patterns",
    NodeType.VALUE.value: "Core Values",
    NodeType.RELATIONSHIP.value: "Relationships",
    NodeType.SKILL.value: "Skills & Competencies",
    NodeType.CUSTOM.value: "Other Memories",
}


class ContextFormatter:
    """Formats ranked memory nodes into structured text for LLM context injection.

    Supports multiple output formats (markdown, xml, plain) and groups
    memories by type with appropriate headers and metadata annotations.
    """

    def __init__(self, default_format: str = "markdown") -> None:
        """Initialize the formatter with the default output format."""
        self._default_format = default_format  # Store default format preference
        logger.info("context_formatter_initialized", format=default_format)  # Log init

    def format_memories(
        self,
        memories: list[tuple[MemoryNode, float]],
        output_format: str | None = None,
    ) -> str:
        """Format a ranked list of memories into a context injection text block.

        Groups memories by node type and includes confidence annotations.
        """
        if not memories:  # No memories to format
            return ""  # Return empty string

        # Use specified format or fall back to default
        fmt = output_format or self._default_format  # Resolve format

        # Dispatch to the appropriate formatter
        if fmt == "markdown":  # Markdown format
            return self._format_markdown(memories)  # Delegate to markdown
        elif fmt == "xml":  # XML format
            return self._format_xml(memories)  # Delegate to XML
        else:  # Plain text format
            return self._format_plain(memories)  # Delegate to plain

    def _format_markdown(self, memories: list[tuple[MemoryNode, float]]) -> str:
        """Format memories as a markdown text block grouped by type."""
        # Group memories by their node type
        grouped = self._group_by_type(memories)  # Dict of type -> list of (node, score)

        # Build the markdown output
        lines: list[str] = []  # Accumulate output lines
        lines.append("<memory_context>")  # Opening tag for LLM to identify context block

        for node_type, type_memories in grouped.items():  # Process each type group
            header = NODE_TYPE_HEADERS.get(node_type, "Other Memories")  # Get section header
            lines.append(f"\n## {header}")  # Add section header

            for node, score in type_memories:  # Process each memory in the group
                confidence_label = self._confidence_label(node.confidence)  # Human-friendly confidence
                lines.append(f"- {node.summary} [{confidence_label}]")  # Bullet with confidence

        lines.append("\n</memory_context>")  # Closing tag

        formatted_text = "\n".join(lines)  # Join all lines
        logger.debug("memories_formatted_markdown", memory_count=len(memories))  # Log formatting
        return formatted_text  # Return the formatted block

    def _format_xml(self, memories: list[tuple[MemoryNode, float]]) -> str:
        """Format memories as an XML structure for LLM parsing."""
        # Group memories by their node type
        grouped = self._group_by_type(memories)  # Dict of type -> list

        # Build the XML output
        lines: list[str] = []  # Accumulate output lines
        lines.append("<memory_context>")  # Root element

        for node_type, type_memories in grouped.items():  # Process each type
            lines.append(f'  <category type="{node_type}">')  # Category element

            for node, score in type_memories:  # Process each memory
                lines.append(f'    <memory confidence="{node.confidence:.2f}">')  # Memory element
                lines.append(f"      {node.summary}")  # Memory content
                lines.append("    </memory>")  # Close memory element

            lines.append("  </category>")  # Close category

        lines.append("</memory_context>")  # Close root

        formatted_text = "\n".join(lines)  # Join all lines
        logger.debug("memories_formatted_xml", memory_count=len(memories))  # Log formatting
        return formatted_text  # Return the XML block

    def _format_plain(self, memories: list[tuple[MemoryNode, float]]) -> str:
        """Format memories as plain text with minimal formatting."""
        # Group memories by their node type
        grouped = self._group_by_type(memories)  # Dict of type -> list

        # Build plain text output
        lines: list[str] = []  # Accumulate output lines
        lines.append("[Memory Context]")  # Header marker

        for node_type, type_memories in grouped.items():  # Process each type
            header = NODE_TYPE_HEADERS.get(node_type, "Other")  # Section header
            lines.append(f"\n{header}:")  # Add section header

            for node, score in type_memories:  # Process each memory
                lines.append(f"  * {node.summary}")  # Indented bullet

        formatted_text = "\n".join(lines)  # Join all lines
        logger.debug("memories_formatted_plain", memory_count=len(memories))  # Log formatting
        return formatted_text  # Return the plain text block

    def _group_by_type(
        self, memories: list[tuple[MemoryNode, float]]
    ) -> dict[str, list[tuple[MemoryNode, float]]]:
        """Group memories by their node type value string."""
        grouped: dict[str, list[tuple[MemoryNode, float]]] = {}  # Initialize grouping dict

        for node, score in memories:  # Process each memory
            type_key = node.node_type.value  # Get type as string key
            if type_key not in grouped:  # New group
                grouped[type_key] = []  # Initialize list for this type
            grouped[type_key].append((node, score))  # Add to appropriate group

        return grouped  # Return the grouped memories

    def _confidence_label(self, confidence: float) -> str:
        """Convert a numeric confidence score to a human-readable label."""
        if confidence >= 0.9:  # Very high confidence
            return "very confident"  # Label for 90%+
        elif confidence >= 0.7:  # High confidence
            return "confident"  # Label for 70-89%
        elif confidence >= 0.5:  # Moderate confidence
            return "moderate"  # Label for 50-69%
        else:  # Low confidence
            return "uncertain"  # Label for below 50%
