"""Unit tests for the ExtractionParser."""

import pytest  # Test framework

from memory_graph.extraction.extraction_parser import ExtractionParser  # Parser under test


class TestExtractionParserValidJSON:
    """Tests for parsing well-formed JSON extraction responses."""

    def setup_method(self):
        """Create a fresh parser instance for each test."""
        self.parser = ExtractionParser()  # New parser

    def test_parse_valid_response(self):
        """Test parsing a well-formed extraction response."""
        raw = '''{
            "extracted_memories": [
                {
                    "content": "User prefers TypeScript for large projects",
                    "summary": "Prefers TypeScript for large projects",
                    "node_type": "preference",
                    "confidence": 0.9,
                    "emotional_valence": "positive",
                    "intensity": 0.7,
                    "tags": ["programming", "typescript"],
                    "suggested_edges": []
                }
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)  # Parse

        assert len(result) == 1  # One memory extracted
        assert result[0]["content"] == "User prefers TypeScript for large projects"
        assert result[0]["node_type"] == "preference"
        assert result[0]["confidence"] == 0.9

    def test_parse_multiple_memories(self):
        """Test parsing multiple extracted memories."""
        raw = '''{
            "extracted_memories": [
                {"content": "Memory one", "summary": "One", "node_type": "thought",
                 "confidence": 0.8, "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [], "suggested_edges": []},
                {"content": "Memory two", "summary": "Two", "node_type": "belief",
                 "confidence": 0.7, "emotional_valence": "positive", "intensity": 0.6,
                 "tags": ["test"], "suggested_edges": []}
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)  # Parse

        assert len(result) == 2  # Two memories extracted
        assert result[0]["node_type"] == "thought"
        assert result[1]["node_type"] == "belief"


class TestExtractionParserMalformed:
    """Tests for handling malformed or edge-case responses."""

    def setup_method(self):
        """Create a fresh parser instance for each test."""
        self.parser = ExtractionParser()  # New parser

    def test_parse_empty_string(self):
        """Test that empty string returns empty list."""
        result = self.parser.parse_extraction_response("")  # Empty input
        assert result == []  # Empty output

    def test_parse_invalid_json(self):
        """Test that invalid JSON returns empty list."""
        result = self.parser.parse_extraction_response("not json at all")  # Invalid
        assert result == []  # Graceful failure

    def test_parse_json_in_code_fence(self):
        """Test extracting JSON from markdown code fences."""
        raw = '''Here are the extracted memories:
```json
{
    "extracted_memories": [
        {"content": "Fenced memory", "summary": "Fenced", "node_type": "thought",
         "confidence": 0.8, "emotional_valence": "neutral", "intensity": 0.5,
         "tags": [], "suggested_edges": []}
    ]
}
```'''
        result = self.parser.parse_extraction_response(raw)  # Parse with fence

        assert len(result) == 1  # Extracted from fence
        assert result[0]["content"] == "Fenced memory"

    def test_invalid_node_type_defaults_to_thought(self):
        """Test that invalid node_type falls back to 'thought'."""
        raw = '''{
            "extracted_memories": [
                {"content": "Test memory", "summary": "Test", "node_type": "invalid_type",
                 "confidence": 0.5, "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [], "suggested_edges": []}
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)

        assert result[0]["node_type"] == "thought"  # Defaulted

    def test_confidence_clamped_to_range(self):
        """Test that out-of-range confidence is clamped."""
        raw = '''{
            "extracted_memories": [
                {"content": "Test", "summary": "Test", "node_type": "thought",
                 "confidence": 5.0, "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [], "suggested_edges": []}
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)

        assert result[0]["confidence"] == 1.0  # Clamped to max

    def test_missing_content_skipped(self):
        """Test that memories without content are skipped."""
        raw = '''{
            "extracted_memories": [
                {"content": "", "summary": "Empty", "node_type": "thought",
                 "confidence": 0.5, "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [], "suggested_edges": []},
                {"content": "Valid memory", "summary": "Valid", "node_type": "belief",
                 "confidence": 0.8, "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [], "suggested_edges": []}
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)

        assert len(result) == 1  # Only valid one kept
        assert result[0]["content"] == "Valid memory"

    def test_suggested_edges_validated(self):
        """Test that suggested edges are validated and normalized."""
        raw = '''{
            "extracted_memories": [
                {"content": "Memory with edges", "summary": "Edges test",
                 "node_type": "thought", "confidence": 0.8,
                 "emotional_valence": "neutral", "intensity": 0.5,
                 "tags": [],
                 "suggested_edges": [
                     {"target_content_hint": "related memory", "edge_type": "supports"},
                     {"target_content_hint": "", "edge_type": "related_to"},
                     {"target_content_hint": "another", "edge_type": "invalid_type"}
                 ]}
            ]
        }'''
        result = self.parser.parse_extraction_response(raw)
        edges = result[0]["suggested_edges"]

        assert len(edges) == 2  # Empty hint skipped
        assert edges[0]["edge_type"] == "supports"  # Valid preserved
        assert edges[1]["edge_type"] == "related_to"  # Invalid defaulted
