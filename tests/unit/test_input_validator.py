"""Unit tests for the InputValidator security component."""

import pytest  # Test framework

from memory_graph.security.input_validator import InputValidator  # Validator under test
from memory_graph.config.settings import InputSettings  # Settings model


class TestInputValidatorContent:
    """Tests for content validation and sanitization."""

    def setup_method(self):
        """Create a validator with test settings."""
        settings = InputSettings(max_content_length=1000, max_tags=5)
        self.validator = InputValidator(settings)

    def test_valid_content_passes(self):
        """Test that normal content passes validation."""
        result = self.validator.validate_content("This is normal content")
        assert result == "This is normal content"  # Unchanged

    def test_exceeds_max_length_rejected(self):
        """Test that content exceeding max length is rejected."""
        long_content = "a" * 1001  # Exceeds 1000 limit
        with pytest.raises(ValueError, match="exceeds maximum length"):
            self.validator.validate_content(long_content)

    def test_empty_content_rejected(self):
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            self.validator.validate_content("   ")  # Only whitespace

    def test_null_bytes_removed(self):
        """Test that null bytes are stripped from content."""
        content_with_null = "Hello\x00World"
        result = self.validator.validate_content(content_with_null)
        assert "\x00" not in result  # Null bytes removed
        assert "HelloWorld" in result  # Content preserved

    def test_script_injection_escaped(self):
        """Test that script injection attempts are HTML-escaped."""
        malicious = '<script>alert("xss")</script>'
        result = self.validator.validate_content(malicious)
        assert "<script>" not in result  # Script tag escaped
        assert "&lt;script&gt;" in result  # HTML entities used


class TestInputValidatorTags:
    """Tests for tag validation."""

    def setup_method(self):
        """Create a validator with test settings."""
        settings = InputSettings(max_content_length=1000, max_tags=5)
        self.validator = InputValidator(settings)

    def test_valid_tags_pass(self):
        """Test that normal tags pass validation."""
        tags = ["python", "coding", "preferences"]
        result = self.validator.validate_tags(tags)
        assert result == ["python", "coding", "preferences"]  # Unchanged

    def test_exceeds_max_tags_rejected(self):
        """Test that too many tags are rejected."""
        tags = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]  # 6 > 5 limit
        with pytest.raises(ValueError, match="Maximum 5 tags"):
            self.validator.validate_tags(tags)

    def test_empty_tags_removed(self):
        """Test that whitespace-only tags are removed."""
        tags = ["valid", "  ", "", "also_valid"]
        result = self.validator.validate_tags(tags)
        assert result == ["valid", "also_valid"]  # Empty removed

    def test_long_tags_truncated(self):
        """Test that individual tags over 100 chars are truncated."""
        long_tag = "a" * 150  # Over 100 char limit
        result = self.validator.validate_tags([long_tag])
        assert len(result[0]) == 100  # Truncated to 100


class TestInputValidatorNodeId:
    """Tests for node ID validation."""

    def setup_method(self):
        """Create a validator with test settings."""
        settings = InputSettings(max_content_length=1000, max_tags=5)
        self.validator = InputValidator(settings)

    def test_valid_uuid_passes(self):
        """Test that a UUID-format ID passes."""
        result = self.validator.validate_node_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"  # Unchanged

    def test_empty_id_rejected(self):
        """Test that empty ID is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            self.validator.validate_node_id("")

    def test_null_byte_in_id_rejected(self):
        """Test that null bytes in ID are rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            self.validator.validate_node_id("valid\x00id")

    def test_long_id_rejected(self):
        """Test that unreasonably long ID is rejected."""
        with pytest.raises(ValueError, match="exceeds maximum length"):
            self.validator.validate_node_id("a" * 101)
