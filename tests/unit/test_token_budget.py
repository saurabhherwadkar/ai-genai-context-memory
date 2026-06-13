"""Unit tests for the TokenBudget manager."""

import pytest  # Test framework

from memory_graph.config.settings import ContextSettings, RankingWeights  # Config
from memory_graph.context.token_budget import TokenBudget  # Budget under test


class TestTokenBudgetCounting:
    """Tests for token counting functionality."""

    def setup_method(self):
        """Create a token budget with test settings."""
        settings = ContextSettings(
            max_token_budget=500,
            format="markdown",
            max_memories_injected=20,
            ranking_weights=RankingWeights(),
        )
        self.budget = TokenBudget(settings)

    def test_count_tokens_simple_text(self):
        """Test counting tokens in a simple text string."""
        count = self.budget.count_tokens("Hello world")
        assert count > 0  # Should be at least 2 tokens
        assert count < 10  # Should be very few tokens

    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string returns 0."""
        count = self.budget.count_tokens("")
        assert count == 0  # No tokens in empty string

    def test_count_tokens_long_text(self):
        """Test that longer text has proportionally more tokens."""
        short_count = self.budget.count_tokens("Short text")
        long_count = self.budget.count_tokens("This is a much longer piece of text " * 20)
        assert long_count > short_count  # More text = more tokens


class TestTokenBudgetAllocation:
    """Tests for text allocation within budget constraints."""

    def setup_method(self):
        """Create a token budget with small limit for testing."""
        settings = ContextSettings(
            max_token_budget=50,  # Small budget for testing
            format="markdown",
            max_memories_injected=20,
            ranking_weights=RankingWeights(),
        )
        self.budget = TokenBudget(settings)

    def test_allocate_within_budget(self):
        """Test that short texts all fit within budget."""
        texts = ["Hello", "World", "Test"]  # Very short texts
        allocated = self.budget.allocate_texts(texts, reserve_tokens=0)  # No reserve for this test
        assert len(allocated) == 3  # All fit

    def test_allocate_exceeds_budget(self):
        """Test that allocation stops when budget is exhausted."""
        texts = [
            "Short",  # Small
            "This is a longer sentence that takes more tokens",  # Medium
            "Another long sentence with many words to use up tokens quickly" * 3,  # Large
        ]
        allocated = self.budget.allocate_texts(texts)
        assert len(allocated) < len(texts)  # Not all fit

    def test_allocate_empty_list(self):
        """Test allocating empty list returns empty."""
        allocated = self.budget.allocate_texts([])
        assert allocated == []  # Empty in, empty out

    def test_reserve_tokens_reduces_budget(self):
        """Test that reserve_tokens parameter reduces available budget."""
        texts = ["Test " * 10]  # Moderately sized text
        # With high reserve, less budget available
        allocated_high_reserve = self.budget.allocate_texts(texts, reserve_tokens=45)
        # Very little budget left after high reserve
        assert len(allocated_high_reserve) <= 1  # May not fit


class TestTokenBudgetHelpers:
    """Tests for budget helper methods."""

    def setup_method(self):
        """Create a token budget with test settings."""
        settings = ContextSettings(
            max_token_budget=100,
            format="markdown",
            max_memories_injected=20,
            ranking_weights=RankingWeights(),
        )
        self.budget = TokenBudget(settings)

    def test_fits_in_budget_returns_true(self):
        """Test that short text fits in empty budget."""
        result = self.budget.fits_in_budget("Short text", current_usage=0)
        assert result is True  # Fits easily

    def test_fits_in_budget_returns_false(self):
        """Test that text doesn't fit when budget nearly exhausted."""
        result = self.budget.fits_in_budget(
            "A longer piece of text " * 20,  # Many tokens
            current_usage=90,  # Budget nearly used up
        )
        assert result is False  # Doesn't fit

    def test_get_remaining(self):
        """Test calculating remaining budget."""
        remaining = self.budget.get_remaining(current_usage=60)
        assert remaining == 40  # 100 - 60 = 40

    def test_get_remaining_never_negative(self):
        """Test that remaining is never negative."""
        remaining = self.budget.get_remaining(current_usage=150)  # Over budget
        assert remaining == 0  # Clamped to zero

    def test_max_budget_property(self):
        """Test the max_budget property returns configured value."""
        assert self.budget.max_budget == 100  # Configured value
