"""Token counting and budget management for context injection."""

import structlog  # Structured logging
import tiktoken  # OpenAI tokenizer for accurate token counting

from memory_graph.config.settings import ContextSettings  # Context configuration

# Module logger for token budget operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class TokenBudget:
    """Manages token counting and budget allocation for memory context.

    Uses tiktoken for accurate token counting with a configurable
    maximum budget to prevent oversized context injection.
    """

    def __init__(self, settings: ContextSettings) -> None:
        """Initialize the token budget manager with configuration."""
        self._max_budget = settings.max_token_budget  # Maximum tokens allowed
        self._encoding: tiktoken.Encoding | None = None  # Lazy-loaded tokenizer
        logger.info("token_budget_initialized", max_budget=self._max_budget)  # Log init

    @property
    def max_budget(self) -> int:
        """Get the configured maximum token budget."""
        return self._max_budget  # Return configured limit

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Uses tiktoken cl100k_base encoding for accurate counting.
        """
        encoder = self._get_encoder()  # Get or create the tokenizer
        token_count = len(encoder.encode(text))  # Encode and count tokens
        return token_count  # Return the token count

    def fits_in_budget(self, text: str, current_usage: int = 0) -> bool:
        """Check whether a text fits within the remaining token budget.

        Accounts for tokens already allocated to other memories.
        """
        remaining = self._max_budget - current_usage  # Calculate remaining budget
        text_tokens = self.count_tokens(text)  # Count tokens in the candidate text
        return text_tokens <= remaining  # True if text fits within budget

    def allocate_texts(self, texts: list[str], reserve_tokens: int = 50) -> list[str]:
        """Greedily allocate texts within the token budget.

        Selects texts in order until budget is exhausted.
        Reserves a fixed number of tokens for formatting overhead.
        """
        available_budget = self._max_budget - reserve_tokens  # Budget minus formatting reserve
        allocated: list[str] = []  # Accumulate selected texts
        used_tokens = 0  # Track tokens consumed so far

        for text in texts:  # Process texts in priority order
            text_tokens = self.count_tokens(text)  # Count this text's tokens

            if used_tokens + text_tokens > available_budget:  # Would exceed budget
                logger.debug(  # Log budget exhaustion
                    "token_budget_exhausted",
                    allocated_count=len(allocated),
                    remaining_budget=available_budget - used_tokens,
                )
                break  # Stop allocating

            allocated.append(text)  # Add to allocation
            used_tokens += text_tokens  # Track usage

        logger.debug(  # Log allocation results
            "texts_allocated",
            count=len(allocated),
            tokens_used=used_tokens,
            budget=available_budget,
        )
        return allocated  # Return texts that fit within budget

    def get_remaining(self, current_usage: int) -> int:
        """Calculate remaining token budget after current usage."""
        remaining = self._max_budget - current_usage  # Subtract usage from max
        return max(0, remaining)  # Never return negative

    def _get_encoder(self) -> tiktoken.Encoding:
        """Get or lazily initialize the tiktoken encoder."""
        if self._encoding is None:  # Not yet initialized
            self._encoding = tiktoken.get_encoding("cl100k_base")  # Load cl100k encoding
            logger.debug("tiktoken_encoder_loaded")  # Log encoder load
        return self._encoding  # Return the encoder instance
