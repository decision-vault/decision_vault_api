from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    max_input_tokens: int
    max_output_tokens: int


class TokenLimiter:
    def __init__(self, budget: TokenBudget):
        self.budget = budget

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len((text or "").split()) * 1.3))

    @staticmethod
    def compress_text(text: str, max_tokens: int) -> str:
        words = (text or "").split()
        max_words = max(1, int(max_tokens / 1.3))
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def enforce(self, *, input_tokens: int, output_tokens: int) -> None:
        if input_tokens > self.budget.max_input_tokens:
            raise ValueError(
                f"Input token budget exceeded ({input_tokens}>{self.budget.max_input_tokens})"
            )
        if output_tokens > self.budget.max_output_tokens:
            raise ValueError(
                f"Output token budget exceeded ({output_tokens}>{self.budget.max_output_tokens})"
            )
