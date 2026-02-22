from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    max_input_tokens: int
    max_output_tokens: int


class TokenLimiter:
    def __init__(self, budget: TokenBudget):
        self.budget = budget

    def enforce(self, *, input_tokens: int, output_tokens: int) -> None:
        if input_tokens > self.budget.max_input_tokens:
            raise ValueError(
                f"Input token budget exceeded ({input_tokens}>{self.budget.max_input_tokens})"
            )
        if output_tokens > self.budget.max_output_tokens:
            raise ValueError(
                f"Output token budget exceeded ({output_tokens}>{self.budget.max_output_tokens})"
            )

