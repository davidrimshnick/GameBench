"""Token usage tracking for agentic benchmark budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Token counts from a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    """Tracks cumulative token usage against a budget.

    All tokens (input + output) count toward the budget.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.usage = TokenUsage()
        self.call_log: list[TokenUsage] = []

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an API call."""
        call = TokenUsage(prompt_tokens, completion_tokens)
        self.call_log.append(call)
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens

    @property
    def total_used(self) -> int:
        return self.usage.total

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.usage.total)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def num_calls(self) -> int:
        return len(self.call_log)

    def has_budget_for(self, estimated_tokens: int) -> bool:
        """Check if there's enough budget for an estimated call cost."""
        return self.remaining >= estimated_tokens

    def summary(self) -> dict:
        """Return a summary of token usage."""
        return {
            "budget": self.budget,
            "total_used": self.total_used,
            "remaining": self.remaining,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "num_calls": self.num_calls,
        }
