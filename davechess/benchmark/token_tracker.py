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

    def budget_message(self, phase: str = "learning") -> str:
        """Generate a human-readable budget status message.

        Args:
            phase: Current session phase (baseline/learning/evaluation/completed).
        """
        remaining = self.remaining
        budget = self.budget

        if self.total_used == 0:
            return (
                f"You have {budget:,} tokens available. "
                f"Report your usage with --tokens on any command to track your budget."
            )

        if self.exhausted:
            return "Token budget exhausted. Wrapping up evaluation."

        pct_remaining = (remaining / budget * 100) if budget > 0 else 0

        if pct_remaining < 5:
            return (
                f"Budget nearly exhausted ({remaining:,} of {budget:,} tokens remaining). "
                f"Transition to evaluation if you haven't already."
            )

        if phase == "learning":
            if pct_remaining > 70:
                return (
                    f"You have {remaining:,} of {budget:,} tokens remaining ({pct_remaining:.0f}%). "
                    f"Keep studying and practicing — more learning means higher ELO!"
                )
            elif pct_remaining > 30:
                return (
                    f"You have {remaining:,} of {budget:,} tokens remaining ({pct_remaining:.0f}%). "
                    f"Good progress! Continue practicing to build your skills before evaluating."
                )
            else:
                return (
                    f"You have {remaining:,} of {budget:,} tokens remaining ({pct_remaining:.0f}%). "
                    f"Consider transitioning to evaluation soon with the 'evaluate' command."
                )
        elif phase == "evaluation":
            return (
                f"You have {remaining:,} of {budget:,} tokens remaining ({pct_remaining:.0f}%). "
                f"Playing rated evaluation games."
            )
        elif phase == "baseline":
            return (
                f"You have {remaining:,} of {budget:,} tokens remaining ({pct_remaining:.0f}%). "
                f"Playing baseline games to establish your starting ELO."
            )
        else:
            return f"Session complete. Used {self.total_used:,} of {budget:,} tokens."
