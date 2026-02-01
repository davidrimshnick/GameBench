"""Agentic loop: drives autonomous LLM learning via tool use."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from davechess.benchmark.llm_interface import ToolUseLLMClient, LLMResponse, ToolCallResult
from davechess.benchmark.token_tracker import TokenTracker
from davechess.benchmark.tools import ToolExecutor, ToolCall, TOOL_DEFINITIONS
from davechess.benchmark.prompt import build_agentic_system_prompt

logger = logging.getLogger("davechess.benchmark")


class AgentHarness:
    """Runs the autonomous agent learning loop.

    Sends tools to the LLM, executes tool calls, tracks tokens,
    and manages a rolling conversation window.
    """

    def __init__(self, llm_client: ToolUseLLMClient,
                 token_tracker: TokenTracker,
                 tool_executor: ToolExecutor,
                 token_budget: int,
                 eval_reserve: int = 50_000,
                 context_window: int = 20):
        self.llm = llm_client
        self.tracker = token_tracker
        self.executor = tool_executor
        self.system_prompt = build_agentic_system_prompt(token_budget)
        self.eval_reserve = eval_reserve
        self.context_window = context_window
        self.messages: list[dict] = []
        self.turn_count = 0
        self.transcript: list[dict] = []  # Full transcript for saving

    def run_learning_phase(self) -> dict:
        """Run the autonomous learning loop until budget is nearly spent.

        Returns:
            Stats about the learning phase.
        """
        logger.info(f"Starting learning phase. Budget: {self.tracker.budget:,} tokens, "
                     f"reserve: {self.eval_reserve:,} for eval")

        start_time = time.time()

        while self.tracker.remaining > self.eval_reserve:
            success = self._step()
            if not success:
                break

        elapsed = time.time() - start_time
        stats = {
            "turns": self.turn_count,
            "tokens_used": self.tracker.total_used,
            "tokens_remaining": self.tracker.remaining,
            "elapsed_sec": elapsed,
            **self.executor.stats,
        }
        logger.info(f"Learning phase complete: {self.turn_count} turns, "
                     f"{self.tracker.total_used:,} tokens used")
        return stats

    def play_eval_game(self, game_state_msg: str) -> Optional[str]:
        """Play a single move in an evaluation game.

        The harness presents the game state and gets the agent's move.

        Args:
            game_state_msg: Message describing the current game state and legal moves.

        Returns:
            The agent's move in DCN notation, or None if budget exhausted.
        """
        if self.tracker.exhausted:
            return None

        self.messages.append({"role": "user", "content": game_state_msg})
        self._trim_window()

        try:
            response = self.llm.chat_with_tools(
                self.system_prompt, self.messages, TOOL_DEFINITIONS
            )
        except Exception as e:
            logger.error(f"API error during eval: {e}")
            return None

        self.tracker.record(response.usage.prompt_tokens,
                            response.usage.completion_tokens)

        # Handle tool calls (agent might use play_move)
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "play_move":
                    return tc.arguments.get("move_dcn")
            # If no play_move tool call, try to extract from text
        if response.text:
            self.messages.append({"role": "assistant", "content": response.text})
            return _extract_move_from_text(response.text)

        return None

    def _step(self) -> bool:
        """Execute one turn of the agent loop.

        Returns:
            False if we should stop (budget exhausted, agent done, etc.)
        """
        self._trim_window()

        try:
            response = self.llm.chat_with_tools(
                self.system_prompt, self.messages, TOOL_DEFINITIONS
            )
        except Exception as e:
            logger.error(f"API error: {e}")
            return False

        self.tracker.record(response.usage.prompt_tokens,
                            response.usage.completion_tokens)
        self.turn_count += 1

        # Log to transcript
        self.transcript.append({
            "turn": self.turn_count,
            "response_text": response.text,
            "tool_calls": [
                {"name": tc.name, "args": tc.arguments}
                for tc in response.tool_calls
            ],
            "usage": self.tracker.call_log[-1].total if self.tracker.call_log else 0,
            "total_used": self.tracker.total_used,
        })

        if response.tool_calls:
            # Build assistant message with tool calls (OpenAI format)
            assistant_msg: dict = {"role": "assistant", "content": response.text or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            self.messages.append(assistant_msg)

            # Execute each tool call
            for tc in response.tool_calls:
                result = self.executor.execute(ToolCall(
                    id=tc.id, name=tc.name, arguments=tc.arguments
                ))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        elif response.text:
            self.messages.append({"role": "assistant", "content": response.text})

            # If agent says it's done learning, respect that
            lower = response.text.lower()
            if any(phrase in lower for phrase in
                   ["ready for evaluation", "ready to be evaluated",
                    "begin evaluation", "start evaluation"]):
                logger.info("Agent indicated readiness for evaluation")
                return False
        else:
            # Empty response
            return False

        # Check budget
        if self.tracker.remaining <= self.eval_reserve:
            logger.info("Budget reserve reached, ending learning phase")
            return False

        return True

    def _trim_window(self) -> None:
        """Keep only the last N messages in the rolling window."""
        if len(self.messages) > self.context_window:
            self.messages = self.messages[-self.context_window:]


def _extract_move_from_text(text: str) -> Optional[str]:
    """Try to extract a DCN move from free text (fallback for eval)."""
    import re
    patterns = [
        r'`([CWRBL][a-h][1-8][-x][a-h][1-8])`',
        r'`(\+[WRBL]@[a-h][1-8])`',
        r'`(B[a-h][1-8]~[a-h][1-8])`',
        r'([CWRBL][a-h][1-8][-x][a-h][1-8])',
        r'(\+[WRBL]@[a-h][1-8])',
        r'(B[a-h][1-8]~[a-h][1-8])',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None
