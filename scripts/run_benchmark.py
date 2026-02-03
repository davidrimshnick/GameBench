#!/usr/bin/env python3
"""External testing harness for the DaveChess benchmark.

Orchestrates the full benchmark by calling the LLM API directly (via
ToolUseLLMClient) and executing tool calls through agent_cli.py.
Token usage is extracted from every API response and reported to the
session, making the token budget the binding constraint.

Usage:
    python scripts/run_benchmark.py \
      --provider anthropic --model claude-sonnet-4-20250514 \
      --budget 500000 --name "claude-sonnet-500k"
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.benchmark.llm_interface import ToolUseLLMClient, LLMResponse
from davechess.benchmark.prompt import build_agentic_system_prompt
from davechess.benchmark.tools import TOOL_DEFINITIONS, tools_to_anthropic_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark_harness")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, os.path.join(REPO_ROOT, "scripts", "agent_cli.py")]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def cli_run(command: list[str]) -> dict:
    """Run an agent_cli.py command, return parsed JSON output."""
    full_cmd = CLI + command
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        full_cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        logger.error(f"CLI error: {result.stderr}")
        return {"error": result.stderr}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Some commands (like rules) return plain text
        return {"text": result.stdout}


def report_tokens(session_file: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """Report token usage to the session."""
    return cli_run([
        "report-tokens", session_file,
        str(prompt_tokens), str(completion_tokens),
    ])


def get_status(session_file: str) -> dict:
    return cli_run(["status", session_file])


# ---------------------------------------------------------------------------
# Tool call -> CLI mapping
# ---------------------------------------------------------------------------

def execute_tool(session_file: str, tool_name: str, args: dict) -> str:
    """Execute a tool call by mapping it to an agent_cli.py command."""
    if tool_name == "study_games":
        result = cli_run(["study", session_file, str(args.get("n", 1))])
    elif tool_name == "start_practice_game":
        result = cli_run(["practice", session_file, str(args.get("opponent_elo", 1000))])
    elif tool_name == "play_move":
        move_dcn = args.get("move_dcn", "")
        result = cli_run(["move", session_file, move_dcn])
    elif tool_name == "get_game_state":
        result = cli_run(["state", session_file])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """External testing harness that wraps LLM API + agent_cli.py."""

    def __init__(self, llm: ToolUseLLMClient, session_file: str,
                 token_budget: int, eval_reserve: int = 50_000,
                 context_window: int = 20):
        self.llm = llm
        self.session_file = session_file
        self.budget = token_budget
        self.eval_reserve = eval_reserve
        self.context_window = context_window

        self.system_prompt = build_agentic_system_prompt(token_budget)
        self.messages: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.turn_count = 0
        self.transcript: list[dict] = []

        # Get tool definitions in the right format
        if llm.provider == "anthropic":
            self.tools = tools_to_anthropic_format()
        else:
            self.tools = TOOL_DEFINITIONS

    @property
    def tokens_used(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.budget - self.tokens_used)

    def _record_tokens(self, response: LLMResponse):
        """Record token usage from an API response."""
        prompt = response.usage.prompt_tokens
        completion = response.usage.completion_tokens
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        # Report to session
        report_tokens(self.session_file, prompt, completion)
        logger.info(
            f"Turn {self.turn_count}: +{prompt + completion} tokens "
            f"({self.tokens_used:,}/{self.budget:,} used, "
            f"{self.tokens_remaining:,} remaining)"
        )

    def _trim_window(self):
        """Keep only the last N messages."""
        if len(self.messages) > self.context_window:
            self.messages = self.messages[-self.context_window:]

    def _call_llm(self) -> LLMResponse:
        """Make one LLM API call."""
        self._trim_window()
        response = self.llm.chat_with_tools(
            self.system_prompt, self.messages, self.tools,
        )
        self.turn_count += 1
        self._record_tokens(response)
        return response

    def run_learning_phase(self):
        """Run the autonomous learning loop until budget reserve reached."""
        logger.info(f"=== LEARNING PHASE (budget: {self.budget:,}, "
                     f"reserve: {self.eval_reserve:,}) ===")

        while self.tokens_remaining > self.eval_reserve:
            try:
                response = self._call_llm()
            except Exception as e:
                logger.error(f"API error: {e}")
                break

            # Log to transcript
            self.transcript.append({
                "turn": self.turn_count,
                "phase": "learning",
                "text": response.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ],
                "tokens_used": self.tokens_used,
            })

            if response.tool_calls:
                # Build assistant message
                if self.llm.provider == "anthropic":
                    self.messages.append({
                        "role": "assistant",
                        "content": self._build_anthropic_assistant_content(response),
                    })
                else:
                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": response.text or "",
                        "tool_calls": [
                            {
                                "id": tc.id, "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                    self.messages.append(assistant_msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    result_str = execute_tool(
                        self.session_file, tc.name, tc.arguments,
                    )
                    if self.llm.provider == "anthropic":
                        self.messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc.id,
                                    "content": result_str,
                                }
                            ],
                        })
                    else:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        })

            elif response.text:
                self.messages.append({"role": "assistant", "content": response.text})
                # Check if agent wants to move to eval
                lower = response.text.lower()
                if any(p in lower for p in [
                    "ready for evaluation", "begin evaluation",
                    "start evaluation", "ready to be evaluated",
                ]):
                    logger.info("Agent indicated readiness for evaluation")
                    break
            else:
                logger.warning("Empty response from LLM")
                break

        logger.info(f"Learning phase complete: {self.turn_count} turns, "
                     f"{self.tokens_used:,} tokens used")

    def run_eval_phase(self):
        """Transition to evaluation and play rated games."""
        logger.info("=== EVALUATION PHASE ===")

        # Transition to eval
        result = cli_run(["evaluate", self.session_file])
        if "error" in result:
            logger.error(f"Failed to start evaluation: {result['error']}")
            return

        # Play eval games until session completes or budget exhausted
        while self.tokens_remaining > 0:
            status = get_status(self.session_file)
            phase = status.get("phase", "")
            if phase == "completed":
                logger.info("Evaluation complete!")
                break

            # Get current game state
            state = cli_run(["state", self.session_file])
            if "error" in state or state.get("info"):
                logger.info(f"No active game: {state}")
                break

            legal_moves = state.get("legal_moves", [])
            if not legal_moves:
                # Game might have ended, check again
                continue

            # Ask LLM for a move
            game_msg = self._format_eval_state(state)
            self.messages.append({"role": "user", "content": game_msg})

            try:
                response = self._call_llm()
            except Exception as e:
                logger.error(f"API error during eval: {e}")
                break

            self.transcript.append({
                "turn": self.turn_count,
                "phase": "evaluation",
                "text": response.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ],
                "tokens_used": self.tokens_used,
            })

            # Extract move from response
            move = self._extract_move(response, legal_moves)
            if not move:
                logger.warning("Could not extract move, picking first legal move")
                move = legal_moves[0]

            self.messages.append({"role": "assistant", "content": move})

            # Play the move
            move_result = cli_run(["move", self.session_file, move])
            if "error" in move_result:
                logger.warning(f"Move error: {move_result['error']}")
                # Try first legal move as fallback
                if move_result.get("legal_moves"):
                    move = move_result["legal_moves"][0]
                    move_result = cli_run(["move", self.session_file, move])

        logger.info(f"Eval phase complete: {self.tokens_used:,} tokens total")

    def _format_eval_state(self, state: dict) -> str:
        """Format game state for the LLM during evaluation."""
        parts = [
            f"You are playing a rated evaluation game.",
            f"You are {state.get('agent_color', 'white')}.",
            f"Turn: {state.get('turn', '?')}",
            f"Your resources: {state.get('your_resources', 0)}, "
            f"Opponent resources: {state.get('opponent_resources', 0)}",
        ]
        if state.get("board"):
            parts.append(f"\n{state['board']}")
        if state.get("move_history"):
            parts.append(f"\nHistory: {state['move_history']}")
        legal = state.get("legal_moves", [])
        parts.append(f"\nLegal moves ({len(legal)}): {', '.join(legal[:30])}")
        if len(legal) > 30:
            parts.append(f"... ({len(legal)} total)")
        parts.append("\nRespond with ONLY your move in DCN notation.")
        return "\n".join(parts)

    def _extract_move(self, response: LLMResponse, legal_moves: list[str]) -> str | None:
        """Extract a DCN move from the LLM response."""
        # Check tool calls first
        for tc in response.tool_calls:
            if tc.name == "play_move":
                return tc.arguments.get("move_dcn")

        # Try to find a move in the text
        if response.text:
            text = response.text.strip()
            # Direct match
            if text in legal_moves:
                return text
            # Case-insensitive
            for lm in legal_moves:
                if text.lower() == lm.lower():
                    return lm
            # Search for pattern in text
            import re
            for pattern in [
                r'`([CWRBL][a-h][1-8][-x][a-h][1-8])`',
                r'`(\+[WRBL]@[a-h][1-8])`',
                r'([CWRBL][a-h][1-8][-x][a-h][1-8])',
                r'(\+[WRBL]@[a-h][1-8])',
            ]:
                m = re.search(pattern, text)
                if m and m.group(1) in legal_moves:
                    return m.group(1)
        return None

    def _build_anthropic_assistant_content(self, response: LLMResponse) -> list[dict]:
        """Build Anthropic-format assistant content with text + tool_use blocks."""
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return content

    def get_results(self) -> dict:
        """Get final results from the session."""
        status = get_status(self.session_file)
        phase = status.get("phase", "")

        results = {
            "session_file": self.session_file,
            "phase": phase,
            "turns": self.turn_count,
            "total_tokens": self.tokens_used,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "budget": self.budget,
            "baseline_rating": status.get("baseline_rating"),
            "final_rating": status.get("final_rating"),
        }

        if phase == "completed":
            full_result = cli_run(["result", self.session_file])
            results.update({
                "baseline_elo": full_result.get("baseline_elo"),
                "final_elo": full_result.get("final_elo"),
                "elo_gain": full_result.get("elo_gain"),
            })

        return results

    def save_transcript(self, path: str):
        """Save the full transcript to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.transcript, f, indent=2)
        logger.info(f"Transcript saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DaveChess Benchmark Harness")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., claude-sonnet-4-20250514)")
    parser.add_argument("--budget", type=int, required=True,
                        help="Total token budget (learning + evaluation)")
    parser.add_argument("--eval-reserve", type=int, default=50_000,
                        help="Tokens reserved for evaluation phase")
    parser.add_argument("--name", default="benchmark-run",
                        help="Session name")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline phase")
    parser.add_argument("--baseline-games", type=int, default=5)
    parser.add_argument("--eval-min-games", type=int, default=5)
    parser.add_argument("--eval-max-games", type=int, default=30)
    parser.add_argument("--context-window", type=int, default=20)
    parser.add_argument("--results-dir", default="results/harness")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to neural network checkpoint for MCTS opponents")
    parser.add_argument("--calibration", default=None,
                        help="Path to calibration JSON (from calibrate_opponents.py)")
    args = parser.parse_args()

    # Create LLM client
    llm = ToolUseLLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.7,
        max_tokens=4096,
    )

    # Create session
    create_args = [
        "create",
        "--name", args.name,
        "--budget", str(args.budget),
        "--baseline-games", str(args.baseline_games),
        "--eval-min-games", str(args.eval_min_games),
        "--eval-max-games", str(args.eval_max_games),
    ]
    if args.skip_baseline:
        create_args.append("--skip-baseline")
    if args.checkpoint:
        create_args.extend(["--checkpoint", args.checkpoint])
    if args.calibration:
        create_args.extend(["--calibration", args.calibration])

    result = cli_run(create_args)
    session_file = result["session_file"]
    logger.info(f"Session created: {session_file} (phase: {result['phase']})")

    # Run harness
    harness = BenchmarkHarness(
        llm=llm,
        session_file=session_file,
        token_budget=args.budget,
        eval_reserve=args.eval_reserve,
        context_window=args.context_window,
    )

    start_time = time.time()

    # If session starts in baseline, play baseline games first
    if result["phase"] == "baseline":
        logger.info("Playing baseline games (no learning, rules only)...")
        # For baseline, use a simpler prompt
        harness.system_prompt = (
            "You are playing DaveChess. Respond with ONLY your move in "
            "DCN notation. Pick from the legal moves provided."
        )
        harness.run_eval_phase()  # Baseline uses same move-by-move flow
        # Reset for learning
        harness.messages = []
        harness.system_prompt = build_agentic_system_prompt(args.budget)

    # Check current phase
    status = get_status(session_file)
    phase = status.get("phase", "")

    if phase == "learning":
        harness.run_learning_phase()
        harness.run_eval_phase()
    elif phase == "evaluation":
        harness.run_eval_phase()

    elapsed = time.time() - start_time

    # Results
    results = harness.get_results()
    results["elapsed_sec"] = elapsed
    results["model"] = args.model
    results["provider"] = args.provider

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    safe_name = args.name.replace(" ", "_")
    result_path = os.path.join(args.results_dir, f"{safe_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    transcript_path = os.path.join(args.results_dir, f"{safe_name}_transcript.json")
    harness.save_transcript(transcript_path)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Token budget:   {args.budget:,}")
    print(f"Tokens used:    {results['total_tokens']:,}")
    print(f"Turns:          {results['turns']}")
    print(f"Elapsed:        {elapsed:.1f}s")
    if results.get("baseline_elo"):
        print(f"Baseline ELO:   {results['baseline_elo']:.0f}")
    if results.get("final_elo"):
        print(f"Final ELO:      {results['final_elo']:.0f}")
    if results.get("elo_gain") is not None:
        print(f"ELO gain:       {results['elo_gain']:+.0f}")
    print(f"Results saved:  {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
