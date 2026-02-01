"""Orchestrator for the agentic GameBench benchmark.

Runs the full pipeline: learning phase -> evaluation phase -> scoring.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from davechess.benchmark.agent_harness import AgentHarness
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.game_manager import GameManager
from davechess.benchmark.llm_interface import ToolUseLLMClient
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.benchmark.scoring import compute_budget_learning_curve, compute_agentic_score
from davechess.benchmark.sequential_eval import SequentialEvaluator, EvalConfig, EvalResult
from davechess.benchmark.token_tracker import TokenTracker
from davechess.benchmark.tools import ToolExecutor

logger = logging.getLogger("davechess.benchmark")


@dataclass
class AgenticRunResult:
    """Result of a single agentic benchmark run at one budget level."""
    model_name: str
    token_budget: int
    # Learning phase
    learning_stats: dict = field(default_factory=dict)
    # Evaluation phase
    eval_result: Optional[EvalResult] = None
    # Overall
    total_tokens_used: int = 0
    elapsed_sec: float = 0.0
    transcript_path: Optional[str] = None

    @property
    def estimated_elo(self) -> float:
        return self.eval_result.estimated_elo if self.eval_result else 400.0

    @property
    def elo_rd(self) -> float:
        return self.eval_result.rd if self.eval_result else 350.0

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "token_budget": self.token_budget,
            "estimated_elo": self.estimated_elo,
            "elo_rd": self.elo_rd,
            "total_tokens_used": self.total_tokens_used,
            "elapsed_sec": self.elapsed_sec,
            "learning_stats": self.learning_stats,
        }
        if self.eval_result:
            result["eval"] = {
                "games_played": self.eval_result.games_played,
                "wins": self.eval_result.wins,
                "losses": self.eval_result.losses,
                "draws": self.eval_result.draws,
                "tokens_used": self.eval_result.tokens_used,
            }
        if self.transcript_path:
            result["transcript_path"] = self.transcript_path
        return result


class AgenticBenchmarkRunner:
    """Runs the complete agentic benchmark pipeline.

    1. Learning phase: agent autonomously studies games, plays practice games
    2. Evaluation phase: sequential testing to measure ELO with Glicko-2
    3. Scoring: compute agentic score from budget-ELO curve
    """

    def __init__(self,
                 opponent_pool: OpponentPool,
                 game_library_path: str,
                 eval_config: Optional[EvalConfig] = None,
                 max_library_games: int = 200,
                 max_concurrent_games: int = 5,
                 context_window: int = 20,
                 eval_reserve: int = 50_000,
                 results_dir: str = "results/agentic",
                 save_transcripts: bool = True):
        self.opponent_pool = opponent_pool
        self.game_library_path = game_library_path
        self.eval_config = eval_config or EvalConfig()
        self.max_library_games = max_library_games
        self.max_concurrent_games = max_concurrent_games
        self.context_window = context_window
        self.eval_reserve = eval_reserve
        self.results_dir = results_dir
        self.save_transcripts = save_transcripts

    def run(self, llm_client: ToolUseLLMClient, model_name: str,
            token_budget: int) -> AgenticRunResult:
        """Run one complete benchmark at a given token budget.

        Args:
            llm_client: LLM client with tool-use support.
            model_name: Name of the model being evaluated.
            token_budget: Total token budget for learning + evaluation.

        Returns:
            AgenticRunResult with ELO and stats.
        """
        logger.info(f"=== Agentic benchmark: {model_name} @ {token_budget:,} tokens ===")
        start_time = time.time()

        # Set up components
        tracker = TokenTracker(budget=token_budget)

        library = GameLibrary(self.game_library_path,
                              max_games=self.max_library_games)
        library.load()
        logger.info(f"Game library: {library.total_games} games loaded")

        game_manager = GameManager(self.opponent_pool,
                                   max_concurrent=self.max_concurrent_games)
        tool_executor = ToolExecutor(game_manager, library)

        harness = AgentHarness(
            llm_client=llm_client,
            token_tracker=tracker,
            tool_executor=tool_executor,
            token_budget=token_budget,
            eval_reserve=self.eval_reserve,
            context_window=self.context_window,
        )

        # Phase 1: Learning
        logger.info("--- Phase 1: Learning ---")
        learning_stats = harness.run_learning_phase()

        # Phase 2: Evaluation
        logger.info("--- Phase 2: Evaluation ---")
        evaluator = SequentialEvaluator(
            config=self.eval_config,
            opponent_pool=self.opponent_pool,
            token_tracker=tracker,
        )
        eval_result = evaluator.evaluate(harness.play_eval_game)

        elapsed = time.time() - start_time

        # Build result
        result = AgenticRunResult(
            model_name=model_name,
            token_budget=token_budget,
            learning_stats=learning_stats,
            eval_result=eval_result,
            total_tokens_used=tracker.total_used,
            elapsed_sec=elapsed,
        )

        # Save results
        if self.results_dir:
            result.transcript_path = self._save_results(
                result, harness.transcript, evaluator.results, model_name, token_budget
            )

        logger.info(f"Result: ELO {result.estimated_elo:.0f} "
                     f"(+/-{result.elo_rd:.0f}), "
                     f"{tracker.total_used:,}/{token_budget:,} tokens used")

        return result

    def run_multi_budget(self, llm_client: ToolUseLLMClient, model_name: str,
                         budgets: Optional[list[int]] = None) -> dict:
        """Run benchmark at multiple budget levels for a learning curve.

        Each budget level is an independent run.

        Args:
            llm_client: LLM client with tool-use support.
            model_name: Name of the model being evaluated.
            budgets: List of token budgets to test.
                     Defaults to [100_000, 1_000_000, 10_000_000].

        Returns:
            Dict with results per budget and overall agentic score.
        """
        if budgets is None:
            budgets = [100_000, 1_000_000, 10_000_000]

        results: dict[int, AgenticRunResult] = {}
        for budget in sorted(budgets):
            logger.info(f"\n{'='*60}")
            logger.info(f"Budget level: {budget:,} tokens")
            logger.info(f"{'='*60}")

            result = self.run(llm_client, model_name, budget)
            results[budget] = result

        # Compute learning curve and score
        budget_elo_map = {b: r.estimated_elo for b, r in results.items()}
        curve = compute_budget_learning_curve(budget_elo_map)
        score = compute_agentic_score(curve)

        summary = {
            "model_name": model_name,
            "agentic_score": score,
            "learning_curve": curve,
            "runs": {b: r.to_dict() for b, r in results.items()},
        }

        # Save summary
        if self.results_dir:
            self._save_summary(summary, model_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Agentic GameBench Score: {score:.1f}/100")
        logger.info(f"Learning curve: {curve}")
        logger.info(f"{'='*60}")

        return summary

    def _save_results(self, result: AgenticRunResult,
                      transcript: list[dict],
                      eval_games: list[dict],
                      model_name: str,
                      budget: int) -> str:
        """Save run results and transcript to disk."""
        os.makedirs(self.results_dir, exist_ok=True)

        safe_name = model_name.replace("/", "_").replace(" ", "_")
        prefix = f"{safe_name}_{budget}"

        # Save result JSON
        result_path = os.path.join(self.results_dir, f"{prefix}_result.json")
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save transcript
        if self.save_transcripts and transcript:
            transcript_path = os.path.join(self.results_dir, f"{prefix}_transcript.json")
            with open(transcript_path, "w") as f:
                json.dump(transcript, f, indent=2)

        # Save eval game details
        if eval_games:
            eval_path = os.path.join(self.results_dir, f"{prefix}_eval_games.json")
            with open(eval_path, "w") as f:
                json.dump(eval_games, f, indent=2)

        return result_path

    def _save_summary(self, summary: dict, model_name: str) -> None:
        """Save multi-budget summary."""
        os.makedirs(self.results_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        path = os.path.join(self.results_dir, f"{safe_name}_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {path}")
