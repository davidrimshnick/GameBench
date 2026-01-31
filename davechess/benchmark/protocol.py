"""Benchmark orchestration: run full benchmark for a model."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from davechess.benchmark.evaluator import LLMAgent, play_llm_vs_opponent
from davechess.benchmark.llm_interface import LLMClient
from davechess.benchmark.prompt import build_system_prompt
from davechess.benchmark.scoring import (
    compute_llm_elo, compute_learning_curve, compute_gamebench_score,
    format_results,
)
from davechess.data.generator import Agent, create_agent
from davechess.game.state import Move

logger = logging.getLogger("davechess.benchmark")


class BenchmarkRunner:
    """Run the full GameBench benchmark for a model."""

    def __init__(self, config: dict, example_games: list[tuple[list[Move], str]],
                 calibrated_agents: list[tuple[str, Agent, float]]):
        """
        Args:
            config: Benchmark configuration dict.
            example_games: Pool of example games (moves, result_string).
            calibrated_agents: List of (name, agent, elo) at each level.
        """
        self.config = config
        self.example_games = example_games
        self.calibrated_agents = calibrated_agents

    def run(self, llm_client: LLMClient, model_name: str = "unknown") -> dict:
        """Run the full benchmark.

        Returns:
            Dict with learning curve, GameBench score, and raw results.
        """
        bench_cfg = self.config.get("benchmark", {})
        scoring_cfg = self.config.get("scoring", {})

        n_values = bench_cfg.get("n_values", [0, 1, 5, 10, 25, 50, 100])
        games_per_opponent = bench_cfg.get("games_per_opponent", 50)
        max_retries = bench_cfg.get("max_retries", 3)
        opponent_levels = bench_cfg.get("opponent_levels", list(range(len(self.calibrated_agents))))

        results_by_n: dict[int, float] = {}
        raw_results: dict[int, dict] = {}

        for n in n_values:
            logger.info(f"Testing with N={n} example games...")

            # Build system prompt with N examples
            system_prompt = build_system_prompt(self.example_games, num_examples=n)

            # Play against each calibrated opponent
            level_results: dict[int, list[float]] = {}
            n_raw = []

            for level_idx in opponent_levels:
                if level_idx >= len(self.calibrated_agents):
                    continue

                name, opponent, opp_elo = self.calibrated_agents[level_idx]
                logger.info(f"  vs {name} (ELO ~{opp_elo:.0f}), "
                            f"{games_per_opponent} games...")

                scores = []
                for game_idx in range(games_per_opponent):
                    llm_agent = LLMAgent(llm_client, system_prompt,
                                         max_retries=max_retries)
                    llm_plays_white = (game_idx % 2 == 0)

                    result = play_llm_vs_opponent(llm_agent, opponent,
                                                   llm_plays_white)
                    scores.append(result["score"])
                    n_raw.append(result)

                level_results[level_idx] = scores
                avg = sum(scores) / len(scores) if scores else 0
                logger.info(f"    Score: {avg:.3f} ({sum(s == 1.0 for s in scores)}W "
                            f"{sum(s == 0.5 for s in scores)}D "
                            f"{sum(s == 0.0 for s in scores)}L)")

            # Compute LLM ELO for this N
            level_elos = [elo for _, _, elo in self.calibrated_agents]
            llm_elo = compute_llm_elo(level_results, level_elos)
            results_by_n[n] = llm_elo
            raw_results[n] = {
                "elo": llm_elo,
                "level_results": {k: v for k, v in level_results.items()},
                "games": n_raw,
            }
            logger.info(f"  N={n}: Estimated ELO = {llm_elo:.0f}")

        # Compute GameBench Score
        curve = compute_learning_curve(results_by_n)
        score = compute_gamebench_score(
            curve,
            random_elo=scoring_cfg.get("random_elo", 400),
            max_elo=scoring_cfg.get("max_elo", 2700),
            max_n=scoring_cfg.get("max_n", 500),
        )

        logger.info(f"\n{format_results(model_name, curve, score)}")

        return {
            "model": model_name,
            "gamebench_score": score,
            "learning_curve": curve,
            "raw_results": raw_results,
            "timestamp": time.time(),
        }

    def save_results(self, results: dict, output_dir: str):
        """Save benchmark results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        model = results["model"].replace("/", "_")
        path = os.path.join(output_dir, f"{model}_results.json")

        # Convert for JSON serialization
        serializable = {
            "model": results["model"],
            "gamebench_score": results["gamebench_score"],
            "learning_curve": results["learning_curve"],
            "timestamp": results["timestamp"],
            # Don't save full raw game data - too large
            "summary_by_n": {
                str(n): {
                    "elo": data["elo"],
                    "levels": {
                        str(k): {"winrate": sum(v) / len(v), "games": len(v)}
                        for k, v in data["level_results"].items()
                    }
                }
                for n, data in results["raw_results"].items()
            },
        }

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {path}")
