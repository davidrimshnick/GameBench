#!/usr/bin/env python3
"""Run the agentic GameBench benchmark.

Usage:
    python scripts/run_agentic_benchmark.py --model gpt-4 --provider openai --budget 100000
    python scripts/run_agentic_benchmark.py --model claude-3-5-sonnet-20241022 --provider anthropic --multi-budget
    python scripts/run_agentic_benchmark.py --config configs/agentic_benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import yaml

from davechess.benchmark.agentic_protocol import AgenticBenchmarkRunner
from davechess.benchmark.llm_interface import ToolUseLLMClient
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.benchmark.sequential_eval import EvalConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run agentic GameBench benchmark")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g., gpt-4, claude-3-5-sonnet-20241022)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--budget", type=int, default=None,
                        help="Token budget for single run")
    parser.add_argument("--multi-budget", action="store_true",
                        help="Run at multiple budget levels (100K, 1M, 10M)")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated budget list (e.g., 100000,500000,1000000)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to network checkpoint for opponents")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to opponent calibration JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for opponent network (cuda/cpu)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config, then override with CLI args
    config = {}
    if args.config:
        config = load_config(args.config)

    # Resolve settings (CLI > config > defaults)
    provider = args.provider or config.get("api", {}).get("provider", "openai")
    model = args.model or config.get("api", {}).get("model", "gpt-4")
    temperature = config.get("api", {}).get("temperature", 0.7)
    max_tokens = config.get("api", {}).get("max_tokens", 4096)

    checkpoint = args.checkpoint or config.get("opponents", {}).get("checkpoint")
    calibration = args.calibration or config.get("opponents", {}).get("calibration_file")
    device = args.device or config.get("opponents", {}).get("device", "cuda")

    results_dir = args.results_dir or config.get("output", {}).get("results_dir", "results/agentic")
    save_transcripts = config.get("output", {}).get("save_transcripts", True)

    game_library_path = config.get("game_library", {}).get("path", "data/games/grandmaster")
    max_library_games = config.get("game_library", {}).get("max_games", 200)

    eval_cfg = config.get("evaluation", {})
    eval_config = EvalConfig(
        initial_elo=eval_cfg.get("initial_elo", 1000),
        target_rd=eval_cfg.get("target_rd", 50.0),
        max_games=eval_cfg.get("max_games", 200),
        min_games=eval_cfg.get("min_games", 10),
    )

    eval_reserve = config.get("budget", {}).get("eval_reserve", 50_000)
    context_window = config.get("budget", {}).get("context_window", 20)
    max_concurrent = config.get("tools", {}).get("max_concurrent_games", 5)

    # Set up opponent pool
    network = None
    if checkpoint:
        try:
            import torch
            from davechess.engine.network import DaveChessNet, load_checkpoint
            network = DaveChessNet()
            load_checkpoint(network, checkpoint, device=device)
            logging.info(f"Loaded network from {checkpoint}")
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}. Using random opponents only.")

    if calibration:
        opponent_pool = OpponentPool.from_calibration_file(
            calibration, network=network, device=device
        )
    else:
        logging.warning("No calibration file specified. Using default calibration.")
        from davechess.benchmark.opponent_pool import CalibratedLevel
        opponent_pool = OpponentPool(
            network=network, device=device,
            calibration=[
                CalibratedLevel(sim_count=0, measured_elo=400),
                CalibratedLevel(sim_count=1, measured_elo=600),
                CalibratedLevel(sim_count=10, measured_elo=900),
                CalibratedLevel(sim_count=50, measured_elo=1200),
                CalibratedLevel(sim_count=200, measured_elo=1600),
                CalibratedLevel(sim_count=800, measured_elo=2100),
            ],
        )

    # Set up LLM client
    llm_client = ToolUseLLMClient(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Set up runner
    runner = AgenticBenchmarkRunner(
        opponent_pool=opponent_pool,
        game_library_path=game_library_path,
        eval_config=eval_config,
        max_library_games=max_library_games,
        max_concurrent_games=max_concurrent,
        context_window=context_window,
        eval_reserve=eval_reserve,
        results_dir=results_dir,
        save_transcripts=save_transcripts,
    )

    # Run
    if args.multi_budget or args.budgets:
        if args.budgets:
            budgets = [int(b.strip()) for b in args.budgets.split(",")]
        else:
            budgets = config.get("budget", {}).get(
                "token_budgets", [100_000, 1_000_000, 10_000_000]
            )
        summary = runner.run_multi_budget(llm_client, model, budgets)
        print(f"\nAgentic GameBench Score: {summary['agentic_score']:.1f}/100")
        print(f"Learning curve: {summary['learning_curve']}")
    elif args.budget:
        result = runner.run(llm_client, model, args.budget)
        print(f"\nELO: {result.estimated_elo:.0f} (+/-{result.elo_rd:.0f})")
        print(f"Tokens: {result.total_tokens_used:,}/{args.budget:,}")
    else:
        # Default: single run at first configured budget
        budget = config.get("budget", {}).get("token_budgets", [1_000_000])[0]
        result = runner.run(llm_client, model, budget)
        print(f"\nELO: {result.estimated_elo:.0f} (+/-{result.elo_rd:.0f})")
        print(f"Tokens: {result.total_tokens_used:,}/{budget:,}")


if __name__ == "__main__":
    main()
