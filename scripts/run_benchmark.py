#!/usr/bin/env python3
"""Run the full GameBench benchmark for a model.

Usage:
    python scripts/run_benchmark.py --model gpt-4 [--config configs/benchmark.yaml]
    python scripts/run_benchmark.py --model claude-3-opus --base-url https://api.anthropic.com/v1
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.benchmark.llm_interface import LLMClient
from davechess.benchmark.protocol import BenchmarkRunner
from davechess.data.generator import create_agent
from davechess.data.storage import load_games_collection

logger = logging.getLogger("davechess.benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run GameBench benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name/identifier")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--gen-config", type=str, default="configs/generation.yaml",
                        help="Generation config for calibrated agents")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override API base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Override API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--example-games", type=str, default="data/games/grandmaster",
                        help="Directory containing example game files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Network checkpoint for calibrated agents")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    # Load configs
    with open(args.config) as f:
        bench_config = yaml.safe_load(f)
    with open(args.gen_config) as f:
        gen_config = yaml.safe_load(f)

    # Create LLM client
    api_cfg = bench_config.get("api", {})
    llm_client = LLMClient(
        base_url=args.base_url or api_cfg.get("base_url", "https://api.openai.com/v1"),
        api_key=args.api_key,
        model=args.model,
        temperature=api_cfg.get("temperature", 0.3),
        max_tokens=api_cfg.get("max_tokens", 256),
    )

    # Load example games
    example_games = []
    if os.path.isdir(args.example_games):
        for fname in sorted(os.listdir(args.example_games)):
            if fname.endswith(".dcn"):
                from davechess.data.storage import load_game
                fpath = os.path.join(args.example_games, fname)
                try:
                    headers, moves, result = load_game(fpath)
                    if moves and result:
                        example_games.append((moves, result))
                except Exception as e:
                    logger.warning(f"Could not load {fname}: {e}")
    logger.info(f"Loaded {len(example_games)} example games")

    # Load network for calibrated agents
    network = None
    if args.checkpoint:
        try:
            import torch
            from davechess.engine.network import DaveChessNetwork
            network = DaveChessNetwork()
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            network.load_state_dict(ckpt.get("network_state", ckpt))
            network.to(args.device)
            network.eval()
        except Exception as e:
            logger.warning(f"Could not load network: {e}")

    # Create calibrated agents
    levels = gen_config.get("levels", [])
    calibrated_agents = []
    for level in levels:
        agent = create_agent(level, network=network, device=args.device)
        calibrated_agents.append((level["name"], agent, level.get("estimated_elo", 1500)))

    # Run benchmark
    runner = BenchmarkRunner(bench_config, example_games, calibrated_agents)
    results = runner.run(llm_client, model_name=args.model)

    # Save results
    output_dir = args.output_dir or bench_config.get("output", {}).get("results_dir", "results")
    runner.save_results(results, output_dir)

    print(f"\nGameBench Score: {results['gamebench_score']:.1f}/100")


if __name__ == "__main__":
    main()
