#!/usr/bin/env python3
"""Calibrate opponent ELO ratings for the agentic benchmark.

Runs a round-robin tournament between MCTS agents at different simulation
counts and computes Glicko-2 ratings. Saves results to a calibration JSON
file used by the OpponentPool.

Usage:
    python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt
    python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt --output checkpoints/calibration.json
    python scripts/calibrate_opponents.py --no-network  # Random + MCTSLite only
"""

from __future__ import annotations

import argparse
import json
import logging

from davechess.data.elo import Glicko2Rating, glicko2_update
from davechess.data.generator import RandomAgent, MCTSLiteAgent, play_game
from davechess.benchmark.opponent_pool import CalibratedLevel, OpponentPool

logger = logging.getLogger("calibrate")


# Simulation counts to calibrate
DEFAULT_SIM_COUNTS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800]


def create_agent(sim_count: int, network=None, device: str = "cpu"):
    """Create an agent for the given simulation count."""
    if sim_count == 0:
        return RandomAgent()
    if network is not None:
        from davechess.engine.mcts import MCTSAgent
        return MCTSAgent(network, device=device, num_simulations=sim_count)
    return MCTSLiteAgent(num_simulations=sim_count)


def calibrate(sim_counts: list[int], games_per_pair: int = 20,
              network=None, device: str = "cpu") -> list[CalibratedLevel]:
    """Run round-robin tournament and return calibrated levels.

    Args:
        sim_counts: List of simulation counts to calibrate.
        games_per_pair: Games to play between each pair (half as white, half as black).
        network: Optional neural network for MCTS agents.
        device: Device for network inference.

    Returns:
        List of CalibratedLevel with ELO ratings.
    """
    n = len(sim_counts)
    agents = [create_agent(s, network, device) for s in sim_counts]
    ratings = [Glicko2Rating.from_rating(1000, rd=350.0) for _ in range(n)]

    total_games = n * (n - 1) // 2 * games_per_pair
    played = 0

    for i in range(n):
        for j in range(i + 1, n):
            for g in range(games_per_pair):
                # Alternate colors
                if g % 2 == 0:
                    white_idx, black_idx = i, j
                else:
                    white_idx, black_idx = j, i

                result = play_game(agents[white_idx], agents[black_idx])
                played += 1

                # Score from white's perspective
                if result == "1-0":
                    white_score, black_score = 1.0, 0.0
                elif result == "0-1":
                    white_score, black_score = 0.0, 1.0
                else:
                    white_score, black_score = 0.5, 0.5

                # Update ratings
                ratings[white_idx] = glicko2_update(
                    ratings[white_idx], [ratings[black_idx]], [white_score]
                )
                ratings[black_idx] = glicko2_update(
                    ratings[black_idx], [ratings[white_idx]], [black_score]
                )

                if played % 10 == 0:
                    logger.info(f"Calibration: {played}/{total_games} games")

    # Build calibration levels
    levels = []
    for sim_count, rating in zip(sim_counts, ratings):
        levels.append(CalibratedLevel(
            sim_count=sim_count,
            measured_elo=round(rating.rating),
            elo_rd=round(rating.rd, 1),
        ))

    # Sort by ELO
    levels.sort(key=lambda l: l.measured_elo)
    return levels


def main():
    parser = argparse.ArgumentParser(description="Calibrate opponent ELO ratings")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to neural network checkpoint")
    parser.add_argument("--no-network", action="store_true",
                        help="Use MCTSLite only (no neural network)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for network inference")
    parser.add_argument("--output", "-o", type=str,
                        default="checkpoints/calibration.json",
                        help="Output path for calibration JSON")
    parser.add_argument("--games-per-pair", type=int, default=20,
                        help="Games per pair of agents")
    parser.add_argument("--sim-counts", type=str, default=None,
                        help="Comma-separated simulation counts (e.g., 0,1,10,50,200)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load network
    network = None
    if args.checkpoint and not args.no_network:
        try:
            import torch
            from davechess.engine.network import DaveChessNet, load_checkpoint
            network = DaveChessNet()
            load_checkpoint(network, args.checkpoint, device=args.device)
            logger.info(f"Loaded network from {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    # Parse sim counts
    if args.sim_counts:
        sim_counts = [int(s.strip()) for s in args.sim_counts.split(",")]
    else:
        sim_counts = DEFAULT_SIM_COUNTS

    logger.info(f"Calibrating {len(sim_counts)} levels: {sim_counts}")
    logger.info(f"Games per pair: {args.games_per_pair}")

    levels = calibrate(
        sim_counts, args.games_per_pair,
        network=network, device=args.device,
    )

    # Save
    pool = OpponentPool(network=network, device=args.device, calibration=levels)
    pool.save_calibration(args.output)

    # Print results
    print("\nCalibration Results:")
    print(f"{'Sims':>6s}  {'ELO':>6s}  {'RD':>6s}")
    print(f"{'----':>6s}  {'---':>6s}  {'--':>6s}")
    for level in levels:
        print(f"{level.sim_count:>6d}  {level.measured_elo:>6d}  "
              f"{level.elo_rd:>6.1f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
