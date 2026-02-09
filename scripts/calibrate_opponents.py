#!/usr/bin/env python3
"""Calibrate opponent ELO ratings for the agentic benchmark.

Runs a round-robin tournament between MCTS agents at different simulation
counts and computes Glicko-2 ratings. Saves results to a calibration JSON
file used by the OpponentPool.

Supports --resume to continue from a crashed run (progress is checkpointed
after every pair of agents).

Usage:
    python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt
    python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt --output checkpoints/calibration.json
    python scripts/calibrate_opponents.py --no-network  # Random + MCTSLite only
    python scripts/calibrate_opponents.py --no-network --resume  # Resume crashed run
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from davechess.game.state import Player
from davechess.data.elo import Glicko2Rating, glicko2_update
from davechess.data.generator import RandomAgent, MCTSAgent, MCTSLiteAgent, play_game
from davechess.benchmark.opponent_pool import CalibratedLevel, OpponentPool

logger = logging.getLogger("calibrate")


# Simulation counts to calibrate
DEFAULT_SIM_COUNTS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800]


def create_agent(sim_count: int, network=None, device: str = "cpu"):
    """Create an agent for the given simulation count."""
    if sim_count == 0:
        return RandomAgent()
    if network is not None:
        return MCTSAgent(network, num_simulations=sim_count, device=device)
    return MCTSLiteAgent(num_simulations=sim_count)


PROGRESS_FILE = "checkpoints/calibration_progress.json"


def _save_progress(sim_counts, ratings, completed_pair, played, progress_file):
    """Save calibration progress to a checkpoint file."""
    data = {
        "sim_counts": sim_counts,
        "ratings": [{"mu": r.mu, "phi": r.phi, "sigma": r.sigma} for r in ratings],
        "completed_pair": completed_pair,  # (i, j) of last completed pair
        "played": played,
    }
    tmp = progress_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, progress_file)


def _load_progress(progress_file):
    """Load calibration progress from checkpoint. Returns None if not found."""
    if not os.path.exists(progress_file):
        return None
    with open(progress_file) as f:
        return json.load(f)


def calibrate(sim_counts: list[int], games_per_pair: int = 20,
              network=None, device: str = "cpu",
              resume: bool = False) -> list[CalibratedLevel]:
    """Run round-robin tournament and return calibrated levels.

    Args:
        sim_counts: List of simulation counts to calibrate.
        games_per_pair: Games to play between each pair (half as white, half as black).
        network: Optional neural network for MCTS agents.
        device: Device for network inference.
        resume: If True, resume from progress checkpoint.

    Returns:
        List of CalibratedLevel with ELO ratings.
    """
    n = len(sim_counts)
    agents = [create_agent(s, network, device) for s in sim_counts]
    ratings = [Glicko2Rating.from_rating(1000, rd=350.0) for _ in range(n)]

    total_games = n * (n - 1) // 2 * games_per_pair
    played = 0
    skip_until = None  # (i, j) pair to resume after

    # Resume from checkpoint if available
    if resume:
        progress = _load_progress(PROGRESS_FILE)
        if progress and progress["sim_counts"] == sim_counts:
            for k, rdata in enumerate(progress["ratings"]):
                ratings[k] = Glicko2Rating(mu=rdata["mu"], phi=rdata["phi"],
                                           sigma=rdata["sigma"])
            skip_until = tuple(progress["completed_pair"])
            played = progress["played"]
            logger.info(f"Resuming from pair {skip_until}, {played}/{total_games} games done")
        elif progress:
            logger.warning("Progress file has different sim_counts, starting fresh")

    resuming = skip_until is not None

    for i in range(n):
        for j in range(i + 1, n):
            # Skip already-completed pairs when resuming
            if resuming:
                if (i, j) == skip_until:
                    resuming = False  # Next pair is where we resume
                continue

            for g in range(games_per_pair):
                # Alternate colors
                if g % 2 == 0:
                    white_idx, black_idx = i, j
                else:
                    white_idx, black_idx = j, i

                _moves, winner, _turns = play_game(
                    agents[white_idx], agents[black_idx])
                played += 1

                # Score from white's perspective
                if winner == Player.WHITE:
                    white_score, black_score = 1.0, 0.0
                elif winner == Player.BLACK:
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

            # Checkpoint after each pair
            _save_progress(sim_counts, ratings, [i, j], played, PROGRESS_FILE)
            logger.debug(f"Checkpointed after pair ({i},{j}), {played} games")

    # Clean up progress file on successful completion
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if available")
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
            from davechess.engine.network import DaveChessNetwork
            ckpt = torch.load(args.checkpoint, map_location=args.device,
                              weights_only=False)
            network = DaveChessNetwork()
            network.load_state_dict(ckpt["network_state"])
            network.eval()
            elo = ckpt.get("elo_estimate", "?")
            logger.info(f"Loaded network from {args.checkpoint} (ELO {elo})")
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
        resume=args.resume,
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
