#!/usr/bin/env python3
"""Run round-robin tournament between agents and output ELO ladder.

Usage:
    python scripts/calibrate_elo.py --config configs/generation.yaml [--checkpoint best.pt]
"""

import argparse
import logging
import os
import sys
from itertools import combinations

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import Player
from davechess.data.generator import create_agent, play_game
from davechess.data.elo import calculate_elo_ratings, calculate_glicko2_ratings

logger = logging.getLogger("davechess.calibrate")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ELO ladder")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--games-per-pair", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load network if available
    network = None
    if args.checkpoint:
        try:
            import torch
            from davechess.engine.network import DaveChessNetwork
            network = DaveChessNetwork()
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            if "network_state" in ckpt:
                network.load_state_dict(ckpt["network_state"])
            else:
                network.load_state_dict(ckpt)
            network.to(args.device)
            network.eval()
        except Exception as e:
            logger.warning(f"Could not load network: {e}")
            network = None

    levels = config.get("levels", [])
    cal_cfg = config.get("calibration", {})
    games_per_pair = args.games_per_pair or cal_cfg.get("games_per_pair", 50)

    # Create agents
    agents = []
    for level in levels:
        agent = create_agent(level, network=network, device=args.device)
        agents.append((level["name"], agent))
        logger.info(f"Created agent: {level['name']} ({level['mcts_sims']} sims)")

    # Round-robin tournament
    results = []  # (player_a_id, player_b_id, score)
    num_agents = len(agents)

    for i, j in combinations(range(num_agents), 2):
        name_i, agent_i = agents[i]
        name_j, agent_j = agents[j]
        logger.info(f"Match: {name_i} vs {name_j} ({games_per_pair} games)")

        wins_i = 0
        wins_j = 0
        draws = 0

        for game_idx in range(games_per_pair):
            # Alternate colors
            if game_idx % 2 == 0:
                moves, winner, turns = play_game(agent_i, agent_j)
                if winner == Player.WHITE:
                    results.append((i, j, 1.0))
                    wins_i += 1
                elif winner == Player.BLACK:
                    results.append((i, j, 0.0))
                    wins_j += 1
                else:
                    results.append((i, j, 0.5))
                    draws += 1
            else:
                moves, winner, turns = play_game(agent_j, agent_i)
                if winner == Player.WHITE:
                    results.append((i, j, 0.0))
                    wins_j += 1
                elif winner == Player.BLACK:
                    results.append((i, j, 1.0))
                    wins_i += 1
                else:
                    results.append((i, j, 0.5))
                    draws += 1

        logger.info(f"  {name_i}: {wins_i}W  {name_j}: {wins_j}W  Draws: {draws}")

    # Calculate ratings
    rating_system = cal_cfg.get("rating_system", "elo")

    print("\n" + "=" * 60)
    print("  ELO LADDER")
    print("=" * 60)

    if rating_system == "glicko2":
        ratings = calculate_glicko2_ratings(results, num_agents)
        for idx in sorted(range(num_agents), key=lambda i: ratings[i].rating, reverse=True):
            name = agents[idx][0]
            r = ratings[idx]
            estimated = levels[idx].get("estimated_elo", "?")
            print(f"  {name:15s}  Rating: {r.rating:7.1f} +/- {r.rd:5.1f}  "
                  f"(estimated: {estimated})")
    else:
        elo_ratings = calculate_elo_ratings(results, num_agents)
        for idx in sorted(range(num_agents), key=lambda i: elo_ratings[i], reverse=True):
            name = agents[idx][0]
            r = elo_ratings[idx]
            estimated = levels[idx].get("estimated_elo", "?")
            print(f"  {name:15s}  ELO: {r:7.1f}  (estimated: {estimated})")

    # Check monotonicity
    print("\nMonotonicity check (higher sims → higher rating):")
    if rating_system == "glicko2":
        r_list = [ratings[i].rating for i in range(num_agents)]
    else:
        r_list = [elo_ratings[i] for i in range(num_agents)]

    monotonic = all(r_list[i] <= r_list[i+1] for i in range(len(r_list) - 1))
    print(f"  {'PASS' if monotonic else 'FAIL'}: Ratings are "
          f"{'monotonically increasing' if monotonic else 'NOT monotonic'}")


if __name__ == "__main__":
    main()
