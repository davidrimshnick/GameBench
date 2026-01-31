#!/usr/bin/env python3
"""Generate games at specified strength levels.

Usage:
    python scripts/generate_games.py --config configs/generation.yaml [--checkpoint checkpoints/best.pt]
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import Player
from davechess.data.generator import create_agent, generate_games
from davechess.data.storage import save_game

logger = logging.getLogger("davechess.generate")


def main():
    parser = argparse.ArgumentParser(description="Generate DaveChess games")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to neural network checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--level", type=int, default=None,
                        help="Generate games at this specific level only")
    parser.add_argument("--num-games", type=int, default=None,
                        help="Override number of games")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load network if checkpoint provided
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
            logger.info(f"Loaded network from {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load network: {e}. Using MCTS-lite agents.")
            network = None

    gen_cfg = config.get("generation", {})
    output_dir = args.output_dir or gen_cfg.get("output_dir", "data/games")
    num_games = args.num_games or gen_cfg.get("num_grandmaster_games", 100)
    min_length = gen_cfg.get("min_game_length", 20)
    discard_draws = gen_cfg.get("discard_draws", True)

    levels = config.get("levels", [])

    if args.level is not None:
        # Generate self-play at specific level
        level = levels[args.level]
        agent = create_agent(level, network=network, device=args.device)
        logger.info(f"Generating {num_games} games at {level['name']} "
                    f"({level['mcts_sims']} sims)")

        games = generate_games(agent, agent, num_games,
                               min_length=min_length, discard_draws=discard_draws)

        for i, (moves, winner) in enumerate(games):
            result = "1-0" if winner == Player.WHITE else \
                     "0-1" if winner == Player.BLACK else "1/2-1/2"
            headers = {
                "White": f"MCTS-{level['mcts_sims']}",
                "Black": f"MCTS-{level['mcts_sims']}",
                "Level": level['name'],
            }
            filepath = os.path.join(output_dir, level['name'], f"game_{i:05d}.dcn")
            save_game(filepath, moves, headers=headers, result=result)

        logger.info(f"Saved {len(games)} games to {output_dir}/{level['name']}/")

    else:
        # Generate at the strongest level (grandmaster games)
        strongest = levels[-1]
        agent = create_agent(strongest, network=network, device=args.device)
        logger.info(f"Generating {num_games} grandmaster games at {strongest['name']} "
                    f"({strongest['mcts_sims']} sims)")

        games = generate_games(agent, agent, num_games,
                               min_length=min_length, discard_draws=discard_draws)

        for i, (moves, winner) in enumerate(games):
            result = "1-0" if winner == Player.WHITE else \
                     "0-1" if winner == Player.BLACK else "1/2-1/2"
            headers = {
                "White": "AlphaZero",
                "Black": "AlphaZero",
                "Level": strongest['name'],
            }
            filepath = os.path.join(output_dir, "grandmaster", f"game_{i:05d}.dcn")
            save_game(filepath, moves, headers=headers, result=result)

        logger.info(f"Saved {len(games)} grandmaster games")


if __name__ == "__main__":
    main()
