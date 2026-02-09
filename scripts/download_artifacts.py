#!/usr/bin/env python3
"""Download trained model and generate GM games for the benchmark API.

Usage:
    # Download best model from W&B
    python scripts/download_artifacts.py --download-model

    # Generate GM-level games using HeuristicPlayer (no network needed)
    python scripts/download_artifacts.py --generate-games 50

    # Both
    python scripts/download_artifacts.py --download-model --generate-games 50
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WANDB_PROJECT = "david-rimshnick-david-rimshnick/davechess"
CHECKPOINT_DIR = "checkpoints"
GAMES_DIR = "data/gm_games"


def download_model():
    """Download the latest best model from W&B."""
    try:
        import wandb
    except ImportError:
        log.error("wandb not installed. Run: pip install wandb")
        return False

    api = wandb.Api()
    log.info(f"Searching for model artifacts in {WANDB_PROJECT}...")

    # Find the latest best-model artifact
    try:
        artifacts = list(api.artifacts(type_name="model", project=WANDB_PROJECT))
        if not artifacts:
            log.error("No model artifacts found in W&B")
            return False

        # Sort by version to get latest
        latest = sorted(artifacts, key=lambda a: a.version)[-1]
        log.info(f"Found: {latest.name} (v{latest.version})")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        artifact_dir = latest.download(root=CHECKPOINT_DIR)
        log.info(f"Downloaded to {artifact_dir}")

        # Check for best.pt
        best_pt = os.path.join(CHECKPOINT_DIR, "best.pt")
        if os.path.exists(best_pt):
            size_mb = os.path.getsize(best_pt) / (1024 * 1024)
            log.info(f"Model checkpoint: {best_pt} ({size_mb:.1f} MB)")
        else:
            # List what we got
            for f in os.listdir(artifact_dir):
                log.info(f"  Downloaded: {f}")

        return True

    except Exception as e:
        log.error(f"Failed to download model: {e}")
        return False


def generate_games(num_games: int):
    """Generate GM-level games using HeuristicPlayer vs HeuristicPlayer."""
    from davechess.game.state import GameState, Player
    from davechess.game.rules import generate_legal_moves, apply_move
    from davechess.game.notation import move_to_dcn
    from davechess.data.storage import save_game

    try:
        from davechess.engine.heuristic_player import HeuristicPlayer
    except ImportError:
        log.warning("HeuristicPlayer not available, using MCTS lite agents")
        from davechess.data.generator import MCTSLiteAgent
        white_player = MCTSLiteAgent(num_simulations=50)
        black_player = MCTSLiteAgent(num_simulations=50)
        player_name = "MCTSLite-50"
    else:
        white_player = HeuristicPlayer(Player.WHITE)
        black_player = HeuristicPlayer(Player.BLACK)
        player_name = "Heuristic"

    os.makedirs(GAMES_DIR, exist_ok=True)

    log.info(f"Generating {num_games} games with {player_name} players...")
    generated = 0
    decisive = 0

    for i in range(num_games):
        state = GameState()
        moves = []
        states = []

        while not state.done:
            legal = generate_legal_moves(state)
            if not legal:
                break

            states.append(state.clone())
            if state.current_player == Player.WHITE:
                move = white_player.get_move(state)
            else:
                move = black_player.get_move(state)

            moves.append(move)
            apply_move(state, move)

        if not moves:
            continue

        # Determine result
        if state.winner == Player.WHITE:
            result = "1-0"
            decisive += 1
        elif state.winner == Player.BLACK:
            result = "0-1"
            decisive += 1
        else:
            result = "1/2-1/2"

        headers = {
            "Game": str(i + 1),
            "White": player_name,
            "Black": player_name,
            "Moves": str(len(moves)),
            "Result": result,
        }

        filepath = os.path.join(GAMES_DIR, f"game_{i+1:04d}.dcn")
        save_game(filepath, moves, headers=headers, result=result, states=states)
        generated += 1

        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{num_games} games ({decisive} decisive)")

    log.info(f"Generated {generated} games ({decisive} decisive) in {GAMES_DIR}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download/generate benchmark data")
    parser.add_argument("--download-model", action="store_true",
                        help="Download best model from W&B")
    parser.add_argument("--generate-games", type=int, metavar="N",
                        help="Generate N GM-level games for the study library")
    args = parser.parse_args()

    if not args.download_model and not args.generate_games:
        parser.print_help()
        return

    if args.download_model:
        download_model()

    if args.generate_games:
        generate_games(args.generate_games)


if __name__ == "__main__":
    main()
