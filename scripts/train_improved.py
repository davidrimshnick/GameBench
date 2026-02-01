#!/usr/bin/env python3
"""Improved training script with better seed game strategy."""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.engine.training import AlphaZeroTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("davechess.train")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train DaveChess AlphaZero network")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--improved-seeding", action="store_true", default=True,
                        help="Use improved seed game strategy (default: True)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["device"] = args.device

    # Initialize trainer
    trainer = AlphaZeroTrainer(config, use_wandb=args.wandb)

    # Configure W&B if enabled
    if args.wandb:
        import wandb
        wandb_config = {
            **config,
            "improved_seeding": args.improved_seeding,
        }
        wandb.init(
            project="davechess-training",
            config=wandb_config,
            resume="allow",
            name=f"training-improved-{wandb.util.generate_id()}"
        )
        logger.info(f"W&B logging enabled — dashboard at https://wandb.ai")

    # Setup TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=config.get("paths", {}).get("log_dir", "logs") + "/tensorboard")
    trainer.tb_writer = tb_writer
    logger.info(f"TensorBoard logging to {tb_writer.log_dir}")

    # Start simple HTTP server for TensorBoard
    import subprocess
    import threading
    def run_tensorboard():
        subprocess.run(["tensorboard", "--logdir", tb_writer.log_dir,
                       "--port", "6006", "--bind_all"],
                      capture_output=True)
    tb_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tb_thread.start()
    logger.info(f"TensorBoard started at http://0.0.0.0:6006")

    try:
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from {args.resume}")
        else:
            logger.info("Starting fresh training")
            trainer.save_best()  # Save initial model as best

            if args.improved_seeding:
                # IMPROVED SEEDING STRATEGY
                logger.info("Using improved seed game strategy")

                # 1. Generate high-quality seed games with more simulations
                logger.info("Phase 1: High-quality seed games (100 games, 50 sims)")
                trainer.seed_buffer(num_games=100, mcts_sims=50)

                # 2. Add some medium-quality games for diversity
                logger.info("Phase 2: Medium-quality seed games (50 games, 25 sims)")
                trainer.seed_buffer(num_games=50, mcts_sims=25)

                logger.info(f"Initial seeding complete. Buffer size: {len(trainer.replay_buffer.buffer)}")
            else:
                # Original seeding (for comparison)
                trainer.seed_buffer(num_games=30, mcts_sims=10)

        # Training loop with periodic re-seeding
        for iteration in range(1, config["training"]["max_iterations"] + 1):
            logger.info(f"=== Iteration {iteration} ===")

            # Periodic high-quality game injection (every 5 iterations)
            if args.improved_seeding and iteration > 1 and iteration % 5 == 0:
                logger.info(f"Adding high-quality MCTS games at iteration {iteration}")
                trainer.seed_buffer(num_games=20, mcts_sims=100)

            # Self-play phase
            trainer.run_self_play(iteration)

            # Training phase
            trainer.train_network(iteration)

            # Evaluation phase
            if trainer.evaluate_network(iteration):
                logger.info("New network accepted!")
                trainer.save_best()
                # After acceptance, add some high-quality games to celebrate
                if args.improved_seeding:
                    trainer.seed_buffer(num_games=10, mcts_sims=75)
            else:
                logger.info("New network rejected, keeping previous best.")

            trainer.save_checkpoint(iteration)

            # Stop if we've achieved high performance
            if trainer.elo_estimate > 200:
                logger.info(f"Reached ELO {trainer.elo_estimate}, stopping training")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
    finally:
        if args.wandb:
            wandb.finish()
        tb_writer.close()

    logger.info("Training complete!")

if __name__ == "__main__":
    main()