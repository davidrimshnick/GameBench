#!/usr/bin/env python3
"""Entry point for AlphaZero training on DaveChess.

Usage:
    python scripts/train.py [--config configs/training.yaml] [--device cuda]
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from davechess.engine.training import Trainer


def setup_logging(log_dir: str):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train DaveChess AlphaZero")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to training config")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda, auto-detected if not specified)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override max training iterations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_dir = config.get("paths", {}).get("log_dir", "logs")
    setup_logging(log_dir)

    logger = logging.getLogger("davechess.train")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Config: {args.config}")

    trainer = Trainer(config, device=device)

    param_count = sum(p.numel() for p in trainer.network.parameters())
    logger.info(f"Network parameters: {param_count:,}")

    trainer.train(max_iterations=args.max_iterations)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
