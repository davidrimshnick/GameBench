#!/usr/bin/env python3
"""Entry point for AlphaZero training on DaveChess.

Usage:
    python scripts/train.py [--config configs/training.yaml] [--device cuda]
    python scripts/train.py --no-wandb   # disable W&B logging
"""

import argparse
import logging
import os
import sys

# Reduce CUDA memory on Jetson (shared CPU/GPU memory)
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

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


def get_wandb_run_id(checkpoint_dir: str) -> str | None:
    """Try to recover W&B run ID from the latest checkpoint."""
    from pathlib import Path
    checkpoints = sorted(Path(checkpoint_dir).glob("step_*.pt"),
                         key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    try:
        ckpt = torch.load(str(checkpoints[-1]), map_location="cpu", weights_only=False)
        return ckpt.get("wandb_run_id")
    except Exception:
        return None


def init_wandb(config: dict, device: str) -> bool:
    """Initialize Weights & Biases. Returns True if successful.

    Resumes the previous W&B run if a run ID is found in the latest checkpoint.
    """
    try:
        import wandb
    except ImportError:
        return False

    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    run_id = get_wandb_run_id(checkpoint_dir)

    try:
        wandb.init(
            project="davechess",
            id=run_id,
            resume="allow",
            config={
                "network": config.get("network", {}),
                "mcts": config.get("mcts", {}),
                "selfplay": config.get("selfplay", {}),
                "training": config.get("training", {}),
                "device": device,
            },
            tags=["alphazero", "jetson"],
            save_code=False,
        )
        # Per-iteration metrics use iteration as x-axis for cleaner charts
        wandb.define_metric("iteration")
        wandb.define_metric("eval/*", step_metric="iteration")
        wandb.define_metric("selfplay/*", step_metric="iteration")
        wandb.define_metric("elo", step_metric="iteration")
        return True
    except Exception as e:
        logging.getLogger("davechess.train").warning(f"W&B init failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train DaveChess AlphaZero")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to training config")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda, auto-detected if not specified)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override max training iterations")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
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
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Config: {args.config}")

    # Initialize W&B
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(config, device)
        if use_wandb:
            logger.info("W&B logging enabled — dashboard at https://wandb.ai")
        else:
            logger.info("W&B not available — logging to file only")

    trainer = Trainer(config, device=device, use_wandb=use_wandb)

    param_count = sum(p.numel() for p in trainer.network.parameters())
    logger.info(f"Network parameters: {param_count:,}")

    # Start TensorBoard server in background (use absolute path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tb_dir = os.path.join(project_root, config.get("paths", {}).get("log_dir", "logs"), "tensorboard")
    if os.path.isdir(tb_dir):
        try:
            import subprocess
            tb_port = 6006
            tb_proc = subprocess.Popen(
                ["tensorboard", "--logdir", tb_dir, "--port", str(tb_port),
                 "--bind_all", "--reload_interval", "30"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info(f"TensorBoard started at http://0.0.0.0:{tb_port}")
        except Exception as e:
            tb_proc = None
            logger.warning(f"Could not start TensorBoard: {e}")
    else:
        tb_proc = None

    try:
        trainer.train(max_iterations=args.max_iterations)
        logger.info("Training complete!")
    finally:
        if use_wandb:
            import wandb
            try:
                wandb.finish()
            except Exception:
                pass  # W&B service may already be dead
        if tb_proc is not None:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
