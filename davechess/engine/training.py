"""AlphaZero training loop with checkpointing, evaluation, and crash recovery."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from davechess.engine.network import DaveChessNetwork, POLICY_SIZE
from davechess.engine.selfplay import ReplayBuffer, run_selfplay_batch
from davechess.engine.mcts import MCTS
from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move

logger = logging.getLogger("davechess.training")


class Trainer:
    """AlphaZero training loop."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        net_cfg = config.get("network", {})
        self.network = DaveChessNetwork(
            num_res_blocks=net_cfg.get("num_res_blocks", 5),
            num_filters=net_cfg.get("num_filters", 64),
            input_planes=net_cfg.get("input_planes", 12),
        ).to(device)

        # Keep best_network on CPU to save GPU memory on Jetson (shared memory)
        self.best_network = DaveChessNetwork(
            num_res_blocks=net_cfg.get("num_res_blocks", 5),
            num_filters=net_cfg.get("num_filters", 64),
            input_planes=net_cfg.get("input_planes", 12),
        )  # CPU only
        self.best_network.load_state_dict(self.network.state_dict())

        train_cfg = config.get("training", {})
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=train_cfg.get("learning_rate", 0.01),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        self.replay_buffer = ReplayBuffer(
            max_size=config.get("selfplay", {}).get("replay_buffer_size", 500_000)
        )

        self.scaler = GradScaler("cuda") if HAS_TORCH and device != "cpu" else None
        self.training_step = 0
        self.iteration = 0
        self.best_elo_estimate = 0

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(paths.get("log_dir", "logs"))
        self.training_log_path = paths.get("training_log", "training_log.jsonl")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, tag: str = ""):
        """Save a training checkpoint."""
        name = f"step_{self.training_step}" if not tag else tag
        path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "iteration": self.iteration,
            "best_elo_estimate": self.best_elo_estimate,
        }
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Save replay buffer data alongside
        buf_path = self.checkpoint_dir / f"{name}_buffer.npz"
        self.replay_buffer.save_data(str(buf_path))

        return path

    def save_best(self):
        """Save the best network."""
        path = self.checkpoint_dir / "best.pt"
        torch.save({
            "network_state": self.best_network.state_dict(),
            "training_step": self.training_step,
            "iteration": self.iteration,
            "elo_estimate": self.best_elo_estimate,
        }, path)
        logger.info(f"Saved best model: {path}")

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load the latest checkpoint, or a specific one.

        Returns True if a checkpoint was loaded.
        """
        if path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"),
                                 key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                return False
            path = str(checkpoints[-1])

        logger.info(f"Loading checkpoint: {path}")
        # Load to CPU first to avoid GPU memory spike, then copy
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.network.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_step = checkpoint["training_step"]
        self.iteration = checkpoint["iteration"]
        self.best_elo_estimate = checkpoint.get("best_elo_estimate", 0)

        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        del checkpoint
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

        # Try loading replay buffer
        buf_path = path.replace(".pt", "_buffer.npz")
        if os.path.exists(buf_path):
            self.replay_buffer.load_data(buf_path)
            logger.info(f"Loaded replay buffer: {len(self.replay_buffer)} positions")

        # Load best network on CPU (moved to GPU only when needed)
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            self.best_network.load_state_dict(best_ckpt["network_state"])
            del best_ckpt
            gc.collect()

        logger.info(f"Resumed from step {self.training_step}, iteration {self.iteration}")
        return True

    def train_step(self, batch_size: int) -> dict:
        """Perform one training step on a mini-batch from the replay buffer.

        Returns dict with loss values.
        """
        if len(self.replay_buffer) < batch_size:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        planes, policies, values = self.replay_buffer.sample(batch_size)

        planes_t = torch.from_numpy(planes).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)
        values_t = torch.from_numpy(values).unsqueeze(1).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()

        if self.scaler is not None:
            with autocast("cuda"):
                policy_logits, value_pred = self.network(planes_t)
                policy_loss = -torch.mean(
                    torch.sum(policies_t * torch.log_softmax(policy_logits, dim=1), dim=1)
                )
                value_loss = torch.mean((value_pred - values_t) ** 2)
                total_loss = policy_loss + value_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value_pred = self.network(planes_t)
            policy_loss = -torch.mean(
                torch.sum(policies_t * torch.log_softmax(policy_logits, dim=1), dim=1)
            )
            value_loss = torch.mean((value_pred - values_t) ** 2)
            total_loss = policy_loss + value_loss

            total_loss.backward()
            self.optimizer.step()

        self.training_step += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def evaluate_network(self, num_games: int = 40,
                         num_simulations: int = 100) -> float:
        """Evaluate current network against best network.

        Returns win rate of current network.
        """
        current_mcts = MCTS(self.network, num_simulations=num_simulations,
                            temperature=0.1, device=self.device)
        best_mcts = MCTS(self.best_network, num_simulations=num_simulations,
                         temperature=0.1, device=self.device)

        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            state = GameState()
            # Alternate colors
            current_is_white = (game_idx % 2 == 0)

            while not state.done:
                moves = generate_legal_moves(state)
                if not moves:
                    break

                is_current_turn = (
                    (state.current_player == Player.WHITE and current_is_white) or
                    (state.current_player == Player.BLACK and not current_is_white)
                )

                if is_current_turn:
                    move, _ = current_mcts.get_move(state, add_noise=False)
                else:
                    move, _ = best_mcts.get_move(state, add_noise=False)

                apply_move(state, move)

            if state.winner is not None:
                current_won = (
                    (state.winner == Player.WHITE and current_is_white) or
                    (state.winner == Player.BLACK and not current_is_white)
                )
                if current_won:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1

        total_decisive = wins + losses
        if total_decisive == 0:
            return 0.5
        return wins / total_decisive

    def log_metrics(self, metrics: dict):
        """Append metrics to the training log."""
        metrics["timestamp"] = time.time()
        metrics["training_step"] = self.training_step
        metrics["iteration"] = self.iteration

        with open(self.training_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def run_iteration(self):
        """Run one training iteration: self-play + training + evaluation."""
        sp_cfg = self.config.get("selfplay", {})
        train_cfg = self.config.get("training", {})
        mcts_cfg = self.config.get("mcts", {})

        self.iteration += 1
        logger.info(f"=== Iteration {self.iteration} ===")

        # Self-play phase — temporarily move best_network to GPU
        logger.info("Self-play phase...")
        self.best_network.to(self.device)
        examples = run_selfplay_batch(
            network=self.best_network,
            num_games=sp_cfg.get("num_games_per_iteration", 100),
            num_simulations=mcts_cfg.get("num_simulations", 200),
            temperature_threshold=mcts_cfg.get("temperature_threshold", 30),
            device=self.device,
        )
        self.best_network.to("cpu")

        num_new_examples = len(examples)
        for planes, policy, value in examples:
            self.replay_buffer.push(planes, policy, value)
        del examples
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        logger.info(f"Generated {num_new_examples} training examples. "
                    f"Buffer size: {len(self.replay_buffer)}")

        # Training phase
        logger.info("Training phase...")
        batch_size = train_cfg.get("batch_size", 256)
        steps = train_cfg.get("steps_per_iteration", 1000)
        checkpoint_interval = train_cfg.get("checkpoint_interval", 500)

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0

        for step in range(steps):
            losses = self.train_step(batch_size)
            total_loss_sum += losses["total_loss"]
            policy_loss_sum += losses["policy_loss"]
            value_loss_sum += losses["value_loss"]

            if (self.training_step % checkpoint_interval == 0):
                self.save_checkpoint()

        avg_losses = {
            "avg_total_loss": total_loss_sum / steps if steps > 0 else 0,
            "avg_policy_loss": policy_loss_sum / steps if steps > 0 else 0,
            "avg_value_loss": value_loss_sum / steps if steps > 0 else 0,
        }
        logger.info(f"Training: {avg_losses}")

        # Detect loss spikes (divergence)
        if avg_losses["avg_total_loss"] > 10.0:
            logger.warning("Loss spike detected! Rolling back to best model.")
            self.network.load_state_dict(self.best_network.state_dict())
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.5
                logger.info(f"Reduced LR to {param_group['lr']}")

        # Evaluation phase — temporarily move best_network to GPU
        logger.info("Evaluation phase...")
        eval_games = train_cfg.get("eval_games", 40)
        eval_sims = train_cfg.get("eval_simulations", 50)
        eval_threshold = train_cfg.get("eval_threshold", 0.55)
        self.best_network.to(self.device)
        win_rate = self.evaluate_network(num_games=eval_games,
                                          num_simulations=eval_sims)
        self.best_network.to("cpu")
        logger.info(f"Evaluation win rate: {win_rate:.3f}")

        if win_rate >= eval_threshold:
            logger.info("New best network accepted!")
            self.best_network.load_state_dict(self.network.state_dict())
            self.save_best()
        else:
            logger.info("New network rejected, keeping previous best.")

        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

        # Log metrics
        self.log_metrics({
            "phase": "iteration",
            "num_examples": num_new_examples,
            "buffer_size": len(self.replay_buffer),
            "eval_win_rate": win_rate,
            **avg_losses,
        })

        # Save checkpoint at end of iteration
        self.save_checkpoint()

    def train(self, max_iterations: Optional[int] = None):
        """Run the full training loop."""
        max_iter = max_iterations or self.config.get("training", {}).get("max_iterations", 200)

        # Try to resume
        if self.load_checkpoint():
            logger.info(f"Resumed from step {self.training_step}, iteration {self.iteration}")
        else:
            logger.info("Starting fresh training")
            self.save_best()  # Save initial model as best

        while self.iteration < max_iter:
            try:
                self.run_iteration()
            except KeyboardInterrupt:
                logger.info("Training interrupted. Saving checkpoint...")
                self.save_checkpoint(tag="interrupted")
                break
            except Exception as e:
                logger.error(f"Error during iteration {self.iteration}: {e}")
                self.save_checkpoint(tag="error")
                raise
