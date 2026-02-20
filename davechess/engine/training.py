"""AlphaZero training loop with checkpointing, evaluation, and crash recovery."""

from __future__ import annotations

import gc
import json
import logging
import math
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from muon import zeropower_via_newtonschulz5
    HAS_MUON = True
except ImportError:
    HAS_MUON = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from davechess.engine.network import DaveChessNetwork, POLICY_SIZE, NUM_INPUT_PLANES, state_to_planes, move_to_policy_index
from davechess.engine.selfplay import (
    ReplayBuffer, StructuredReplayBuffer, run_selfplay_batch,
    run_selfplay_batch_parallel, run_selfplay_multiprocess,
)
from davechess.engine.mcts import MCTS
from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.data.generator import MCTSLiteAgent, play_game
from davechess.engine.smart_seeds import generate_smart_seeds
from davechess.game.notation import game_to_dcn

logger = logging.getLogger("davechess.training")


def _safe_wandb_log(*args, **kwargs):
    """Log to wandb, catching connection errors so training doesn't crash."""
    try:
        wandb.log(*args, **kwargs)
    except Exception as e:
        logger.warning(f"W&B log failed (training continues): {e}")


def _safe_wandb_log_artifact(artifact):
    """Upload wandb artifact, catching errors."""
    try:
        wandb.run.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"W&B artifact upload failed: {e}")


def _get_rss_mb() -> float:
    """Return current RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB→MB
    except Exception:
        return 0.0


def _log_memory(label: str):
    """Log current RSS and GPU memory at a phase transition point."""
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB→MB
        msg = f"[mem] {label}: RSS={rss_mb:.0f}MB"
    except Exception:
        msg = f"[mem] {label}"
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            msg += f", GPU alloc={alloc:.0f}MB, reserved={reserved:.0f}MB"
    except Exception:
        pass
    logger.info(msg)


def adaptive_simulations(elo: float, min_sims: int = 2, max_sims: int = 200,
                         elo_min: float = 0, elo_max: float = 2000) -> int:
    """Compute MCTS sim count from ELO via log-scale interpolation.

    At elo_min → min_sims, at elo_max → max_sims.
    Clamped to [min_sims, max_sims].
    """
    t = max(0.0, min(1.0, (elo - elo_min) / (elo_max - elo_min)))
    sims = min_sims * (max_sims / min_sims) ** t
    return max(min_sims, min(max_sims, int(round(sims))))


def win_rate_to_elo_diff(win_rate: float) -> float:
    """Convert a win rate to an ELO difference.

    ELO_diff = -400 * log10(1/win_rate - 1)
    """
    if win_rate <= 0.0:
        return -400.0
    if win_rate >= 1.0:
        return 400.0
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


class MuonSGD:
    """Dual optimizer: Muon for conv/hidden weights, SGD for heads/biases/BN.

    Single-GPU implementation — uses the Newton-Schulz orthogonalization from
    the muon package but without distributed all_gather.
    """

    def __init__(self, network: nn.Module, muon_lr: float = 0.02,
                 muon_momentum: float = 0.95, sgd_lr: float = 0.001,
                 sgd_momentum: float = 0.9, weight_decay: float = 1e-4):
        # Split parameters: ResNet trunk conv weights → Muon, everything else → SGD
        # Muon's Newton-Schulz orthogonalization is designed for intermediate conv layers.
        # It MUST NOT be applied to classification heads (policy_fc, value_fc) because
        # orthogonalizing those gradients destroys the per-class learning signal.
        muon_params = []
        sgd_params = []
        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            # Only trunk conv weights get Muon — everything else to SGD
            is_trunk_conv = (param.ndim >= 2
                             and "bn" not in name
                             and "bias" not in name
                             and "fc" not in name
                             and "policy" not in name
                             and "value" not in name)
            if is_trunk_conv:
                muon_params.append(param)
            else:
                sgd_params.append(param)

        self.muon_params = muon_params
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_wd = weight_decay

        # Momentum buffers for Muon params
        self.muon_state = {}
        for p in muon_params:
            self.muon_state[p] = {"momentum_buffer": torch.zeros_like(p)}

        # Standard SGD for everything else
        self.sgd = optim.SGD(sgd_params, lr=sgd_lr, momentum=sgd_momentum,
                             weight_decay=weight_decay)

        logger.info(f"MuonSGD: {len(muon_params)} Muon params, "
                    f"{len(sgd_params)} SGD params, "
                    f"muon_lr={muon_lr}, sgd_lr={sgd_lr}")

    @property
    def param_groups(self):
        """Expose param_groups for LR logging compatibility."""
        return [{"lr": self.muon_lr, "type": "muon"}] + self.sgd.param_groups

    def zero_grad(self):
        for p in self.muon_params:
            if p.grad is not None:
                p.grad.zero_()
        self.sgd.zero_grad()

    @torch.no_grad()
    def step(self):
        # Muon update for conv/hidden weights
        for p in self.muon_params:
            if p.grad is None:
                continue
            state = self.muon_state[p]
            buf = state["momentum_buffer"]
            grad = p.grad

            # Momentum + Nesterov
            buf.lerp_(grad, 1 - self.muon_momentum)
            update = grad.lerp_(buf, self.muon_momentum)  # Nesterov

            # Reshape for Newton-Schulz (conv4D → 2D)
            orig_shape = update.shape
            if update.ndim == 4:
                update = update.view(len(update), -1)

            # Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, steps=5)
            scale = max(1, update.size(-2) / update.size(-1)) ** 0.5
            update = update * scale

            update = update.reshape(orig_shape)

            # Weight decay + update
            p.mul_(1 - self.muon_lr * self.muon_wd)
            p.add_(update, alpha=-self.muon_lr)

        # SGD update for heads/biases/BN
        self.sgd.step()

    def state_dict(self):
        muon_buffers = {}
        for i, p in enumerate(self.muon_params):
            muon_buffers[i] = self.muon_state[p]["momentum_buffer"]
        return {
            "muon_lr": self.muon_lr,
            "muon_momentum": self.muon_momentum,
            "muon_wd": self.muon_wd,
            "muon_buffers": muon_buffers,
            "sgd_state": self.sgd.state_dict(),
        }

    def load_state_dict(self, state):
        if "sgd_state" in state:
            self.sgd.load_state_dict(state["sgd_state"])
        if "muon_buffers" in state:
            for i, p in enumerate(self.muon_params):
                if i in state["muon_buffers"]:
                    self.muon_state[p]["momentum_buffer"].copy_(
                        state["muon_buffers"][i]
                    )
        self.muon_lr = state.get("muon_lr", self.muon_lr)
        self.muon_momentum = state.get("muon_momentum", self.muon_momentum)
        self.muon_wd = state.get("muon_wd", self.muon_wd)


class Trainer:
    """AlphaZero training loop."""

    def __init__(self, config: dict, device: str = "cpu", use_wandb: bool = False,
                 config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path  # For hot-reloading between iterations
        self.device = device
        self.use_wandb = use_wandb and HAS_WANDB
        self.tb_writer = None

        # Create networks on CPU first — moved to GPU after checkpoint loading
        net_cfg = config.get("network", {})
        self.network = DaveChessNetwork(
            num_res_blocks=net_cfg.get("num_res_blocks", 5),
            num_filters=net_cfg.get("num_filters", 64),
            input_planes=net_cfg.get("input_planes", NUM_INPUT_PLANES),
        )  # CPU initially, moved to GPU in train()

        train_cfg = config.get("training", {})
        optimizer_type = train_cfg.get("optimizer", "sgd").lower()
        if optimizer_type == "muon" and HAS_MUON:
            self.optimizer = MuonSGD(
                self.network,
                muon_lr=train_cfg.get("learning_rate", 0.02),
                muon_momentum=train_cfg.get("muon_momentum", 0.95),
                sgd_lr=train_cfg.get("head_lr", 0.001),
                sgd_momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )
        else:
            if optimizer_type == "muon" and not HAS_MUON:
                logger.warning("Muon requested but not installed, falling back to SGD")
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=train_cfg.get("learning_rate", 0.01),
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )

        sp_cfg = config.get("selfplay", {})
        self.replay_buffer = StructuredReplayBuffer(
            seed_size=sp_cfg.get("buffer_seed_size", 20_000),
            decisive_size=sp_cfg.get("buffer_decisive_size", 20_000),
            draw_size=sp_cfg.get("buffer_draw_size", 10_000),
        )

        self.scaler = GradScaler("cuda") if HAS_TORCH and device != "cpu" else None
        self.training_step = 0
        self.iteration = 0
        self.elo_estimate = 0  # Running ELO estimate from MCTSLite probes
        # Smoothed ELO used for adaptive self-play sims (less jittery than probes)
        self.elo_for_sims = 0.0

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(paths.get("log_dir", "logs"))
        self.training_log_path = paths.get("training_log", "training_log.jsonl")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if HAS_TB:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging to {tb_dir}")

    def save_checkpoint(self, tag: str = "", save_buffer: bool = False):
        """Save a training checkpoint.

        Args:
            tag: Optional tag for the checkpoint name.
            save_buffer: If True, also save the replay buffer (expensive on Jetson).
                Only set this for end-of-iteration or error checkpoints.
        """
        name = f"step_{self.training_step}" if not tag else tag
        path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "iteration": self.iteration,
            "elo_estimate": self.elo_estimate,
            "elo_for_sims": self.elo_for_sims,
        }
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        if self.use_wandb:
            checkpoint["wandb_run_id"] = wandb.run.id

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if save_buffer:
            buf_path = self.checkpoint_dir / f"{name}_buffer.npz"
            self.replay_buffer.save_data(str(buf_path))
            # Backup buffer to W&B so it survives OOM crashes
            if self.use_wandb:
                artifact = wandb.Artifact(
                    "replay-buffer",
                    type="replay-buffer",
                    metadata={
                        "step": self.training_step,
                        "iteration": self.iteration,
                        "size": len(self.replay_buffer),
                    },
                )
                artifact.add_file(str(buf_path))
                _safe_wandb_log_artifact(artifact)

        return path

    def save_best(self):
        """Save the current network as best.pt (for benchmark use)."""
        path = self.checkpoint_dir / "best.pt"
        torch.save({
            "network_state": self.network.state_dict(),
            "training_step": self.training_step,
            "iteration": self.iteration,
            "elo_estimate": self.elo_estimate,
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
        # Load everything on CPU — network moves to GPU later in train()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Could not load optimizer state (optimizer type changed?): {e}")
            logger.warning("Starting with fresh optimizer state — momentum will rebuild in ~50 steps")
        self.training_step = checkpoint["training_step"]
        self.iteration = checkpoint["iteration"]
        self.elo_estimate = checkpoint.get("elo_estimate", checkpoint.get("best_elo_estimate", 0))
        self.elo_for_sims = checkpoint.get("elo_for_sims", float(self.elo_estimate))

        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        del checkpoint
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

        # Try loading replay buffer (local file first, then W&B artifact fallback)
        buf_path = path.replace(".pt", "_buffer.npz")
        if os.path.exists(buf_path):
            self.replay_buffer.load_data(buf_path)
            parts = self.replay_buffer.partition_sizes()
            logger.info(f"Loaded replay buffer: {len(self.replay_buffer)} positions "
                        f"(seeds={parts['seeds']}, decisive={parts['decisive']}, "
                        f"draws={parts['draws']})")
        elif self.use_wandb:
            try:
                artifact = wandb.use_artifact("replay-buffer:latest")
                artifact_dir = artifact.download()
                # Find the npz file in the artifact
                for f in os.listdir(artifact_dir):
                    if f.endswith("_buffer.npz"):
                        self.replay_buffer.load_data(os.path.join(artifact_dir, f))
                        logger.info(f"Loaded replay buffer from W&B artifact: "
                                    f"{len(self.replay_buffer)} positions")
                        break
            except Exception as e:
                logger.warning(f"Could not load replay buffer from W&B: {e}")

        logger.info(f"Resumed from step {self.training_step}, iteration {self.iteration}")
        return True

    def train_step(self, batch_size: int) -> dict:
        """Perform one training step on a mini-batch from the replay buffer.

        Returns dict with loss values.
        """
        if len(self.replay_buffer) < batch_size:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        # Decay seed sampling weight over iterations so self-play data
        # gradually dominates over stale heuristic seed positions.
        seed_cfg = self.config.get("training", {})
        seed_w_init = float(seed_cfg.get("seed_sample_weight_init", 1.0))
        seed_w_decay = float(seed_cfg.get("seed_sample_weight_decay", 1.0))
        seed_w_min = float(seed_cfg.get("seed_sample_weight_min", 0.1))
        seed_weight = max(seed_w_min, seed_w_init * (seed_w_decay ** self.iteration))

        planes, policies, values = self.replay_buffer.sample(batch_size, seed_weight=seed_weight)

        planes_t = torch.from_numpy(planes).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)
        values_t = torch.from_numpy(values).unsqueeze(1).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()
        draw_sample_weight = float(self.config.get("training", {}).get("draw_sample_weight", 1.0))

        def _compute_losses(policy_logits, value_pred):
            per_example_policy_loss = -torch.sum(
                policies_t * torch.log_softmax(policy_logits, dim=1), dim=1
            )
            per_example_value_loss = (value_pred - values_t) ** 2

            if draw_sample_weight != 1.0:
                # Downweight draw positions so decisive positions contribute
                # proportionally more gradient signal. Match buffer's threshold.
                draw_mask = (values_t.abs() <= 0.5)
                sample_weights = torch.where(
                    draw_mask,
                    torch.full_like(values_t, draw_sample_weight),
                    torch.ones_like(values_t),
                )
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
            else:
                sample_weights = torch.ones_like(values_t)

            policy_loss = torch.mean(per_example_policy_loss * sample_weights.squeeze(1))
            value_loss = torch.mean(per_example_value_loss * sample_weights)
            # Fixed weight to balance policy CE (~5-6) vs value MSE (~0.1)
            value_weight = float(self.config.get("training", {}).get("value_loss_weight", 20.0))
            total_loss = policy_loss + value_weight * value_loss
            return policy_loss, value_loss, total_loss, value_weight

        is_muon = isinstance(self.optimizer, MuonSGD)

        if self.scaler is not None:
            with autocast("cuda"):
                policy_logits, value_pred = self.network(planes_t)
                policy_loss, value_loss, total_loss, value_scale = _compute_losses(policy_logits, value_pred)

            self.scaler.scale(total_loss).backward()
            if is_muon:
                # MuonSGD: manually unscale grads then step
                self.scaler.unscale_(self.optimizer.sgd)
                # Unscale Muon params manually
                scale = self.scaler.get_scale()
                if scale > 0:
                    inv_scale = 1.0 / scale
                    for p in self.optimizer.muon_params:
                        if p.grad is not None:
                            p.grad.mul_(inv_scale)
                    # Clip gradients before inf/nan check
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    # Check for inf/nan in grads (skip step if found)
                    found_inf = any(
                        torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
                        for p in self.optimizer.muon_params if p.grad is not None
                    )
                    if found_inf:
                        # Manually back off scale to prevent infinite overflow loop
                        # (scaler.update() won't detect Muon overflows since we bypass scaler.step())
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                    else:
                        self.optimizer.step()
                        self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            policy_logits, value_pred = self.network(planes_t)
            policy_loss, value_loss, total_loss, value_scale = _compute_losses(policy_logits, value_pred)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.training_step += 1

        losses = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "value_scale": float(value_scale),
        }

        if self.training_step % 10 == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            step_metrics = {
                "train/total_loss": losses["total_loss"],
                "train/policy_loss": losses["policy_loss"],
                "train/value_loss": losses["value_loss"],
                "train/value_scale": losses["value_scale"],
                "train/learning_rate": lr,
            }
            if self.use_wandb:
                _safe_wandb_log(step_metrics, step=self.training_step)
            if self.tb_writer:
                for k, v in step_metrics.items():
                    self.tb_writer.add_scalar(k, v, self.training_step)

        return losses

    def estimate_elo_mctslite(self, num_games: int = 4,
                              mctslite_sims: int = 50,
                              nn_sims: Optional[int] = None,
                              max_moves: int = 100) -> dict:
        """Estimate ELO by playing the NN against MCTSLite (no neural network).

        This is a lightweight, non-gating probe. MCTSLite uses random rollouts
        only, so it's cheap and provides an absolute strength reference.
        The NN plays with MCTS (using its own policy/value), MCTSLite uses
        random rollouts. We estimate ELO from the win rate against MCTSLite.

        MCTSLite at 50 sims ≈ ELO 200-400 (a weak but non-trivial baseline).
        """
        from davechess.engine.mcts_lite import MCTSLite

        # Use configured NN-MCTS sims if provided, else a modest default for speed.
        if nn_sims is None:
            nn_sims = max(25, mctslite_sims)
        nn_mcts = MCTS(self.network, num_simulations=nn_sims,
                       temperature=0.1, device=self.device)

        wins = 0
        losses = 0
        draws = 0
        game_lengths = []

        for game_idx in range(num_games):
            state = GameState()
            nn_is_white = (game_idx % 2 == 0)
            mctslite = MCTSLite(num_simulations=mctslite_sims)
            move_count = 0

            while not state.done and move_count < max_moves:
                moves = generate_legal_moves(state)
                if not moves:
                    break

                is_nn_turn = (
                    (state.current_player == Player.WHITE and nn_is_white) or
                    (state.current_player == Player.BLACK and not nn_is_white)
                )

                if is_nn_turn:
                    move, _ = nn_mcts.get_move(state, add_noise=False)
                else:
                    move = mctslite.search(state)

                apply_move(state, move)
                move_count += 1

            game_lengths.append(move_count)

            if state.winner is not None:
                nn_won = (
                    (state.winner == Player.WHITE and nn_is_white) or
                    (state.winner == Player.BLACK and not nn_is_white)
                )
                if nn_won:
                    wins += 1
                    result = "W"
                else:
                    losses += 1
                    result = "L"
            else:
                draws += 1
                result = "D"

            logger.info(f"  MCTSLite probe {game_idx+1}/{num_games}: {result} "
                        f"({move_count} moves, NN as {'W' if nn_is_white else 'B'}) "
                        f"[running: {wins}W {draws}D {losses}L]")

        del nn_mcts
        gc.collect()

        total = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.5

        # Estimate ELO: MCTSLite at 50 sims ≈ 300
        # Rebased from original 650 anchor (was overestimated)
        mctslite_elo = 300
        elo_diff = win_rate_to_elo_diff(win_rate)
        estimated_elo = mctslite_elo + elo_diff

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
            "estimated_elo": estimated_elo,
            "mctslite_sims": mctslite_sims,
            "nn_sims": nn_sims,
        }

    def log_metrics(self, metrics: dict):
        """Append metrics to the training log and wandb."""
        metrics["timestamp"] = time.time()
        metrics["training_step"] = self.training_step
        metrics["iteration"] = self.iteration

        with open(self.training_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        phase = metrics.get("phase", "")
        if self.use_wandb:
            wb_metrics = {}
            for k, v in metrics.items():
                if k in ("timestamp", "phase"):
                    continue
                if isinstance(v, (int, float)):
                    wb_metrics[f"{phase}/{k}" if phase else k] = v
            _safe_wandb_log(wb_metrics, step=self.training_step)

        if self.tb_writer:
            for k, v in metrics.items():
                if k in ("timestamp", "phase"):
                    continue
                if isinstance(v, (int, float)):
                    tag = f"{phase}/{k}" if phase else k
                    self.tb_writer.add_scalar(tag, v, self.training_step)
            self.tb_writer.flush()

    def seed_buffer(self, num_games: int = 50, use_smart_seeds: bool = True,
                    seed_file: str = "checkpoints/smart_seeds.pkl"):
        """Seed replay buffer with intelligent games.

        Uses heuristic players with commander hunting for better quality
        seed games instead of expensive random rollouts.
        """
        if use_smart_seeds:
            # Try to load pre-generated seeds first
            import os
            import pickle

            if os.path.exists(seed_file):
                logger.info(f"Loading pre-generated smart seeds from {seed_file}...")
                with open(seed_file, 'rb') as f:
                    smart_buffer = pickle.load(f)
                total_positions = len(smart_buffer)
            else:
                logger.info(f"Generating {num_games} smart seed games using heuristic players...")
                smart_buffer = generate_smart_seeds(num_games, verbose=False)
                total_positions = len(smart_buffer)

                # Save for future use
                os.makedirs(os.path.dirname(seed_file), exist_ok=True)
                with open(seed_file, 'wb') as f:
                    pickle.dump(smart_buffer, f)
                logger.info(f"Saved seed games to {seed_file}")

            # Copy smart seeds to permanent seed partition, then free the pickle data
            for i in range(total_positions):
                self.replay_buffer.push_seed(
                    smart_buffer.planes[i],
                    smart_buffer.policies[i],
                    smart_buffer.values[i]
                )
            del smart_buffer
            gc.collect()

            logger.info(f"Added {total_positions} positions from smart seed games")
        else:
            # Fallback to MCTSLite (much slower, not recommended)
            logger.info(f"Seeding buffer with {num_games} MCTSLite games...")
            agent = MCTSLiteAgent(num_simulations=50)
            total_positions = 0

            for game_idx in range(num_games):
                moves, winner, turns = play_game(agent, agent)
                if winner is None:
                    continue  # Skip draws

                # Replay game to extract training examples
                state = GameState()
                for move in moves:
                    planes = state_to_planes(state)
                    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                    policy[move_to_policy_index(move)] = 1.0

                    # Value from current player's perspective: +1 win, -1 loss
                    if int(winner) == int(state.current_player):
                        value = 1.0
                    else:
                        value = -1.0

                    self.replay_buffer.push(planes, policy, value)
                    total_positions += 1
                    apply_move(state, move)

            if (game_idx + 1) % 10 == 0:
                logger.info(f"  Seed game {game_idx+1}/{num_games}: "
                            f"{total_positions} total positions")

        logger.info(f"Seeded buffer with {total_positions} positions from "
                    f"{num_games} games. Buffer size: {len(self.replay_buffer)}")

    def _hot_reload_config(self):
        """Re-read config YAML if available. Skips network architecture (unsafe to change)."""
        if not self.config_path:
            return
        try:
            import yaml
            with open(self.config_path) as f:
                new_config = yaml.safe_load(f)
            # Preserve network config (can't change architecture mid-run)
            new_config["network"] = self.config.get("network", {})
            self.config = new_config
            # Resize replay buffer partitions if config changed
            sp_cfg = new_config.get("selfplay", {})
            new_decisive = sp_cfg.get("buffer_decisive_size", self.replay_buffer.decisive_size)
            new_draw = sp_cfg.get("buffer_draw_size", self.replay_buffer.draw_size)
            if new_decisive != self.replay_buffer.decisive_size or new_draw != self.replay_buffer.draw_size:
                old_d, old_dr = self.replay_buffer.decisive_size, self.replay_buffer.draw_size
                old_size = len(self.replay_buffer)
                self.replay_buffer.resize(new_decisive, new_draw)
                logger.info("Resized buffer: decisive %d->%d, draw %d->%d (total %d->%d)",
                            old_d, new_decisive, old_dr, new_draw,
                            old_size, len(self.replay_buffer))
            logger.info("Hot-reloaded config from %s", self.config_path)
        except Exception as e:
            logger.warning("Config hot-reload failed: %s", e)

    def _gpu_defrag(self):
        """Full CPU round-trip to defragment Jetson unified memory."""
        if self.device != "cpu":
            self.network.cpu()
            self.optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.network.to(self.device)

    def run_iteration(self):
        """Run one training iteration: self-play + training + evaluation."""
        self._hot_reload_config()
        sp_cfg = self.config.get("selfplay", {})
        train_cfg = self.config.get("training", {})
        mcts_cfg = self.config.get("mcts", {})

        self.iteration += 1
        iteration_start = time.time()
        logger.info(f"=== Iteration {self.iteration} ===")
        _log_memory("iteration_start")

        # Periodic seed re-injection disabled — at higher ELO the heuristic seed
        # data creates distribution mismatch that hurts eval performance.
        # Seeds are already in the buffer from initial seeding at iteration 0.
        # if self.iteration > 1 and self.iteration % 5 == 0:
        #     logger.info(f"Adding high-quality seed games at iteration {self.iteration}")
        #     self.seed_buffer(num_games=20)
        #     gc.collect()
        #     _log_memory("after_seed_injection")

        # Self-play phase — use current training network (not best_network)
        # This ensures self-play data always matches the network being trained,
        # avoiding distribution mismatch that causes eval regression.
        base_sims = mcts_cfg.get("num_simulations", 200)
        min_sims = int(mcts_cfg.get("min_selfplay_simulations", 25))
        fixed_sims = mcts_cfg.get("fixed_selfplay_simulations")
        if fixed_sims is not None:
            num_sims = int(fixed_sims)
            logger.info(f"Self-play phase... (fixed sims: {num_sims})")
        else:
            num_sims = adaptive_simulations(
                self.elo_for_sims,
                min_sims=min_sims,
                max_sims=base_sims,
            )
            logger.info(
                "Self-play phase... "
                f"(adaptive sims: {num_sims}, probe_ELO={self.elo_estimate:.0f}, "
                f"smoothed_ELO={self.elo_for_sims:.0f})"
            )
        selfplay_start = time.time()
        num_workers = sp_cfg.get("num_workers", 1)
        parallel_games = sp_cfg.get("parallel_games", 0)
        # Build Gumbel config if enabled
        gumbel_cfg = self.config.get("gumbel", {})
        use_gumbel = gumbel_cfg.get("enabled", False)
        gumbel_config = None
        if use_gumbel:
            gumbel_config = {
                "max_num_considered_actions": gumbel_cfg.get("max_num_considered_actions", 16),
                "gumbel_scale": gumbel_cfg.get("gumbel_scale", 1.0),
                "maxvisit_init": gumbel_cfg.get("maxvisit_init", 50.0),
                "value_scale": gumbel_cfg.get("value_scale", 0.1),
            }

        sp_kwargs = dict(
            network=self.network,
            num_games=sp_cfg.get("num_games_per_iteration", 100),
            num_simulations=num_sims,
            temperature_threshold=mcts_cfg.get("temperature_threshold", 30),
            dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
            dirichlet_epsilon=mcts_cfg.get("dirichlet_epsilon", 0.25),
            random_opponent_fraction=sp_cfg.get("random_opponent_fraction", 0.0),
            draw_value_target=float(sp_cfg.get("draw_value_target", 0.0)),
            device=self.device,
        )
        if use_gumbel and num_workers > 1:
            # Multiprocess Gumbel: workers run Gumbel MCTS on CPU,
            # main process runs GPU server batching all NN evaluations
            examples, sp_stats = run_selfplay_multiprocess(
                **sp_kwargs, num_workers=num_workers,
                gumbel_config=gumbel_config,
            )
        elif use_gumbel:
            # Single-process Gumbel: batched evaluation in main process
            examples, sp_stats = run_selfplay_batch_parallel(
                **sp_kwargs, parallel_games=parallel_games,
                gumbel_config=gumbel_config,
            )
        elif num_workers > 1:
            examples, sp_stats = run_selfplay_multiprocess(
                **sp_kwargs, num_workers=num_workers,
            )
        elif parallel_games > 0:
            examples, sp_stats = run_selfplay_batch_parallel(
                **sp_kwargs, parallel_games=parallel_games,
            )
        else:
            examples, sp_stats = run_selfplay_batch(**sp_kwargs)
        selfplay_elapsed = time.time() - selfplay_start

        # Add all positions to buffer — draws carry useful value signal
        # (draw_sample_weight in training config downweights them in the loss).
        num_new_examples = len(examples)
        for planes, policy, value in examples:
            self.replay_buffer.push(planes, policy, value)
        del examples
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        _log_memory("after_selfplay")
        parts = self.replay_buffer.partition_sizes()
        logger.info(f"Generated {num_new_examples} examples. "
                    f"Buffer: {len(self.replay_buffer)} "
                    f"(seeds={parts['seeds']}, decisive={parts['decisive']}, "
                    f"draws={parts['draws']})")
        logger.info(f"Self-play stats: W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
                    f"D:{sp_stats['draws']} white_win%={sp_stats['white_win_pct']:.1%} "
                    f"lengths={sp_stats['min_game_length']}-{sp_stats['max_game_length']} "
                    f"avg={sp_stats['avg_game_length']:.0f}")
        draw_reason_counts = sp_stats.get("draw_reason_counts", {})
        if sp_stats["draws"] > 0:
            logger.info(
                "Draw reasons: "
                f"repetition={draw_reason_counts.get('repetition', 0)}, "
                f"turn_limit={draw_reason_counts.get('turn_limit', 0)}, "
                f"fifty_move={draw_reason_counts.get('fifty_move', 0)}, "
                f"other={draw_reason_counts.get('stalemate_or_other', 0)}"
            )

        total_selfplay_games = (
            sp_stats["white_wins"] + sp_stats["black_wins"] + sp_stats["draws"]
        )
        draw_rate = sp_stats["draws"] / total_selfplay_games if total_selfplay_games else 0.0
        decisive_rate = (
            (sp_stats["white_wins"] + sp_stats["black_wins"]) / total_selfplay_games
            if total_selfplay_games else 0.0
        )

        sp_metrics = {
            "iteration": self.iteration,
            "selfplay/num_examples": num_new_examples,
            "selfplay/buffer_size": len(self.replay_buffer),
            "selfplay/buffer_seeds": parts["seeds"],
            "selfplay/buffer_decisive": parts["decisive"],
            "selfplay/buffer_draws": parts["draws"],
            "selfplay/elapsed_sec": selfplay_elapsed,
            "selfplay/avg_game_length": sp_stats["avg_game_length"],
            "selfplay/min_game_length": sp_stats["min_game_length"],
            "selfplay/max_game_length": sp_stats["max_game_length"],
            "selfplay/white_wins": sp_stats["white_wins"],
            "selfplay/black_wins": sp_stats["black_wins"],
            "selfplay/draws": sp_stats["draws"],
            "selfplay/total_games": total_selfplay_games,
            "selfplay/draw_rate": draw_rate,
            "selfplay/decisive_rate": decisive_rate,
            "selfplay/draw_reason_turn_limit": draw_reason_counts.get("turn_limit", 0),
            "selfplay/draw_reason_fifty_move": draw_reason_counts.get("fifty_move", 0),
            "selfplay/draw_reason_repetition": draw_reason_counts.get("repetition", 0),
            "selfplay/draw_reason_other": draw_reason_counts.get("stalemate_or_other", 0),
            "selfplay/white_win_pct": sp_stats["white_win_pct"],
            "selfplay/num_simulations": num_sims,
        }
        if self.use_wandb:
            _safe_wandb_log(sp_metrics, step=self.training_step)
            # Log per-game details as a table
            if "game_details" in sp_stats:
                table = wandb.Table(columns=["game", "length", "winner", "draw_reason"])
                for g in sp_stats["game_details"]:
                    table.add_data(g["game"], g["length"], g["winner"], g.get("draw_reason"))
                _safe_wandb_log({"selfplay/games": table}, step=self.training_step)
        if self.tb_writer:
            for k, v in sp_metrics.items():
                self.tb_writer.add_scalar(k, v, self.training_step)

        # Log games in DCN notation for strategy analysis
        if "game_records" in sp_stats:
            games_dir = os.path.join(self.log_dir, "games")
            os.makedirs(games_dir, exist_ok=True)
            dcn_path = os.path.join(games_dir, f"iter_{self.iteration:04d}.dcn")
            try:
                dcn_games = []
                for i, rec in enumerate(sp_stats["game_records"]):
                    headers = {
                        "Iteration": str(self.iteration),
                        "Game": str(i + 1),
                        "Moves": str(rec["length"]),
                        "ELO": f"{self.elo_estimate:.0f}",
                    }
                    result_map = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}
                    result = result_map.get(rec["winner"], "1/2-1/2")
                    dcn_games.append(game_to_dcn(rec["moves"], headers, result))
                with open(dcn_path, "w") as f:
                    f.write("\n\n".join(dcn_games))
                logger.info(f"Saved {len(dcn_games)} games to {dcn_path}")
                # Backup game log to W&B
                if self.use_wandb:
                    try:
                        artifact = wandb.Artifact(
                            f"game-log-iter{self.iteration:04d}",
                            type="game-log",
                            metadata={
                                "iteration": self.iteration,
                                "num_games": len(dcn_games),
                                "elo": self.elo_estimate,
                            },
                        )
                        artifact.add_file(dcn_path)
                        _safe_wandb_log_artifact(artifact)
                    except Exception as e:
                        logger.warning(f"Failed to upload game log to W&B: {e}")
            except Exception as e:
                logger.warning(f"Failed to save game log: {e}")
            # Free game records to reclaim memory (state objects are large)
            del sp_stats["game_records"]
            gc.collect()

        # Training phase — skip if buffer too small
        min_buffer = sp_cfg.get("min_buffer_size", 0)
        if len(self.replay_buffer) < min_buffer:
            logger.info(f"Buffer size {len(self.replay_buffer)} < {min_buffer}, "
                        "skipping training this iteration")
            self.save_checkpoint(save_buffer=True)
            return

        logger.info("Training phase...")
        batch_size = train_cfg.get("batch_size", 256)
        max_steps = train_cfg.get("steps_per_iteration", 1000)
        # Scale steps to buffer size: ~2 passes through the data, capped at max_steps.
        # Prevents overfitting when buffer is small (e.g. 2K positions × 300 steps = 16x overfit).
        buf_size = len(self.replay_buffer)
        steps = min(max_steps, max(1, (buf_size * 2) // batch_size))
        if steps < max_steps:
            logger.info(f"Scaled training steps {max_steps} -> {steps} for buffer size {buf_size}")
        checkpoint_interval = train_cfg.get("checkpoint_interval", 500)

        # Log seed sampling weight for this iteration
        seed_cfg = train_cfg
        seed_w_init = float(seed_cfg.get("seed_sample_weight_init", 1.0))
        seed_w_decay = float(seed_cfg.get("seed_sample_weight_decay", 1.0))
        seed_w_min = float(seed_cfg.get("seed_sample_weight_min", 0.1))
        current_seed_weight = max(seed_w_min, seed_w_init * (seed_w_decay ** self.iteration))
        if seed_w_decay < 1.0:
            logger.info(f"Seed sampling weight: {current_seed_weight:.3f} "
                        f"(init={seed_w_init}, decay={seed_w_decay}, iter={self.iteration})")

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        value_scale_sum = 0.0

        # NVML retry: defrag + train, retry up to 3 times without redoing self-play
        for _train_attempt in range(3):
            try:
                self._gpu_defrag()
                for step in range(steps):
                    losses = self.train_step(batch_size)
                    total_loss_sum += losses["total_loss"]
                    policy_loss_sum += losses["policy_loss"]
                    value_loss_sum += losses["value_loss"]
                    value_scale_sum += losses["value_scale"]

                    if (self.training_step % checkpoint_interval == 0):
                        self.save_checkpoint()
                break  # Training succeeded
            except RuntimeError as e:
                if "NVML_SUCCESS" in str(e) and _train_attempt < 2:
                    logger.warning(
                        f"CUDA NVML crash during training (attempt {_train_attempt+1}/3). "
                        f"Full GPU defrag and retrying training only..."
                    )
                    time.sleep(3)
                    # Reset loss accumulators for clean retry
                    total_loss_sum = 0.0
                    policy_loss_sum = 0.0
                    value_loss_sum = 0.0
                    value_scale_sum = 0.0
                else:
                    raise

        avg_losses = {
            "avg_total_loss": total_loss_sum / steps if steps > 0 else 0,
            "avg_policy_loss": policy_loss_sum / steps if steps > 0 else 0,
            "avg_value_loss": value_loss_sum / steps if steps > 0 else 0,
            "avg_value_scale": value_scale_sum / steps if steps > 0 else 0,
        }
        logger.info(f"Training: policy={avg_losses['avg_policy_loss']:.3f} "
                     f"value={avg_losses['avg_value_loss']:.4f}")

        # Pure AlphaZero: no eval gatekeeper. Save best.pt every iteration.
        self.save_best()

        # Periodic MCTSLite ELO probe (every 5 iterations, non-gating)
        elo_interval = train_cfg.get("elo_probe_interval", 5)
        elo_results = None
        if self.iteration % elo_interval == 0:
            _log_memory("before_elo_probe")
            logger.info("MCTSLite ELO probe...")
            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()
            probe_num_games = int(train_cfg.get("elo_probe_games", 20))
            probe_mctslite_sims = int(train_cfg.get("elo_probe_mctslite_sims", 50))
            probe_nn_sims = int(train_cfg.get("elo_probe_nn_sims", max(min_sims, probe_mctslite_sims)))
            probe_max_moves = int(train_cfg.get("elo_probe_max_moves", 100))
            elo_results = self.estimate_elo_mctslite(
                num_games=probe_num_games,
                mctslite_sims=probe_mctslite_sims,
                nn_sims=probe_nn_sims,
                max_moves=probe_max_moves,
            )
            self.elo_estimate = elo_results["estimated_elo"]
            smoothing = float(train_cfg.get("adaptive_elo_smoothing", 0.7))
            smoothing = min(max(smoothing, 0.0), 0.99)
            self.elo_for_sims = (
                smoothing * self.elo_for_sims + (1.0 - smoothing) * self.elo_estimate
            )
            logger.info(f"MCTSLite ELO estimate: {self.elo_estimate:.0f} "
                        f"(W:{elo_results['wins']} L:{elo_results['losses']} D:{elo_results['draws']} "
                        f"win_rate={elo_results['win_rate']:.2f} avg_len={elo_results['avg_game_length']:.0f}, "
                        f"nn_sims={elo_results['nn_sims']}, mctslite_sims={elo_results['mctslite_sims']})")

            # Clear seed partition once model is strong enough to self-improve
            seed_elo_threshold = float(train_cfg.get("seed_removal_elo", 1000))
            parts = self.replay_buffer.partition_sizes()
            if self.elo_estimate >= seed_elo_threshold and parts["seeds"] > 0:
                n_cleared = self.replay_buffer.clear_seeds()
                logger.info(f"Cleared {n_cleared} seed positions (ELO {self.elo_estimate:.0f} "
                            f">= threshold {seed_elo_threshold:.0f})")

            if self.use_wandb:
                elo_metrics = {
                    "elo/estimate": self.elo_estimate,
                    "elo/for_sims": self.elo_for_sims,
                    "elo/vs_mctslite_win_rate": elo_results["win_rate"],
                    "elo/vs_mctslite_wins": elo_results["wins"],
                    "elo/vs_mctslite_losses": elo_results["losses"],
                    "elo/vs_mctslite_draws": elo_results["draws"],
                    "elo/vs_mctslite_avg_length": elo_results["avg_game_length"],
                    "elo/vs_mctslite_num_games": probe_num_games,
                    "elo/vs_mctslite_nn_sims": elo_results["nn_sims"],
                    "elo/vs_mctslite_sims": elo_results["mctslite_sims"],
                }
                _safe_wandb_log(elo_metrics, step=self.training_step)
                wandb.run.summary["elo_estimate"] = self.elo_estimate
                wandb.run.summary["total_iterations"] = self.iteration

            # Upload best model to W&B on ELO probe iterations
            if self.use_wandb:
                artifact = wandb.Artifact(
                    f"best-model-iter{self.iteration}",
                    type="model",
                    metadata={
                        "iteration": self.iteration,
                        "training_step": self.training_step,
                        "elo_estimate": self.elo_estimate,
                    },
                )
                artifact.add_file(str(self.checkpoint_dir / "best.pt"))
                _safe_wandb_log_artifact(artifact)

            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()

        # W&B summary update (every iteration) + alert (only on ELO probe iterations)
        if self.use_wandb:
            wandb.run.summary["total_iterations"] = self.iteration
            if elo_results:
                total_sp = total_selfplay_games
                sp_draw_pct = draw_rate * 100
                iter_elapsed = time.time() - iteration_start
                wandb.alert(
                    title=f"Iter {self.iteration}: ELO {self.elo_estimate:.0f}",
                    text=(
                        f"ELO: {self.elo_estimate:.0f} (W:{elo_results['wins']} L:{elo_results['losses']} D:{elo_results['draws']}) | Sims: {num_sims}\n"
                        f"Self-play: {sp_stats['white_wins']}W/{sp_stats['black_wins']}B/{sp_stats['draws']}D "
                        f"({sp_draw_pct:.0f}% draws) avg={sp_stats['avg_game_length']:.0f} moves "
                        f"[{sp_stats['min_game_length']}-{sp_stats['max_game_length']}]\n"
                        f"Loss: policy={avg_losses['avg_policy_loss']:.3f} value={avg_losses['avg_value_loss']:.4f}\n"
                        f"Buffer: {len(self.replay_buffer)} | Mem: {_get_rss_mb():.0f}MB | Time: {iter_elapsed/60:.1f}min"
                    ),
                    wait_duration=0,
                )
                for threshold in [500, 1000, 1500, 2000]:
                    if self.elo_estimate >= threshold:
                        milestone_key = f"elo_milestone_{threshold}"
                        if not wandb.run.summary.get(milestone_key):
                            wandb.run.summary[milestone_key] = True
                            wandb.alert(
                                title=f"ELO milestone: {threshold}",
                                text=f"Model reached ELO {self.elo_estimate:.0f} "
                                     f"at iteration {self.iteration}",
                            )

        # Log GPU memory stats
        if self.device != "cpu":
            gpu_metrics = {
                "system/gpu_allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "system/gpu_reserved_mb": torch.cuda.memory_reserved() / 1e6,
            }
            if self.use_wandb:
                _safe_wandb_log(gpu_metrics, step=self.training_step)
            if self.tb_writer:
                for k, v in gpu_metrics.items():
                    self.tb_writer.add_scalar(k, v, self.training_step)

        # Log metrics
        iteration_elapsed = time.time() - iteration_start
        self.log_metrics({
            "phase": "iteration",
            "num_examples": num_new_examples,
            "buffer_size": len(self.replay_buffer),
            "elo_estimate": self.elo_estimate,
            "elo_for_sims": self.elo_for_sims,
            "selfplay_draw_rate": draw_rate,
            "selfplay_decisive_rate": decisive_rate,
            "selfplay_draw_reason_repetition": draw_reason_counts.get("repetition", 0),
            "selfplay_draw_reason_turn_limit": draw_reason_counts.get("turn_limit", 0),
            "selfplay_draw_reason_fifty_move": draw_reason_counts.get("fifty_move", 0),
            "selfplay_draw_reason_other": draw_reason_counts.get("stalemate_or_other", 0),
            "elapsed_sec": iteration_elapsed,
            **avg_losses,
        })

        # Save checkpoint at end of iteration (with buffer for full resume)
        _log_memory("before_checkpoint_save")
        self.save_checkpoint(save_buffer=True)
        _log_memory("after_checkpoint_save")

    def train(self, max_iterations: Optional[int] = None):
        """Run the full training loop."""
        max_iter = max_iterations or self.config.get("training", {}).get("max_iterations", 200)

        # Try to resume (everything on CPU at this point)
        warm_started = False
        resumed = False
        if self.load_checkpoint():
            logger.info(f"Resumed from step {self.training_step}, iteration {self.iteration}")
            resumed = True
        else:
            logger.info("Starting fresh training")
            # If best.pt already exists (e.g. restored from W&B), warm-start
            # the network from it instead of starting from random weights
            best_path = self.checkpoint_dir / "best.pt"
            if best_path.exists():
                try:
                    best_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
                    self.network.load_state_dict(best_ckpt["network_state"])
                    old_elo = best_ckpt.get("elo_estimate", 0)
                    self.elo_estimate = 0
                    self.elo_for_sims = 0.0
                    logger.info(f"Warm-starting from existing best.pt (old ELO {old_elo:.0f}, reset to 0)")
                    warm_started = True
                    del best_ckpt
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Could not load best.pt for warm start: {e}")
                    self.save_best()
            else:
                self.save_best()  # Save initial model as best
            # Seed the replay buffer with heuristic games
            logger.info("Seeding replay buffer with heuristic games")
            self.seed_buffer(num_games=100)
            # Check for existing buffer to carry over self-play data
            existing_buf = self.checkpoint_dir / "existing_buffer.npz"
            if existing_buf.exists():
                logger.info(f"Loading existing buffer from {existing_buf}")
                self.replay_buffer.load_data(str(existing_buf))
                parts = self.replay_buffer.partition_sizes()
                logger.info(f"Loaded existing buffer: {len(self.replay_buffer)} positions "
                            f"(seeds={parts['seeds']}, decisive={parts['decisive']}, "
                            f"draws={parts['draws']})")
                existing_buf.unlink()  # One-shot: don't reload on next fresh start
            logger.info(f"Initial seeding complete. Buffer size: {len(self.replay_buffer)}")
            # Save initial checkpoint so resume works if we crash mid-iteration
            self.save_checkpoint(save_buffer=True)

        # Now move training network to GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.network.to(self.device)
        # Rebuild optimizer to point at GPU params
        train_cfg = self.config.get("training", {})
        old_state = self.optimizer.state_dict()
        ot = train_cfg.get("optimizer", "sgd").lower()
        if ot == "muon" and HAS_MUON:
            self.optimizer = MuonSGD(
                self.network,
                muon_lr=train_cfg.get("learning_rate", 0.02),
                muon_momentum=train_cfg.get("muon_momentum", 0.95),
                sgd_lr=train_cfg.get("head_lr", 0.001),
                sgd_momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )
        else:
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=train_cfg.get("learning_rate", 0.01),
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )
        try:
            self.optimizer.load_state_dict(old_state)
        except Exception:
            pass  # Fresh optimizer ok (SGD->Muon switch)
        del old_state
        gc.collect()

        # Pre-train on seed data before self-play so the model
        # has learned something before generating its own games.
        # Skip if warm-starting from a trained model — re-pre-training on
        # seeds would overwrite learned features and degrade the model.
        if self.iteration == 0 and len(self.replay_buffer) > 0 and not warm_started and not resumed:
            batch_size = train_cfg.get("batch_size", 128)
            # ~4 passes through the full seed buffer
            steps = (len(self.replay_buffer) * 4) // batch_size
            logger.info(f"Pre-training on {len(self.replay_buffer)} seed positions ({steps} steps, ~4 passes)...")
            total_loss = 0.0
            for step in range(steps):
                losses = self.train_step(batch_size)
                total_loss += losses["total_loss"]
                if (step + 1) % 200 == 0:
                    logger.info(f"  Pre-training step {step+1}/{steps}: avg_loss={total_loss/(step+1):.4f}")
            avg_loss = total_loss / steps
            logger.info(f"Pre-training complete: avg_loss={avg_loss:.4f} over {steps} steps")
            self.save_best()

        cuda_crash_retries = 0
        max_cuda_retries = 3

        while self.iteration < max_iter:
            try:
                self.run_iteration()
                cuda_crash_retries = 0  # Reset on success
            except KeyboardInterrupt:
                logger.info("Training interrupted. Saving checkpoint...")
                self.save_checkpoint(tag="interrupted", save_buffer=True)
                break
            except RuntimeError as e:
                if "NVML_SUCCESS" in str(e) and cuda_crash_retries < max_cuda_retries:
                    cuda_crash_retries += 1
                    logger.warning(
                        f"CUDA NVML crash during iteration {self.iteration} "
                        f"(retry {cuda_crash_retries}/{max_cuda_retries}). "
                        f"Full GPU defrag and retrying..."
                    )
                    self._gpu_defrag()
                    time.sleep(2)
                    # Undo the iteration increment from run_iteration() so we retry
                    # the same iteration instead of skipping to the next one
                    self.iteration -= 1
                    continue
                else:
                    logger.error(f"Error during iteration {self.iteration}: {e}")
                    self.save_checkpoint(tag="error", save_buffer=True)
                    raise
            except Exception as e:
                logger.error(f"Error during iteration {self.iteration}: {e}")
                self.save_checkpoint(tag="error", save_buffer=True)
                raise
