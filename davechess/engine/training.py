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
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from davechess.engine.network import DaveChessNetwork, POLICY_SIZE, state_to_planes, move_to_policy_index
from davechess.engine.selfplay import ReplayBuffer, run_selfplay_batch
from davechess.engine.mcts import MCTS
from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.data.generator import MCTSLiteAgent, play_game
from davechess.engine.smart_seeds import generate_smart_seeds

logger = logging.getLogger("davechess.training")


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


class Trainer:
    """AlphaZero training loop."""

    def __init__(self, config: dict, device: str = "cpu", use_wandb: bool = False):
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and HAS_WANDB
        self.tb_writer = None

        # Create networks on CPU first — moved to GPU after checkpoint loading
        net_cfg = config.get("network", {})
        self.network = DaveChessNetwork(
            num_res_blocks=net_cfg.get("num_res_blocks", 5),
            num_filters=net_cfg.get("num_filters", 64),
            input_planes=net_cfg.get("input_planes", 12),
        )  # CPU initially, moved to GPU in train()

        self.best_network = DaveChessNetwork(
            num_res_blocks=net_cfg.get("num_res_blocks", 5),
            num_filters=net_cfg.get("num_filters", 64),
            input_planes=net_cfg.get("input_planes", 12),
        )  # CPU always
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
        self.elo_estimate = 0  # Current network's estimated ELO (baseline=0)
        self.consecutive_rejections = 0

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
            "best_elo_estimate": self.best_elo_estimate,
            "consecutive_rejections": self.consecutive_rejections,
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
                wandb.log_artifact(artifact)

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
        # Load everything on CPU — network moves to GPU later in train()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_step = checkpoint["training_step"]
        self.iteration = checkpoint["iteration"]
        self.best_elo_estimate = checkpoint.get("best_elo_estimate", 0)
        self.consecutive_rejections = checkpoint.get("consecutive_rejections", 0)

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
            logger.info(f"Loaded replay buffer: {len(self.replay_buffer)} positions")
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

        losses = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

        if self.training_step % 10 == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            step_metrics = {
                "train/total_loss": losses["total_loss"],
                "train/policy_loss": losses["policy_loss"],
                "train/value_loss": losses["value_loss"],
                "train/learning_rate": lr,
            }
            if self.use_wandb:
                wandb.log(step_metrics, step=self.training_step)
            if self.tb_writer:
                for k, v in step_metrics.items():
                    self.tb_writer.add_scalar(k, v, self.training_step)

        return losses

    def evaluate_network(self, num_games: int = 40,
                         num_simulations: int = 100) -> dict:
        """Evaluate current network against best network.

        Returns dict with win_rate and detailed results.
        """
        current_mcts = MCTS(self.network, num_simulations=num_simulations,
                            temperature=0.1, device=self.device)
        best_mcts = MCTS(self.best_network, num_simulations=num_simulations,
                         temperature=0.1, device=self.device)

        wins = 0
        losses = 0
        draws = 0
        game_lengths = []

        for game_idx in range(num_games):
            state = GameState()
            current_is_white = (game_idx % 2 == 0)
            move_count = 0

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
                move_count += 1

            game_lengths.append(move_count)

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
        win_rate = wins / total_decisive if total_decisive > 0 else 0.5

        # Explicitly free MCTS trees (circular parent↔child refs)
        del current_mcts, best_mcts
        gc.collect()

        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
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
            wandb.log(wb_metrics, step=self.training_step)

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

            # Copy smart seeds to our replay buffer, then free the pickle data
            for i in range(total_positions):
                self.replay_buffer.push(
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

    def run_iteration(self):
        """Run one training iteration: self-play + training + evaluation."""
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
        num_sims = adaptive_simulations(self.best_elo_estimate, min_sims=25, max_sims=base_sims)
        logger.info(f"Self-play phase... (adaptive sims: {num_sims}, ELO={self.best_elo_estimate:.0f})")
        selfplay_start = time.time()
        examples, sp_stats = run_selfplay_batch(
            network=self.network,
            num_games=sp_cfg.get("num_games_per_iteration", 100),
            num_simulations=num_sims,
            temperature_threshold=mcts_cfg.get("temperature_threshold", 30),
            device=self.device,
        )
        selfplay_elapsed = time.time() - selfplay_start

        num_new_examples = len(examples)
        for planes, policy, value in examples:
            self.replay_buffer.push(planes, policy, value)
        del examples
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        _log_memory("after_selfplay")
        logger.info(f"Generated {num_new_examples} training examples. "
                    f"Buffer size: {len(self.replay_buffer)}")
        logger.info(f"Self-play stats: W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
                    f"D:{sp_stats['draws']} white_win%={sp_stats['white_win_pct']:.1%} "
                    f"lengths={sp_stats['min_game_length']}-{sp_stats['max_game_length']} "
                    f"avg={sp_stats['avg_game_length']:.0f}")

        sp_metrics = {
            "iteration": self.iteration,
            "selfplay/num_examples": num_new_examples,
            "selfplay/buffer_size": len(self.replay_buffer),
            "selfplay/elapsed_sec": selfplay_elapsed,
            "selfplay/avg_game_length": sp_stats["avg_game_length"],
            "selfplay/min_game_length": sp_stats["min_game_length"],
            "selfplay/max_game_length": sp_stats["max_game_length"],
            "selfplay/white_wins": sp_stats["white_wins"],
            "selfplay/black_wins": sp_stats["black_wins"],
            "selfplay/draws": sp_stats["draws"],
            "selfplay/white_win_pct": sp_stats["white_win_pct"],
            "selfplay/num_simulations": num_sims,
        }
        if self.use_wandb:
            wandb.log(sp_metrics, step=self.training_step)
            # Log per-game details as a table
            if "game_details" in sp_stats:
                table = wandb.Table(columns=["game", "length", "winner"])
                for g in sp_stats["game_details"]:
                    table.add_data(g["game"], g["length"], g["winner"])
                wandb.log({"selfplay/games": table}, step=self.training_step)
        if self.tb_writer:
            for k, v in sp_metrics.items():
                self.tb_writer.add_scalar(k, v, self.training_step)

        # Training phase — skip if buffer too small
        min_buffer = sp_cfg.get("min_buffer_size", 0)
        if len(self.replay_buffer) < min_buffer:
            logger.info(f"Buffer size {len(self.replay_buffer)} < {min_buffer}, "
                        "skipping training this iteration")
            self.save_checkpoint(save_buffer=True)
            return

        logger.info("Training phase...")
        # Clear GPU cache before training to avoid CUDA fragmentation after self-play
        if self.device != "cpu":
            torch.cuda.empty_cache()
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
        # Networks are ~3MB each, so dual-GPU is fine. Clear cache first
        # to defragment after training phase.
        eval_start = time.time()
        eval_games = train_cfg.get("eval_games", 40)
        base_eval_sims = train_cfg.get("eval_simulations", 50)
        eval_sims = adaptive_simulations(self.best_elo_estimate, min_sims=30, max_sims=base_eval_sims)
        _log_memory("before_eval")
        logger.info(f"Evaluation phase... (adaptive sims: {eval_sims})")
        eval_threshold = train_cfg.get("eval_threshold", 0.55)
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        self.best_network.to(self.device)
        eval_results = self.evaluate_network(num_games=eval_games,
                                              num_simulations=eval_sims)
        self.best_network.to("cpu")
        eval_elapsed = time.time() - eval_start
        win_rate = eval_results["win_rate"]
        logger.info(f"Evaluation: win_rate={win_rate:.3f} "
                    f"(W:{eval_results['wins']} L:{eval_results['losses']} D:{eval_results['draws']}) "
                    f"avg_length={eval_results['avg_game_length']:.0f}")

        # Update ELO estimate: current network vs best network
        elo_diff = win_rate_to_elo_diff(win_rate)
        if accepted := (win_rate >= (0.5 if self.iteration <= 10 else eval_threshold)):
            # ELO gain relative to previous best
            self.elo_estimate = self.best_elo_estimate + elo_diff
            self.best_elo_estimate = self.elo_estimate
            logger.info(f"New best network accepted! ELO: {self.elo_estimate:.0f} (+{elo_diff:.0f})")
            self.best_network.load_state_dict(self.network.state_dict())
            self.consecutive_rejections = 0
            self.save_best()
            # Backup best model to W&B
            if self.use_wandb:
                artifact = wandb.Artifact(
                    f"best-model-iter{self.iteration}",
                    type="model",
                    metadata={
                        "iteration": self.iteration,
                        "training_step": self.training_step,
                        "eval_win_rate": win_rate,
                    },
                )
                artifact.add_file(str(self.checkpoint_dir / "best.pt"))
                wandb.log_artifact(artifact)
                logger.info("Uploaded best model to W&B artifacts")
        else:
            self.consecutive_rejections += 1
            logger.info(f"New network rejected, keeping previous best. "
                        f"({self.consecutive_rejections} consecutive rejections)")
            # Auto-reset: after 7 consecutive rejections, reset training network
            # to best model to escape local minima
            max_rejections = train_cfg.get("max_consecutive_rejections", 7)
            if self.consecutive_rejections >= max_rejections:
                logger.warning(f"Resetting training network to best model after "
                               f"{self.consecutive_rejections} consecutive rejections")
                self.network.load_state_dict(self.best_network.state_dict())
                self.consecutive_rejections = 0
                if self.use_wandb:
                    wandb.alert(
                        title=f"Auto-reset at iter {self.iteration}",
                        text=f"Training network reset to best model (ELO {self.best_elo_estimate:.0f}) "
                             f"after {max_rejections} consecutive rejections",
                        wait_duration=0,
                    )

        eval_metrics = {
            "iteration": self.iteration,
            "eval/win_rate": win_rate,
            "eval/wins": eval_results["wins"],
            "eval/losses": eval_results["losses"],
            "eval/draws": eval_results["draws"],
            "eval/avg_game_length": eval_results["avg_game_length"],
            "eval/accepted": int(accepted),
            "eval/elapsed_sec": eval_elapsed,
            "elo": self.best_elo_estimate,
        }
        if self.use_wandb:
            wandb.log(eval_metrics, step=self.training_step)
            # Update summary with best values
            wandb.run.summary["best_elo"] = self.best_elo_estimate
            wandb.run.summary["best_eval_win_rate"] = max(
                wandb.run.summary.get("best_eval_win_rate", 0), win_rate)
            wandb.run.summary["total_iterations"] = self.iteration
            # Email summary after every iteration
            status = f"ACCEPTED ELO {self.best_elo_estimate:.0f} (+{elo_diff:.0f})" if accepted else "rejected"
            wandb.alert(
                title=f"Iter {self.iteration}: {status}",
                text=(
                    f"SP: {sp_stats['white_wins']}W/{sp_stats['black_wins']}B/{sp_stats['draws']}D "
                    f"avg={sp_stats['avg_game_length']:.0f} moves\n"
                    f"Loss: p={avg_losses['avg_policy_loss']:.4f} v={avg_losses['avg_value_loss']:.4f}\n"
                    f"Eval: {win_rate:.3f} (W:{eval_results['wins']} L:{eval_results['losses']} D:{eval_results['draws']})\n"
                    f"ELO: {self.best_elo_estimate:.0f} | Mem: RSS={_get_rss_mb():.0f}MB"
                ),
                wait_duration=0,
            )
            # Alert on ELO milestones
            if accepted:
                for threshold in [500, 1000, 1500, 2000]:
                    prev_elo = self.best_elo_estimate - elo_diff
                    if prev_elo < threshold <= self.best_elo_estimate:
                        wandb.alert(
                            title=f"ELO milestone: {threshold}",
                            text=f"Model reached ELO {self.best_elo_estimate:.0f} "
                                 f"at iteration {self.iteration}",
                        )
        if self.tb_writer:
            for k, v in eval_metrics.items():
                self.tb_writer.add_scalar(k, v, self.training_step)

        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

        # Log GPU memory stats
        if self.device != "cpu":
            gpu_metrics = {
                "system/gpu_allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "system/gpu_reserved_mb": torch.cuda.memory_reserved() / 1e6,
            }
            if self.use_wandb:
                wandb.log(gpu_metrics, step=self.training_step)
            if self.tb_writer:
                for k, v in gpu_metrics.items():
                    self.tb_writer.add_scalar(k, v, self.training_step)

        # Log metrics
        iteration_elapsed = time.time() - iteration_start
        self.log_metrics({
            "phase": "iteration",
            "num_examples": num_new_examples,
            "buffer_size": len(self.replay_buffer),
            "eval_win_rate": win_rate,
            "accepted": int(accepted),
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
        if self.load_checkpoint():
            logger.info(f"Resumed from step {self.training_step}, iteration {self.iteration}")
        else:
            logger.info("Starting fresh training")
            self.save_best()  # Save initial model as best
            # Improved seeding strategy to bootstrap learning
            logger.info("Using improved seed game strategy")
            self.seed_buffer(num_games=100)
            logger.info(f"Initial seeding complete. Buffer size: {len(self.replay_buffer)}")

        # Now move training network to GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.network.to(self.device)
        # Rebuild optimizer to point at GPU params
        train_cfg = self.config.get("training", {})
        old_state = self.optimizer.state_dict()
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=train_cfg.get("learning_rate", 0.01),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        try:
            self.optimizer.load_state_dict(old_state)
        except Exception:
            pass  # Fresh optimizer if state doesn't match
        del old_state
        gc.collect()

        # Pre-train on seed data before self-play so the model
        # has learned something before generating its own games.
        # Multiple passes through the seed data to fully absorb patterns.
        if self.iteration == 0 and len(self.replay_buffer) > 0:
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

        while self.iteration < max_iter:
            try:
                self.run_iteration()
            except KeyboardInterrupt:
                logger.info("Training interrupted. Saving checkpoint...")
                self.save_checkpoint(tag="interrupted", save_buffer=True)
                break
            except Exception as e:
                logger.error(f"Error during iteration {self.iteration}: {e}")
                self.save_checkpoint(tag="error", save_buffer=True)
                raise
