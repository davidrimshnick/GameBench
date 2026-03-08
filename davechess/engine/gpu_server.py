"""GPU inference server for multiprocess MCTS self-play.

The main process runs the GPU server, collecting leaf evaluation requests
from CPU worker processes and batching them into single forward passes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from queue import Empty

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from davechess.engine.network import POLICY_SIZE

logger = logging.getLogger("davechess.gpu_server")


@dataclass
class BatchRequest:
    """Request from a worker to evaluate leaf states."""
    worker_id: int
    planes_list: list  # list of np.ndarray, each (NUM_INPUT_PLANES, 8, 8) float32
    network_id: int = 0  # 0 = primary, 1 = secondary (for probe)


@dataclass
class BatchResponse:
    """Response to a worker with evaluation results."""
    policies: np.ndarray  # (N, POLICY_SIZE) float32 — softmax'd probabilities
    values: list  # list of float, length N
    logits: np.ndarray = None  # (N, POLICY_SIZE) float32 — raw logits (for Gumbel MCTS)


class WorkerDone:
    """Sentinel indicating a worker has finished all its games."""
    __slots__ = ["worker_id"]

    def __init__(self, worker_id: int):
        self.worker_id = worker_id


class GameCompleted:
    """Notification that a worker finished one game (for progress tracking)."""
    __slots__ = ["worker_id", "game_idx", "game_type", "length", "winner"]

    def __init__(self, worker_id: int, game_idx: int, game_type: str,
                 length: int, winner: str):
        self.worker_id = worker_id
        self.game_idx = game_idx
        self.game_type = game_type
        self.length = length
        self.winner = winner


@dataclass
class ProgressTracker:
    """Aggregate partial game results for mid-run logging."""
    games_completed: int = 0
    total_completed_length: int = 0
    vr_wins: int = 0
    vr_losses: int = 0
    vr_draws: int = 0
    sp_white_wins: int = 0
    sp_black_wins: int = 0
    sp_draws: int = 0

    def record(self, msg: GameCompleted):
        """Ingest one finished game notification."""
        self.games_completed += 1
        self.total_completed_length += int(msg.length)

        if msg.game_type == "vs_random":
            nn_plays_white = (msg.game_idx % 2 == 0)
            if msg.winner == "draw":
                self.vr_draws += 1
            elif ((msg.winner == "white" and nn_plays_white) or
                  (msg.winner == "black" and not nn_plays_white)):
                self.vr_wins += 1
            else:
                self.vr_losses += 1
            return

        if msg.winner == "white":
            self.sp_white_wins += 1
        elif msg.winner == "black":
            self.sp_black_wins += 1
        else:
            self.sp_draws += 1

    def summary(self) -> str:
        """Human-readable partial results summary."""
        parts = []
        if self.games_completed > 0:
            parts.append(
                f"avg_len={self.total_completed_length / self.games_completed:.0f}"
            )

        vr_total = self.vr_wins + self.vr_losses + self.vr_draws
        if vr_total > 0:
            vr_score = (self.vr_wins + 0.5 * self.vr_draws) / vr_total
            parts.append(
                f"vs_random W:{self.vr_wins} L:{self.vr_losses} D:{self.vr_draws} "
                f"score={vr_score:.1%}"
            )

        sp_total = self.sp_white_wins + self.sp_black_wins + self.sp_draws
        if sp_total > 0:
            parts.append(
                f"selfplay W:{self.sp_white_wins} B:{self.sp_black_wins} D:{self.sp_draws}"
            )

        return ", ".join(parts)


def run_gpu_server(network, device: str, request_queue, response_queues: list,
                   num_workers: int, workers: list = None,
                   drain_timeout_ms: float = 5.0,
                   num_games: int = 0,
                   secondary_network=None):
    """Run the GPU inference server loop.

    Blocks until all workers send WorkerDone. Batches requests from multiple
    workers into single GPU forward passes.

    Args:
        network: PyTorch model on device.
        device: CUDA device string.
        request_queue: Shared queue receiving BatchRequest/WorkerDone from workers.
        response_queues: Per-worker queues for sending BatchResponse back.
        num_workers: Number of worker processes.
        workers: List of Process objects (for crash detection).
        drain_timeout_ms: Max time in ms to wait for additional requests after
            the first one arrives, to build bigger batches.
        num_games: Total number of games being played (for progress logging).
        secondary_network: Optional second model for dual-network probes.
            Requests with network_id=1 are evaluated against this network.
    """
    workers_done = set()
    drain_timeout_sec = drain_timeout_ms / 1000.0

    # Progress tracking
    total_batches = 0
    total_evals = 0
    start_time = time.monotonic()
    last_log_time = start_time
    log_interval = 30.0  # seconds
    progress = ProgressTracker()

    def _handle_game_completed(msg: GameCompleted):
        progress.record(msg)
        game_label = f"{num_games}" if num_games else "?"
        logger.info(f"  Game {progress.games_completed}/{game_label} done "
                    f"(w{msg.worker_id} g{msg.game_idx} {msg.game_type}): "
                    f"{msg.length} moves, winner={msg.winner}")
        partial = progress.summary()
        if partial:
            logger.info(f"    Partial: {partial}")

    networks = [network]
    if HAS_TORCH and network is not None:
        network.to(device)
        network.eval()
    if HAS_TORCH and secondary_network is not None:
        secondary_network.to(device)
        secondary_network.eval()
        networks.append(secondary_network)

    while len(workers_done) < num_workers:
        # Block until first request arrives
        try:
            msg = request_queue.get(timeout=5.0)
        except Empty:
            # Check for dead workers
            if workers:
                for i, w in enumerate(workers):
                    if i not in workers_done and not w.is_alive():
                        logger.warning(f"Worker {i} died unexpectedly")
                        workers_done.add(i)
            continue

        if isinstance(msg, WorkerDone):
            workers_done.add(msg.worker_id)
            logger.info(f"GPU server: worker {msg.worker_id} done "
                        f"({len(workers_done)}/{num_workers} workers)")
            continue

        if isinstance(msg, GameCompleted):
            _handle_game_completed(msg)
            continue

        # Collect this request and greedily drain for more
        pending: list[BatchRequest] = [msg]
        deadline = time.monotonic() + drain_timeout_sec

        while time.monotonic() < deadline:
            try:
                msg = request_queue.get_nowait()
            except Empty:
                break
            if isinstance(msg, WorkerDone):
                workers_done.add(msg.worker_id)
                logger.info(f"GPU server: worker {msg.worker_id} done "
                            f"({len(workers_done)}/{num_workers} workers)")
                continue
            if isinstance(msg, GameCompleted):
                _handle_game_completed(msg)
                continue
            pending.append(msg)

        # Build combined batch, tracking per-request metadata
        all_planes = []
        # (worker_id, start, end, network_id)
        request_slices: list[tuple[int, int, int, int]] = []
        offset = 0
        for req in pending:
            n = len(req.planes_list)
            request_slices.append((req.worker_id, offset, offset + n, req.network_id))
            all_planes.extend(req.planes_list)
            offset += n

        if not all_planes:
            continue

        n_total = len(all_planes)
        all_logits = np.zeros((n_total, POLICY_SIZE), dtype=np.float32)
        all_policies = np.ones((n_total, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE
        all_values = np.zeros(n_total, dtype=np.float32)

        if HAS_TORCH and network is not None:
            # Partition by network_id for separate forward passes
            net_ids_used = set(r[3] for r in request_slices)
            for net_id in sorted(net_ids_used):
                net = networks[net_id] if net_id < len(networks) else network
                # Gather indices for this network
                indices = []
                for _, start, end, nid in request_slices:
                    if nid == net_id:
                        indices.extend(range(start, end))
                if not indices:
                    continue
                planes_batch = np.stack([all_planes[i] for i in indices])
                x = torch.from_numpy(planes_batch).to(device)
                with torch.no_grad():
                    logits, values = net(x)
                logits_np = logits.cpu().numpy()
                policies_np = torch.softmax(logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()
                for j, idx in enumerate(indices):
                    all_logits[idx] = logits_np[j]
                    all_policies[idx] = policies_np[j]
                    all_values[idx] = values_np[j]

        # Distribute results back to workers
        for worker_id, start, end, _ in request_slices:
            resp = BatchResponse(
                policies=all_policies[start:end],
                values=[float(v) for v in all_values[start:end]],
                logits=all_logits[start:end],
            )
            response_queues[worker_id].put(resp)

        total_batches += 1
        total_evals += len(all_planes)

        # Periodic progress logging
        now = time.monotonic()
        if now - last_log_time >= log_interval:
            elapsed = now - start_time
            rate = total_evals / elapsed if elapsed > 0 else 0
            partial = progress.summary()
            partial_suffix = f", {partial}" if partial else ""
            logger.info(f"GPU server: {total_batches} batches, "
                        f"{total_evals} evals ({rate:.0f}/s), "
                        f"{progress.games_completed}/{num_games or '?'} games done"
                        f"{partial_suffix}, "
                        f"{elapsed:.0f}s elapsed")
            last_log_time = now

    elapsed = time.monotonic() - start_time
    rate = total_evals / elapsed if elapsed > 0 else 0
    logger.info(f"GPU server: all workers done — {total_batches} batches, "
                f"{total_evals} evals ({rate:.0f}/s) in {elapsed:.0f}s")
