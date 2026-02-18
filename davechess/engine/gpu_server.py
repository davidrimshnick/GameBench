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


def run_gpu_server(network, device: str, request_queue, response_queues: list,
                   num_workers: int, workers: list = None,
                   drain_timeout_ms: float = 5.0):
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
    """
    workers_done = set()
    drain_timeout_sec = drain_timeout_ms / 1000.0

    if HAS_TORCH and network is not None:
        network.to(device)
        network.eval()

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
                continue
            pending.append(msg)

        # Build combined batch
        all_planes = []
        # Track which planes belong to which worker request
        request_slices: list[tuple[int, int, int]] = []  # (worker_id, start, end)
        offset = 0
        for req in pending:
            n = len(req.planes_list)
            request_slices.append((req.worker_id, offset, offset + n))
            all_planes.extend(req.planes_list)
            offset += n

        if not all_planes:
            continue

        # GPU forward pass
        if HAS_TORCH and network is not None:
            planes_batch = np.stack(all_planes)
            x = torch.from_numpy(planes_batch).to(device)
            with torch.no_grad():
                logits, values = network(x)
            all_logits = logits.cpu().numpy()
            all_policies = torch.softmax(logits, dim=1).cpu().numpy()
            all_values = values.cpu().numpy().flatten()
        else:
            n_total = len(all_planes)
            all_logits = np.zeros((n_total, POLICY_SIZE), dtype=np.float32)
            all_policies = np.ones((n_total, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE
            all_values = np.zeros(n_total, dtype=np.float32)

        # Distribute results back to workers
        for worker_id, start, end in request_slices:
            resp = BatchResponse(
                policies=all_policies[start:end],
                values=[float(v) for v in all_values[start:end]],
                logits=all_logits[start:end],
            )
            response_queues[worker_id].put(resp)

    logger.info("GPU server: all workers done")
