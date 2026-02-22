"""MCTS worker process for multiprocess self-play.

Each worker runs MCTS tree traversal on a subset of games, sending leaf
evaluation requests to the GPU server via IPC queues.

Supports both standard MCTS and Gumbel MCTS modes.
"""

from __future__ import annotations

import os

# Must be set before any torch import to prevent CUDA context in workers
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import numpy as np
from typing import Optional

from davechess.engine.network import POLICY_SIZE, state_to_planes
from davechess.engine.mcts import MCTS, MCTSNode
from davechess.engine.selfplay import (
    _ActiveGame, _play_wave, _finalize_game,
)
from davechess.engine.gpu_server import BatchRequest, BatchResponse, WorkerDone

logger = logging.getLogger("davechess.mcts_worker")


class RemoteBatchedEvaluator:
    """Drop-in replacement for BatchedEvaluator that routes through IPC.

    Same interface as BatchedEvaluator (submit/evaluate_batch/pending_count)
    so MCTS.batched_search() works without modification.
    """

    def __init__(self, worker_id: int, request_queue, response_queue,
                 use_network: bool = True):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.use_network = use_network
        self._pending: list[tuple[MCTSNode, np.ndarray]] = []

    def submit(self, node: MCTSNode, planes: np.ndarray):
        """Queue a leaf node for evaluation."""
        self._pending.append((node, planes))

    def evaluate_batch(self) -> list[tuple[np.ndarray, float]]:
        """Send pending leaves to GPU server, block until results arrive."""
        if not self._pending:
            return []

        if not self.use_network:
            # Random opponent: uniform policy, zero value (no GPU needed)
            results = [(np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE, 0.0)
                       for _ in self._pending]
            self._pending.clear()
            return results

        planes_list = [p for _, p in self._pending]
        self.request_queue.put(BatchRequest(
            worker_id=self.worker_id,
            planes_list=planes_list,
        ))

        resp: BatchResponse = self.response_queue.get(timeout=60)

        results = [(resp.policies[i], float(resp.values[i]))
                   for i in range(len(self._pending))]
        self._pending.clear()
        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def _gumbel_remote_evaluator(worker_id, request_queue, response_queue):
    """Create a batch evaluator callable for GumbelBatchedSearch.

    Returns a function: list[GameState] -> list[(policy, logits, value)]
    that routes through the GPU server via IPC queues.
    """
    def evaluate(states):
        if not states:
            return []

        planes_list = [state_to_planes(s) for s in states]
        request_queue.put(BatchRequest(
            worker_id=worker_id,
            planes_list=planes_list,
        ))

        resp: BatchResponse = response_queue.get(timeout=60)

        results = []
        for i in range(len(states)):
            logits = resp.logits[i]
            policy = np.exp(logits - logits.max())
            policy = policy / policy.sum()
            results.append((policy, logits, float(resp.values[i])))
        return results

    return evaluate


def worker_entry(worker_id: int, request_queue, response_queue, results_queue,
                 game_assignments: list[dict], mcts_config: dict):
    """Entry point for a worker process.

    Args:
        worker_id: Unique worker identifier.
        request_queue: Shared queue for sending BatchRequest to GPU server.
        response_queue: Dedicated queue for receiving BatchResponse from GPU server.
        results_queue: Queue for sending completed game data back to main process.
        game_assignments: List of dicts, each with keys:
            - game_idx: Global game index
            - is_random: Whether this is a random-opponent game
            - nn_plays_white: Whether NN plays white in random games
        mcts_config: Dict with MCTS parameters (num_simulations, cpuct,
            dirichlet_alpha, dirichlet_epsilon, temperature_threshold).
            If gumbel_config key is present, uses Gumbel MCTS.
    """
    try:
        gumbel_config = mcts_config.get("gumbel_config")
        temperature_threshold = mcts_config["temperature_threshold"]
        draw_value_target = mcts_config.get("draw_value_target", 0.0)

        # Set up Gumbel or standard MCTS
        gumbel_search = None
        evaluator = None
        nn_mcts = None

        if gumbel_config:
            from davechess.engine.gumbel_mcts import GumbelBatchedSearch

            remote_eval = _gumbel_remote_evaluator(
                worker_id, request_queue, response_queue)

            gumbel_search = GumbelBatchedSearch(
                network=None,
                num_simulations=mcts_config["num_simulations"],
                max_num_considered_actions=gumbel_config.get(
                    "max_num_considered_actions", 16),
                cpuct=mcts_config.get("cpuct", 1.5),
                gumbel_scale=gumbel_config.get("gumbel_scale", 1.0),
                maxvisit_init=gumbel_config.get("maxvisit_init", 50.0),
                value_scale=gumbel_config.get("value_scale", 0.1),
                device="cpu",
                evaluator=remote_eval,
            )
            # Standard MCTS + evaluator for vs-random games (Gumbel only for training)
            evaluator = RemoteBatchedEvaluator(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
                use_network=True,
            )
            nn_mcts = MCTS(
                network=None,
                num_simulations=mcts_config["num_simulations"],
                cpuct=mcts_config.get("cpuct", 1.5),
                dirichlet_alpha=mcts_config.get("dirichlet_alpha", 0.3),
                dirichlet_epsilon=mcts_config.get("dirichlet_epsilon", 0.25),
                device="cpu",
            )
        else:
            evaluator = RemoteBatchedEvaluator(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
                use_network=True,
            )
            nn_mcts = MCTS(
                network=None,
                num_simulations=mcts_config["num_simulations"],
                cpuct=mcts_config.get("cpuct", 1.5),
                dirichlet_alpha=mcts_config["dirichlet_alpha"],
                dirichlet_epsilon=mcts_config["dirichlet_epsilon"],
                device="cpu",
            )

        # Create per-sim-level random MCTS instances
        random_mcts_by_sims: dict[int, MCTS] = {}
        for g in game_assignments:
            if g["is_random"]:
                sims = g.get("random_sims", 25)
                if sims not in random_mcts_by_sims:
                    random_mcts_by_sims[sims] = MCTS(
                        None, num_simulations=sims, device="cpu",
                    )
        random_mcts = next(iter(random_mcts_by_sims.values()), None)

        wave_games: list[_ActiveGame] = []
        for g in game_assignments:
            if g["is_random"]:
                sims = g.get("random_sims", 25)
                opp = random_mcts_by_sims[sims]
            else:
                opp = None
            wave_games.append(_ActiveGame(
                game_idx=g["game_idx"],
                nn_engine=nn_mcts,
                opponent_engine=opp,
                nn_plays_white=g["nn_plays_white"],
                temperature_threshold=temperature_threshold,
                game_type="vs_random" if g["is_random"] else "selfplay",
            ))

        _play_wave(wave_games, nn_mcts, random_mcts, evaluator,
                   temperature_threshold, gumbel_search=gumbel_search)

        # Finalize games and send results
        worker_results = []
        for g in wave_games:
            training_data, game_record = _finalize_game(
                g, draw_value_target=draw_value_target
            )
            worker_results.append({
                "game_idx": g.game_idx,
                "game_type": g.game_type,
                "training_data": training_data,
                "game_record": game_record,
            })

        results_queue.put((worker_id, worker_results))

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        results_queue.put((worker_id, []))

    finally:
        request_queue.put(WorkerDone(worker_id))
