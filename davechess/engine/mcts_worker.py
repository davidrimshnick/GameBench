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
from davechess.engine.gpu_server import BatchRequest, BatchResponse, WorkerDone, GameCompleted

logger = logging.getLogger("davechess.mcts_worker")


class RemoteBatchedEvaluator:
    """Drop-in replacement for BatchedEvaluator that routes through IPC.

    Same interface as BatchedEvaluator (submit/evaluate_batch/pending_count)
    so MCTS.batched_search() works without modification.
    """

    def __init__(self, worker_id: int, request_queue, response_queue,
                 use_network: bool = True, network_id: int = 0):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.use_network = use_network
        self.network_id = network_id
        self._pending: list[tuple[MCTSNode, np.ndarray]] = []

    def submit(self, node: MCTSNode, planes: np.ndarray):
        """Queue a leaf node for evaluation."""
        self._pending.append((node, planes))

    def evaluate_batch(self) -> list[tuple[np.ndarray, float]]:
        """Send pending leaves to GPU server, block until results arrive."""
        if not self._pending:
            return []

        if not self.use_network:
            # Random opponent: uniform logits, zero value (no GPU needed)
            results = [(np.zeros(POLICY_SIZE, dtype=np.float32), 0.0)
                       for _ in self._pending]
            self._pending.clear()
            return results

        planes_list = [p for _, p in self._pending]
        self.request_queue.put(BatchRequest(
            worker_id=self.worker_id,
            planes_list=planes_list,
            network_id=self.network_id,
        ))

        resp: BatchResponse = self.response_queue.get(timeout=60)

        # Return raw logits so expand() can do legal-move-masked softmax
        logits_source = resp.logits if resp.logits is not None else resp.policies
        results = [(logits_source[i], float(resp.values[i]))
                   for i in range(len(self._pending))]
        self._pending.clear()
        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def _gumbel_remote_evaluator(worker_id, request_queue, response_queue,
                             network_id: int = 0):
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
            network_id=network_id,
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
        policy_target_smoothing = mcts_config.get("policy_target_smoothing", 0.0)

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
                backprop_value_scale=gumbel_config.get("backprop_value_scale", 1.0),
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
                value_scale=mcts_config.get("value_scale", 1.0),
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
                value_scale=mcts_config.get("value_scale", 1.0),
            )

        # Random opponent: same Gumbel MCTS config but with no network
        # (uniform policy, zero value). Tests whether learned network beats
        # random given identical search budget.
        random_opp = None
        has_random = any(g["is_random"] for g in game_assignments)
        if has_random:
            if gumbel_config:
                from davechess.engine.gumbel_mcts import GumbelMCTS as _GumbelMCTS
                random_opp = _GumbelMCTS(
                    network=None,
                    num_simulations=mcts_config["num_simulations"],
                    max_num_considered_actions=gumbel_config.get(
                        "max_num_considered_actions", 48),
                    cpuct=mcts_config.get("cpuct", 1.5),
                    gumbel_scale=gumbel_config.get("gumbel_scale", 1.0),
                    maxvisit_init=gumbel_config.get("maxvisit_init", 50.0),
                    value_scale=gumbel_config.get("value_scale", 0.1),
                    device="cpu",
                )
            else:
                random_opp = MCTS(
                    None, num_simulations=mcts_config["num_simulations"],
                    cpuct=mcts_config.get("cpuct", 1.5), device="cpu",
                )
        random_mcts = random_opp

        wave_games: list[_ActiveGame] = []
        for g in game_assignments:
            opp = random_opp if g["is_random"] else None
            wave_games.append(_ActiveGame(
                game_idx=g["game_idx"],
                nn_engine=nn_mcts,
                opponent_engine=opp,
                nn_plays_white=g["nn_plays_white"],
                temperature_threshold=temperature_threshold,
                game_type="vs_random" if g["is_random"] else "selfplay",
            ))

        def on_game_finished(g: _ActiveGame):
            request_queue.put(GameCompleted(
                worker_id=worker_id,
                game_idx=g.game_idx,
                game_type=g.game_type,
                length=g.move_count,
                winner=g.winner_str,
            ))

        _play_wave(wave_games, nn_mcts, random_mcts, evaluator,
                   temperature_threshold, gumbel_search=gumbel_search,
                   on_game_finished=on_game_finished)

        # Finalize games and send results
        worker_results = []
        for g in wave_games:
            training_data, game_record = _finalize_game(
                g, draw_value_target=draw_value_target,
                policy_target_smoothing=policy_target_smoothing,
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


def probe_worker_entry(worker_id: int, request_queue, response_queue, results_queue,
                       game_assignments: list[dict], mcts_config: dict):
    """Entry point for ELO probe worker. Plays current vs reference network.

    Each game assignment has current_is_white indicating which side uses
    network_id=0 (current) vs network_id=1 (reference). Games are played
    sequentially within each worker; parallelism comes from multiple workers
    sharing the GPU server.

    Uses standard MCTS with RemoteBatchedEvaluator (one per network).
    Each move: batched_search with a single state and the appropriate evaluator.
    """
    from davechess.game.state import GameState
    from davechess.game.rules import generate_legal_moves, apply_move
    from davechess.game.state import Player

    try:
        num_sims = mcts_config["num_simulations"]
        cpuct = mcts_config.get("cpuct", 4.0)
        max_moves = mcts_config.get("max_moves", 200)
        value_scale = mcts_config.get("value_scale", 1.0)

        # Two evaluators routing to different networks on GPU
        current_eval = RemoteBatchedEvaluator(
            worker_id, request_queue, response_queue, network_id=0)
        ref_eval = RemoteBatchedEvaluator(
            worker_id, request_queue, response_queue, network_id=1)

        # Standard MCTS engines (no Dirichlet noise, low temperature)
        current_mcts = MCTS(
            network=None, num_simulations=num_sims,
            cpuct=cpuct, temperature=0.1, device="cpu",
            value_scale=value_scale,
        )
        ref_mcts = MCTS(
            network=None, num_simulations=num_sims,
            cpuct=cpuct, temperature=0.1, device="cpu",
            value_scale=value_scale,
        )

        worker_results = []
        for g in game_assignments:
            game_idx = g["game_idx"]
            current_is_white = g["current_is_white"]
            state = GameState()
            move_count = 0

            while not state.done and move_count < max_moves:
                moves = generate_legal_moves(state)
                if not moves:
                    break

                is_current_turn = (
                    (state.current_player == Player.WHITE and current_is_white) or
                    (state.current_player == Player.BLACK and not current_is_white)
                )

                if is_current_turn:
                    mcts_engine = current_mcts
                    evaluator = current_eval
                else:
                    mcts_engine = ref_mcts
                    evaluator = ref_eval

                # batched_search with single state
                roots = MCTS.batched_search(
                    [mcts_engine], [state], evaluator, [False])
                move, _ = mcts_engine.get_move_from_root(roots[0], state)

                apply_move(state, move)
                move_count += 1

            # Determine winner from current's perspective
            if state.winner is not None:
                current_won = (
                    (state.winner == Player.WHITE and current_is_white) or
                    (state.winner == Player.BLACK and not current_is_white)
                )
                winner_str = "current" if current_won else "reference"
            else:
                winner_str = "draw"

            worker_results.append({
                "game_idx": game_idx,
                "current_is_white": current_is_white,
                "winner": winner_str,
                "length": move_count,
            })

            # Notify GPU server of completion
            request_queue.put(GameCompleted(
                worker_id=worker_id,
                game_idx=game_idx,
                game_type="probe",
                length=move_count,
                winner=winner_str,
            ))

        results_queue.put((worker_id, worker_results))

    except Exception as e:
        logger.error(f"Probe worker {worker_id} failed: {e}", exc_info=True)
        results_queue.put((worker_id, []))

    finally:
        request_queue.put(WorkerDone(worker_id))
