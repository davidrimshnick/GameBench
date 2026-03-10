"""Self-play game generation and replay buffer for AlphaZero training."""

from __future__ import annotations

import gc
import logging
import os
import json
import random
import numpy as np
from typing import Optional
from collections import deque

logger = logging.getLogger("davechess.selfplay")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.network import state_to_planes, move_to_policy_index, POLICY_SIZE, NUM_INPUT_PLANES
from davechess.engine.mcts import MCTS, BatchedEvaluator
from davechess.engine.gumbel_mcts import GumbelMCTS, GumbelBatchedSearch


def build_legal_move_mask(state: GameState,
                          legal_moves: Optional[list[Move]] = None) -> np.ndarray:
    """Build a dense legal-move mask aligned with the policy encoding."""
    if legal_moves is None:
        legal_moves = generate_legal_moves(state)
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    flip = state.current_player == Player.BLACK
    for move in legal_moves:
        mask[move_to_policy_index(move, flip=flip)] = True
    return mask


def _build_policy_target(policy_dict: dict, smoothing: float = 0.0) -> np.ndarray:
    """Build dense policy target from sparse visit proportions.

    Args:
        policy_dict: {policy_index: visit_proportion} from MCTS
        smoothing: Mix with uniform over legal moves. 0.0 = raw visits,
            0.5 = 50% visits + 50% uniform. Prevents policy from becoming
            too peaked, breaking the PUCT feedback loop where peaked priors
            concentrate visits which produce even more peaked targets.
    """
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    n_legal = len(policy_dict)
    if n_legal == 0:
        return policy
    for idx, prob in policy_dict.items():
        if smoothing > 0 and n_legal > 1:
            policy[idx] = (1.0 - smoothing) * prob + smoothing / n_legal
        else:
            policy[idx] = prob
    return policy


def classify_draw_reason(state: GameState) -> str:
    """Classify why a finished game ended in a draw."""
    if state.winner is not None:
        return "not_draw"
    if state.turn > 100:
        return "turn_limit"
    if state.halfmove_clock >= 100:
        return "fifty_move"
    if state.position_counts and max(state.position_counts.values()) >= 3:
        return "repetition"
    return "stalemate_or_other"


class ReplayBuffer:
    """Circular replay buffer for training data.

    Stores (planes, policy_target, value_target, legal_mask) tuples.
    """

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.planes: deque[np.ndarray] = deque(maxlen=max_size)
        self.policies: deque[np.ndarray] = deque(maxlen=max_size)
        self.values: deque[float] = deque(maxlen=max_size)
        self.legal_masks: deque[np.ndarray] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.planes)

    def push(self, planes: np.ndarray, policy: np.ndarray, value: float,
             legal_mask: Optional[np.ndarray] = None):
        """Add a single training example."""
        self.planes.append(planes.astype(np.float32, copy=False))
        self.policies.append(policy.astype(np.float32, copy=False))
        self.values.append(value)
        if legal_mask is None:
            # Backward compatibility for older callers / buffers.
            legal_mask = policy > 0
        self.legal_masks.append(legal_mask.astype(np.bool_, copy=False))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch.

        Returns:
            (planes_batch, policy_batch, value_batch, legal_mask_batch)
        """
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        planes = np.stack([self.planes[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        legal_masks = np.stack([self.legal_masks[i] for i in indices])
        return planes, policies, values, legal_masks

    def save(self, path: str):
        """Save buffer metadata (not full data, for memory efficiency)."""
        meta = {"size": len(self), "max_size": self.max_size}
        with open(path, "w") as f:
            json.dump(meta, f)

    def save_data(self, path: str, chunk_size: int = 5000):
        """Save full buffer data to disk using chunked writes to limit peak memory.

        Writes arrays in chunks to avoid allocating the full buffer as a single
        contiguous array. Peak temp memory is ~chunk_size worth of data (~85MB
        for 5K positions) instead of the full buffer (~1GB+ at 60K positions).
        """
        import tempfile
        import zipfile

        n = len(self)
        if n == 0:
            np.savez_compressed(path, planes=np.empty((0, NUM_INPUT_PLANES, 8, 8)),
                                policies=np.empty((0, POLICY_SIZE)),
                                values=np.empty(0),
                                legal_masks=np.empty((0, POLICY_SIZE), dtype=np.bool_))
            return

        with tempfile.TemporaryDirectory() as tmp:
            # Write each array in chunks using memory-mapped files
            planes_path = os.path.join(tmp, "planes.npy")
            planes_mmap = np.lib.format.open_memmap(
                planes_path, mode="w+", dtype=np.float32, shape=(n, NUM_INPUT_PLANES, 8, 8))
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = np.stack([self.planes[i] for i in range(start, end)])
                planes_mmap[start:end] = chunk
            del planes_mmap, chunk
            gc.collect()

            policies_path = os.path.join(tmp, "policies.npy")
            policies_mmap = np.lib.format.open_memmap(
                policies_path, mode="w+", dtype=np.float32, shape=(n, POLICY_SIZE))
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = np.stack([self.policies[i] for i in range(start, end)])
                policies_mmap[start:end] = chunk
            del policies_mmap, chunk
            gc.collect()

            values_path = os.path.join(tmp, "values.npy")
            arr = np.array(list(self.values), dtype=np.float32)
            np.save(values_path, arr)
            del arr
            gc.collect()

            legal_masks_path = os.path.join(tmp, "legal_masks.npy")
            legal_masks_mmap = np.lib.format.open_memmap(
                legal_masks_path, mode="w+", dtype=np.bool_, shape=(n, POLICY_SIZE))
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = np.stack([self.legal_masks[i] for i in range(start, end)])
                legal_masks_mmap[start:end] = chunk
            del legal_masks_mmap, chunk
            gc.collect()

            # Combine into compressed npz
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(planes_path, "planes.npy")
                zf.write(policies_path, "policies.npy")
                zf.write(values_path, "values.npy")
                zf.write(legal_masks_path, "legal_masks.npy")

    def load_data(self, path: str, chunk_size: int = 5000):
        """Load buffer data from disk using chunked reads to limit peak memory.

        Processes each array in chunks, deleting consumed slices to keep peak
        memory at ~chunk_size worth of data rather than the full array.
        """
        data = np.load(path)

        # Values are small — load all at once
        values_arr = data["values"]
        n = len(values_arr)
        for i in range(n):
            self.values.append(float(values_arr[i]))
        del values_arr
        gc.collect()

        # Planes: load full array then consume in chunks to free memory gradually
        planes_arr = data["planes"]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = planes_arr[start:end]
            for i in range(len(chunk)):
                self.planes.append(chunk[i].astype(np.float32, copy=False))
            del chunk
        del planes_arr
        gc.collect()

        # Policies: same chunked approach
        policies_arr = data["policies"]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = policies_arr[start:end]
            for i in range(len(chunk)):
                self.policies.append(chunk[i].astype(np.float32, copy=False))
            del chunk
        del policies_arr

        legal_masks_arr = data["legal_masks"] if "legal_masks" in data else None
        if legal_masks_arr is not None:
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = legal_masks_arr[start:end]
                for i in range(len(chunk)):
                    self.legal_masks.append(chunk[i].astype(np.bool_, copy=False))
                del chunk
            del legal_masks_arr
        else:
            for policy in self.policies:
                self.legal_masks.append((policy > 0).astype(np.bool_, copy=False))
        del data
        gc.collect()


class StructuredReplayBuffer:
    """Replay buffer with three managed partitions: seeds, decisive, draws.

    Seeds are permanent (never evicted). Decisive and draw positions use
    separate circular buffers so draw-heavy self-play can't displace
    tactical/checkmate signal.

    Default partition sizes: 20K seeds + 20K decisive + 10K draws = 50K total.
    """

    def __init__(self, seed_size: int = 20_000, decisive_size: int = 20_000,
                 draw_size: int = 10_000):
        self.seed_size = seed_size
        self.decisive_size = decisive_size
        self.draw_size = draw_size

        self._seeds = ReplayBuffer(max_size=seed_size)
        self._decisive = ReplayBuffer(max_size=decisive_size)
        self._draws = ReplayBuffer(max_size=draw_size)

    def __len__(self) -> int:
        return len(self._seeds) + len(self._decisive) + len(self._draws)

    @property
    def max_size(self) -> int:
        return self.seed_size + self.decisive_size + self.draw_size

    def push_seed(self, planes: np.ndarray, policy: np.ndarray, value: float,
                  legal_mask: Optional[np.ndarray] = None):
        """Add a seed position (permanent, never evicted by self-play)."""
        self._seeds.push(planes, policy, value, legal_mask=legal_mask)

    def push(self, planes: np.ndarray, policy: np.ndarray, value: float,
             legal_mask: Optional[np.ndarray] = None):
        """Add a self-play position, routed by value magnitude."""
        if abs(value) > 0.5:
            self._decisive.push(planes, policy, value, legal_mask=legal_mask)
        else:
            self._draws.push(planes, policy, value, legal_mask=legal_mask)

    def resize(self, decisive_size: int, draw_size: int):
        """Resize decisive and draw partitions, keeping most recent data."""
        if decisive_size != self.decisive_size:
            old = self._decisive
            self.decisive_size = decisive_size
            self._decisive = ReplayBuffer(max_size=decisive_size)
            # Copy most recent data (deque keeps oldest at front)
            start = max(0, len(old) - decisive_size)
            for i in range(start, len(old)):
                self._decisive.push(
                    old.planes[i], old.policies[i], old.values[i],
                    legal_mask=old.legal_masks[i],
                )
            del old
        if draw_size != self.draw_size:
            old = self._draws
            self.draw_size = draw_size
            self._draws = ReplayBuffer(max_size=draw_size)
            start = max(0, len(old) - draw_size)
            for i in range(start, len(old)):
                self._draws.push(
                    old.planes[i], old.policies[i], old.values[i],
                    legal_mask=old.legal_masks[i],
                )
            del old
        gc.collect()

    def clear_seeds(self):
        """Remove all seed positions, freeing the partition for self-play data."""
        n = len(self._seeds)
        self._seeds = ReplayBuffer(max_size=self.seed_size)
        gc.collect()
        return n

    def sample(self, batch_size: int, seed_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Weighted random sample across all three partitions.

        Args:
            batch_size: Number of samples to return.
            seed_weight: Weight for seed partition relative to self-play partitions.
                1.0 = seeds sampled proportionally to their size (uniform).
                0.5 = seeds get half the samples they'd get under uniform.
                0.0 = no seed samples (only decisive + draws).
        """
        total = len(self)
        if total == 0:
            return (np.empty((0, NUM_INPUT_PLANES, 8, 8), dtype=np.float32),
                    np.empty((0, POLICY_SIZE), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty((0, POLICY_SIZE), dtype=np.bool_))

        n = min(batch_size, total)
        len_s = len(self._seeds)
        len_d = len(self._decisive)
        len_dr = len(self._draws)

        # Compute per-partition sample counts based on seed_weight
        seed_weight = max(0.0, min(1.0, seed_weight))
        if len_s > 0 and (len_d + len_dr) > 0 and seed_weight < 1.0:
            # Weighted allocation: seeds get reduced share, rest gets proportionally more
            w_s = len_s * seed_weight
            w_d = float(len_d)
            w_dr = float(len_dr)
            w_total = w_s + w_d + w_dr
            n_s = min(int(round(n * w_s / w_total)), len_s)
            n_d = min(int(round(n * w_d / w_total)), len_d)
            n_dr = min(n - n_s - n_d, len_dr)
        else:
            # Uniform fallback (seed_weight=1.0 or only one partition has data)
            n_s = min(int(round(n * len_s / total)), len_s)
            n_d = min(int(round(n * len_d / total)), len_d)
            n_dr = min(n - n_s - n_d, len_dr)

        # Sample from each partition independently
        planes_list = []
        policies_list = []
        values_list = []
        legal_masks_list = []
        for buf, count in [(self._seeds, n_s), (self._decisive, n_d), (self._draws, n_dr)]:
            if count <= 0 or len(buf) == 0:
                continue
            idxs = random.sample(range(len(buf)), count)
            for i in idxs:
                planes_list.append(buf.planes[i])
                policies_list.append(buf.policies[i])
                values_list.append(buf.values[i])
                legal_masks_list.append(buf.legal_masks[i])

        if not planes_list:
            return (np.empty((0, NUM_INPUT_PLANES, 8, 8), dtype=np.float32),
                    np.empty((0, POLICY_SIZE), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty((0, POLICY_SIZE), dtype=np.bool_))

        return (np.stack(planes_list),
                np.stack(policies_list),
                np.array(values_list, dtype=np.float32),
                np.stack(legal_masks_list))

    def partition_sizes(self) -> dict:
        """Return sizes of each partition."""
        return {
            "seeds": len(self._seeds),
            "decisive": len(self._decisive),
            "draws": len(self._draws),
        }

    def save(self, path: str):
        """Save buffer metadata."""
        meta = {
            "size": len(self),
            "max_size": self.max_size,
            "partitions": self.partition_sizes(),
        }
        with open(path, "w") as f:
            json.dump(meta, f)

    def save_data(self, path: str, chunk_size: int = 5000):
        """Save all partitions to a single compressed npz file."""
        import tempfile
        import zipfile

        total = len(self)
        if total == 0:
            np.savez_compressed(path,
                                planes=np.empty((0, NUM_INPUT_PLANES, 8, 8)),
                                policies=np.empty((0, POLICY_SIZE)),
                                values=np.empty(0),
                                legal_masks=np.empty((0, POLICY_SIZE), dtype=np.bool_),
                                partition_offsets=np.array([0, 0, 0, 0]))
            return

        len_s = len(self._seeds)
        len_d = len(self._decisive)
        offsets = np.array([0, len_s, len_s + len_d, total], dtype=np.int64)

        all_bufs = [self._seeds, self._decisive, self._draws]

        with tempfile.TemporaryDirectory() as tmp:
            planes_path = os.path.join(tmp, "planes.npy")
            planes_mmap = np.lib.format.open_memmap(
                planes_path, mode="w+", dtype=np.float32, shape=(total, NUM_INPUT_PLANES, 8, 8))
            pos = 0
            for buf in all_bufs:
                n = len(buf)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = np.stack([buf.planes[i] for i in range(start, end)])
                    planes_mmap[pos + start:pos + end] = chunk
                    del chunk
                pos += n
            del planes_mmap
            gc.collect()

            policies_path = os.path.join(tmp, "policies.npy")
            policies_mmap = np.lib.format.open_memmap(
                policies_path, mode="w+", dtype=np.float32, shape=(total, POLICY_SIZE))
            pos = 0
            for buf in all_bufs:
                n = len(buf)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = np.stack([buf.policies[i] for i in range(start, end)])
                    policies_mmap[pos + start:pos + end] = chunk
                    del chunk
                pos += n
            del policies_mmap
            gc.collect()

            values_list = []
            for buf in all_bufs:
                values_list.extend(buf.values)
            values_arr = np.array(values_list, dtype=np.float32)
            values_path = os.path.join(tmp, "values.npy")
            np.save(values_path, values_arr)
            del values_list, values_arr
            gc.collect()

            legal_masks_path = os.path.join(tmp, "legal_masks.npy")
            legal_masks_mmap = np.lib.format.open_memmap(
                legal_masks_path, mode="w+", dtype=np.bool_, shape=(total, POLICY_SIZE))
            pos = 0
            for buf in all_bufs:
                n = len(buf)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = np.stack([buf.legal_masks[i] for i in range(start, end)])
                    legal_masks_mmap[pos + start:pos + end] = chunk
                    del chunk
                pos += n
            del legal_masks_mmap
            gc.collect()

            offsets_path = os.path.join(tmp, "partition_offsets.npy")
            np.save(offsets_path, offsets)

            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(planes_path, "planes.npy")
                zf.write(policies_path, "policies.npy")
                zf.write(values_path, "values.npy")
                zf.write(legal_masks_path, "legal_masks.npy")
                zf.write(offsets_path, "partition_offsets.npy")

    def load_data(self, path: str, chunk_size: int = 5000):
        """Load partitioned buffer from disk.

        Handles both new format (with partition_offsets) and old flat format.
        """
        data = np.load(path)

        values_arr = data["values"]
        planes_arr = data["planes"]
        policies_arr = data["policies"]
        legal_masks_arr = data["legal_masks"] if "legal_masks" in data else None
        total = len(values_arr)

        if "partition_offsets" in data:
            offsets = data["partition_offsets"]
            ranges = [
                (0, int(offsets[1]), self._seeds),
                (int(offsets[1]), int(offsets[2]), self._decisive),
                (int(offsets[2]), int(offsets[3]), self._draws),
            ]
        else:
            # Legacy flat format — route by value magnitude
            ranges = None

        if ranges is not None:
            for start_off, end_off, buf in ranges:
                for i in range(start_off, end_off):
                    buf.push(
                        planes_arr[i].astype(np.float32, copy=False),
                        policies_arr[i].astype(np.float32, copy=False),
                        float(values_arr[i]),
                        legal_mask=(
                            legal_masks_arr[i].astype(np.bool_, copy=False)
                            if legal_masks_arr is not None else None
                        ),
                    )
        else:
            for i in range(total):
                v = float(values_arr[i])
                p = planes_arr[i].astype(np.float32, copy=False)
                pol = policies_arr[i].astype(np.float32, copy=False)
                legal_mask = (
                    legal_masks_arr[i].astype(np.bool_, copy=False)
                    if legal_masks_arr is not None else None
                )
                if abs(v) > 0.5:
                    self._decisive.push(p, pol, v, legal_mask=legal_mask)
                else:
                    self._draws.push(p, pol, v, legal_mask=legal_mask)

        del values_arr, planes_arr, policies_arr, data
        if legal_masks_arr is not None:
            del legal_masks_arr
        gc.collect()


def play_selfplay_game(mcts_engine: MCTS,
                       temperature_threshold: int = 30,
                       opponent_mcts: MCTS = None,
                       nn_plays_white: bool = True,
                       draw_value_target: float = 0.0) -> tuple[list, dict]:
    """Play one self-play game and return training examples + game record.

    Args:
        mcts_engine: Primary MCTS engine (with neural network).
        temperature_threshold: Move number after which temperature drops.
        opponent_mcts: Optional second MCTS engine (e.g. random/no-NN).
            When provided, mcts_engine plays one side and opponent_mcts
            plays the other. Training examples are only collected from
            the mcts_engine's turns.
        nn_plays_white: When opponent_mcts is set, which side the NN plays.
        draw_value_target: Value target assigned to drawn positions.

    Returns:
        (training_data, game_record) where:
        - training_data: list of (planes, policy_target, value_target) tuples
          from NN turns in the game
        - game_record: dict with "moves" (list of (state, move) pairs),
          "winner" ("white"/"black"/"draw"), "length" (int),
          and "draw_reason" (or None for decisive games)
    """
    state = GameState()
    examples: list[tuple[np.ndarray, dict, int, np.ndarray]] = []  # (planes, policy_dict, player, legal_mask)
    game_moves: list[tuple[GameState, Move]] = []  # For DCN logging
    move_count = 0
    nn_player = Player.WHITE if nn_plays_white else Player.BLACK

    while not state.done:
        moves = generate_legal_moves(state)
        if not moves:
            break

        # Choose which engine plays this move
        if opponent_mcts is not None and state.current_player != nn_player:
            # Opponent's turn — use opponent engine, no training data
            engine = opponent_mcts
            is_nn_turn = False
        else:
            engine = mcts_engine
            is_nn_turn = True

        # Adjust temperature
        if move_count < temperature_threshold:
            engine.temperature = 1.0
        else:
            engine.temperature = 0.1  # Near-greedy

        # Noise only for self-play exploration (both sides); no noise in vs-random
        use_noise = (opponent_mcts is None)
        move, info = engine.get_move(state, add_noise=use_noise)

        # Only record training examples from the NN engine's turns
        if is_nn_turn:
            planes = state_to_planes(state)
            legal_mask = build_legal_move_mask(state, legal_moves=moves)
            examples.append((planes, info["policy_target"], int(state.current_player), legal_mask))

        # Record move for game log (apply_move mutates in-place, so clone first)
        game_moves.append((state.clone(), move))

        state = apply_move(state, move)
        move_count += 1

    # Determine game outcome
    if state.winner is not None:
        winner = int(state.winner)
        winner_str = "white" if state.winner == Player.WHITE else "black"
        draw_reason = None
    else:
        winner = -1
        winner_str = "draw"
        draw_reason = classify_draw_reason(state)

    game_record = {
        "moves": game_moves,
        "winner": winner_str,
        "length": move_count,
        "draw_reason": draw_reason,
    }

    # Assign value targets based on outcome
    training_data = []
    for planes, policy_dict, player, legal_mask in examples:
        policy = _build_policy_target(policy_dict)

        # Value from this player's perspective: +1 win, -1 loss, draw_value_target draw
        if winner == -1:
            value = draw_value_target
        elif winner == player:
            value = 1.0
        else:
            value = -1.0

        training_data.append((planes, policy, value, legal_mask))

    return training_data, game_record


def run_selfplay_batch(network, num_games: int, num_simulations: int = 200,
                       cpuct: float = 1.5,
                       temperature_threshold: int = 30,
                       dirichlet_alpha: float = 0.3,
                       dirichlet_epsilon: float = 0.25,
                       random_opponent_fraction: float = 0.0,
                       draw_value_target: float = 0.0,
                       device: str = "cpu",
                       policy_target_smoothing: float = 0.0,
                       value_scale: float = 1.0) -> tuple[list, dict]:
    """Run a batch of self-play games sequentially.

    Args:
        network: The neural network for evaluation.
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        cpuct: PUCT exploration constant.
        temperature_threshold: Move number after which temperature drops.
        dirichlet_alpha: Dirichlet noise concentration parameter.
        dirichlet_epsilon: Fraction of Dirichlet noise to blend with policy.
        random_opponent_fraction: Fraction of games played vs random MCTS
            (no neural network) to prevent self-play overfitting.
        draw_value_target: Value target assigned to drawn positions.
        device: Torch device string.
        value_scale: Scale factor for NN value predictions in MCTS (0-1).

    Returns:
        (all_examples, stats) where stats has game-level statistics.
    """
    all_examples = []
    mcts = MCTS(network, num_simulations=num_simulations,
                cpuct=cpuct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                device=device,
                value_scale=value_scale)
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    game_details = []
    draw_reason_counts = {
        "turn_limit": 0,
        "fifty_move": 0,
        "repetition": 0,
        "stalemate_or_other": 0,
    }

    game_records = []

    # Random opponent: same search budget, no network (uniform policy, zero value)
    num_random_games = int(num_games * random_opponent_fraction)
    random_opp = None
    if num_random_games > 0:
        random_opp = MCTS(None, num_simulations=num_simulations, cpuct=cpuct, device=device)

    for game_idx in range(num_games):
        if game_idx < num_random_games:
            nn_plays_white = (game_idx % 2 == 0)
            examples, game_record = play_selfplay_game(
                mcts, temperature_threshold,
                opponent_mcts=random_opp,
                nn_plays_white=nn_plays_white,
                draw_value_target=draw_value_target,
            )
            game_type = "vs_random"
        else:
            # Standard self-play
            examples, game_record = play_selfplay_game(
                mcts, temperature_threshold, draw_value_target=draw_value_target
            )
            game_type = "selfplay"

        game_len = game_record["length"]
        # Include all game types in buffer — vs-random provides decisive
        # outcomes that bootstrap value learning from random init.
        all_examples.extend(examples)
        game_lengths.append(game_len)

        winner = game_record["winner"]
        if winner == "white":
            white_wins += 1
        elif winner == "black":
            black_wins += 1
        else:
            draws += 1
            draw_reason = game_record.get("draw_reason", "stalemate_or_other")
            draw_reason_counts[draw_reason] = draw_reason_counts.get(draw_reason, 0) + 1

        game_details.append({"game": game_idx + 1, "length": game_record["length"],
                             "winner": winner, "type": game_type,
                             "draw_reason": game_record.get("draw_reason")})
        game_records.append(game_record)

        logger.info(f"  Self-play game {game_idx+1}/{num_games} ({game_type}): "
                    f"{game_len} moves, {len(all_examples)} total positions")
        gc.collect()  # Free MCTS tree circular references

    stats = {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_pct": white_wins / max(white_wins + black_wins, 1),
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "min_game_length": min(game_lengths) if game_lengths else 0,
        "max_game_length": max(game_lengths) if game_lengths else 0,
        "game_details": game_details,
        "game_records": game_records,
        "num_random_games": num_random_games,
        "draw_reason_counts": draw_reason_counts,
    }

    return all_examples, stats


class _ActiveGame:
    """Tracks state for one game in the parallel self-play batch."""
    __slots__ = [
        "game_idx", "state", "nn_engine", "opponent_engine",
        "nn_plays_white", "move_count", "temperature_threshold",
        "examples", "game_moves", "game_type", "finished", "winner_str",
        "draw_reason",
    ]

    def __init__(self, game_idx: int, nn_engine: MCTS,
                 opponent_engine: Optional[MCTS],
                 nn_plays_white: bool, temperature_threshold: int,
                 game_type: str):
        self.game_idx = game_idx
        self.state = GameState()
        self.nn_engine = nn_engine
        self.opponent_engine = opponent_engine
        self.nn_plays_white = nn_plays_white
        self.move_count = 0
        self.temperature_threshold = temperature_threshold
        self.examples: list[tuple[np.ndarray, dict, int]] = []
        self.game_moves: list[tuple[GameState, Move]] = []
        self.game_type = game_type
        self.finished = False
        self.winner_str = "draw"
        self.draw_reason = None


def _record_winner(g: _ActiveGame):
    """Set the winner string on a finished game."""
    if g.state.winner is not None:
        g.winner_str = "white" if g.state.winner == Player.WHITE else "black"
        g.draw_reason = None
    else:
        g.winner_str = "draw"
        g.draw_reason = classify_draw_reason(g.state)


def _finalize_game(g: _ActiveGame,
                   draw_value_target: float = 0.0,
                   policy_target_smoothing: float = 0.0) -> tuple[list, dict]:
    """Convert a finished game into training data and game record.

    Returns (training_data, game_record) in same format as play_selfplay_game().
    """
    winner_str = g.winner_str
    if winner_str == "white":
        winner = int(Player.WHITE)
    elif winner_str == "black":
        winner = int(Player.BLACK)
    else:
        winner = -1

    training_data = []
    for planes, policy_dict, player, legal_mask in g.examples:
        policy = _build_policy_target(policy_dict, smoothing=policy_target_smoothing)
        if winner == -1:
            value = draw_value_target
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((planes, policy, value, legal_mask))

    game_record = {"moves": g.game_moves, "winner": winner_str,
                   "length": g.move_count, "draw_reason": g.draw_reason}
    return training_data, game_record


def _apply_game_move(g: _ActiveGame, move: Move, info: dict, is_nn_turn: bool,
                     on_game_finished=None):
    """Apply a move to an active game, collect training data if NN's turn."""
    if is_nn_turn:
        planes = state_to_planes(g.state)
        legal_mask = build_legal_move_mask(g.state)
        g.examples.append((planes, info["policy_target"], int(g.state.current_player), legal_mask))

    g.game_moves.append((g.state.clone(), move))
    g.state = apply_move(g.state, move)
    g.move_count += 1

    if g.state.done or not generate_legal_moves(g.state):
        g.finished = True
        _record_winner(g)
        if on_game_finished is not None:
            on_game_finished(g)


def _play_wave(wave_games: list[_ActiveGame], nn_mcts,
               random_mcts: Optional[MCTS], evaluator: BatchedEvaluator,
               temperature_threshold: int,
               gumbel_search: Optional[GumbelBatchedSearch] = None,
               on_game_finished=None):
    """Play a wave of games to completion with batched evaluation.

    At each step:
    1. Partition active games by engine type (NN vs random).
    2. NN games: Gumbel batched search (if gumbel_search provided) or
       standard MCTS.batched_search().
    3. Random games: sequential MCTS search (no NN needed).
    4. Select moves, update states, check for game end.
    """
    while True:
        nn_games: list[_ActiveGame] = []
        random_games: list[_ActiveGame] = []

        for g in wave_games:
            if g.finished:
                continue

            moves = generate_legal_moves(g.state)
            if not moves or g.state.done:
                g.finished = True
                _record_winner(g)
                if on_game_finished is not None:
                    on_game_finished(g)
                continue

            # Determine whose turn it is
            if g.opponent_engine is not None:
                nn_player = Player.WHITE if g.nn_plays_white else Player.BLACK
                is_nn_turn = (g.state.current_player == nn_player)
            else:
                is_nn_turn = True

            if is_nn_turn:
                nn_games.append(g)
            else:
                random_games.append(g)

        if not nn_games and not random_games:
            break

        # Batched search for NN games
        if nn_games:
            # Use Gumbel for all NN turns (self-play and vs-random alike)
            gumbel_games = []
            standard_games = []
            if gumbel_search is not None:
                gumbel_games = nn_games
            else:
                standard_games = nn_games

            if gumbel_games:
                # Gumbel MCTS — batched Sequential Halving (self-play only)
                states = [g.state for g in gumbel_games]
                temps = [1.0 if g.move_count < temperature_threshold else 0.1
                         for g in gumbel_games]
                noise_flags = [g.game_type != "vs_random" for g in gumbel_games]
                results = gumbel_search.batched_search(
                    states, temps, add_noise_flags=noise_flags
                )

                for g, (move, info) in zip(gumbel_games, results):
                    if move is not None:
                        _apply_game_move(
                            g, move, info, is_nn_turn=True,
                            on_game_finished=on_game_finished,
                        )
                    else:
                        g.finished = True
                        _record_winner(g)
                        if on_game_finished is not None:
                            on_game_finished(g)

            if standard_games:
                # Standard MCTS — batched PUCT search (vs-random + non-gumbel)
                engines: list[MCTS] = []
                for g in standard_games:
                    temp = 1.0 if g.move_count < temperature_threshold else 0.1
                    eng = MCTS(nn_mcts.network, num_simulations=nn_mcts.num_simulations,
                               cpuct=nn_mcts.cpuct, dirichlet_alpha=nn_mcts.dirichlet_alpha,
                               dirichlet_epsilon=nn_mcts.dirichlet_epsilon,
                               temperature=temp, device=nn_mcts.device,
                               value_scale=nn_mcts.value_scale)
                    engines.append(eng)

                states = [g.state for g in standard_games]
                noise_flags = [g.game_type != "vs_random" for g in standard_games]

                roots = MCTS.batched_search(engines, states, evaluator, noise_flags)

                for g, eng, root in zip(standard_games, engines, roots):
                    move, info = eng.get_move_from_root(root, g.state)
                    _apply_game_move(
                        g, move, info, is_nn_turn=True,
                        on_game_finished=on_game_finished,
                    )

        # Sequential MCTS for random-opponent games
        for g in random_games:
            temp = 1.0 if g.move_count < temperature_threshold else 0.1
            g.opponent_engine.temperature = temp
            move, info = g.opponent_engine.get_move(g.state, add_noise=False)
            _apply_game_move(
                g, move, info, is_nn_turn=False,
                on_game_finished=on_game_finished,
            )


def run_selfplay_batch_parallel(network, num_games: int, num_simulations: int = 200,
                                 cpuct: float = 1.5,
                                 temperature_threshold: int = 30,
                                 dirichlet_alpha: float = 0.3,
                                 dirichlet_epsilon: float = 0.25,
                                 random_opponent_fraction: float = 0.0,
                                 draw_value_target: float = 0.0,
                                 device: str = "cpu",
                                 parallel_games: int = 10,
                                 gumbel_config: Optional[dict] = None,
                                 policy_target_smoothing: float = 0.0,
                                 value_scale: float = 1.0) -> tuple[list, dict]:
    """Run self-play games with batched NN evaluation for GPU efficiency.

    Plays multiple games simultaneously, collecting leaf evaluations from
    all active MCTS searches into a single batched NN forward pass.

    Args:
        Same as run_selfplay_batch(), plus:
        parallel_games: Number of games to play simultaneously.
        gumbel_config: If provided, use Gumbel MCTS instead of standard MCTS.
            Keys: max_num_considered_actions, gumbel_scale, maxvisit_init, value_scale.

    Returns:
        (all_examples, stats) — same format as run_selfplay_batch().
    """
    all_examples = []
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    game_details = []
    game_records = []
    draw_reason_counts = {
        "turn_limit": 0,
        "fifty_move": 0,
        "repetition": 0,
        "stalemate_or_other": 0,
    }

    evaluator = BatchedEvaluator(network, device=device)

    # Create Gumbel batched search if configured
    gumbel_search: Optional[GumbelBatchedSearch] = None
    if gumbel_config is not None:
        gumbel_search = GumbelBatchedSearch(
            network=network,
            num_simulations=num_simulations,
            max_num_considered_actions=gumbel_config.get("max_num_considered_actions", 16),
            cpuct=cpuct,
            gumbel_scale=gumbel_config.get("gumbel_scale", 1.0),
            maxvisit_init=gumbel_config.get("maxvisit_init", 50.0),
            value_scale=gumbel_config.get("value_scale", 0.1),
            backprop_value_scale=gumbel_config.get("backprop_value_scale", 1.0),
            device=device,
        )
        logger.info(f"Using Gumbel MCTS (k={gumbel_search.max_num_considered_actions}, "
                     f"sims={num_simulations})")

    nn_mcts = MCTS(network, num_simulations=num_simulations,
                   cpuct=cpuct,
                   dirichlet_alpha=dirichlet_alpha,
                   dirichlet_epsilon=dirichlet_epsilon,
                   device=device,
                   value_scale=value_scale)
    num_random_games = int(num_games * random_opponent_fraction)
    # Random opponent: same search config but with no network (uniform policy, zero value)
    random_opp = None
    if num_random_games > 0:
        if gumbel_config is not None:
            random_opp = GumbelMCTS(
                network=None, num_simulations=num_simulations,
                max_num_considered_actions=gumbel_config.get("max_num_considered_actions", 48),
                cpuct=cpuct,
                gumbel_scale=gumbel_config.get("gumbel_scale", 1.0),
                maxvisit_init=gumbel_config.get("maxvisit_init", 50.0),
                value_scale=gumbel_config.get("value_scale", 0.1),
                device=device,
            )
        else:
            random_opp = MCTS(None, num_simulations=num_simulations, cpuct=cpuct, device=device)
    random_mcts = random_opp

    games_launched = 0
    while games_launched < num_games:
        wave_size = min(parallel_games, num_games - games_launched)
        wave_games: list[_ActiveGame] = []

        for i in range(wave_size):
            game_global_idx = games_launched + i
            is_random_game = game_global_idx < num_random_games
            nn_plays_white = (game_global_idx % 2 == 0)
            opp = random_opp if is_random_game else None

            wave_games.append(_ActiveGame(
                game_idx=game_global_idx,
                nn_engine=nn_mcts,
                opponent_engine=opp,
                nn_plays_white=nn_plays_white,
                temperature_threshold=temperature_threshold,
                game_type="vs_random" if is_random_game else "selfplay",
            ))

        _play_wave(wave_games, nn_mcts, random_mcts, evaluator,
                   temperature_threshold, gumbel_search=gumbel_search)

        for g in wave_games:
            training_data, game_record = _finalize_game(
                g, draw_value_target=draw_value_target,
                policy_target_smoothing=policy_target_smoothing,
            )
            # Include all game types — vs-random provides decisive outcomes
            # that bootstrap value learning from random init.
            all_examples.extend(training_data)
            game_lengths.append(game_record["length"])

            winner = game_record["winner"]
            if winner == "white":
                white_wins += 1
            elif winner == "black":
                black_wins += 1
            else:
                draws += 1
                draw_reason = game_record.get("draw_reason", "stalemate_or_other")
                draw_reason_counts[draw_reason] = draw_reason_counts.get(draw_reason, 0) + 1

            game_details.append({"game": g.game_idx + 1,
                                 "length": game_record["length"],
                                 "winner": winner, "type": g.game_type,
                                 "draw_reason": game_record.get("draw_reason")})
            game_records.append(game_record)

            logger.info(f"  Self-play game {g.game_idx+1}/{num_games} ({g.game_type}): "
                        f"{game_record['length']} moves, {len(all_examples)} total positions")

        games_launched += wave_size
        gc.collect()

    stats = {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_pct": white_wins / max(white_wins + black_wins, 1),
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "min_game_length": min(game_lengths) if game_lengths else 0,
        "max_game_length": max(game_lengths) if game_lengths else 0,
        "game_details": game_details,
        "game_records": game_records,
        "num_random_games": num_random_games,
        "draw_reason_counts": draw_reason_counts,
    }

    return all_examples, stats


def run_selfplay_multiprocess(network, num_games: int, num_simulations: int = 200,
                               cpuct: float = 1.5,
                               temperature_threshold: int = 30,
                               dirichlet_alpha: float = 0.3,
                               dirichlet_epsilon: float = 0.25,
                               random_opponent_fraction: float = 0.0,
                               draw_value_target: float = 0.0,
                               device: str = "cpu",
                               num_workers: int = 4,
                               gumbel_config: Optional[dict] = None,
                               policy_target_smoothing: float = 0.0,
                               value_scale: float = 1.0) -> tuple[list, dict]:
    """Run self-play with multiprocess CPU workers and centralized GPU inference.

    Each worker process runs MCTS tree traversal on a subset of games.
    The main process runs the GPU inference server, batching leaf evaluations
    from all workers into single forward passes.

    Supports both standard MCTS and Gumbel MCTS (when gumbel_config is provided).

    Args:
        Same as run_selfplay_batch_parallel(), plus:
        num_workers: Number of CPU worker processes.
        gumbel_config: If provided, workers use Gumbel MCTS instead of standard MCTS.

    Returns:
        (all_examples, stats) — same format as run_selfplay_batch().
    """
    import multiprocessing as mp
    from davechess.engine.gpu_server import run_gpu_server
    from davechess.engine.mcts_worker import worker_entry

    if gumbel_config is not None:
        logger.info(f"Multiprocess Gumbel MCTS (k={gumbel_config.get('max_num_considered_actions', 16)}, "
                     f"sims={num_simulations})")

    num_random_games = int(num_games * random_opponent_fraction)

    # Distribute games across workers
    game_assignments = _distribute_games(num_games, num_workers, num_random_games,
                                         nn_sims=num_simulations)

    mcts_config = {
        "num_simulations": num_simulations,
        "temperature_threshold": temperature_threshold,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_epsilon": dirichlet_epsilon,
        "draw_value_target": draw_value_target,
        "cpuct": cpuct,
        "policy_target_smoothing": policy_target_smoothing,
        "value_scale": value_scale,
    }
    if gumbel_config is not None:
        mcts_config["gumbel_config"] = gumbel_config

    # Create IPC queues
    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    results_queue = mp.Queue()

    # Spawn workers
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_entry,
            args=(wid, request_queue, response_queues[wid], results_queue,
                  game_assignments[wid], mcts_config),
            daemon=True,
        )
        p.start()
        workers.append(p)

    logger.info(f"Multiprocess self-play: {num_workers} workers, "
                f"{num_games} games ({num_random_games} vs random)")

    # Main process runs GPU inference server (blocks until all workers done)
    run_gpu_server(network, device, request_queue, response_queues,
                   num_workers, workers, num_games=num_games)

    # Collect results from all workers
    all_worker_results = {}
    for _ in range(num_workers):
        try:
            worker_id, worker_results = results_queue.get(timeout=30)
            all_worker_results[worker_id] = worker_results
        except Exception as e:
            logger.warning(f"Timeout waiting for worker results: {e}")

    missing_workers = sorted(set(range(num_workers)) - set(all_worker_results.keys()))
    if missing_workers:
        raise RuntimeError(
            "Multiprocess self-play missing worker results "
            f"from workers {missing_workers} "
            f"({len(all_worker_results)}/{num_workers} reported)"
        )

    # Wait for workers to exit
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} did not exit, terminating")
            p.terminate()

    # Aggregate results
    return _aggregate_multiprocess_results(all_worker_results, num_games,
                                            num_random_games)


def run_probe_multiprocess(current_network, reference_network,
                           num_games: int, num_simulations: int,
                           cpuct: float, device: str,
                           num_workers: int = 4,
                           max_moves: int = 200,
                           value_scale: float = 1.0) -> list[dict]:
    """Run ELO probe games via multiprocess workers with dual-network GPU server.

    Both networks are loaded on GPU. Workers run MCTS tree traversal on CPU,
    routing evaluation requests to the appropriate network via network_id.

    Returns list of game result dicts with keys:
        game_idx, current_is_white, winner ("current"/"reference"/"draw"), length.
    """
    import multiprocessing as mp
    from davechess.engine.gpu_server import run_gpu_server
    from davechess.engine.mcts_worker import probe_worker_entry

    num_workers = min(num_workers, num_games)

    # Distribute games across workers, alternating colors
    assignments: list[list[dict]] = [[] for _ in range(num_workers)]
    for i in range(num_games):
        assignments[i % num_workers].append({
            "game_idx": i,
            "current_is_white": (i % 2 == 0),
        })

    mcts_config = {
        "num_simulations": num_simulations,
        "cpuct": cpuct,
        "max_moves": max_moves,
        "value_scale": value_scale,
    }

    # Create IPC queues
    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    results_queue = mp.Queue()

    # Spawn workers
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=probe_worker_entry,
            args=(wid, request_queue, response_queues[wid], results_queue,
                  assignments[wid], mcts_config),
            daemon=True,
        )
        p.start()
        workers.append(p)

    logger.info(f"Multiprocess probe: {num_workers} workers, "
                f"{num_games} games, {num_simulations} sims")

    # Main process runs dual-network GPU server
    run_gpu_server(current_network, device, request_queue, response_queues,
                   num_workers, workers, num_games=num_games,
                   secondary_network=reference_network)

    # Collect results
    all_worker_results = {}
    for _ in range(num_workers):
        try:
            worker_id, worker_results = results_queue.get(timeout=30)
            all_worker_results[worker_id] = worker_results
        except Exception as e:
            logger.warning(f"Timeout waiting for probe worker results: {e}")

    missing = sorted(set(range(num_workers)) - set(all_worker_results.keys()))
    if missing:
        raise RuntimeError(
            f"Probe missing results from workers {missing} "
            f"({len(all_worker_results)}/{num_workers} reported)")

    # Wait for workers to exit
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Flatten and sort by game_idx
    all_results = []
    for results in all_worker_results.values():
        all_results.extend(results)
    all_results.sort(key=lambda r: r["game_idx"])

    if len(all_results) != num_games:
        raise RuntimeError(
            f"Probe returned {len(all_results)} games, expected {num_games}")

    return all_results


def _distribute_games(num_games: int, num_workers: int,
                      num_random_games: int,
                      nn_sims: int = 100) -> list[list[dict]]:
    """Distribute games evenly across workers.

    Returns list of game assignments per worker, each a list of dicts with
    game_idx, is_random, nn_plays_white.
    """
    assignments: list[list[dict]] = [[] for _ in range(num_workers)]

    for i in range(num_games):
        worker_id = i % num_workers
        is_random = i < num_random_games
        nn_plays_white = (i % 2 == 0)
        assignments[worker_id].append({
            "game_idx": i,
            "is_random": is_random,
            "nn_plays_white": nn_plays_white,
        })

    return assignments


def _aggregate_multiprocess_results(all_worker_results: dict, num_games: int,
                                     num_random_games: int) -> tuple[list, dict]:
    """Aggregate results from all workers into the standard output format."""
    all_examples = []
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    game_details = []
    game_records = []
    draw_reason_counts = {
        "turn_limit": 0,
        "fifty_move": 0,
        "repetition": 0,
        "stalemate_or_other": 0,
    }

    # Sort results by game_idx for deterministic ordering
    all_game_results = []
    for worker_id, results in all_worker_results.items():
        all_game_results.extend(results)
    all_game_results.sort(key=lambda r: r["game_idx"])

    if len(all_game_results) != num_games:
        raise RuntimeError(
            "Multiprocess self-play returned incomplete game results: "
            f"got {len(all_game_results)} games, expected {num_games}"
        )

    for r in all_game_results:
        # Include all game types — vs-random provides decisive outcomes
        # that bootstrap value learning from random init.
        all_examples.extend(r["training_data"])
        game_lengths.append(r["game_record"]["length"])

        winner = r["game_record"]["winner"]
        if winner == "white":
            white_wins += 1
        elif winner == "black":
            black_wins += 1
        else:
            draws += 1
            draw_reason = r["game_record"].get("draw_reason", "stalemate_or_other")
            draw_reason_counts[draw_reason] = draw_reason_counts.get(draw_reason, 0) + 1

        game_details.append({
            "game": r["game_idx"] + 1,
            "length": r["game_record"]["length"],
            "winner": winner,
            "type": r["game_type"],
            "draw_reason": r["game_record"].get("draw_reason"),
        })
        game_records.append(r["game_record"])

        logger.info(f"  Self-play game {r['game_idx']+1}/{num_games} ({r['game_type']}): "
                    f"{r['game_record']['length']} moves, {len(all_examples)} total positions")

    stats = {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_pct": white_wins / max(white_wins + black_wins, 1),
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "min_game_length": min(game_lengths) if game_lengths else 0,
        "max_game_length": max(game_lengths) if game_lengths else 0,
        "game_details": game_details,
        "game_records": game_records,
        "num_random_games": num_random_games,
        "draw_reason_counts": draw_reason_counts,
    }

    return all_examples, stats
