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
from davechess.engine.network import state_to_planes, move_to_policy_index, POLICY_SIZE
from davechess.engine.mcts import MCTS


class ReplayBuffer:
    """Circular replay buffer for training data.

    Stores (planes, policy_target, value_target) tuples.
    """

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.planes: deque[np.ndarray] = deque(maxlen=max_size)
        self.policies: deque[np.ndarray] = deque(maxlen=max_size)
        self.values: deque[float] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.planes)

    def push(self, planes: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example."""
        self.planes.append(planes.astype(np.float32, copy=False))
        self.policies.append(policy.astype(np.float32, copy=False))
        self.values.append(value)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch.

        Returns:
            (planes_batch, policy_batch, value_batch)
        """
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        planes = np.stack([self.planes[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        return planes, policies, values

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
            np.savez_compressed(path, planes=np.empty((0, 15, 8, 8)),
                                policies=np.empty((0, POLICY_SIZE)),
                                values=np.empty(0))
            return

        with tempfile.TemporaryDirectory() as tmp:
            # Write each array in chunks using memory-mapped files
            planes_path = os.path.join(tmp, "planes.npy")
            planes_mmap = np.lib.format.open_memmap(
                planes_path, mode="w+", dtype=np.float32, shape=(n, 15, 8, 8))
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

            # Combine into compressed npz
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(planes_path, "planes.npy")
                zf.write(policies_path, "policies.npy")
                zf.write(values_path, "values.npy")

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
        del data
        gc.collect()


def play_selfplay_game(mcts_engine: MCTS,
                       temperature_threshold: int = 30,
                       opponent_mcts: MCTS = None,
                       nn_plays_white: bool = True) -> tuple[list, dict]:
    """Play one self-play game and return training examples + game record.

    Args:
        mcts_engine: Primary MCTS engine (with neural network).
        temperature_threshold: Move number after which temperature drops.
        opponent_mcts: Optional second MCTS engine (e.g. random/no-NN).
            When provided, mcts_engine plays one side and opponent_mcts
            plays the other. Training examples are only collected from
            the mcts_engine's turns.
        nn_plays_white: When opponent_mcts is set, which side the NN plays.

    Returns:
        (training_data, game_record) where:
        - training_data: list of (planes, policy_target, value_target) tuples
          (empty for draws)
        - game_record: dict with "moves" (list of (state, move) pairs),
          "winner" ("white"/"black"/"draw"), "length" (int)
    """
    state = GameState()
    examples: list[tuple[np.ndarray, dict, int]] = []  # (planes, policy_dict, player)
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

        move, info = engine.get_move(state, add_noise=True)

        # Only record training examples from the NN engine's turns
        if is_nn_turn:
            planes = state_to_planes(state)
            examples.append((planes, info["policy_target"], int(state.current_player)))

        # Record move for game log (apply_move mutates in-place, so clone first)
        game_moves.append((state.clone(), move))

        state = apply_move(state, move)
        move_count += 1

    # Determine game outcome
    if state.winner is not None:
        winner = int(state.winner)
        winner_str = "white" if state.winner == Player.WHITE else "black"
    else:
        winner = -1
        winner_str = "draw"

    game_record = {"moves": game_moves, "winner": winner_str, "length": move_count}

    # Assign value targets based on outcome
    training_data = []
    for planes, policy_dict, player in examples:
        # Build full policy vector
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, prob in policy_dict.items():
            policy[idx] = prob

        # Value from this player's perspective: +1 win, -1 loss, 0 draw
        if winner == -1:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0

        training_data.append((planes, policy, value))

    return training_data, game_record


def run_selfplay_batch(network, num_games: int, num_simulations: int = 200,
                       temperature_threshold: int = 30,
                       dirichlet_alpha: float = 0.3,
                       dirichlet_epsilon: float = 0.25,
                       random_opponent_fraction: float = 0.0,
                       device: str = "cpu") -> tuple[list, dict]:
    """Run a batch of self-play games sequentially.

    Args:
        network: The neural network for evaluation.
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        temperature_threshold: Move number after which temperature drops.
        dirichlet_alpha: Dirichlet noise concentration parameter.
        dirichlet_epsilon: Fraction of Dirichlet noise to blend with policy.
        random_opponent_fraction: Fraction of games played vs random MCTS
            (no neural network) to prevent self-play overfitting.
        device: Torch device string.

    Returns:
        (all_examples, stats) where stats has game-level statistics.
    """
    all_examples = []
    mcts = MCTS(network, num_simulations=num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                device=device)
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    game_details = []

    game_records = []

    # Create random opponent if needed
    num_random_games = int(num_games * random_opponent_fraction)
    random_mcts = None
    if num_random_games > 0:
        random_mcts = MCTS(None, num_simulations=num_simulations, device=device)

    for game_idx in range(num_games):
        if game_idx < num_random_games:
            # Play against random MCTS — alternate sides
            nn_plays_white = (game_idx % 2 == 0)
            examples, game_record = play_selfplay_game(
                mcts, temperature_threshold,
                opponent_mcts=random_mcts,
                nn_plays_white=nn_plays_white,
            )
            game_type = "vs_random"
        else:
            # Standard self-play
            examples, game_record = play_selfplay_game(mcts, temperature_threshold)
            game_type = "selfplay"

        game_len = len(examples)
        all_examples.extend(examples)
        game_lengths.append(game_len)

        winner = game_record["winner"]
        if winner == "white":
            white_wins += 1
        elif winner == "black":
            black_wins += 1
        else:
            draws += 1

        game_details.append({"game": game_idx + 1, "length": game_record["length"],
                             "winner": winner, "type": game_type})
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
    }

    return all_examples, stats
