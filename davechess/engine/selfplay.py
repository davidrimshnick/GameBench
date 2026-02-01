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

from davechess.game.state import GameState, Player
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
        self.planes.append(planes)
        self.policies.append(policy)
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

    def save_data(self, path: str):
        """Save full buffer data to disk."""
        np.savez_compressed(
            path,
            planes=np.stack(list(self.planes)) if self.planes else np.empty((0, 12, 8, 8)),
            policies=np.stack(list(self.policies)) if self.policies else np.empty((0, POLICY_SIZE)),
            values=np.array(list(self.values), dtype=np.float32) if self.values else np.empty(0),
        )

    def load_data(self, path: str):
        """Load buffer data from disk, one array at a time to limit peak memory.

        For compressed NPZ, each data[key] access decompresses the full array.
        By loading/consuming/deleting one key at a time, peak memory is
        ~max(single_array + deque) rather than all arrays at once.
        """
        import gc
        data = np.load(path)

        values_arr = data["values"]
        n = len(values_arr)
        for i in range(n):
            self.values.append(float(values_arr[i]))
        del values_arr
        gc.collect()

        planes_arr = data["planes"]
        for i in range(n):
            self.planes.append(planes_arr[i])
        del planes_arr
        gc.collect()

        policies_arr = data["policies"]
        for i in range(n):
            self.policies.append(policies_arr[i])
        del policies_arr
        del data
        gc.collect()


def play_selfplay_game(mcts_engine: MCTS,
                       temperature_threshold: int = 30) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game and return training examples.

    Returns:
        List of (planes, policy_target, value_target) for each position.
        value_target is set after the game ends based on the outcome.
    """
    state = GameState()
    examples: list[tuple[np.ndarray, dict, int]] = []  # (planes, policy_dict, player)
    move_count = 0

    while not state.done:
        moves = generate_legal_moves(state)
        if not moves:
            break

        # Adjust temperature
        if move_count < temperature_threshold:
            mcts_engine.temperature = 1.0
        else:
            mcts_engine.temperature = 0.1  # Near-greedy

        move, info = mcts_engine.get_move(state, add_noise=True)

        # Record training example
        planes = state_to_planes(state)
        examples.append((planes, info["policy_target"], int(state.current_player)))

        apply_move(state, move)
        move_count += 1

    # Determine game outcome
    if state.winner is not None:
        winner = int(state.winner)
    else:
        winner = -1  # Draw

    # Assign value targets based on outcome
    training_data = []
    for planes, policy_dict, player in examples:
        # Build full policy vector
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, prob in policy_dict.items():
            policy[idx] = prob

        # Value from this player's perspective - only wins get 1.0
        if winner == player:
            value = 1.0
        else:
            value = 0.0  # Losses and draws both get 0

        training_data.append((planes, policy, value))

    return training_data


def run_selfplay_batch(network, num_games: int, num_simulations: int = 200,
                       temperature_threshold: int = 30,
                       device: str = "cpu") -> tuple[list, dict]:
    """Run a batch of self-play games sequentially.

    Args:
        network: The neural network for evaluation.
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        temperature_threshold: Move number after which temperature drops.
        device: Torch device string.

    Returns:
        (all_examples, stats) where stats has game-level statistics.
    """
    all_examples = []
    mcts = MCTS(network, num_simulations=num_simulations, device=device)
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []

    for game_idx in range(num_games):
        examples = play_selfplay_game(mcts, temperature_threshold)
        game_len = len(examples)
        all_examples.extend(examples)
        game_lengths.append(game_len)

        # Determine winner from value targets: last position's value
        # tells us the outcome (1.0 = that player won, -1.0 = lost, 0 = draw)
        if examples:
            last_value = examples[-1][2]  # value_target of last position
            last_player = 0  # We need to check who moved last
            # Even positions (0, 2, 4...) = White's turn, odd = Black's
            if game_len % 2 == 1:  # odd length = White moved last
                if last_value > 0:
                    white_wins += 1
                elif last_value < 0:
                    black_wins += 1
                else:
                    draws += 1
            else:  # even length = Black moved last
                if last_value > 0:
                    black_wins += 1
                elif last_value < 0:
                    white_wins += 1
                else:
                    draws += 1

        logger.info(f"  Self-play game {game_idx+1}/{num_games}: "
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
    }

    return all_examples, stats
