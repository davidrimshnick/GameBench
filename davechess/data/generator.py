"""Generate games at specified strength levels for training data and benchmarking."""

from __future__ import annotations

import logging
import random
from typing import Optional

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.notation import move_to_dcn
from davechess.engine.mcts import MCTS

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("davechess.generator")


class Agent:
    """Base agent interface."""

    def get_move(self, state: GameState) -> Move:
        raise NotImplementedError


class RandomAgent(Agent):
    """Plays random legal moves."""

    def get_move(self, state: GameState) -> Move:
        moves = generate_legal_moves(state)
        if not moves:
            raise ValueError("No legal moves")
        return random.choice(moves)


class MCTSAgent(Agent):
    """Plays using MCTS with neural network."""

    def __init__(self, network, num_simulations: int, device: str = "cpu"):
        self.mcts = MCTS(
            network, num_simulations=num_simulations,
            temperature=0.1, device=device,
        )

    def get_move(self, state: GameState) -> Move:
        move, _ = self.mcts.get_move(state, add_noise=False)
        return move


class MCTSLiteAgent(Agent):
    """Plays using lightweight MCTS (no neural network)."""

    def __init__(self, num_simulations: int):
        from davechess.engine.mcts_lite import MCTSLite
        self.mcts = MCTSLite(num_simulations=num_simulations)

    def get_move(self, state: GameState) -> Move:
        return self.mcts.search(state)


def create_agent(level: dict, network=None, device: str = "cpu") -> Agent:
    """Create an agent from a level configuration.

    Args:
        level: Dict with 'name' and 'mcts_sims' keys.
        network: Optional neural network for MCTS agents.
        device: Torch device.
    """
    sims = level.get("mcts_sims", 0)
    if sims == 0:
        return RandomAgent()
    elif network is not None:
        return MCTSAgent(network, sims, device)
    else:
        return MCTSLiteAgent(sims)


def play_game(white_agent: Agent, black_agent: Agent,
              max_moves: int = 400) -> tuple[list[Move], Optional[Player], int]:
    """Play a complete game between two agents.

    Returns:
        (moves, winner, num_turns)
    """
    state = GameState()
    moves_played: list[Move] = []

    while not state.done and len(moves_played) < max_moves:
        legal = generate_legal_moves(state)
        if not legal:
            break

        agent = white_agent if state.current_player == Player.WHITE else black_agent
        move = agent.get_move(state)

        moves_played.append(move)
        apply_move(state, move)

    return moves_played, state.winner, state.turn


def generate_games(white_agent: Agent, black_agent: Agent,
                   num_games: int, min_length: int = 0,
                   discard_draws: bool = False) -> list[tuple[list[Move], Optional[Player]]]:
    """Generate multiple games, with optional filtering.

    Args:
        white_agent: Agent playing White.
        black_agent: Agent playing Black.
        num_games: Number of games to generate.
        min_length: Minimum moves for a game to be kept.
        discard_draws: Whether to discard drawn games.

    Returns:
        List of (moves, winner) tuples.
    """
    games = []
    attempts = 0
    max_attempts = num_games * 3  # Allow some waste from filtering

    while len(games) < num_games and attempts < max_attempts:
        moves, winner, turns = play_game(white_agent, black_agent)
        attempts += 1

        if len(moves) < min_length:
            continue
        if discard_draws and winner is None:
            continue

        games.append((moves, winner))

        if len(games) % 100 == 0:
            logger.info(f"Generated {len(games)}/{num_games} games "
                        f"({attempts} attempts)")

    return games
