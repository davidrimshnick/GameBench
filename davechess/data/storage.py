"""Save and load games in DCN (DaveChess Notation) format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from davechess.game.state import GameState, Move
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.notation import game_to_dcn, dcn_to_game, move_to_dcn


def save_game(filepath: str, moves: list[Move], headers: Optional[dict] = None,
              result: Optional[str] = None,
              states: Optional[list[GameState]] = None):
    """Save a game to a DCN file.

    Args:
        filepath: Output file path.
        moves: List of moves played.
        headers: Optional header metadata.
        result: Game result string.
        states: Optional list of states before each move (for notation).
                If not provided, the game is replayed from initial state.
    """
    if states is None:
        # Replay to get states
        states = []
        state = GameState()
        for move in moves:
            states.append(state.clone())
            apply_move(state, move)

    pairs = list(zip(states, moves))
    dcn_text = game_to_dcn(pairs, headers=headers, result=result)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(dcn_text)


def load_game(filepath: str) -> tuple[dict, list[Move], Optional[str]]:
    """Load a game from a DCN file.

    Returns:
        (headers, moves, result)
    """
    with open(filepath) as f:
        dcn_text = f.read()
    return dcn_to_game(dcn_text)


def save_games_collection(filepath: str,
                          games: list[tuple[list[Move], dict, str]]):
    """Save multiple games to a single file, separated by blank lines.

    Args:
        filepath: Output file path.
        games: List of (moves, headers, result) tuples.
    """
    sections = []
    for moves, headers, result in games:
        state = GameState()
        pairs = []
        for move in moves:
            pairs.append((state.clone(), move))
            apply_move(state, move)
        sections.append(game_to_dcn(pairs, headers=headers, result=result))

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n\n".join(sections))


def load_games_collection(filepath: str) -> list[tuple[dict, list[Move], Optional[str]]]:
    """Load multiple games from a collection file.

    Returns:
        List of (headers, moves, result) tuples.
    """
    with open(filepath) as f:
        content = f.read()

    # Split on double blank lines
    sections = content.split("\n\n\n")
    if len(sections) == 1:
        # Try splitting on double newline with bracket start
        parts = []
        current = []
        for line in content.split("\n"):
            if line.startswith("[") and current and not current[-1].strip():
                parts.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            parts.append("\n".join(current))
        sections = parts

    games = []
    for section in sections:
        section = section.strip()
        if section:
            try:
                headers, moves, result = dcn_to_game(section)
                if moves:
                    games.append((headers, moves, result))
            except Exception:
                continue

    return games


def replay_game(moves: list[Move]) -> tuple[list[GameState], GameState]:
    """Replay a sequence of moves from initial position.

    Returns:
        (states_before_each_move, final_state)
    """
    states = []
    state = GameState()
    for move in moves:
        states.append(state.clone())
        apply_move(state, move)
    return states, state
