"""Deterministic opening suite for network-vs-network probes."""

from __future__ import annotations

from davechess.game.rules import apply_move, generate_legal_moves
from davechess.game.state import BombardAttack, GameState, Move, MoveStep, Promote


def probe_opening_seed_for_game(game_idx: int) -> int:
    """Pair color-swapped probe games on the same opening."""
    return game_idx // 2


def build_probe_start_state(opening_seed: int, opening_plies: int = 4) -> GameState:
    """Build a deterministic non-starting probe position.

    The opening seed is decoded in mixed radix against the legal-move counts
    seen along the line. This yields a stable opening suite without needing
    handcrafted book moves, while still pairing two games on the same opening
    with colors reversed.
    """
    if opening_seed < 0:
        raise ValueError("opening_seed must be non-negative")

    state = GameState()
    code = opening_seed

    for _ in range(max(0, opening_plies)):
        legal = generate_legal_moves(state)
        if not legal or state.done:
            break
        legal = sorted(legal, key=_move_sort_key)
        move_idx = code % len(legal)
        code //= len(legal)
        apply_move(state, legal[move_idx])

    return state


def _move_sort_key(move: Move) -> tuple[int, ...]:
    """Stable ordering for deterministic opening generation."""
    if isinstance(move, MoveStep):
        return (
            0,
            move.from_rc[0],
            move.from_rc[1],
            move.to_rc[0],
            move.to_rc[1],
            int(move.is_capture),
        )
    if isinstance(move, Promote):
        return (1, move.from_rc[0], move.from_rc[1], int(move.to_type))
    if isinstance(move, BombardAttack):
        return (
            2,
            move.from_rc[0],
            move.from_rc[1],
            move.target_rc[0],
            move.target_rc[1],
        )
    raise TypeError(f"Unsupported move type for probe ordering: {type(move)!r}")
