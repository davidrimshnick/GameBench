"""DaveChess Notation (DCN) parser and emitter.

Move formats:
  Wa2-a3     Move Warrior from a2 to a3
  Rb1xd3     Capture: Rider at b1 captures piece at d3
  Wa1>R      Promote Warrior at a1 to Rider
  Bc3~e3     Bombard ranged attack from c3 targeting e3

Game format (similar to PGN):
  [White "AlphaZero"]
  [Black "AlphaZero"]
  [Result "1-0"]
  [Date "2025.01.15"]

  1. Wa2-a3 Wa7-a6
  2. Rc1-c3 Rc8-c6
  ...
"""

from __future__ import annotations

import re
from typing import Optional

from davechess.game.board import rc_to_notation, notation_to_rc
from davechess.game.state import (
    GameState, Move, MoveStep, Promote, BombardAttack,
    PieceType, Player, PIECE_CHARS, PIECE_NAMES,
)


def move_to_dcn(state: GameState, move: Move) -> str:
    """Convert a move to DCN string.

    Args:
        state: The game state BEFORE the move is applied.
        move: The move to convert.
    """
    if isinstance(move, MoveStep):
        piece = state.board[move.from_rc[0]][move.from_rc[1]]
        piece_char = piece.char if piece else "?"
        from_sq = rc_to_notation(*move.from_rc)
        to_sq = rc_to_notation(*move.to_rc)
        sep = "x" if move.is_capture else "-"
        return f"{piece_char}{from_sq}{sep}{to_sq}"

    elif isinstance(move, Promote):
        piece = state.board[move.from_rc[0]][move.from_rc[1]]
        piece_char = piece.char if piece else "?"
        from_sq = rc_to_notation(*move.from_rc)
        target_char = PIECE_NAMES[move.to_type]
        return f"{piece_char}{from_sq}>{target_char}"

    elif isinstance(move, BombardAttack):
        from_sq = rc_to_notation(*move.from_rc)
        target_sq = rc_to_notation(*move.target_rc)
        return f"B{from_sq}~{target_sq}"

    raise ValueError(f"Unknown move type: {type(move)}")


# Regex patterns for parsing
_MOVE_RE = re.compile(r"^([CWRBL])([a-h][1-8])([-x])([a-h][1-8])$")
_PROMOTE_RE = re.compile(r"^([CWRBL])([a-h][1-8])>([RBL])$")
_BOMBARD_RE = re.compile(r"^B([a-h][1-8])~([a-h][1-8])$")


def dcn_to_move(dcn: str) -> Move:
    """Parse a DCN string into a Move object.

    Args:
        dcn: DCN notation string.

    Returns:
        A Move object.

    Raises:
        ValueError: If the notation is invalid.
    """
    dcn = dcn.strip()

    # Try promotion
    m = _PROMOTE_RE.match(dcn)
    if m:
        from_rc = notation_to_rc(m.group(2))
        to_type = PIECE_CHARS[m.group(3)]
        return Promote(from_rc, to_type)

    # Try bombard
    m = _BOMBARD_RE.match(dcn)
    if m:
        from_rc = notation_to_rc(m.group(1))
        target_rc = notation_to_rc(m.group(2))
        return BombardAttack(from_rc, target_rc)

    # Try normal move/capture
    m = _MOVE_RE.match(dcn)
    if m:
        from_rc = notation_to_rc(m.group(2))
        is_capture = m.group(3) == "x"
        to_rc = notation_to_rc(m.group(4))
        return MoveStep(from_rc, to_rc, is_capture=is_capture)

    raise ValueError(f"Invalid DCN notation: {dcn!r}")


def game_to_dcn(states_and_moves: list[tuple[GameState, Move]],
                headers: Optional[dict[str, str]] = None,
                result: Optional[str] = None) -> str:
    """Convert a sequence of (state_before_move, move) pairs to a DCN game record.

    Args:
        states_and_moves: List of (state, move) pairs.
        headers: Optional dict of header key-value pairs.
        result: Game result string ("1-0", "0-1", "1/2-1/2").
    """
    lines = []

    # Headers
    if headers:
        for key, value in headers.items():
            lines.append(f'[{key} "{value}"]')
    if result:
        lines.append(f'[Result "{result}"]')
    if headers or result:
        lines.append("")

    # Moves
    move_strs = []
    for state, move in states_and_moves:
        move_strs.append(move_to_dcn(state, move))

    # Format as numbered move pairs
    move_lines = []
    i = 0
    move_num = 1
    while i < len(move_strs):
        if i + 1 < len(move_strs):
            move_lines.append(f"{move_num}. {move_strs[i]} {move_strs[i+1]}")
            i += 2
        else:
            move_lines.append(f"{move_num}. {move_strs[i]}")
            i += 1
        move_num += 1

    lines.extend(move_lines)

    if result:
        lines.append(result)

    return "\n".join(lines)


def dcn_to_game(dcn_text: str) -> tuple[dict[str, str], list[Move], Optional[str]]:
    """Parse a DCN game record.

    Returns:
        (headers, moves, result)
    """
    headers: dict[str, str] = {}
    moves: list[Move] = []
    result: Optional[str] = None

    lines = dcn_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Header
        if line.startswith("[") and line.endswith("]"):
            inner = line[1:-1]
            # Parse: Key "Value"
            m = re.match(r'(\w+)\s+"([^"]*)"', inner)
            if m:
                headers[m.group(1)] = m.group(2)
            continue

        # Result line
        if line in ("1-0", "0-1", "1/2-1/2", "*"):
            result = line
            continue

        # Move line: "1. Wa2-a3 Wa7-a6" or "1. Wa2-a3"
        # Strip move number prefix
        line = re.sub(r"^\d+\.\s*", "", line)
        if not line:
            continue

        # Split into individual moves
        tokens = line.split()
        for token in tokens:
            token = token.strip()
            if not token or token in ("1-0", "0-1", "1/2-1/2", "*"):
                if token in ("1-0", "0-1", "1/2-1/2", "*"):
                    result = token
                continue
            try:
                moves.append(dcn_to_move(token))
            except ValueError:
                pass  # Skip unparseable tokens

    return headers, moves, result
