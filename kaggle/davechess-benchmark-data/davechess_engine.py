"""Consolidated DaveChess engine -- single-file, zero external dependencies.

Merged from:
  davechess/game/board.py
  davechess/game/state.py
  davechess/game/rules.py
  davechess/game/notation.py
  davechess/engine/mcts_lite.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ============================================================================
# === board.py ===
# ============================================================================

BOARD_SIZE = 8

# Gold nodes: give +1 resource per turn (central positions)
GOLD_NODES: list[tuple[int, int]] = [
    (3, 3), (3, 4),  # Central gold nodes
    (4, 3), (4, 4),  # Central gold nodes
]

# All nodes (just Gold nodes now -- Power nodes removed in v2)
ALL_NODES: list[tuple[int, int]] = GOLD_NODES

# Backward compatibility alias
RESOURCE_NODES = ALL_NODES

# Starting positions: dict mapping (row, col) -> (piece_type_char, player)
# White on rows 0-1 (bottom), Black on rows 6-7 (top)
# 12 pieces per side: 1 Commander, 3 Riders, 2 Bombards, 6 Warriors
# Back rank: officers. Front rank: Warrior screen.
STARTING_POSITIONS: dict[tuple[int, int], tuple[str, int]] = {
    # White (player 0) - rows 0-1
    # Row 0 (back rank): R B R C R B (heavy firepower)
    (0, 1): ("R", 0),
    (0, 2): ("B", 0),
    (0, 3): ("R", 0),
    (0, 4): ("C", 0),
    (0, 5): ("R", 0),
    (0, 6): ("B", 0),
    # Row 1 (front rank): 6 Warriors as pawn screen
    (1, 1): ("W", 0),
    (1, 2): ("W", 0),
    (1, 3): ("W", 0),
    (1, 4): ("W", 0),
    (1, 5): ("W", 0),
    (1, 6): ("W", 0),
    # Black (player 1) - rows 6-7 (mirrors White)
    # Row 7 (back rank): mirrored
    (7, 1): ("R", 1),
    (7, 2): ("B", 1),
    (7, 3): ("R", 1),
    (7, 4): ("C", 1),
    (7, 5): ("R", 1),
    (7, 6): ("B", 1),
    # Row 6 (front rank): 6 Warriors as pawn screen
    (6, 1): ("W", 1),
    (6, 2): ("W", 1),
    (6, 3): ("W", 1),
    (6, 4): ("W", 1),
    (6, 5): ("W", 1),
    (6, 6): ("W", 1),
}

# Column labels for notation
COL_LABELS = "abcdefgh"
# Row labels for notation (1-indexed, row 0 = "1", row 7 = "8")
ROW_LABELS = "12345678"


def rc_to_notation(row: int, col: int) -> str:
    """Convert (row, col) to algebraic notation like 'a1'."""
    return COL_LABELS[col] + ROW_LABELS[row]


def notation_to_rc(sq: str) -> tuple[int, int]:
    """Convert algebraic notation like 'a1' to (row, col)."""
    col = COL_LABELS.index(sq[0])
    row = ROW_LABELS.index(sq[1])
    return (row, col)


def render_board(board, resource_counts: tuple[int, int] | None = None,
                 turn: int | None = None, current_player: int | None = None) -> str:
    """Render the board as a text string.

    Args:
        board: 8x8 list of lists. Each cell is None or (piece_type_char, player).
        resource_counts: Optional (white_resources, black_resources).
        turn: Optional turn number.
        current_player: Optional current player (0=White, 1=Black).
    """
    lines = []

    if turn is not None:
        player_name = "White" if current_player == 0 else "Black"
        lines.append(f"Turn {turn} - {player_name} to move")
    if resource_counts is not None:
        lines.append(f"Resources: White={resource_counts[0]}  Black={resource_counts[1]}")
    lines.append("")

    gold_set = set(GOLD_NODES)

    lines.append("    a   b   c   d   e   f   g   h")
    lines.append("  +---+---+---+---+---+---+---+---+")

    for row in range(BOARD_SIZE - 1, -1, -1):
        row_str = f"{row + 1} |"
        for col in range(BOARD_SIZE):
            cell = board[row][col]
            pos = (row, col)
            marker = "$" if pos in gold_set else None
            if cell is not None:
                piece_char, player = cell
                # Lowercase for black, uppercase for white
                display = piece_char if player == 0 else piece_char.lower()
                if marker:
                    row_str += f"{marker}{display}{marker}|"
                else:
                    row_str += f" {display} |"
            else:
                if marker:
                    row_str += f" {marker} |"
                else:
                    row_str += "   |"
        row_str += f" {row + 1}"
        lines.append(row_str)
        lines.append("  +---+---+---+---+---+---+---+---+")

    lines.append("    a   b   c   d   e   f   g   h")

    return "\n".join(lines)


# ============================================================================
# === state.py ===
# ============================================================================

class Player(IntEnum):
    WHITE = 0
    BLACK = 1


class PieceType(IntEnum):
    COMMANDER = 0
    WARRIOR = 1
    RIDER = 2
    BOMBARD = 3
    LANCER = 4


# Map character codes to PieceType
PIECE_CHARS = {
    "C": PieceType.COMMANDER,
    "W": PieceType.WARRIOR,
    "R": PieceType.RIDER,
    "B": PieceType.BOMBARD,
    "L": PieceType.LANCER,
}
PIECE_NAMES = {v: k for k, v in PIECE_CHARS.items()}

# Promotion cost per target type (spend resources to upgrade a piece in place)
# Commander cannot be a promotion target. Warriors are the base unit.
PROMOTION_COST = {
    PieceType.RIDER: 3,
    PieceType.BOMBARD: 5,
    PieceType.LANCER: 7,
}


@dataclass
class Piece:
    piece_type: PieceType
    player: Player

    @property
    def char(self) -> str:
        return PIECE_NAMES[self.piece_type]

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return NotImplemented
        return self.piece_type == other.piece_type and self.player == other.player

    def __hash__(self):
        return hash((self.piece_type, self.player))


# Move types
@dataclass
class Move:
    """Represents a single move in DaveChess."""
    pass


@dataclass
class MoveStep(Move):
    """Move a piece from one square to another."""
    from_rc: tuple[int, int]
    to_rc: tuple[int, int]
    is_capture: bool = False

    def __eq__(self, other):
        if not isinstance(other, MoveStep):
            return NotImplemented
        return self.from_rc == other.from_rc and self.to_rc == other.to_rc

    def __hash__(self):
        return hash(("move", self.from_rc, self.to_rc))


@dataclass
class Promote(Move):
    """Promote a piece to a higher-cost type in place."""
    from_rc: tuple[int, int]
    to_type: PieceType

    def __eq__(self, other):
        if not isinstance(other, Promote):
            return NotImplemented
        return self.from_rc == other.from_rc and self.to_type == other.to_type

    def __hash__(self):
        return hash(("promote", self.from_rc, self.to_type))


@dataclass
class BombardAttack(Move):
    """Bombard ranged attack (piece stays, target removed)."""
    from_rc: tuple[int, int]
    target_rc: tuple[int, int]

    def __eq__(self, other):
        if not isinstance(other, BombardAttack):
            return NotImplemented
        return self.from_rc == other.from_rc and self.target_rc == other.target_rc

    def __hash__(self):
        return hash(("bombard", self.from_rc, self.target_rc))


class GameState:
    """Complete game state for DaveChess."""

    def __init__(self):
        self.board: list[list[Optional[Piece]]] = [
            [None] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self.resources: list[int] = [0, 0]  # [White, Black]
        self.current_player: Player = Player.WHITE
        self.turn: int = 1
        self.done: bool = False
        self.winner: Optional[Player] = None  # None = draw if done
        self.move_history: list[Move] = []
        self.position_counts: dict[tuple, int] = {}
        self.halfmove_clock: int = 0  # Moves since last capture or deploy (50-move rule)
        self.last_move: Optional[Move] = None  # Last move played (for NN input planes)
        self._setup_starting_position()
        # Record starting position for threefold repetition detection
        self.position_counts[self.get_position_key()] = 1

    def _setup_starting_position(self):
        """Place pieces in their starting positions."""
        for (row, col), (char, player) in STARTING_POSITIONS.items():
            piece_type = PIECE_CHARS[char]
            self.board[row][col] = Piece(piece_type, Player(player))

    def clone(self) -> GameState:
        """Return a deep copy of this state."""
        new = GameState.__new__(GameState)
        new.board = [[cell if cell is None else Piece(cell.piece_type, cell.player)
                       for cell in row] for row in self.board]
        new.resources = self.resources.copy()
        new.current_player = self.current_player
        new.turn = self.turn
        new.done = self.done
        new.winner = self.winner
        new.move_history = []  # Don't copy history for MCTS clones
        new.position_counts = self.position_counts.copy()
        new.halfmove_clock = self.halfmove_clock
        new.last_move = self.last_move
        return new

    def get_piece_at(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at position, or None."""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None

    def get_board_tuple(self) -> tuple:
        """Return a hashable representation of the board for state comparison.
        Includes resources -- use get_position_key() for repetition detection.
        """
        cells = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.board[row][col]
                if cell is None:
                    cells.append(None)
                else:
                    cells.append((cell.piece_type, cell.player))
        return (tuple(cells), self.current_player, self.resources[0], self.resources[1])

    def get_position_key(self) -> tuple:
        """Return a hashable key for threefold repetition detection.

        Includes coarse resource buckets (per player) based on promotion
        affordability so positions with materially different options don't
        collapse into the same repetition key.
        """
        def _resource_bucket(resource: int) -> int:
            # Promotion thresholds: Rider=3, Bombard=5, Lancer=7
            if resource < 3:
                return 0
            if resource < 5:
                return 1
            if resource < 7:
                return 2
            return 3

        cells = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.board[row][col]
                if cell is None:
                    cells.append(None)
                else:
                    cells.append((cell.piece_type, cell.player))
        return (
            tuple(cells),
            self.current_player,
            _resource_bucket(self.resources[Player.WHITE]),
            _resource_bucket(self.resources[Player.BLACK]),
        )

    def serialize(self) -> str:
        """Serialize game state to JSON string."""
        board_data = []
        for row in range(BOARD_SIZE):
            row_data = []
            for col in range(BOARD_SIZE):
                cell = self.board[row][col]
                if cell is None:
                    row_data.append(None)
                else:
                    row_data.append({"type": int(cell.piece_type), "player": int(cell.player)})
            board_data.append(row_data)

        return json.dumps({
            "board": board_data,
            "resources": self.resources,
            "current_player": int(self.current_player),
            "turn": self.turn,
            "done": self.done,
            "winner": int(self.winner) if self.winner is not None else None,
        })

    @classmethod
    def deserialize(cls, data: str) -> GameState:
        """Deserialize game state from JSON string."""
        d = json.loads(data)
        state = cls.__new__(cls)
        state.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = d["board"][row][col]
                if cell is not None:
                    state.board[row][col] = Piece(
                        PieceType(cell["type"]), Player(cell["player"])
                    )
        state.resources = d["resources"]
        state.current_player = Player(d["current_player"])
        state.turn = d["turn"]
        state.done = d["done"]
        state.winner = Player(d["winner"]) if d["winner"] is not None else None
        state.move_history = []
        state.position_counts = {}
        state.halfmove_clock = 0
        state.last_move = None
        return state

    def to_display_board(self) -> list[list]:
        """Convert to the format expected by render_board."""
        display = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.board[row][col]
                if cell is not None:
                    display[row][col] = (cell.char, int(cell.player))
        return display


# ============================================================================
# === rules.py ===
# ============================================================================

# Orthogonal directions
ORTHOGONAL = [(0, 1), (0, -1), (1, 0), (-1, 0)]
# All 8 directions
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# Straight-line directions (orthogonal + diagonal)
STRAIGHT_DIRS = ALL_DIRS
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

_GOLD_SET = frozenset(GOLD_NODES)
_ALL_NODES_SET = frozenset(ALL_NODES)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_resource_income(state: GameState, player: Player) -> int:
    """Calculate resource income for a player.

    +1 per Gold node that the player has a piece directly on.
    """
    income = 0
    for nr, nc in GOLD_NODES:
        piece = state.board[nr][nc]
        if piece is not None and piece.player == player:
            income += 1
    return income


def _player_controls_node(state: GameState, player: Player, nr: int, nc: int) -> bool:
    """Check if player has a piece on or orthogonally adjacent to node at (nr, nc).

    Note: resource income only requires being ON the node. This broader check
    is used for node-control queries (e.g. UI, analysis).
    """
    # On the node
    piece = state.board[nr][nc]
    if piece is not None and piece.player == player:
        return True
    # Orthogonally adjacent
    for dr, dc in ORTHOGONAL:
        r2, c2 = nr + dr, nc + dc
        if _in_bounds(r2, c2):
            p2 = state.board[r2][c2]
            if p2 is not None and p2.player == player:
                return True
    return False


def _count_controlled_nodes(state: GameState, player: Player) -> int:
    """Count resource nodes exclusively controlled by player."""
    opponent = Player(1 - player)
    count = 0
    for nr, nc in ALL_NODES:
        if _player_controls_node(state, player, nr, nc) and \
           not _player_controls_node(state, opponent, nr, nc):
            count += 1
    return count


def _find_commander(state: GameState, player: Player) -> tuple[int, int] | None:
    """Find the Commander's position for a player."""
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = state.board[row][col]
            if p is not None and p.player == player and p.piece_type == PieceType.COMMANDER:
                return (row, col)
    return None


def _is_square_attacked(state: GameState, tr: int, tc: int, by_player: Player) -> bool:
    """Check if any piece of by_player can attack the square (tr, tc).

    Checks outward from the target square along attack patterns.
    """
    board = state.board

    # Check all 8 directions for short-range pieces and Rider
    for dr, dc in ALL_DIRS:
        is_orthogonal = (dr == 0 or dc == 0)
        rider_max = 8 if is_orthogonal else 4  # ortho range 7, diag range 3

        # Distance 1: Commander (any dir), Bombard (any dir melee), Rider (any dir)
        r1, c1 = tr + dr, tc + dc
        if _in_bounds(r1, c1):
            p = board[r1][c1]
            if p is not None and p.player == by_player:
                pt = p.piece_type
                if pt == PieceType.COMMANDER or pt == PieceType.BOMBARD or pt == PieceType.RIDER:
                    return True
                # Warrior captures diagonally forward only
                if pt == PieceType.WARRIOR:
                    move_dr = -dr
                    move_dc = -dc
                    if move_dr != 0 and move_dc != 0:
                        forward = 1 if by_player == Player.WHITE else -1
                        if move_dr == forward:
                            return True

        # Distance 2+: Rider only (straight line, clear path, no jumping)
        # Orthogonal: up to 7 squares, Diagonal: up to 3 squares
        path_clear = True
        for dist in range(2, rider_max):
            mid_r, mid_c = tr + dr * (dist - 1), tc + dc * (dist - 1)
            if not _in_bounds(mid_r, mid_c) or board[mid_r][mid_c] is not None:
                path_clear = False
            if not path_clear:
                break
            rd, cd = tr + dr * dist, tc + dc * dist
            if _in_bounds(rd, cd):
                pd = board[rd][cd]
                if pd is not None and pd.player == by_player and pd.piece_type == PieceType.RIDER:
                    return True

    # Lancer: all 8 directions, distance 1-7, can jump over one piece
    for dr, dc in STRAIGHT_DIRS:
        blocking = 0
        for dist in range(1, 8):
            r, c = tr + dr * dist, tc + dc * dist
            if not _in_bounds(r, c):
                break
            p = board[r][c]
            if p is not None:
                if p.player == by_player and p.piece_type == PieceType.LANCER and blocking <= 1:
                    return True
                blocking += 1
                if blocking > 1:
                    break

    # Bombard ranged: exactly 2 squares orthogonal, clear path, but NOT against Commander
    # We only need this for general "is square attacked" -- but Bombard can't target Commander
    # So for check detection this doesn't apply (Commanders can't be bombarded)
    # For other purposes (like protecting squares), we check:
    target_piece = board[tr][tc]
    if target_piece is None or target_piece.piece_type != PieceType.COMMANDER:
        for dr, dc in STRAIGHT_DIRS:
            br, bc = tr + dr * 2, tc + dc * 2
            if _in_bounds(br, bc):
                bp = board[br][bc]
                if bp is not None and bp.player == by_player and bp.piece_type == PieceType.BOMBARD:
                    # Check clear path
                    mr, mc = tr + dr, tc + dc
                    if _in_bounds(mr, mc) and board[mr][mc] is None:
                        return True

    return False


def is_in_check(state: GameState, player: Player) -> bool:
    """Check if the given player's Commander is under attack."""
    cmd_pos = _find_commander(state, player)
    if cmd_pos is None:
        return False  # Commander already captured
    opponent = Player(1 - player)
    return _is_square_attacked(state, cmd_pos[0], cmd_pos[1], opponent)


def _generate_pseudo_legal_moves(state: GameState) -> list[Move]:
    """Generate all pseudo-legal moves (ignoring check)."""
    player = state.current_player
    moves: list[Move] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None or piece.player != player:
                continue

            if piece.piece_type == PieceType.COMMANDER:
                _gen_commander_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.WARRIOR:
                _gen_warrior_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.RIDER:
                _gen_rider_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.BOMBARD:
                _gen_bombard_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.LANCER:
                _gen_lancer_moves(state, row, col, player, moves)

    _gen_promotion_moves(state, player, moves)
    return moves


def _apply_move_no_checks(state: GameState, move: Move) -> GameState:
    """Apply a move without win-condition or turn-switching logic.

    Used internally for check detection (just updates the board).
    Modifies state in place.
    """
    player = state.current_player

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]
        # Chess-style: attacker always takes the square
        state.board[tr][tc] = attacker
        state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        state.board[tr][tc] = None

    return state


def generate_legal_moves(state: GameState) -> list[Move]:
    """Generate all legal moves for the current player.

    Filters out moves that leave the player's own Commander in check.
    Also detects checkmate/stalemate: if no legal moves exist, sets state.done.
    """
    if state.done:
        return []

    player = state.current_player
    pseudo_moves = _generate_pseudo_legal_moves(state)

    legal_moves = []
    for move in pseudo_moves:
        if _is_move_legal(state, move, player):
            legal_moves.append(move)

    # Detect checkmate/stalemate
    if not legal_moves:
        state.done = True
        if is_in_check(state, player):
            # Checkmate: current player loses
            state.winner = Player(1 - player)
        else:
            # Stalemate: draw
            state.winner = None

    return legal_moves


def _is_move_legal(state: GameState, move: Move, player: Player) -> bool:
    """Check if a move is legal (doesn't leave own Commander in check).

    Uses make/unmake on the board to avoid cloning.
    """
    board = state.board

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        moving_piece = board[fr][fc]
        captured_piece = board[tr][tc]

        # Chess-style: attacker always takes the square
        board[fr][fc] = None
        board[tr][tc] = moving_piece

        # Check if our Commander is safe
        safe = not is_in_check(state, player)

        # Unmake
        board[fr][fc] = moving_piece
        board[tr][tc] = captured_piece
        return safe

    elif isinstance(move, Promote):
        r, c = move.from_rc
        old_piece = board[r][c]
        board[r][c] = Piece(move.to_type, player)
        safe = not is_in_check(state, player)
        board[r][c] = old_piece  # unmake
        return safe

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        captured_piece = board[tr][tc]
        board[tr][tc] = None  # target removed
        safe = not is_in_check(state, player)
        board[tr][tc] = captured_piece  # unmake
        return safe

    return True


def _gen_commander_moves(state: GameState, row: int, col: int, player: Player,
                         moves: list[Move]):
    """Commander: 1 square, any direction. Captures same as movement."""
    for dr, dc in ALL_DIRS:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))


def _gen_warrior_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Warrior: moves 1 square forward, captures 1 square diagonal-forward.

    Like a chess pawn. Forward is +row for White, -row for Black.
    """
    forward = 1 if player == Player.WHITE else -1

    # Forward move (non-capture only)
    r2 = row + forward
    if _in_bounds(r2, col) and state.board[r2][col] is None:
        moves.append(MoveStep((row, col), (r2, col)))

    # Diagonal-forward captures
    for dc in (-1, 1):
        c2 = col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is not None and target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))


def _gen_rider_moves(state: GameState, row: int, col: int, player: Player,
                     moves: list[Move]):
    """Rider: up to 7 squares orthogonal, up to 3 squares diagonal, no jumping."""
    for dr, dc in STRAIGHT_DIRS:
        is_orthogonal = (dr == 0 or dc == 0)
        max_dist = 8 if is_orthogonal else 4  # range(1,8)=7, range(1,4)=3
        for dist in range(1, max_dist):
            r2, c2 = row + dr * dist, col + dc * dist
            if not _in_bounds(r2, c2):
                break
            target = state.board[r2][c2]
            if target is None:
                moves.append(MoveStep((row, col), (r2, c2)))
            elif target.player != player:
                moves.append(MoveStep((row, col), (r2, c2), is_capture=True))
                break
            else:
                break  # Blocked by friendly piece


def _gen_bombard_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Bombard: 1 square movement (any direction) + ranged capture at exactly 2 squares.

    Melee: moves 1 square any direction, can capture adjacent enemies.
    Ranged: attacks at exactly 2 squares orthogonal/diagonal, clear path,
            Bombard stays in place. Cannot target Commanders.
    """
    # Normal movement/capture: 1 square, any direction
    for dr, dc in ALL_DIRS:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))

    # Ranged attack: exactly 2 squares away, straight line, clear path
    for dr, dc in STRAIGHT_DIRS:
        # Check intermediate square is clear
        mid_r, mid_c = row + dr, col + dc
        if not _in_bounds(mid_r, mid_c):
            continue
        if state.board[mid_r][mid_c] is not None:
            continue  # Path blocked

        target_r, target_c = row + dr * 2, col + dc * 2
        if not _in_bounds(target_r, target_c):
            continue
        target = state.board[target_r][target_c]
        if target is not None and target.player != player \
                and target.piece_type != PieceType.COMMANDER:
            moves.append(BombardAttack((row, col), (target_r, target_c)))


def _gen_lancer_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Lancer: up to 7 squares any direction, can jump over exactly one piece (any color)."""
    for dr, dc in STRAIGHT_DIRS:
        pieces_in_way = 0
        for dist in range(1, 8):
            r2, c2 = row + dr * dist, col + dc * dist
            if not _in_bounds(r2, c2):
                break
            occupant = state.board[r2][c2]
            if occupant is None:
                # Empty: can land here
                moves.append(MoveStep((row, col), (r2, c2)))
            elif occupant.player == player:
                # Friendly piece: jump over if first, blocked if second
                pieces_in_way += 1
                if pieces_in_way > 1:
                    break
            else:
                # Enemy piece: can capture, then this square blocks further
                if pieces_in_way <= 1:
                    moves.append(MoveStep((row, col), (r2, c2), is_capture=True))
                pieces_in_way += 1
                if pieces_in_way > 1:
                    break


def _gen_promotion_moves(state: GameState, player: Player, moves: list[Move]):
    """Generate promotion moves: upgrade a friendly piece in place.

    Any non-Commander piece can promote to a higher-cost type.
    Cost = full price of target type.
    """
    resources = state.resources[player]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None or piece.player != player:
                continue
            if piece.piece_type == PieceType.COMMANDER:
                continue  # Commander can't promote

            for target_type, cost in PROMOTION_COST.items():
                if resources < cost:
                    continue
                # Can only promote to a different type
                if piece.piece_type == target_type:
                    continue
                moves.append(Promote((row, col), target_type))


def apply_move(state: GameState, move: Move) -> GameState:
    """Apply a move and return the new state.

    This modifies the state in-place for performance, so clone first if needed.
    Chess-style capture: attacker always takes the defender's square.
    """
    player = state.current_player
    opponent = Player(1 - player)

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]

        if move.is_capture:
            defender = state.board[tr][tc]
            # Chess-style: attacker always wins
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None
            # Check if defender was Commander
            if defender.piece_type == PieceType.COMMANDER:
                state.done = True
                state.winner = player
        else:
            # Simple move
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        cost = PROMOTION_COST[move.to_type]
        state.resources[player] -= cost
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        # Ranged capture: target is simply removed, bombard stays
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    state.move_history.append(move)
    state.last_move = move

    # Update halfmove clock (50-move rule): reset on capture/promote/bombard, else increment
    if isinstance(move, MoveStep) and move.is_capture:
        state.halfmove_clock = 0
    elif isinstance(move, (Promote, BombardAttack)):
        state.halfmove_clock = 0
    else:
        state.halfmove_clock += 1

    # Switch player and advance turn
    if not state.done:
        state.current_player = opponent
        if player == Player.BLACK:
            state.turn += 1

        # Gain resources at start of new player's turn
        income = get_resource_income(state, state.current_player)
        state.resources[state.current_player] += income

        # Check turn limit -- draw if no checkmate by turn 100
        if state.turn > 100:
            state.done = True
            state.winner = None  # Draw

        # Check 50-move rule -- draw if 100 halfmoves with no capture or promotion
        if not state.done and state.halfmove_clock >= 100:
            state.done = True
            state.winner = None  # Draw by 50-move rule

        # Check threefold repetition -- draw if same position occurs 3 times
        if not state.done:
            pos_key = state.get_position_key()
            state.position_counts[pos_key] = state.position_counts.get(pos_key, 0) + 1
            if state.position_counts[pos_key] >= 3:
                state.done = True
                state.winner = None  # Draw by repetition

    return state


def generate_pseudo_legal_moves(state: GameState) -> list[Move]:
    """Generate pseudo-legal moves (no check filtering). For fast rollouts."""
    if state.done:
        return []
    return _generate_pseudo_legal_moves(state)


def apply_move_fast(state: GameState, move: Move) -> GameState:
    """Apply a move without checkmate/stalemate detection. For fast rollouts.

    Still handles captures, win conditions, resource income, and turn switching,
    but skips the expensive check for whether the opponent has legal moves.
    """
    player = state.current_player
    opponent = Player(1 - player)

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]

        if move.is_capture:
            defender = state.board[tr][tc]
            # Chess-style: attacker always wins
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None
            if defender.piece_type == PieceType.COMMANDER:
                state.done = True
                state.winner = player
        else:
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        cost = PROMOTION_COST[move.to_type]
        state.resources[player] -= cost
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    state.last_move = move

    # Update halfmove clock (50-move rule)
    if isinstance(move, MoveStep) and move.is_capture:
        state.halfmove_clock = 0
    elif isinstance(move, (Promote, BombardAttack)):
        state.halfmove_clock = 0
    else:
        state.halfmove_clock += 1

    if not state.done:
        state.current_player = opponent
        if player == Player.BLACK:
            state.turn += 1

        income = get_resource_income(state, state.current_player)
        state.resources[state.current_player] += income

        # Check turn limit -- draw if no checkmate by turn 100
        if state.turn > 100:
            state.done = True
            state.winner = None  # Draw

        # Check 50-move rule -- draw if 100 halfmoves with no capture or promotion
        if not state.done and state.halfmove_clock >= 100:
            state.done = True
            state.winner = None  # Draw by 50-move rule

        # Check threefold repetition (same logic as apply_move)
        if not state.done:
            pos_key = state.get_position_key()
            state.position_counts[pos_key] = state.position_counts.get(pos_key, 0) + 1
            if state.position_counts[pos_key] >= 3:
                state.done = True
                state.winner = None  # Draw by repetition

    return state


def check_winner(state: GameState) -> tuple[bool, Optional[Player]]:
    """Check if the game is over.

    Returns (is_done, winner) where winner is None for draw.
    """
    return state.done, state.winner


def get_controlled_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of all nodes controlled by player (on or adjacent)."""
    return [(nr, nc) for nr, nc in ALL_NODES
            if _player_controls_node(state, player, nr, nc)]


def get_exclusive_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of all nodes exclusively controlled by player."""
    opponent = Player(1 - player)
    return [(nr, nc) for nr, nc in ALL_NODES
            if _player_controls_node(state, player, nr, nc)
            and not _player_controls_node(state, opponent, nr, nc)]


# ============================================================================
# === notation.py ===
# ============================================================================

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


# ============================================================================
# === mcts_lite.py ===
# ============================================================================

@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""
    state: GameState
    parent: Optional[MCTSNode] = None
    move: Optional[Move] = None  # Move that led to this node
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0  # From the perspective of the node's parent's player
    untried_moves: Optional[list[Move]] = None

    @property
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.done

    def ucb1(self, exploration: float = 1.41) -> float:
        """Upper confidence bound for trees."""
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSLite:
    """Lightweight MCTS engine using random rollouts."""

    def __init__(self, num_simulations: int = 100, max_rollout_depth: int = 100,
                 exploration: float = 1.41):
        self.num_simulations = num_simulations
        self.max_rollout_depth = max_rollout_depth
        self.exploration = exploration

    def search(self, state: GameState) -> Move:
        """Run MCTS and return the best move."""
        root = MCTSNode(state=state.clone())
        root.untried_moves = generate_legal_moves(root.state)

        if not root.untried_moves:
            raise ValueError("No legal moves available")

        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            winner = self._rollout(node)
            self._backpropagate(node, winner)

        # Select child with most visits
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def search_with_policy(self, state: GameState) -> tuple[Move, dict[Move, float]]:
        """Run MCTS and return best move plus visit-count policy."""
        root = MCTSNode(state=state.clone())
        root.untried_moves = generate_legal_moves(root.state)

        if not root.untried_moves:
            raise ValueError("No legal moves available")

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            winner = self._rollout(node)
            self._backpropagate(node, winner)

        total_visits = sum(c.visits for c in root.children)
        policy = {}
        for child in root.children:
            policy[child.move] = child.visits / total_visits if total_visits > 0 else 0

        best = max(root.children, key=lambda c: c.visits)
        return best.move, policy

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand one untried move."""
        if node.untried_moves is None:
            node.untried_moves = generate_legal_moves(node.state)

        if not node.untried_moves:
            return node

        move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
        new_state = node.state.clone()
        apply_move(new_state, move)

        child = MCTSNode(state=new_state, parent=node, move=move)
        child.untried_moves = generate_legal_moves(child.state)
        node.children.append(child)
        return child

    def _rollout(self, node: MCTSNode) -> Optional[Player]:
        """Random rollout from node's state. Returns the winner (or None for draw).

        Uses pseudo-legal moves and fast apply for performance (skips check
        enforcement in random rollouts -- doesn't significantly affect value
        estimates).
        """
        state = node.state.clone()
        depth = 0

        while not state.done and depth < self.max_rollout_depth:
            moves = generate_pseudo_legal_moves(state)
            if not moves:
                break
            move = random.choice(moves)
            apply_move_fast(state, move)
            depth += 1

        return state.winner  # Player enum or None

    def _backpropagate(self, node: MCTSNode, winner: Optional[Player]):
        """Backpropagate the rollout result up the tree.

        Each node stores wins from the perspective of the player who chose
        to go to this node (i.e., the node's parent's current player).
        """
        while node is not None:
            node.visits += 1
            if winner is not None:
                if node.parent is not None:
                    # Store from parent's player's perspective
                    parent_player = node.parent.state.current_player
                    if winner == parent_player:
                        node.wins += 1.0
                    # If opponent won, wins stays 0 (loss)
                    # Draw = 0 (already handled by winner being None)
                else:
                    # Root node: store from root player's perspective
                    if winner == node.state.current_player:
                        node.wins += 1.0
            else:
                # Draw: half point
                node.wins += 0.5
            node = node.parent


def play_random_game(state: Optional[GameState] = None,
                     max_moves: int = 400) -> GameState:
    """Play a game with random moves. Useful for testing."""
    if state is None:
        state = GameState()

    move_count = 0
    while not state.done and move_count < max_moves:
        moves = generate_legal_moves(state)
        if not moves:
            break
        move = random.choice(moves)
        apply_move(state, move)
        move_count += 1

    return state
