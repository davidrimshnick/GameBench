"""Game state representation for DaveChess."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from davechess.game.board import BOARD_SIZE, STARTING_POSITIONS


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

# Base strength per piece type
BASE_STRENGTH = {
    PieceType.COMMANDER: 2,
    PieceType.WARRIOR: 1,
    PieceType.RIDER: 2,
    PieceType.BOMBARD: 0,
    PieceType.LANCER: 3,
}

# Deploy cost per piece type (Commander cannot be deployed)
DEPLOY_COST = {
    PieceType.WARRIOR: 2,
    PieceType.RIDER: 4,
    PieceType.BOMBARD: 5,
    PieceType.LANCER: 6,
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
class Deploy(Move):
    """Deploy a new piece onto the board."""
    piece_type: PieceType
    to_rc: tuple[int, int]

    def __eq__(self, other):
        if not isinstance(other, Deploy):
            return NotImplemented
        return self.piece_type == other.piece_type and self.to_rc == other.to_rc

    def __hash__(self):
        return hash(("deploy", self.piece_type, self.to_rc))


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
        return new

    def get_piece_at(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at position, or None."""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None

    def get_board_tuple(self) -> tuple:
        """Return a hashable representation of the board for state comparison.
        Includes resources — use get_position_key() for repetition detection.
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
        Excludes resources (they change every turn via income, preventing repeats).
        """
        cells = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.board[row][col]
                if cell is None:
                    cells.append(None)
                else:
                    cells.append((cell.piece_type, cell.player))
        return (tuple(cells), self.current_player)

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
