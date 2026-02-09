"""Board constants, starting positions, and text-based rendering."""

from __future__ import annotations

BOARD_SIZE = 8

# Gold nodes: give +1 resource per turn (central positions)
GOLD_NODES: list[tuple[int, int]] = [
    (3, 3), (3, 4),  # Central gold nodes
    (4, 3), (4, 4),  # Central gold nodes
]

# All nodes (just Gold nodes now — Power nodes removed in v2)
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
