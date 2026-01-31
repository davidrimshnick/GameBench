"""Construct prompts for LLM benchmark: rules text + N example games."""

from __future__ import annotations

from davechess.game.state import GameState, Move
from davechess.game.rules import generate_legal_moves
from davechess.game.board import RESOURCE_NODES, rc_to_notation
from davechess.game.notation import move_to_dcn, game_to_dcn
from davechess.data.storage import replay_game

RULES_TEXT = """# DaveChess Rules

## Board
8x8 grid with 8 resource nodes at fixed positions: {resource_positions}.

## Pieces
| Piece | Symbol | Move | Base Strength | Deploy Cost |
|-------|--------|------|---------------|-------------|
| Commander | C | 1-2 squares, any direction | 1 | Cannot be deployed |
| Warrior | W | 1 square, orthogonal only | 1 (+1 per adjacent friendly Warrior) | 2 resources |
| Rider | R | Up to 3 squares, straight line (no jumping) | 2 | 4 resources |
| Bombard | B | 1 square, any direction | 0 (melee) | 5 resources |

## Starting Position
White (row 1): W at c1, C at d1, R at e1, W at f1
Black (row 8): W at c8, R at d8, C at e8, W at f8

## Turn Structure
1. Gain resources: +1 per resource node you have a piece on or orthogonally adjacent to
2. One action: Move a piece OR Deploy a new piece on your back 2 rows (empty cell)

## Capture
Attacker moves onto defender. Compare total strength. Higher wins. Tie = both removed.
Bombard special: ranged capture at exactly 2 squares distance, straight line, clear path. Target is simply removed (Bombard stays). Bombard melee capture uses strength 0.

## Notation (DCN)
- Move: `Wa2-a3` (Warrior moves from a2 to a3)
- Capture: `Rb1xd3` (Rider captures piece at d3)
- Deploy: `+W@c2` (Deploy Warrior at c2)
- Bombard ranged: `Bc3~e3` (Bombard at c3 attacks target at e3)

Move numbering: `1. <White move> <Black move>  2. <White move> <Black move> ...`

## Win Conditions (checked in order)
1. Capture opponent's Commander → you win
2. Control 6+ of 8 resource nodes exclusively (you control it, opponent doesn't) → you win
3. Turn 200 → most exclusive resource nodes wins, tiebreak by piece count, then draw

## Result
- `1-0` = White wins, `0-1` = Black wins, `1/2-1/2` = Draw
"""


def get_rules_prompt() -> str:
    """Get the rules description with resource node positions filled in."""
    positions = ", ".join(rc_to_notation(r, c) for r, c in RESOURCE_NODES)
    return RULES_TEXT.format(resource_positions=positions)


def format_example_games(games: list[tuple[list[Move], str]],
                         max_games: int | None = None) -> str:
    """Format example games for inclusion in the prompt.

    Args:
        games: List of (moves, result) tuples.
        max_games: Maximum number of games to include.

    Returns:
        Formatted text with numbered example games.
    """
    if max_games is not None:
        games = games[:max_games]

    if not games:
        return ""

    sections = []
    for i, (moves, result) in enumerate(games, 1):
        states, final = replay_game(moves)
        pairs = list(zip(states, moves))
        headers = {"Game": str(i)}
        dcn = game_to_dcn(pairs, headers=headers, result=result)
        sections.append(dcn)

    return "# Example Games\n\n" + "\n\n".join(sections)


def build_system_prompt(example_games: list[tuple[list[Move], str]],
                        num_examples: int = 0) -> str:
    """Build the full system prompt with rules and examples.

    Args:
        example_games: Pool of example games to draw from.
        num_examples: Number of example games to include (0 = no examples).
    """
    prompt = get_rules_prompt()

    if num_examples > 0 and example_games:
        examples_text = format_example_games(example_games, max_games=num_examples)
        prompt += "\n\n" + examples_text

    prompt += "\n\n# Instructions\n"
    prompt += "You are playing DaveChess. On each turn, respond with ONLY your move "
    prompt += "in DCN notation (e.g., `Wa2-a3` or `+W@c2`). No explanation needed.\n"

    return prompt


def build_game_state_message(state: GameState, move_history_dcn: list[str],
                             legal_moves: list[Move]) -> str:
    """Build a user message describing the current game state.

    Args:
        state: Current game state.
        move_history_dcn: List of moves in DCN notation so far.
        legal_moves: List of legal moves available.
    """
    from davechess.game.board import render_board

    parts = []

    # Move history
    if move_history_dcn:
        # Format as numbered pairs
        move_lines = []
        for i in range(0, len(move_history_dcn), 2):
            num = i // 2 + 1
            if i + 1 < len(move_history_dcn):
                move_lines.append(f"{num}. {move_history_dcn[i]} {move_history_dcn[i+1]}")
            else:
                move_lines.append(f"{num}. {move_history_dcn[i]}")
        parts.append("Game so far:\n" + "\n".join(move_lines))

    # Board state
    board = state.to_display_board()
    board_text = render_board(board, tuple(state.resources), state.turn,
                              int(state.current_player))
    parts.append(f"Current position:\n```\n{board_text}\n```")

    # Legal moves summary
    player = "White" if state.current_player == 0 else "Black"
    legal_dcn = []
    for m in legal_moves:
        legal_dcn.append(move_to_dcn(state, m))
    parts.append(f"You are {player}. Legal moves: {', '.join(legal_dcn[:30])}"
                 + (f"... ({len(legal_dcn)} total)" if len(legal_dcn) > 30 else ""))
    parts.append("Your move:")

    return "\n\n".join(parts)
