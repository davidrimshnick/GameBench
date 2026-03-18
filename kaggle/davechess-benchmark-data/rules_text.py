"""Construct prompts for LLM benchmark: rules text + N example games.

Extracted from davechess.benchmark.prompt for standalone use in the
Kaggle benchmark dataset.  Imports are written against the
``davechess_engine`` compiled package; a fallback to the source
``davechess`` tree is provided for local development.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports – prefer the compiled Kaggle package, fall back to source tree
# ---------------------------------------------------------------------------

try:
    from davechess_engine import (
        GameState,
        Move,
        generate_legal_moves,
        apply_move,
        GOLD_NODES,
        rc_to_notation,
        render_board,
        move_to_dcn,
        game_to_dcn,
    )
except ImportError:
    # Development fallback: import from the source davechess package
    from davechess.game.state import GameState, Move
    from davechess.game.rules import generate_legal_moves, apply_move
    from davechess.game.board import GOLD_NODES, rc_to_notation, render_board
    from davechess.game.notation import move_to_dcn, game_to_dcn


# ---------------------------------------------------------------------------
# replay_game  (inlined from davechess.data.storage to avoid extra dep)
# ---------------------------------------------------------------------------

def replay_game(moves: list[Move]) -> tuple[list[GameState], GameState]:
    """Replay a sequence of moves from the initial position.

    Returns:
        (states_before_each_move, final_state)
    """
    states: list[GameState] = []
    state = GameState()
    for move in moves:
        states.append(state.clone())
        apply_move(state, move)
    return states, state


# ---------------------------------------------------------------------------
# Rules text
# ---------------------------------------------------------------------------

RULES_TEXT = """# DaveChess Rules

## Board
8x8 grid with 4 Gold nodes (resource income) at {gold_positions}.

## Pieces
| Piece | Symbol | Move | Capture | Promotion Cost |
|-------|--------|------|---------|----------------|
| Commander | C | 1 square, any direction | Same as move | Cannot promote |
| Warrior | W | 1 square forward | 1 square diagonal-forward | Base piece |
| Rider | R | Up to 7 squares orthogonal, up to 3 squares diagonal (no jumping) | Same as move | 3 resources |
| Bombard | B | 1 square, any direction | Melee: same as move. Ranged: exactly 2 squares, straight line, clear path (stays in place, cannot target Commanders) | 5 resources |
| Lancer | L | Up to 7 squares any direction, can jump one piece | Same as move | 7 resources |

## Starting Position
Each side starts with 12 pieces on their back two rows:
White (rows 1-2): R at b1, B at c1, R at d1, C at e1, R at f1, B at g1; Warriors at b2-g2
Black (rows 7-8): R at b8, B at c8, R at d8, C at e8, R at f8, B at g8; Warriors at b7-g7

## Turn Structure
1. Gain resources: +1 per Gold node you have a piece directly on
2. One action: Move a piece OR Promote a piece (upgrade it in place by spending resources)

## Promotion
Spend resources to upgrade any non-Commander piece to a higher type, in place. The piece stays on its square and changes type. Cost = full price of the target type. Any piece can promote to Rider (3), Bombard (5), or Lancer (7). You cannot promote a Commander.

## Capture
Attacker moves onto defender's square. The defender is removed, the attacker takes its place. Any piece can capture any piece (like chess).
Bombard ranged: attacks at exactly 2 squares distance, straight line, clear path. Target is removed, Bombard stays in place. Cannot target Commanders with ranged attacks.

## Warriors
Warriors move 1 square forward (toward row 8 for White, toward row 1 for Black). They capture 1 square diagonally forward — like chess pawns. Warriors cannot move backward or sideways.

## Lancer
The Lancer moves up to 7 squares in any straight line (orthogonal or diagonal). It can jump over exactly one piece (friendly or enemy) in its path. It captures by landing on an enemy piece.

## Notation (DCN)
- Move: `Wa2-a3` (Warrior moves from a2 to a3)
- Capture: `Rb1xd3` (Rider captures piece at d3)
- Promote: `Wa1>R` (Warrior at a1 promotes to Rider)
- Bombard ranged: `Bc3~e3` (Bombard at c3 attacks target at e3)

Move numbering: `1. <White move> <Black move>  2. <White move> <Black move> ...`

## Check
If your Commander is under attack (an opponent piece could capture it), you are in check.
You MUST resolve check on your turn (move Commander, block, or capture the attacker).
If you cannot resolve check, it is checkmate and you lose.
You cannot make a move that leaves your own Commander in check.

## Win Conditions
1. Checkmate opponent's Commander (they have no legal move to escape check) → you win
2. Turn 100 with no checkmate → draw
3. Threefold repetition of the same position (board + side to move + promotion-affordability buckets) → draw
4. 50-move rule: 50 moves per side with no capture or promotion → draw

## Result
- `1-0` = White wins, `0-1` = Black wins, `1/2-1/2` = Draw

## Benchmark Rules
You must reason about each move yourself. Do NOT write scripts, game engines,
search algorithms (e.g. minimax, MCTS), or any automated move-selection code.
The benchmark measures YOUR strategic reasoning ability, not your ability to
write a chess engine. Pick your moves by studying the games, learning the
patterns, and thinking through the position.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rules_prompt() -> str:
    """Get the rules description with gold-node positions filled in."""
    gold_pos = ", ".join(rc_to_notation(r, c) for r, c in GOLD_NODES)
    return RULES_TEXT.format(gold_positions=gold_pos)


def format_example_games(
    games: list[tuple[list[Move], str]],
    max_games: int | None = None,
) -> str:
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


def build_system_prompt(
    example_games: list[tuple[list[Move], str]],
    num_examples: int = 0,
) -> str:
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
    prompt += "in DCN notation (e.g., `Wa2-a3` or `Wa1>R`). No explanation needed.\n"

    return prompt


def build_game_state_message(
    state: GameState,
    move_history_dcn: list[str],
    legal_moves: list[Move],
) -> str:
    """Build a user message describing the current game state.

    Args:
        state: Current game state.
        move_history_dcn: List of moves in DCN notation so far.
        legal_moves: List of legal moves available.
    """
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
