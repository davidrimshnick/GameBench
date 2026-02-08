"""Construct prompts for LLM benchmark: rules text + N example games."""

from __future__ import annotations

from davechess.game.state import GameState, Move
from davechess.game.rules import generate_legal_moves
from davechess.game.board import GOLD_NODES, rc_to_notation
from davechess.game.notation import move_to_dcn, game_to_dcn
from davechess.data.storage import replay_game

RULES_TEXT = """# DaveChess Rules

## Board
8x8 grid with 4 Gold nodes (resource income) at {gold_positions}.

## Pieces
| Piece | Symbol | Move | Capture | Deploy Cost |
|-------|--------|------|---------|-------------|
| Commander | C | 1 square, any direction | Same as move | Cannot be deployed |
| Warrior | W | 1 square forward | 1 square diagonal-forward | 2 resources |
| Rider | R | Up to 2 squares, any straight line (no jumping) | Same as move | 3 resources |
| Bombard | B | 1 square, any direction | Melee: same as move. Ranged: exactly 2 squares, straight line, clear path (stays in place, cannot target Commanders) | 4 resources |
| Lancer | L | Up to 4 squares diagonal, can jump one piece | Same as move | 5 resources |

## Starting Position
White (rows 1-2): W at c1, C at d1, R at e1, W at f1, W at d2, W at e2
Black (rows 7-8): W at c8, C at d8, R at e8, W at f8, W at d7, W at e7

## Turn Structure
1. Gain resources: +1 per Gold node you have a piece on or orthogonally adjacent to
2. One action: Move a piece OR Deploy a new piece on your back 2 rows (empty cell)

## Capture
Attacker moves onto defender's square. The defender is removed, the attacker takes its place. Any piece can capture any piece (like chess).
Bombard ranged: attacks at exactly 2 squares distance, straight line, clear path. Target is removed, Bombard stays in place. Cannot target Commanders with ranged attacks.

## Warriors
Warriors move 1 square forward (toward row 8 for White, toward row 1 for Black). They capture 1 square diagonally forward — like chess pawns. Warriors cannot move backward or sideways.

## Lancer
The Lancer moves diagonally up to 4 squares. It can jump over exactly one piece (friendly or enemy) in its path. It captures by landing on an enemy piece. It cannot move orthogonally.

## Notation (DCN)
- Move: `Wa2-a3` (Warrior moves from a2 to a3)
- Capture: `Rb1xd3` (Rider captures piece at d3)
- Deploy: `+W@c2` (Deploy Warrior at c2)
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
3. Threefold repetition of the same board position → draw
4. 50-move rule: 50 moves per side with no capture or deploy → draw

## Result
- `1-0` = White wins, `0-1` = Black wins, `1/2-1/2` = Draw
"""


def get_rules_prompt() -> str:
    """Get the rules description with node positions filled in."""
    gold_pos = ", ".join(rc_to_notation(r, c) for r, c in GOLD_NODES)
    return RULES_TEXT.format(gold_positions=gold_pos)


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


def build_agentic_system_prompt(token_budget: int) -> str:
    """Build system prompt for the agentic benchmark.

    Includes rules, budget info, and instructions about available tools.
    """
    rules = get_rules_prompt()
    return f"""{rules}

---

You are an AI agent learning to play DaveChess. Your goal is to become as strong \
a player as possible within your token budget.

**Token budget:** {token_budget:,} tokens (input + output combined). All API calls \
count against this budget, including during the evaluation phase.

**Available tools:**
- `study_games(n)`: Retrieve up to N grandmaster games from the library (max 20 per call)
- `start_practice_game(opponent_elo)`: Start a practice game at chosen difficulty (400-2700 ELO)
- `play_move(game_id, move_dcn)`: Make a move in DCN notation. The opponent responds immediately.
- `get_game_state(game_id)`: Check the current board, legal moves, and history of a game

**Strategy is up to you.** You might study games first, practice against weak opponents, \
then work up to stronger ones — or any other approach you think will maximize your skill.

**Important:** Your conversation history is limited to recent messages. If you want to \
remember strategies, patterns, or notes across many turns, use your own note-taking \
capabilities to persist information externally.

When your budget is nearly exhausted, you will enter an **evaluation phase** where you \
play rated games to determine your final ELO score. Evaluation tokens come from the same budget."""
