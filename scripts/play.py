#!/usr/bin/env python3
"""Interactive CLI for playing DaveChess.

Usage:
    python scripts/play.py                  # Human vs Human
    python scripts/play.py --vs-mcts 100    # Human vs MCTS-lite (100 sims)
    python scripts/play.py --mcts-vs-mcts 100 200  # MCTS vs MCTS
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import GameState, Player, MoveStep, Promote, BombardAttack
from davechess.game.rules import generate_legal_moves, apply_move, check_winner
from davechess.game.board import render_board, rc_to_notation
from davechess.game.notation import move_to_dcn, dcn_to_move


def display_state(state: GameState):
    """Print the current board state."""
    board = state.to_display_board()
    print(render_board(board,
                       resource_counts=tuple(state.resources),
                       turn=state.turn,
                       current_player=int(state.current_player)))
    print()


def list_moves(state: GameState, moves: list) -> list[str]:
    """Display numbered legal moves and return DCN strings."""
    dcn_list = []
    for i, move in enumerate(moves):
        dcn = move_to_dcn(state, move)
        dcn_list.append(dcn)
        print(f"  {i+1:3d}. {dcn}")
    return dcn_list


def human_turn(state: GameState) -> int | None:
    """Get a human player's move. Returns move index or None to quit."""
    moves = generate_legal_moves(state)
    if not moves:
        print("No legal moves!")
        return None

    player_name = "White" if state.current_player == Player.WHITE else "Black"
    print(f"\n{player_name}'s turn. Legal moves:")
    dcn_list = list_moves(state, moves)
    print(f"\nEnter move number (1-{len(moves)}), DCN notation, or 'q' to quit:")

    while True:
        inp = input("> ").strip()
        if inp.lower() == "q":
            return None

        # Try as number
        try:
            idx = int(inp) - 1
            if 0 <= idx < len(moves):
                return idx
            print(f"Invalid number. Enter 1-{len(moves)}.")
            continue
        except ValueError:
            pass

        # Try as DCN notation
        try:
            parsed = dcn_to_move(inp)
            for i, move in enumerate(moves):
                if move == parsed:
                    return i
            print("That move is not legal in this position.")
        except ValueError:
            print("Invalid input. Enter a move number or DCN notation.")


def mcts_turn(state: GameState, num_sims: int):
    """Get an MCTS-lite bot's move."""
    from davechess.engine.mcts_lite import MCTSLite

    mcts = MCTSLite(num_simulations=num_sims)
    move = mcts.search(state)
    return move


def play_game(white_type: str = "human", black_type: str = "human",
              white_sims: int = 100, black_sims: int = 100):
    """Play a full game."""
    state = GameState()

    print("=" * 60)
    print("  DaveChess")
    print("=" * 60)
    print(f"  White: {white_type}" + (f" ({white_sims} sims)" if white_type == "mcts" else ""))
    print(f"  Black: {black_type}" + (f" ({black_sims} sims)" if black_type == "mcts" else ""))
    print("=" * 60)

    while not state.done:
        display_state(state)

        current = state.current_player
        if current == Player.WHITE:
            ptype, sims = white_type, white_sims
        else:
            ptype, sims = black_type, black_sims

        moves = generate_legal_moves(state)
        if not moves:
            print("No legal moves - game over!")
            break

        if ptype == "human":
            idx = human_turn(state)
            if idx is None:
                print("Game aborted.")
                return
            move = moves[idx]
        else:
            move = mcts_turn(state, sims)
            dcn = move_to_dcn(state, move)
            player_name = "White" if current == Player.WHITE else "Black"
            print(f"{player_name} (MCTS) plays: {dcn}")

        state = apply_move(state, move)

    # Game over
    display_state(state)
    done, winner = check_winner(state)
    if winner == Player.WHITE:
        print("White wins!")
    elif winner == Player.BLACK:
        print("Black wins!")
    else:
        print("Draw!")
    print(f"Game ended on turn {state.turn} ({len(state.move_history)} moves)")


def main():
    parser = argparse.ArgumentParser(description="Play DaveChess")
    parser.add_argument("--vs-mcts", type=int, metavar="SIMS",
                        help="Play as White against MCTS-lite bot with N simulations")
    parser.add_argument("--mcts-vs-mcts", type=int, nargs=2, metavar=("W_SIMS", "B_SIMS"),
                        help="Watch MCTS vs MCTS")
    args = parser.parse_args()

    if args.mcts_vs_mcts:
        play_game("mcts", "mcts", args.mcts_vs_mcts[0], args.mcts_vs_mcts[1])
    elif args.vs_mcts:
        play_game("human", "mcts", black_sims=args.vs_mcts)
    else:
        play_game("human", "human")


if __name__ == "__main__":
    main()
