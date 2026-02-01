#!/usr/bin/env python3
"""Test a simple aggressive commander-hunting strategy."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.game.state import GameState, Player, PieceType
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.heuristic_player import HeuristicPlayer
import random


class CommanderHunter:
    """Ultra-aggressive player that always tries to kill the enemy commander."""

    def get_move(self, state: GameState):
        """Get move that most threatens enemy commander."""
        moves = generate_legal_moves(state)
        if not moves:
            return None

        # Find enemy commander
        enemy_commander_pos = None
        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece and piece.piece_type == PieceType.COMMANDER:
                    if piece.player != state.current_player:
                        enemy_commander_pos = (r, c)
                        break

        if not enemy_commander_pos:
            # Commander dead? Just pick random move
            return random.choice(moves)

        # Score moves by how close they get to enemy commander
        best_moves = []
        best_score = -999

        for move in moves:
            score = 0

            # Check if this directly captures commander
            if hasattr(move, 'to_rc') and move.to_rc == enemy_commander_pos:
                return move  # Instant win!

            # Otherwise score by manhattan distance to commander
            if hasattr(move, 'to_rc'):
                to_r, to_c = move.to_rc
                cmd_r, cmd_c = enemy_commander_pos
                dist = abs(to_r - cmd_r) + abs(to_c - cmd_c)
                score = -dist  # Closer is better

                # Bonus for threatening commander (adjacent)
                if dist == 1:
                    score += 10

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(moves)


def test_commander_hunter():
    """Play games with commander hunters."""
    hunter1 = CommanderHunter()
    hunter2 = CommanderHunter()

    print("Playing 10 games with CommanderHunter vs CommanderHunter...")

    wins = {'white': 0, 'black': 0, 'draw': 0}
    lengths = []

    for i in range(10):
        state = GameState()
        moves = 0

        while not state.done and moves < 100:
            if state.current_player == Player.WHITE:
                move = hunter1.get_move(state)
            else:
                move = hunter2.get_move(state)

            if move is None:
                break

            apply_move(state, move)
            moves += 1

        lengths.append(moves)
        if state.winner == Player.WHITE:
            wins['white'] += 1
        elif state.winner == Player.BLACK:
            wins['black'] += 1
        else:
            wins['draw'] += 1

        print(f"  Game {i+1}: {moves} moves, winner: {state.winner if state.winner else 'Draw'}")

    print(f"\nResults:")
    print(f"  White: {wins['white']}")
    print(f"  Black: {wins['black']}")
    print(f"  Draws: {wins['draw']}")
    print(f"  Avg length: {sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    test_commander_hunter()