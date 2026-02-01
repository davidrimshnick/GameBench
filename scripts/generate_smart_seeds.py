#!/usr/bin/env python3
"""Generate high-quality seed games using heuristic players."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import time
from typing import List, Tuple

from davechess.game.state import GameState, Player, Move, PieceType
from davechess.game.rules import apply_move, generate_legal_moves
from davechess.engine.heuristic_player import HeuristicPlayer, SmartMCTS
from davechess.engine.network import state_to_planes, move_to_policy_index, POLICY_SIZE
from davechess.engine.selfplay import ReplayBuffer
import random


class CommanderHunter:
    """Ultra-aggressive player that hunts enemy commander."""

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
            # Commander dead, pick random
            return random.choice(moves)

        # Score moves by distance to enemy commander
        best_moves = []
        best_score = -999

        for move in moves:
            score = 0

            # Check if this directly captures commander
            if hasattr(move, 'to_rc') and move.to_rc == enemy_commander_pos:
                return move  # Instant win!

            # Score by manhattan distance
            if hasattr(move, 'to_rc'):
                to_r, to_c = move.to_rc
                cmd_r, cmd_c = enemy_commander_pos
                dist = abs(to_r - cmd_r) + abs(to_c - cmd_c)
                score = -dist  # Closer is better

                # Big bonus for threatening commander
                if dist == 1:
                    score += 10

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(moves)


def play_heuristic_game(white_player, black_player, max_moves: int = 100) -> Tuple[List[Move], Player, List[GameState]]:
    """Play a game between two heuristic players."""
    state = GameState()
    moves = []
    states = []

    for turn in range(max_moves):
        states.append(state.clone())

        # Get move from current player
        if state.current_player == Player.WHITE:
            move = white_player.get_move(state)
        else:
            move = black_player.get_move(state)

        # If no legal moves, game ends
        if move is None:
            break

        moves.append(move)
        apply_move(state, move)

        if state.done:
            break

    return moves, state.winner, states


def generate_smart_seeds(num_games: int = 50, verbose: bool = True) -> ReplayBuffer:
    """Generate seed games using different heuristic strategies."""
    buffer = ReplayBuffer(max_size=100000)

    # Mix of strategies for diverse games
    players = [
        ("Commander Hunter", CommanderHunter()),
        ("Aggressive", HeuristicPlayer(exploration=0.05, aggression=0.9)),
        ("Balanced", HeuristicPlayer(exploration=0.1, aggression=0.6)),
        ("Smart MCTS", SmartMCTS(num_simulations=15)),  # Fast MCTS
    ]

    game_count = 0
    total_positions = 0
    start_time = time.time()

    # Statistics
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []

    if verbose:
        print(f"Generating {num_games} smart seed games...")
        print("-" * 60)

    while game_count < num_games:
        # Pick two players (can be same type)
        import random
        white_name, white_player = random.choice(players)
        black_name, black_player = random.choice(players)

        if verbose and (game_count + 1) % 10 == 0:
            print(f"Game {game_count + 1}/{num_games}: {white_name} vs {black_name}")

        # Play game
        moves, winner, states = play_heuristic_game(white_player, black_player)
        game_lengths.append(len(moves))

        # Skip games that hit max length (likely stalemates)
        if len(moves) >= 100:
            if verbose:
                print(f"  Skipping max-length game")
            continue

        # Update statistics
        if winner == Player.WHITE:
            white_wins += 1
        elif winner == Player.BLACK:
            black_wins += 1
        else:
            draws += 1

        # Convert to training examples
        for i, (state, move) in enumerate(zip(states, moves)):
            # Create planes
            planes = state_to_planes(state)

            # Create policy (one-hot for actual move)
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            policy[move_to_policy_index(move)] = 1.0

            # Create value from game outcome - only wins get 1.0
            if winner == state.current_player:
                value = 1.0  # Win
            else:
                value = 0.0  # Loss or draw

            buffer.push(planes, policy, value)
            total_positions += 1

        game_count += 1

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print("SEED GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Games generated: {game_count}")
        print(f"Training positions: {total_positions}")
        print(f"Time: {elapsed:.1f}s ({elapsed/game_count:.2f}s per game)")

        print(f"\nGame Statistics:")
        print(f"  White wins: {white_wins} ({100*white_wins/game_count:.1f}%)")
        print(f"  Black wins: {black_wins} ({100*black_wins/game_count:.1f}%)")
        print(f"  Draws: {draws} ({100*draws/game_count:.1f}%)")

        print(f"\nGame Lengths:")
        print(f"  Average: {np.mean(game_lengths):.1f} moves")
        print(f"  Min: {min(game_lengths)} moves")
        print(f"  Max: {max(game_lengths)} moves")

        balance = 1 - abs(white_wins - black_wins) / game_count
        print(f"\nBalance score: {balance:.2f} (1.0 = perfect)")

    return buffer


def main():
    """Test the smart seed generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate smart seed games")
    parser.add_argument("--num-games", type=int, default=50,
                        help="Number of games to generate")
    parser.add_argument("--save", type=str, default=None,
                        help="Save buffer to file")
    args = parser.parse_args()

    buffer = generate_smart_seeds(args.num_games)

    if args.save:
        # Save the buffer
        import pickle
        with open(args.save, 'wb') as f:
            pickle.dump(buffer, f)
        print(f"\nSaved {len(buffer)} positions to {args.save}")


if __name__ == "__main__":
    main()