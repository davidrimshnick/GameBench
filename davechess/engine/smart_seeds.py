"""Smart seed game generation using heuristic strategies."""

import time
import random
import numpy as np
from typing import List, Tuple
from davechess.game.state import GameState, Player, Move, PieceType
from davechess.game.rules import apply_move, generate_legal_moves
from davechess.engine.heuristic_player import HeuristicPlayer, SmartMCTS
from davechess.engine.network import state_to_planes, move_to_policy_index, POLICY_SIZE
from davechess.engine.selfplay import ReplayBuffer


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
                score = -dist

                # Bonus for threatening
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


def generate_smart_seeds(num_games: int = 50, verbose: bool = False) -> ReplayBuffer:
    """Generate seed games using smart heuristic strategies."""
    buffer = ReplayBuffer(max_size=100000)

    # Mostly CommanderHunter matchups since those actually finish
    players = [
        ("Commander Hunter", CommanderHunter()),
        ("Commander Hunter", CommanderHunter()),
        ("Commander Hunter", CommanderHunter()),
        ("Aggressive", HeuristicPlayer(exploration=0.05, aggression=0.95)),
    ]

    game_count = 0
    total_positions = 0
    skipped = 0

    if verbose:
        print(f"Generating {num_games} smart seed games...")

    while game_count < num_games:
        white_name, white_player = random.choice(players)
        black_name, black_player = random.choice(players)

        # Play game
        moves, winner, states = play_heuristic_game(white_player, black_player)

        # Skip games that hit max length
        if len(moves) >= 100:
            skipped += 1
            if verbose:
                print(f"  [skip {skipped}] {white_name} vs {black_name}: {len(moves)} moves (max length) | {game_count}/{num_games} done", flush=True)
            continue

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

        if verbose:
            winner_str = "White" if winner == Player.WHITE else "Black" if winner == Player.BLACK else "Draw"
            print(f"  [game {game_count}/{num_games}] {white_name} vs {black_name}: {len(moves)} moves, {winner_str} | {total_positions} positions, {skipped} skipped", flush=True)

    if verbose:
        print(f"Complete: {game_count} games, {total_positions} positions, {skipped} skipped")

    return buffer