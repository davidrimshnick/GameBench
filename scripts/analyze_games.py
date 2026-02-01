#!/usr/bin/env python3
"""Analyze game balance and statistics from seed games."""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.data.generator import MCTSLiteAgent, play_game
from davechess.game.state import GameState, Player, PieceType, MoveStep, Deploy
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.board import ALL_NODES

def analyze_games(num_games: int = 100, simulations: int = 50):
    """Run games and collect statistics."""
    agent = MCTSLiteAgent(num_simulations=simulations)

    stats = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'game_lengths': [],
        'win_types': defaultdict(int),
        'first_capture_turn': [],
        'commander_captures': defaultdict(int),
        'resource_control': [],
        'piece_survival': defaultdict(list),
    }

    print(f"\nRunning {num_games} games with {simulations} simulations each...")
    print("-" * 60)

    for game_idx in range(num_games):
        if (game_idx + 1) % 10 == 0:
            print(f"Game {game_idx + 1}/{num_games}...")

        # Play a game
        moves, winner, turns = play_game(agent, agent, max_moves=200)

        # Update basic stats
        stats['game_lengths'].append(len(moves))

        if winner == Player.WHITE:
            stats['white_wins'] += 1
        elif winner == Player.BLACK:
            stats['black_wins'] += 1
        else:
            stats['draws'] += 1

        # Replay game to analyze details
        state = GameState()
        first_capture = None
        commander_white_alive = True
        commander_black_alive = True
        resource_counts = []

        for turn, move in enumerate(moves):
            # Handle different move types

            if isinstance(move, MoveStep):
                # Regular move
                from_r, from_c = move.from_rc
                to_r, to_c = move.to_rc

                # Check for first capture
                if first_capture is None and state.board[to_r][to_c] is not None:
                    first_capture = turn

                # Check commander survival
                piece = state.board[from_r][from_c]
                if piece and piece.piece_type == PieceType.COMMANDER:  # Commander move
                    target = state.board[to_r][to_c]
                    if target is not None:
                        if state.current_player == Player.WHITE:
                            stats['commander_captures']['white'] += 1
                        else:
                            stats['commander_captures']['black'] += 1
            elif isinstance(move, Deploy):
                # Deployment move - no capture possible
                pass

            # Count resources
            white_res = 0
            black_res = 0
            for r in range(8):
                for c in range(8):
                    piece = state.board[r][c]
                    # Resource nodes are location-based, not pieces
                    # Check if this is a resource location
                    if (r, c) in ALL_NODES:
                        # Check which player controls it
                        if piece:
                            if piece.player == Player.WHITE:
                                white_res += 1
                            else:
                                black_res += 1
            resource_counts.append((white_res, black_res))

            apply_move(state, move)

            # Check if commander died
            commander_found = [False, False]
            for r in range(8):
                for c in range(8):
                    piece = state.board[r][c]
                    if piece and piece.piece_type == PieceType.COMMANDER:
                        if piece.player == Player.WHITE:
                            commander_found[0] = True
                        else:
                            commander_found[1] = True

            if commander_white_alive and not commander_found[0]:
                commander_white_alive = False
                stats['piece_survival']['white_commander'].append(turn)
            if commander_black_alive and not commander_found[1]:
                commander_black_alive = False
                stats['piece_survival']['black_commander'].append(turn)

        if first_capture:
            stats['first_capture_turn'].append(first_capture)

        # Determine win type
        if winner is not None:
            if not commander_white_alive or not commander_black_alive:
                stats['win_types']['commander_death'] += 1
            else:
                stats['win_types']['unknown'] += 1

        stats['resource_control'].append(resource_counts[-1] if resource_counts else (0, 0))

    return stats

def print_analysis(stats, label=""):
    """Print analysis results."""
    total_games = stats['white_wins'] + stats['black_wins'] + stats['draws']

    print(f"\n{'='*60}")
    print(f"GAME BALANCE ANALYSIS {label}")
    print(f"{'='*60}")

    # Win rates
    print(f"\n📊 Win Rates:")
    print(f"  White wins: {stats['white_wins']:3d} ({100*stats['white_wins']/total_games:.1f}%)")
    print(f"  Black wins: {stats['black_wins']:3d} ({100*stats['black_wins']/total_games:.1f}%)")
    print(f"  Draws:      {stats['draws']:3d} ({100*stats['draws']/total_games:.1f}%)")

    # Balance score (0 = perfect balance, 1 = total imbalance)
    balance = abs(stats['white_wins'] - stats['black_wins']) / total_games
    print(f"  Balance score: {1-balance:.2f} (1.0 = perfect)")

    # Game lengths
    lengths = stats['game_lengths']
    print(f"\n📏 Game Lengths:")
    print(f"  Min:    {min(lengths):3d} moves")
    print(f"  Max:    {max(lengths):3d} moves")
    print(f"  Mean:   {np.mean(lengths):3.1f} moves")
    print(f"  Median: {np.median(lengths):3.0f} moves")
    print(f"  At max (200): {sum(1 for l in lengths if l >= 200)} games")

    # First capture timing
    if stats['first_capture_turn']:
        captures = stats['first_capture_turn']
        print(f"\n⚔️  First Capture:")
        print(f"  Mean turn:   {np.mean(captures):3.1f}")
        print(f"  Median turn: {np.median(captures):3.0f}")

    # Win conditions
    print(f"\n🏆 Win Conditions:")
    for win_type, count in stats['win_types'].items():
        if count > 0:
            print(f"  {win_type}: {count} ({100*count/(total_games-stats['draws']):.1f}%)")

    # Commander activity
    print(f"\n👑 Commander Captures:")
    print(f"  White commander: {stats['commander_captures'].get('white', 0)} captures")
    print(f"  Black commander: {stats['commander_captures'].get('black', 0)} captures")

    # Commander survival
    if stats['piece_survival'].get('white_commander'):
        white_surv = stats['piece_survival']['white_commander']
        print(f"\n💀 Commander Deaths:")
        print(f"  White commander died in {len(white_surv)} games")
        if white_surv:
            print(f"    Average death turn: {np.mean(white_surv):.1f}")
    if stats['piece_survival'].get('black_commander'):
        black_surv = stats['piece_survival']['black_commander']
        print(f"  Black commander died in {len(black_surv)} games")
        if black_surv:
            print(f"    Average death turn: {np.mean(black_surv):.1f}")

    # Resource control
    final_resources = stats['resource_control']
    white_res = [w for w, b in final_resources]
    black_res = [b for w, b in final_resources]
    print(f"\n💎 Final Resource Control:")
    print(f"  White avg: {np.mean(white_res):.1f} nodes")
    print(f"  Black avg: {np.mean(black_res):.1f} nodes")

    # Warnings
    print(f"\n⚠️  Potential Issues:")
    if balance < 0.6:
        print(f"  - SEVERE IMBALANCE: {max(stats['white_wins'], stats['black_wins'])/total_games:.0%} win rate")
    if np.mean(lengths) > 180:
        print(f"  - Games too long (avg {np.mean(lengths):.0f} moves)")
    if sum(1 for l in lengths if l >= 200) > total_games * 0.3:
        print(f"  - Too many games hitting max length ({sum(1 for l in lengths if l >= 200)})")
    if stats['commander_captures']['white'] + stats['commander_captures']['black'] < total_games * 0.1:
        print(f"  - Commanders too passive (only {stats['commander_captures']['white'] + stats['commander_captures']['black']} total captures)")

def main():
    """Run multiple analyses with different configurations."""

    # Test with different simulation counts
    configs = [
        (50, 10, "Low quality (10 sims)"),
        (50, 25, "Medium quality (25 sims)"),
        (50, 50, "High quality (50 sims)"),
        (30, 100, "Very high quality (100 sims)"),
    ]

    all_stats = []
    for num_games, sims, label in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")
        start = time.time()
        stats = analyze_games(num_games, sims)
        elapsed = time.time() - start
        print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
        print_analysis(stats, label)
        all_stats.append((label, stats))

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'White%':>7} {'Black%':>7} {'Draw%':>6} {'AvgLen':>7} {'Balance':>8}")
    print("-" * 60)

    for label, stats in all_stats:
        total = stats['white_wins'] + stats['black_wins'] + stats['draws']
        white_pct = 100 * stats['white_wins'] / total
        black_pct = 100 * stats['black_wins'] / total
        draw_pct = 100 * stats['draws'] / total
        avg_len = np.mean(stats['game_lengths'])
        balance = 1 - abs(stats['white_wins'] - stats['black_wins']) / total

        print(f"{label:<25} {white_pct:6.1f}% {black_pct:6.1f}% {draw_pct:5.1f}% {avg_len:7.1f} {balance:8.2f}")

if __name__ == "__main__":
    main()