#!/usr/bin/env python3
"""Quick balance check for DaveChess."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.data.generator import MCTSLiteAgent, play_game
from davechess.game.state import Player
import time

def test_balance(num_games: int = 50, sims: int = 50):
    """Run games and check basic balance."""
    agent = MCTSLiteAgent(num_simulations=sims)

    white_wins = 0
    black_wins = 0
    draws = 0
    lengths = []

    print(f"Running {num_games} games with {sims} simulations...")
    start = time.time()

    for i in range(num_games):
        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{num_games}...")

        moves, winner, _ = play_game(agent, agent, max_moves=200)
        lengths.append(len(moves))

        if winner == Player.WHITE:
            white_wins += 1
        elif winner == Player.BLACK:
            black_wins += 1
        else:
            draws += 1

    elapsed = time.time() - start

    print(f"\n📊 Results ({num_games} games @ {sims} sims):")
    print(f"  White: {white_wins} ({100*white_wins/num_games:.1f}%)")
    print(f"  Black: {black_wins} ({100*black_wins/num_games:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/num_games:.1f}%)")
    print(f"  Avg length: {sum(lengths)/len(lengths):.1f} moves")
    print(f"  Max length games: {sum(1 for l in lengths if l >= 200)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")

    balance = 1 - abs(white_wins - black_wins) / num_games
    print(f"\n  Balance score: {balance:.2f} (1.0 = perfect)")

    if balance < 0.6:
        print(f"  ⚠️  SEVERE IMBALANCE!")
    if sum(lengths)/len(lengths) > 180:
        print(f"  ⚠️  Games too long!")
    if sum(1 for l in lengths if l >= 200) > num_games * 0.3:
        print(f"  ⚠️  Too many max-length games!")

if __name__ == "__main__":
    # Quick test with different configs
    configs = [
        (30, 10),
        (30, 25),
        (30, 50),
        (20, 100),
    ]

    for num_games, sims in configs:
        print(f"\n{'='*50}")
        test_balance(num_games, sims)