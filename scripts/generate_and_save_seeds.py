#!/usr/bin/env python3
"""Generate and save smart seed games for training.

Generates two types of seeds:
1. Heuristic games (CommanderHunter vs Aggressive) - full games with middlegame patterns
2. Endgame seeds (R+C vs C, L+C vs C, 2R+C vs C) - MCTS-solved mating sequences
"""

import sys
import os
import pickle
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.engine.smart_seeds import generate_smart_seeds, generate_endgame_seeds


def main():
    parser = argparse.ArgumentParser(description="Generate and save smart seed games")
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of heuristic games to generate (default: 200)")
    parser.add_argument("--num-endgames", type=int, default=200,
                        help="Number of endgame wins to generate (default: 200)")
    parser.add_argument("--endgame-sims", type=int, default=400,
                        help="MCTS simulations per move in endgames (default: 400)")
    parser.add_argument("--output", type=str, default="checkpoints/smart_seeds.pkl",
                        help="Output file path (default: checkpoints/smart_seeds.pkl)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing file if it exists")
    parser.add_argument("--endgames-only", action="store_true",
                        help="Only generate endgame seeds (skip heuristic games)")
    parser.add_argument("--append", action="store_true",
                        help="Append new endgame seeds to existing pickle file")
    args = parser.parse_args()

    # Append mode: load existing, add more endgames
    if args.append:
        if not os.path.exists(args.output):
            print(f"Error: {args.output} does not exist. Cannot append.")
            return
        with open(args.output, 'rb') as f:
            buffer = pickle.load(f)
        existing_count = len(buffer)
        print(f"Loaded existing seeds: {existing_count} positions ({os.path.getsize(args.output) / (1024*1024):.1f} MB)")

        print(f"\n=== Generating {args.num_endgames} additional endgame wins ({args.endgame_sims} sims) ===")
        endgame_buffer = generate_endgame_seeds(
            num_positions=args.num_endgames,
            mcts_sims=args.endgame_sims,
            verbose=True,
        )
        endgame_count = len(endgame_buffer)

        for i in range(endgame_count):
            buffer.push(
                endgame_buffer.planes[i],
                endgame_buffer.policies[i],
                endgame_buffer.values[i],
            )
        del endgame_buffer

        total = len(buffer)
        print(f"\n=== Summary ===")
        print(f"Existing positions:     {existing_count}")
        print(f"New endgame positions:  {endgame_count}")
        print(f"Total positions:        {total}")

    else:
        # Check if file exists
        if os.path.exists(args.output) and not args.force:
            print(f"File {args.output} already exists. Use --force to overwrite.")
            print(f"Current file size: {os.path.getsize(args.output) / (1024*1024):.1f} MB")

            # Load and show stats
            with open(args.output, 'rb') as f:
                buffer = pickle.load(f)
            print(f"Contains {len(buffer)} positions from seed games")
            return

        # Generate heuristic seeds
        if not args.endgames_only:
            print(f"=== Phase 1: Generating {args.num_games} heuristic games ===")
            buffer = generate_smart_seeds(args.num_games, verbose=True)
            heuristic_count = len(buffer)
            print(f"\nHeuristic seeds: {heuristic_count} positions")
        else:
            from davechess.engine.selfplay import ReplayBuffer
            buffer = ReplayBuffer(max_size=200000)
            heuristic_count = 0

        # Generate endgame seeds
        print(f"\n=== Phase 2: Generating {args.num_endgames} endgame wins ({args.endgame_sims} sims) ===")
        endgame_buffer = generate_endgame_seeds(
            num_positions=args.num_endgames,
            mcts_sims=args.endgame_sims,
            verbose=True,
        )
        endgame_count = len(endgame_buffer)

        # Merge endgame seeds into main buffer
        for i in range(endgame_count):
            buffer.push(
                endgame_buffer.planes[i],
                endgame_buffer.policies[i],
                endgame_buffer.values[i],
            )
        del endgame_buffer

        total = len(buffer)
        print(f"\n=== Summary ===")
        print(f"Heuristic positions: {heuristic_count}")
        print(f"Endgame positions:   {endgame_count}")
        print(f"Total positions:     {total}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(buffer, f)

    file_size = os.path.getsize(args.output) / (1024*1024)
    print(f"Saved {total} positions ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
