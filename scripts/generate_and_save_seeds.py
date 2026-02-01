#!/usr/bin/env python3
"""Generate and save smart seed games for training."""

import sys
import os
import pickle
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from davechess.engine.smart_seeds import generate_smart_seeds


def main():
    parser = argparse.ArgumentParser(description="Generate and save smart seed games")
    parser.add_argument("--num-games", type=int, default=150,
                        help="Number of games to generate (default: 150)")
    parser.add_argument("--output", type=str, default="checkpoints/smart_seeds.pkl",
                        help="Output file path (default: checkpoints/smart_seeds.pkl)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing file if it exists")
    args = parser.parse_args()

    # Check if file exists
    if os.path.exists(args.output) and not args.force:
        print(f"File {args.output} already exists. Use --force to overwrite.")
        print(f"Current file size: {os.path.getsize(args.output) / (1024*1024):.1f} MB")

        # Load and show stats
        with open(args.output, 'rb') as f:
            buffer = pickle.load(f)
        print(f"Contains {len(buffer)} positions from seed games")
        return

    # Generate seeds
    print(f"Generating {args.num_games} smart seed games...")
    buffer = generate_smart_seeds(args.num_games, verbose=True)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(buffer, f)

    file_size = os.path.getsize(args.output) / (1024*1024)
    print(f"Saved {len(buffer)} positions ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()