#!/usr/bin/env python3
"""Compare benchmark results from multiple models and generate charts.

Usage:
    python scripts/compare_models.py results/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> list[dict]:
    """Load all result files from a directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_results.json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                results.append(json.load(f))
    return results


def print_comparison(results: list[dict]):
    """Print a comparison table."""
    print("=" * 70)
    print(f"  {'Model':30s}  {'GameBench Score':>15s}  {'Best ELO':>10s}")
    print("=" * 70)

    for r in sorted(results, key=lambda x: x["gamebench_score"], reverse=True):
        curve = r.get("learning_curve", [])
        best_elo = max((e for _, e in curve), default=0)
        print(f"  {r['model']:30s}  {r['gamebench_score']:>14.1f}  {best_elo:>10.0f}")

    print("=" * 70)


def generate_chart(results: list[dict], output_path: str):
    """Generate a learning curve comparison chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        curve = r.get("learning_curve", [])
        if curve:
            ns = [n for n, _ in curve]
            elos = [e for _, e in curve]
            label = f"{r['model']} (Score: {r['gamebench_score']:.1f})"
            ax.plot(ns, elos, marker="o", label=label)

    ax.set_xlabel("Number of Example Games (N)")
    ax.set_ylabel("Estimated ELO")
    ax.set_title("GameBench: ELO vs Number of Examples")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Reference lines
    ax.axhline(y=400, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.axhline(y=2700, color="gray", linestyle="--", alpha=0.5, label="Max")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare GameBench results")
    parser.add_argument("results_dir", type=str, help="Directory with result JSON files")
    parser.add_argument("--chart", type=str, default=None,
                        help="Output chart path (default: results_dir/comparison.png)")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} model results\n")
    print_comparison(results)

    chart_path = args.chart or os.path.join(args.results_dir, "charts", "comparison.png")
    generate_chart(results, chart_path)


if __name__ == "__main__":
    main()
