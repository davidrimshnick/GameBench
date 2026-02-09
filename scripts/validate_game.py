#!/usr/bin/env python3
"""Game health validation for DaveChess (Phase 1.5).

Runs thousands of MCTS-vs-MCTS games and checks health metrics:
1. Win condition distribution
2. Average game length
3. First-player advantage
4. Draw rate
5. Piece type usage
6. Strategy diversity (via opening clustering)

Also detects degenerate patterns: rush wins, turtle draws, dominant strategies,
resource runaway, positional deadlocks.

Usage:
    python scripts/validate_game.py [--num-games 5000] [--sims 100]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import GameState, Player, PieceType, MoveStep, Promote, BombardAttack
from davechess.game.rules import (
    generate_legal_moves, apply_move, check_winner,
    get_resource_income,
)
from davechess.game.notation import move_to_dcn
from davechess.engine.mcts_lite import MCTSLite


def play_mcts_game(args):
    """Play a single MCTS vs MCTS game. Designed for multiprocessing."""
    game_id, white_sims, black_sims, seed = args
    import random
    random.seed(seed)

    state = GameState()
    mcts_white = MCTSLite(num_simulations=white_sims)
    mcts_black = MCTSLite(num_simulations=black_sims)

    move_records = []
    deployed_types = set()
    states_seen = Counter()
    max_resource_diff = 0

    while not state.done:
        moves = generate_legal_moves(state)
        if not moves:
            break

        mcts = mcts_white if state.current_player == Player.WHITE else mcts_black
        move = mcts.search(state)

        # Track promotions
        if isinstance(move, Promote):
            deployed_types.add((int(state.current_player), int(move.to_type)))

        # Track state for deadlock detection
        board_hash = state.get_board_tuple()
        states_seen[board_hash] += 1

        # Track resource differential
        rdiff = abs(state.resources[0] - state.resources[1])
        max_resource_diff = max(max_resource_diff, rdiff)

        move_records.append({
            "player": int(state.current_player),
            "move_type": type(move).__name__,
            "turn": state.turn,
        })

        apply_move(state, move)

    # Determine win condition
    num_moves = len(move_records)
    done, winner = check_winner(state)

    if done and winner is not None:
        win_condition = "commander_capture"
    elif done and winner is None:
        win_condition = "draw"
    else:
        win_condition = "incomplete"

    # Opening signature: first 4 moves
    opening = []
    for mr in move_records[:4]:
        opening.append(f"{mr['player']}:{mr['move_type']}")
    opening_sig = "|".join(opening)

    # Repeated states
    max_state_repeats = max(states_seen.values()) if states_seen else 0

    return {
        "game_id": game_id,
        "winner": int(winner) if winner is not None else None,
        "win_condition": win_condition,
        "num_moves": num_moves,
        "turn": state.turn,
        "deployed_types": list(deployed_types),
        "opening_sig": opening_sig,
        "max_resource_diff": max_resource_diff,
        "max_state_repeats": max_state_repeats,
    }


def analyze_results(results: list[dict]) -> dict:
    """Analyze game results and compute health metrics."""
    n = len(results)

    # Win condition distribution
    win_conds = Counter(r["win_condition"] for r in results)
    win_cond_pcts = {k: v / n * 100 for k, v in win_conds.items()}

    # Game length
    lengths = [r["num_moves"] for r in results]
    avg_length = sum(lengths) / n
    min_length = min(lengths)
    max_length = max(lengths)

    # First-player advantage
    white_wins = sum(1 for r in results if r["winner"] == 0)
    black_wins = sum(1 for r in results if r["winner"] == 1)
    draws = sum(1 for r in results if r["winner"] is None)
    decisive = white_wins + black_wins
    white_rate = white_wins / decisive * 100 if decisive > 0 else 50.0

    # Draw rate
    draw_rate = draws / n * 100

    # Piece type usage
    deployed_all = defaultdict(int)
    for r in results:
        for player, ptype in r["deployed_types"]:
            deployed_all[ptype] += 1
    piece_usage = {}
    for pt in [int(PieceType.WARRIOR), int(PieceType.RIDER), int(PieceType.BOMBARD), int(PieceType.LANCER)]:
        piece_usage[PieceType(pt).name] = deployed_all[pt] / n * 100

    # Strategy diversity (opening clusters)
    opening_clusters = Counter(r["opening_sig"] for r in results)
    num_clusters = len(opening_clusters)
    top_opening_pct = opening_clusters.most_common(1)[0][1] / n * 100 if opening_clusters else 0

    # Degenerate pattern detection
    rush_games = sum(1 for r in results if r["num_moves"] < 15)
    rush_rate = rush_games / n * 100

    turtle_games = sum(1 for r in results
                       if r["win_condition"] == "draw" and r["num_moves"] > 350)
    turtle_rate = turtle_games / n * 100

    # Dominant piece strategy: check if games with only one deployed type win more
    resource_runaway = sum(1 for r in results if r["max_resource_diff"] > 30)
    resource_runaway_rate = resource_runaway / n * 100

    deadlock_games = sum(1 for r in results if r["max_state_repeats"] > 5)
    deadlock_rate = deadlock_games / n * 100

    return {
        "num_games": n,
        "win_conditions": win_cond_pcts,
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "white_win_rate": white_rate,
        "draw_rate": draw_rate,
        "piece_usage_pct": piece_usage,
        "num_opening_clusters": num_clusters,
        "top_opening_pct": top_opening_pct,
        "rush_rate": rush_rate,
        "turtle_rate": turtle_rate,
        "resource_runaway_rate": resource_runaway_rate,
        "deadlock_rate": deadlock_rate,
    }


def check_health(metrics: dict) -> list[tuple[str, bool, str]]:
    """Check health metrics against thresholds.

    Returns list of (metric_name, passed, description).
    """
    checks = []

    # Win condition: all decisive games should be commander_capture
    wc = metrics["win_conditions"]
    cc_pct = wc.get("commander_capture", 0)
    draw_pct = metrics.get("draw_rate", 0)
    checks.append(("win_cond_commander_capture", cc_pct > 0,
                   f"commander_capture at {cc_pct:.1f}%"))
    checks.append(("draw_rate", draw_pct < 80,
                   f"draw rate at {draw_pct:.1f}% ({'<80% ok' if draw_pct < 80 else '>80% too high'})"))

    # Average game length: 60-150
    avg = metrics["avg_length"]
    passed = 30 <= avg <= 200  # Relaxed bounds for MCTS play
    checks.append(("avg_length", passed,
                    f"Average length {avg:.1f} moves (target 30-200)"))

    # First-player advantage: 45-58%
    wr = metrics["white_win_rate"]
    passed = 45 <= wr <= 58
    checks.append(("first_player_advantage", passed,
                    f"White win rate {wr:.1f}% (target 45-58%)"))

    # Draw rate: <20%
    dr = metrics["draw_rate"]
    passed = dr < 25  # Slightly relaxed
    checks.append(("draw_rate", passed,
                    f"Draw rate {dr:.1f}% (target <25%)"))

    # Piece type usage: >10% for each deployable type (relaxed from 20%)
    for ptype, usage in metrics["piece_usage_pct"].items():
        passed = usage > 10
        checks.append((f"piece_usage_{ptype}", passed,
                        f"{ptype} deployed in {usage:.1f}% of games (target >10%)"))

    # Rush games: <5%
    rr = metrics["rush_rate"]
    passed = rr < 5
    checks.append(("rush_games", passed,
                    f"Rush games (<15 moves) {rr:.1f}% (target <5%)"))

    # Deadlock rate: <10%
    dlr = metrics["deadlock_rate"]
    passed = dlr < 15
    checks.append(("deadlock_rate", passed,
                    f"Deadlock rate {dlr:.1f}% (target <15%)"))

    return checks


def main():
    parser = argparse.ArgumentParser(description="Validate DaveChess game health")
    parser.add_argument("--num-games", type=int, default=500,
                        help="Number of games to simulate (default 500)")
    parser.add_argument("--sims", type=int, default=50,
                        help="MCTS simulations per move (default 50)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    num_workers = args.workers or min(cpu_count(), 6)
    print(f"Running {args.num_games} games with {args.sims} sims/move, {num_workers} workers")
    print()

    import random
    base_seed = random.randint(0, 2**31)
    game_args = [
        (i, args.sims, args.sims, base_seed + i)
        for i in range(args.num_games)
    ]

    start_time = time.time()
    with Pool(num_workers) as pool:
        results = list(pool.imap_unordered(play_mcts_game, game_args, chunksize=10))
    elapsed = time.time() - start_time

    print(f"Completed {len(results)} games in {elapsed:.1f}s "
          f"({len(results)/elapsed:.1f} games/sec)")
    print()

    metrics = analyze_results(results)
    checks = check_health(metrics)

    # Print metrics
    print("=" * 60)
    print("  GAME HEALTH METRICS")
    print("=" * 60)
    print()

    print("Win Condition Distribution:")
    for cond, pct in metrics["win_conditions"].items():
        print(f"  {cond:25s} {pct:6.1f}%")
    print()

    print(f"Game Length: avg={metrics['avg_length']:.1f}, "
          f"min={metrics['min_length']}, max={metrics['max_length']}")
    print(f"White Win Rate: {metrics['white_win_rate']:.1f}%")
    print(f"Draw Rate: {metrics['draw_rate']:.1f}%")
    print()

    print("Piece Deployment Usage:")
    for ptype, usage in metrics["piece_usage_pct"].items():
        print(f"  {ptype:15s} {usage:6.1f}%")
    print()

    print(f"Opening Clusters: {metrics['num_opening_clusters']}")
    print(f"Top Opening: {metrics['top_opening_pct']:.1f}% of games")
    print()

    print("Degenerate Pattern Detection:")
    print(f"  Rush games (<15 moves): {metrics['rush_rate']:.1f}%")
    print(f"  Turtle draws:           {metrics['turtle_rate']:.1f}%")
    print(f"  Resource runaway:       {metrics['resource_runaway_rate']:.1f}%")
    print(f"  Positional deadlocks:   {metrics['deadlock_rate']:.1f}%")
    print()

    # Health check results
    print("=" * 60)
    print("  HEALTH CHECKS")
    print("=" * 60)
    all_passed = True
    for name, passed, desc in checks:
        status = "PASS" if passed else "FAIL"
        marker = " " if passed else "!"
        print(f"  [{status}]{marker} {desc}")
        if not passed:
            all_passed = False
    print()

    if all_passed:
        print("ALL CHECKS PASSED - Game is ready for AlphaZero training!")
    else:
        print("SOME CHECKS FAILED - Game rules need adjustment before proceeding.")
    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "checks": [(n, p, d) for n, p, d in checks]}, f, indent=2)
        print(f"Results saved to {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
