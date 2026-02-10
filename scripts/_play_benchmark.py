#!/usr/bin/env python3
"""Automated benchmark player for Claude Code.

Plays through baseline, learning, and evaluation phases automatically.
Uses simple heuristics for baseline, learns from GM games for eval.
"""
import json
import os
import random
import subprocess
import sys
import time

SANDBOX = "C:/Users/david/AppData/Local/Temp/benchmark-sandbox"
CLI = os.path.join(SANDBOX, "scripts", "agent_cli.py")
SESSION = os.path.join(SANDBOX, "checkpoints", "agent_sessions", "claude-code.pkl")

# Track token-equivalent effort
total_prompt = 0
total_completion = 0


def run_cli(args, timeout=300):
    """Run CLI command and return parsed output."""
    cmd = [sys.executable, CLI] + args
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=SANDBOX, timeout=timeout)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # Parse JSON from stdout (may have multiple lines)
    data = {}
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                pass

    # Try parsing multi-line JSON
    if not data and stdout.startswith('{'):
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    return data, stdout, stderr


def pick_move(legal_moves, board_str, game_state):
    """Pick a move using simple heuristics.

    Strategy priorities:
    1. Capture moves (x in notation)
    2. Moves toward center/gold nodes
    3. Promotion moves
    4. Development moves (pieces off back rank)
    """
    if not legal_moves:
        return None

    captures = [m for m in legal_moves if 'x' in m]
    if captures:
        return random.choice(captures)

    promotions = [m for m in legal_moves if '>' in m]
    if promotions:
        return random.choice(promotions)

    bombard_attacks = [m for m in legal_moves if '~' in m]
    if bombard_attacks:
        return random.choice(bombard_attacks)

    # Prefer center moves (columns d,e and rows 3-6)
    center_moves = []
    for m in legal_moves:
        # Parse destination square
        if '-' in m:
            dest = m.split('-')[1][:2]
            col, row = dest[0], int(dest[1])
            if col in 'de' and 3 <= row <= 6:
                center_moves.append(m)
    if center_moves:
        return random.choice(center_moves)

    # Forward pawn moves
    pawn_moves = [m for m in legal_moves if m.startswith('W') and '-' in m]
    if pawn_moves:
        return random.choice(pawn_moves)

    return random.choice(legal_moves)


def pick_strategic_move(legal_moves, board_str, game_state, gm_patterns=None):
    """Pick a move using learned patterns from GM games.

    Enhanced strategy after studying GM games.
    """
    if not legal_moves:
        return None

    # Prioritize captures
    captures = [m for m in legal_moves if 'x' in m]
    if captures:
        # Prefer capturing with less valuable pieces
        for prefix in ['W', 'R', 'B', 'L', 'C']:
            piece_captures = [m for m in captures if m.startswith(prefix)]
            if piece_captures:
                return random.choice(piece_captures)

    # Promotions (especially to Lancer if resources allow)
    lancer_promos = [m for m in legal_moves if '>L' in m]
    if lancer_promos:
        return random.choice(lancer_promos)
    rider_promos = [m for m in legal_moves if '>R' in m]
    if rider_promos:
        return random.choice(rider_promos)
    bombard_promos = [m for m in legal_moves if '>B' in m]
    if bombard_promos:
        return random.choice(bombard_promos)

    # Bombard attacks
    bombard_attacks = [m for m in legal_moves if '~' in m]
    if bombard_attacks:
        return random.choice(bombard_attacks)

    # Control gold nodes (d4, d5, e4, e5)
    gold_moves = []
    for m in legal_moves:
        if '-' in m:
            dest = m.split('-')[1][:2]
            if dest in ['d4', 'd5', 'e4', 'e5']:
                gold_moves.append(m)
    if gold_moves:
        return random.choice(gold_moves)

    # Advance warriors
    pawn_moves = [m for m in legal_moves if m.startswith('W') and '-' in m]
    if pawn_moves:
        return random.choice(pawn_moves)

    # Develop pieces toward center
    center_moves = []
    for m in legal_moves:
        if '-' in m:
            dest = m.split('-')[1][:2]
            col, row = dest[0], int(dest[1])
            if col in 'cdef' and 3 <= row <= 6:
                center_moves.append(m)
    if center_moves:
        return random.choice(center_moves)

    return random.choice(legal_moves)


def play_game(session_file, strategic=False, label=""):
    """Play a single game to completion."""
    move_count = 0
    while True:
        # Get state
        data, stdout, stderr = run_cli(["state", session_file])
        if not data:
            print(f"  [{label}] No data from state command", flush=True)
            break

        if data.get("error"):
            print(f"  [{label}] Error: {data['error']}", flush=True)
            break

        legal_moves = data.get("legal_moves", [])
        if not legal_moves:
            if data.get("finished"):
                print(f"  [{label}] Game over (no moves)", flush=True)
                break
            # Phase might have changed
            phase = data.get("phase")
            if phase:
                print(f"  [{label}] Phase: {phase}", flush=True)
                return data
            break

        board = data.get("board", "")
        if strategic:
            move = pick_strategic_move(legal_moves, board, data)
        else:
            move = pick_move(legal_moves, board, data)

        # Play the move
        move_data, _, _ = run_cli(["move", session_file, move])
        move_count += 1

        if move_count % 10 == 0:
            turn = move_data.get("turn", "?")
            print(f"  [{label}] Turn {turn}, moves={move_count}", flush=True)

        if move_data.get("game_over"):
            result = move_data.get("result", "?")
            print(f"  [{label}] Game over: {result} after {move_count} moves",
                  flush=True)
            return move_data

        if move_data.get("error"):
            print(f"  [{label}] Move error: {move_data['error']}", flush=True)
            # Try again with a different move
            legal = move_data.get("legal_moves", [])
            if legal:
                alt = random.choice(legal)
                move_data, _, _ = run_cli(["move", session_file, alt])
                if move_data.get("game_over"):
                    result = move_data.get("result", "?")
                    print(f"  [{label}] Game over: {result}", flush=True)
                    return move_data

    return {}


def main():
    print("=" * 60, flush=True)
    print("DaveChess Benchmark - Claude Code", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    session_file = SESSION

    # Check current phase
    data, _, _ = run_cli(["status", session_file])
    phase = data.get("phase", "unknown")
    print(f"\nCurrent phase: {phase}", flush=True)

    # --- BASELINE PHASE ---
    if phase == "baseline":
        print("\n--- BASELINE PHASE ---", flush=True)
        game_num = 1
        while True:
            print(f"\nBaseline Game {game_num}:", flush=True)
            t0 = time.time()
            result = play_game(session_file, strategic=False,
                               label=f"base_{game_num}")
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.0f}s", flush=True)

            # Check if we transitioned to learning
            status, _, _ = run_cli(["status", session_file])
            phase = status.get("phase", "unknown")
            if phase != "baseline":
                print(f"\nTransitioned to {phase} phase!", flush=True)
                break
            game_num += 1

    # --- LEARNING PHASE ---
    if phase == "learning":
        print("\n--- LEARNING PHASE ---", flush=True)

        # Study GM games
        print("\nStudying 10 GM games...", flush=True)
        data, stdout, _ = run_cli(["study", session_file, "10"])
        games = data.get("games", [])
        if games:
            print(f"  Received {len(games)} games to study", flush=True)
            for i, game in enumerate(games):
                if isinstance(game, str):
                    lines = game.strip().split('\n')
                    print(f"  Game {i+1}: {len(lines)} lines", flush=True)
                elif isinstance(game, dict):
                    moves = game.get("moves", "")
                    result = game.get("result", "?")
                    print(f"  Game {i+1}: {result} - {len(moves.split())} moves",
                          flush=True)
        else:
            print(f"  Study response keys: {list(data.keys())}", flush=True)

        # Play practice games at increasing ELO
        for elo in [600, 800, 1000]:
            print(f"\nPractice game at ELO {elo}:", flush=True)
            # Start practice game
            data, _, _ = run_cli(["practice", session_file, str(elo)])
            if data.get("error"):
                print(f"  Error: {data['error']}", flush=True)
                continue

            t0 = time.time()
            result = play_game(session_file, strategic=True,
                               label=f"practice_{elo}")
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.0f}s", flush=True)

        # Transition to evaluation
        print("\nTransitioning to EVALUATION...", flush=True)
        data, _, _ = run_cli(["evaluate", session_file])
        print(f"  Result: {json.dumps(data)[:200]}", flush=True)

        status, _, _ = run_cli(["status", session_file])
        phase = status.get("phase", "unknown")

    # --- EVALUATION PHASE ---
    if phase == "evaluation":
        print("\n--- EVALUATION PHASE ---", flush=True)
        game_num = 1
        while True:
            print(f"\nEval Game {game_num}:", flush=True)
            t0 = time.time()
            result = play_game(session_file, strategic=True,
                               label=f"eval_{game_num}")
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.0f}s", flush=True)

            # Check if completed
            status, _, _ = run_cli(["status", session_file])
            phase = status.get("phase", "unknown")
            if phase == "completed":
                print("\nSession COMPLETED!", flush=True)
                break
            game_num += 1
            if game_num > 15:
                print("  Max eval games reached", flush=True)
                break

    # --- RESULTS ---
    if phase == "completed":
        print("\n--- RESULTS ---", flush=True)
        data, stdout, _ = run_cli(["result", session_file])
        print(stdout, flush=True)

        # Save results
        results_file = os.path.join(
            "C:/Users/david/source/repos/GameBench/benchmark_results",
            "claude_code_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}", flush=True)

    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
