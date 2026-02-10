#!/usr/bin/env python3
"""Launch Gemini CLI for DaveChess benchmark."""
import os
import subprocess
import sys
import time

SANDBOX = os.environ.get("BENCHMARK_SANDBOX", "/tmp/benchmark-sandbox")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_DIR, "benchmark_logs")

PROMPT = """You are in a sandbox directory with a DaveChess benchmark CLI.

STEP 1: Read scripts/agent_cli.py to understand the interface (the docstring has all docs).

STEP 2: Create a session:
python scripts/agent_cli.py create --name "gemini-cli" --budget 500000 --baseline-games 3 --eval-min-games 5 --eval-max-games 10 --checkpoint checkpoints/best.pt --calibration checkpoints/calibration.json

STEP 3: Play baseline games. You start in BASELINE phase. Get the board with `state`, then play moves from the legal_moves list using `move`. Keep playing until the phase changes to LEARNING.

STEP 4: Read the rules: python scripts/agent_cli.py rules <session_file>

STEP 5: Study GM games: python scripts/agent_cli.py study <session_file> 10

STEP 6: Play 3 practice games at increasing ELO (600, 800, 1000). Use `practice` to start, then `move` to play each move until game_over.

STEP 7: Call evaluate to transition to EVALUATION phase.

STEP 8: Play rated evaluation games until the session auto-completes.

STEP 9: Get final results: python scripts/agent_cli.py result <session_file>

IMPORTANT:
- Pick moves ONLY from the legal_moves list in each response
- Do NOT write scripts or game engines to select moves - reason about each move yourself
- When a game ends (game_over=true), check for next_game_id and continue
- Use `state` if you need to see the current board"""

os.makedirs(LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"gemini_{timestamp}.log")

print(f"Launching Gemini CLI benchmark at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log: {log_file}")

with open(log_file, "w") as log:
    proc = subprocess.Popen(
        ["gemini", PROMPT, "--yolo", "--model", "gemini-3-pro-preview"],
        stdout=log, stderr=subprocess.STDOUT,
        cwd=SANDBOX
    )
    print(f"PID: {proc.pid}")
    proc.wait()
    print(f"Gemini exited with code {proc.returncode}")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
