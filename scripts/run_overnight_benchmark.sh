#!/usr/bin/env bash
# run_overnight_benchmark.sh — Run DaveChess benchmark with 3 coding agents
# This script runs sequentially to avoid memory issues from concurrent NN loading.
# Expected runtime: 4-12 hours total (varies by agent speed and game length)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SANDBOX="${BENCHMARK_SANDBOX:-/tmp/benchmark-sandbox}"
LOG_DIR="$REPO_DIR/benchmark_logs"
RESULTS_DIR="$REPO_DIR/benchmark_results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== DaveChess Overnight Benchmark ==="
echo "Started: $(date)"
echo "Sandbox: $SANDBOX"
echo "Logs: $LOG_DIR"
echo ""

# Common prompt for all agents
PROMPT='You are in a sandbox directory with a DaveChess benchmark CLI.

STEP 1: Read scripts/agent_cli.py to learn the full interface (the docstring has everything).

STEP 2: Create a session:
  python scripts/agent_cli.py create --name "AGENT_NAME" --budget 500000 --baseline-games 3 --eval-min-games 5 --eval-max-games 10 --checkpoint checkpoints/best.pt --calibration checkpoints/calibration.json

STEP 3: Play baseline games. You are in the BASELINE phase. Use `state` to see the board, then play moves from legal_moves using `move`. Keep playing until the phase transitions to LEARNING. Report tokens with --tokens on every command.

STEP 4: Read rules (python scripts/agent_cli.py rules <session_file>)

STEP 5: Study GM games (python scripts/agent_cli.py study <session_file> 10)

STEP 6: Play 3-5 practice games at increasing ELO (600, 800, 1000). Use `practice` then `move` until game_over.

STEP 7: Transition to evaluation (python scripts/agent_cli.py evaluate <session_file>)

STEP 8: Play rated evaluation games. Keep playing moves until the session reaches COMPLETED.

STEP 9: Get results (python scripts/agent_cli.py result <session_file>)

IMPORTANT RULES:
- Pick moves ONLY from the legal_moves list in each response
- Report tokens with --tokens on EVERY command (estimate: 3000 prompt, 500 completion per call)
- Do NOT write scripts or game engines to select moves — reason about each move yourself
- When a game ends (game_over=true), check for next_game_id and continue
- Use `state` if you need to see the current board
- The session file path is in the response of each command'

# --- Agent 1: Claude Code ---
echo "=== Agent 1: Claude Code ==="
echo "Starting at $(date)"
CLAUDE_PROMPT="${PROMPT//AGENT_NAME/claude-code}"

cd "$SANDBOX"
claude -p "$CLAUDE_PROMPT" \
  --allowedTools "Bash(run benchmark commands:*)" \
  --output-format stream-json \
  2>&1 | tee "$LOG_DIR/claude_${TIMESTAMP}.log"

# Extract results
echo ""
echo "Claude Code finished at $(date)"
CLAUDE_SESSION=$(ls -t "$SANDBOX/checkpoints/agent_sessions/claude-code"*.pkl 2>/dev/null | head -1)
if [ -n "$CLAUDE_SESSION" ]; then
  python "$SANDBOX/scripts/agent_cli.py" result "$CLAUDE_SESSION" > "$RESULTS_DIR/claude_${TIMESTAMP}.json" 2>/dev/null || true
  echo "Results saved to $RESULTS_DIR/claude_${TIMESTAMP}.json"
fi

# --- Agent 2: Codex CLI ---
echo ""
echo "=== Agent 2: Codex CLI ==="
echo "Starting at $(date)"
CODEX_PROMPT="${PROMPT//AGENT_NAME/codex-cli}"

cd "$SANDBOX"
codex exec "$CODEX_PROMPT" \
  -m gpt-5.3-codex \
  -C "$SANDBOX" \
  --skip-git-repo-check \
  --json \
  2>&1 | tee "$LOG_DIR/codex_${TIMESTAMP}.log"

echo ""
echo "Codex CLI finished at $(date)"
CODEX_SESSION=$(ls -t "$SANDBOX/checkpoints/agent_sessions/codex-cli"*.pkl 2>/dev/null | head -1)
if [ -n "$CODEX_SESSION" ]; then
  python "$SANDBOX/scripts/agent_cli.py" result "$CODEX_SESSION" > "$RESULTS_DIR/codex_${TIMESTAMP}.json" 2>/dev/null || true
  echo "Results saved to $RESULTS_DIR/codex_${TIMESTAMP}.json"
fi

# --- Agent 3: Gemini CLI ---
echo ""
echo "=== Agent 3: Gemini CLI ==="
echo "Starting at $(date)"
GEMINI_PROMPT="${PROMPT//AGENT_NAME/gemini-cli}"

cd "$SANDBOX"
gemini "$GEMINI_PROMPT" \
  --yolo \
  --model gemini-3-pro-preview \
  2>&1 | tee "$LOG_DIR/gemini_${TIMESTAMP}.log"

echo ""
echo "Gemini CLI finished at $(date)"
GEMINI_SESSION=$(ls -t "$SANDBOX/checkpoints/agent_sessions/gemini-cli"*.pkl 2>/dev/null | head -1)
if [ -n "$GEMINI_SESSION" ]; then
  python "$SANDBOX/scripts/agent_cli.py" result "$GEMINI_SESSION" > "$RESULTS_DIR/gemini_${TIMESTAMP}.json" 2>/dev/null || true
  echo "Results saved to $RESULTS_DIR/gemini_${TIMESTAMP}.json"
fi

echo ""
echo "=== All Benchmarks Complete ==="
echo "Finished at $(date)"
echo ""
echo "Results:"
for f in "$RESULTS_DIR"/*_${TIMESTAMP}.json; do
  if [ -f "$f" ]; then
    echo "  $(basename "$f"):"
    cat "$f" | python -c "import json,sys; d=json.load(sys.stdin); print(f'    Baseline ELO: {d.get(\"baseline_elo\",\"?\")}, Final ELO: {d.get(\"final_elo\",\"?\")}, Gain: {d.get(\"elo_gain\",\"?\")}')" 2>/dev/null || echo "    (could not parse)"
  fi
done
