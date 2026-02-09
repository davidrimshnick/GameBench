# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GameBench is a benchmark measuring how efficiently LLMs learn novel strategic reasoning from examples using DaveChess, a custom board game designed to be absent from training data. The system uses AlphaZero self-play training on a Jetson Orin Nano to create synthetic grandmaster games for LLM evaluation.

## Key Commands

### Development & Testing
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests (149 tests across 6 suites)
pytest

# Run specific test suite
pytest tests/test_game.py
pytest tests/test_mcts.py -v

# Interactive play modes
python scripts/play.py                    # Human vs Human
python scripts/play.py --vs-mcts 100      # Human vs MCTS (100 simulations)
python scripts/play.py --mcts-vs-mcts 50 50  # Watch MCTS vs MCTS
```

### Training
```bash
# IMPORTANT: Always use W&B for monitoring! The API key is already configured locally.
# Start AlphaZero training (requires CUDA)
python scripts/train.py --config configs/training.yaml

# Monitor training with TensorBoard (runs on port 6006)
tensorboard --logdir logs/tensorboard

# Resume training from checkpoint
python scripts/train.py --config configs/training.yaml  # Automatically resumes

# If W&B fails to connect, check credentials:
wandb login  # API key is in ~/.netrc
```

### Benchmark CLI (for any coding agent)
```bash
# Install davechess as a package first: pip install -e . (from repo root)
# All docs are in scripts/agent_cli.py — read the docstring for full reference.

# Create a session (baseline measures starting ELO for gain calculation)
python scripts/agent_cli.py create --name "test" --budget 500000

# Study GM games, practice, play moves, evaluate
python scripts/agent_cli.py study <session_file> 5
python scripts/agent_cli.py practice <session_file> 800
python scripts/agent_cli.py move <session_file> "Wd2-d3"
python scripts/agent_cli.py evaluate <session_file>
python scripts/agent_cli.py result <session_file>

# Launch a coding agent in a sandbox — see "Running the Benchmark with a Coding Agent" below
```

### REST API Server
```bash
# Start benchmark REST API (FastAPI)
pip install -e ".[api]"  # Install fastapi, uvicorn, pydantic
python scripts/run_api_server.py --config configs/api_server.yaml

# Use Python SDK client
from davechess.benchmark.sdk import BenchmarkClient
client = BenchmarkClient("http://localhost:8000")
```

### Legacy Harness (API-level token tracking)
```bash
# Run benchmark with direct API calls and token budget enforcement
python scripts/run_benchmark.py --provider anthropic --model claude-sonnet-4-20250514 --budget 500000

# Calibrate opponent pool (generates checkpoints/calibration.json)
python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt
```

### Game Analysis & Seed Generation
```bash
# Validate game balance with MCTSLite
python scripts/validate_game.py --num-games 500 --sims 50

# Generate full seed set (heuristic games + endgame wins)
python scripts/generate_and_save_seeds.py --num-games 100 --num-endgames 100 --endgame-sims 100 --force

# Append more endgame seeds to existing pickle (incremental)
python scripts/generate_and_save_seeds.py --append --num-endgames 150 --endgame-sims 100

# Check existing seed games
python scripts/generate_and_save_seeds.py  # Shows stats without regenerating

# Seeds are also stored as W&B artifact (smart-seeds:latest)

# Quick balance check
python scripts/quick_balance_check.py

# Detailed game analysis
python scripts/analyze_games.py

# Endgame analysis (theoretical checkmate positions, R+C vs C barriers)
python scripts/endgame_analysis.py
python scripts/endgame_barrier_analysis.py
```

## Architecture & Key Design Decisions

### Training Pipeline

The AlphaZero implementation has several critical modifications for DaveChess:

1. **Value Target Structure**: Uses standard AlphaZero three-valued targets (+1.0 for wins, 0.0 for draws, -1.0 for losses) with tanh output head. Drawn games are included in training data so the model learns that draw-prone positions are worse than winning ones. Earlier versions incorrectly used 0/1 targets with a tanh head, causing the network to treat losses as neutral (tanh midpoint = 0 = "uncertain"). Another earlier version discarded draws entirely, but with the threefold repetition rule, draws carry meaningful signal.

2. **Smart Seed Generation**: Two types of seeds bootstrap learning:
   - **Heuristic games**: `CommanderHunter` and `HeuristicPlayer` provide strategic full-game seeds with middlegame patterns.
   - **Endgame curriculum**: MCTS-solved mating sequences teach the network what checkmate looks like. Endgame types: 2R+C, 2L+C, R+L+C, R+B+C, L+B+C, 2R+L+C, R+L+B+C, 2R+B+C — all vs lone Commander. Black Commander biased to edges/corners for realistic mating scenarios.

   Seeds are saved to `checkpoints/smart_seeds.pkl` (~8-10K positions, ~170MB, backed up as W&B artifact `smart-seeds:latest`). **Not committed to git** — download from W&B if missing. Training automatically loads these instead of regenerating. The model pre-trains on seed data before starting self-play. Located in:
   - `davechess/engine/heuristic_player.py` - Position evaluation and strategic play
   - `davechess/engine/smart_seeds.py` - Seed game generation + endgame position generator
   - `scripts/generate_and_save_seeds.py` - Standalone seed generator (`--append` to add endgame seeds to existing pickle)

3. **Warm-Start from W&B**: If `checkpoints/best.pt` exists when starting "fresh" training (no step checkpoint), the trainer loads those weights into both networks instead of starting from random. This enables restoring a model from W&B artifacts and injecting new seed data without losing learned knowledge.

4. **Adaptive MCTS Simulations**: The `adaptive_simulations()` function in `training.py` scales simulation count based on model ELO (25 sims at ELO 0 → 100 sims at ELO 2000) to speed up early training. Min sims raised from 10→25 (selfplay) and 15→30 (eval) because too-shallow search produced low-quality data that couldn't learn from seed strategies.

### Game State Representation

- **Board**: 8x8 grid stored as `state.board[row][col]` containing `Piece` objects
- **Nodes**: 4 Gold nodes (give resource income for promotion) at (3,3), (3,4), (4,3), (4,4)
- **Neural Network Input**: 14 planes (5 piece types × 2 players + gold nodes + player + 2 resources) via `state_to_planes()`
- **Move Encoding**: Policy size of 4288 (64×67 move slots per square) via `move_to_policy_index()`. Slots 0-55 = direction moves (8 dirs × 7 dist), 56-63 = bombard ranged, 64-66 = promotion targets (R, B, L).

### Critical Game Rules

**When changing game rules, update ALL three locations: `davechess/game/rules.py`, `davechess/benchmark/prompt.py` (RULES_TEXT), and `README.md` (DaveChess section).**

1. **Capture (Chess-style)**: Any piece can capture any piece by moving onto it. Attacker always takes the defender's square — no strength comparison. Enables sacrifices, forks, pins, and tactical depth.
2. **Commander Safety**: Must resolve check immediately (move/block/capture). Cannot make a move that leaves own Commander in check.
3. **Win Conditions**: Checkmate opponent's Commander (only way to win). Turn 100 with no checkmate = draw. Threefold repetition of the same position (board + player, excluding resources) = draw. 50-move rule: 50 moves per side (100 halfmoves) with no capture or promotion = draw.
4. **Piece Types**: Commander (C), Warrior (W, pawn-like), Rider (R, up to 7 squares orthogonal / 3 diagonal), Bombard (B, 1 sq move + ranged attack), Lancer (L, up to 7 squares any direction with jump)
5. **Promotion Costs**: R=3, B=5, L=7. No deployment — pieces promote in place by spending Gold resources.
6. **Starting Army**: 12 pieces per side — 1 Commander, 3 Riders, 2 Bombards, 6 Warriors. No new pieces are ever added.
7. **Warrior Movement**: Forward only (like chess pawns). Captures diagonal-forward only. White Warriors move toward row 8, Black toward row 1. No retreat, no sideways movement.
8. **Bombard**: 1 square movement any direction. Ranged attack at exactly 2 squares (straight line, clear path). Stays in place when attacking. Can't use ranged attack against Commander.
9. **Lancer**: Moves up to 7 squares in any direction, can jump over exactly one piece (any color)

### Known Issues & Solutions

1. **Defensive Stalemates (v1)**: Strength-based capture made defense always optimal — no sacrifices possible, material snowballed. Warrior-toggle pattern exploited sideways movement.
   - Solution (v2): Complete game redesign — chess-style capture (attacker always wins), pawn-like Warriors (forward move, diagonal-forward capture only), removed strength stat and Power nodes. Threefold repetition draw rule, correct +1/-1 value targets with tanh head.

2. **Value Target / Activation Mismatch**: Using 0/1 value targets with tanh output [-1,+1] caused the network to learn that losing positions are neutral (0 = tanh midpoint). The network couldn't distinguish losses from draws, contributing to defensive play.
   - Solution: Standard AlphaZero targets (+1 win, -1 loss) matching the tanh output range. All value target assignments must use +1/-1 (selfplay.py, smart_seeds.py, training.py MCTSLite fallback).

2. **Slow Seed Generation**: MCTSLite with random rollouts takes minutes per game
   - Solution: Heuristic players provide 10x faster seed generation

4. **Training Instability**: Early iterations produce all draws/max-length games
   - Solution: Smart seeds with strategic play, adaptive simulation counts

5. **Draw Data Flooding Replay Buffer**: When 60-70% of self-play games are draws, the buffer fills with low-information positions that dilute the signal from decisive games.
   - Solution (previous): Discarded draws entirely. (Current): With threefold repetition rule, draws carry meaningful signal. Draws are now kept with value target 0.0 (standard AlphaZero three-valued targets). The repetition rule shortens the worst-case draws, and the model learns to avoid draw-prone positions.

6. **Consecutive Rejection Stalls**: Training network can get stuck in local minima, failing eval repeatedly against the best network (7+ consecutive rejections observed).
   - Solution: Auto-reset mechanism resets training network to best model weights after `max_consecutive_rejections` (default 7) consecutive failures. Counter persists across restarts via checkpoint.

7. **Seed Re-injection Distribution Mismatch**: At higher ELO, periodic re-injection of heuristic seed data (every 5 iterations) creates a distribution mismatch that hurts eval performance.
   - Solution: Disabled periodic seed re-injection. Seeds are only used for initial buffer seeding at iteration 0.

8. **Replay Buffer OOM on Checkpoint Load/Save**: The buffer at 30K+ positions contains ~605MB of planes and ~515MB of policies (4288-wide). Stacking the full buffer into a contiguous array for save/load caused OOM on the 8GB Jetson.
   - Solution: Chunked save using memory-mapped files (5K positions per chunk, ~85MB peak). Chunked load processes arrays in slices. All buffer data enforced as float32 on push to prevent accidental float64 doubling.

9. **CUDA Fragmentation After Self-play**: Higher adaptive sims fragment GPU memory, causing `NVML_SUCCESS` assert failures when transitioning to training.
   - Solution: `torch.cuda.empty_cache()` before training phase and before eval phase

10. **Temperature Threshold Too Low**: At `temperature_threshold: 30`, the model played near-deterministically after move 30 with noisy policies, entrenching defensive patterns.
   - Solution: Raised to 60 to maintain exploration longer in self-play

11. **Seed Re-injection Memory Spike**: Periodic seed injection (every 5 iterations) loads a 233MB pickle. Without cleanup, the pickle data persists in memory alongside the replay buffer.
   - Solution: Explicit `del smart_buffer; gc.collect()` after copying seeds to replay buffer

12. **Memory Diagnostics**: Silent OOM kills during training are hard to diagnose on Jetson (kernel kills process without error in app logs).
   - Solution: `_log_memory()` helper logs RSS and GPU memory at phase transitions (after self-play, before/after eval, before/after checkpoint save). Explicit `del current_mcts, best_mcts; gc.collect()` after eval to free MCTS circular references.

13. **Threefold Repetition**: At high ELO (2000+), 75% of self-play games end in draws via Warrior-toggle oscillation (Wb1-c1/Wc1-b1 repeated to turn 100). The policy network couldn't learn to avoid repetition because the rule wasn't in `apply_move()`.
   - Solution: Added `position_counts` dict to `GameState` tracking `get_position_key()` occurrences (board + current_player only, excluding resources — resources change every turn via income, preventing repeats). In both `apply_move()` and `apply_move_fast()`, draw is declared when any position occurs 3 times. MCTS search sees repetition draws as terminal nodes (value 0.0). The `clone()` method copies `position_counts` so game loops from cloned states track correctly.

14. **Self-Play Overfitting**: At ELO 4000+, the network learned to exploit its own play patterns but lost to random MCTS rollouts. The policy became too narrow — all 20 games per iteration were homogeneous self-play with modest Dirichlet noise (alpha=0.3, epsilon=0.25).
   - Solution: Three changes to increase diversity: (1) Lowered `dirichlet_alpha` from 0.3→0.15 for spikier noise (DaveChess has ~30-80 legal moves, not Go's ~400). (2) Raised `dirichlet_epsilon` from 0.25→0.4 so 40% of the root policy comes from noise. (3) Added `random_opponent_fraction` (default 0.25): 25% of self-play games are played against a no-NN MCTS opponent (uniform policy + zero value), forcing the network to handle non-standard play. Training examples are only collected from the NN side in these games. The NN alternates White/Black across random-opponent games.

### Hardware Constraints (Jetson Orin Nano)

- 8GB shared RAM (CPU + GPU)
- Multiprocess MCTS: 4 CPU workers + GPU inference server in main process
- Network: 20 ResBlocks, 256 filters (~24M params, ~1GB GPU with training)
- Training uses mixed precision (FP16) when available
- Replay buffer capped at 30K positions (~700MB with 4288-wide policy vectors). Save/load uses chunked I/O (5K chunks) to avoid temp array spikes
- All buffer data enforced as float32 on push — prevents accidental float64 doubling
- Always call `torch.cuda.empty_cache()` before switching between self-play, training, and eval phases
- Memory usage logged at phase transitions (`_log_memory()`) for OOM diagnosis
- MCTS trees explicitly freed after eval via `del` + `gc.collect()` (circular parent↔child refs)

### Benchmark CLI (`scripts/agent_cli.py`)

**`scripts/agent_cli.py` is the single source of truth for the benchmark interface.** All documentation, commands, DCN notation, calibration tables, and agent tips are in its docstring. Any coding agent reads this one file to understand and run the full benchmark.

Session lifecycle: `BASELINE -> LEARNING -> EVALUATION -> COMPLETED`

Key commands: `create`, `rules`, `study`, `practice`, `move`, `state`, `evaluate`, `result`, `report-tokens`

Opponents use MCTSLite (random rollouts, no neural network). The neural network checkpoint (`checkpoints/best.pt`) plays too defensively against out-of-distribution opponents (draws 50% vs random) because it was trained via self-play. MCTSLite produces decisive games with clear winners, which is better for ELO measurement.

### Running the Benchmark with a Coding Agent

To test the benchmark with a coding agent (Claude Code, Codex CLI, etc.), launch a separate instance in an isolated sandbox directory so it can't read the game engine source code.

**Sandbox setup:**
```bash
# Create minimal sandbox with only the CLI script and game data
SANDBOX="/tmp/benchmark-sandbox"  # or any temp directory
mkdir -p "$SANDBOX/scripts" "$SANDBOX/data" "$SANDBOX/checkpoints/agent_sessions"
cp scripts/agent_cli.py "$SANDBOX/scripts/"
cp -r data/gm_games "$SANDBOX/data/"

# davechess must be installed as a package: pip install -e . (from repo root)
```

**Launch the agent (with streaming output for monitoring):**
```bash
cd "$SANDBOX" && claude -p "You are in a sandbox directory. \
  Read scripts/agent_cli.py — it has all the docs you need. \
  Run the full DaveChess benchmark: create a session \
  (--budget 500000 --eval-min-games 5 --eval-max-games 15), \
  read the rules, study GM games, play practice games, then evaluate. \
  Report tokens with --tokens on every command. \
  Pick moves from legal_moves. Show the final result." \
  --allowedTools "Bash(run benchmark commands)" \
  --output-format stream-json \
  2>&1 | tee benchmark-output.log
```

The `--output-format stream-json` flag streams JSON chunks in real time so you can monitor progress (without it, `claude -p` buffers all output until completion).

**What the agent should do autonomously:**
1. Read `scripts/agent_cli.py` docstring to learn the interface
2. Create a session, read rules, study GM games
3. Play practice games to develop strategy (Lancers are dominant — see GM games)
4. Transition to evaluation and play rated games
5. Report final ELO via `result` command

Each practice/eval game involves ~30-100 individual `move` commands. A full run with practice + 5-15 eval games can take 10-30 minutes depending on the agent's speed.

**Why a sandbox?** The agent should only see `agent_cli.py` (the interface) and game data. If it has access to the full repo, it could read the game engine, opponent logic, or rules implementation — defeating the purpose of measuring learning from examples.

### Agentic Benchmark Architecture (Legacy Harness)

The `run_benchmark.py` harness gives an LLM a token budget and tools, then measures ELO achieved through autonomous learning. It calls the LLM API directly and enforces token budgets.

Key design decisions:
- **Rolling context window** (20 messages): forces agents to use external memory at large budgets
- **Eval from same budget**: agent must budget tokens wisely between learning and evaluation
- **Opponent pool**: ELO-to-MCTS-sims mapping with log-space interpolation between calibrated points
- **Move-by-move play**: agent plays each move individually via tool calls (not pre-committed strategies)

Key files:
- `scripts/agent_cli.py` — CLI interface (single file with all docs)
- `scripts/run_benchmark.py` — External harness with token budget enforcement
- `davechess/benchmark/api/session.py` — Core session lifecycle
- `davechess/benchmark/sequential_eval.py` — Glicko-2 adaptive ELO measurement
- `davechess/benchmark/opponent_pool.py` — Calibrated opponent ELO interpolation
- `davechess/benchmark/game_library.py` — GM game library (DCN format)
- `davechess/benchmark/token_tracker.py` — Token budget tracking

## File Structure Notes

### Source Code
- `davechess/game/` - Core game engine
  - `board.py` - Board representation, Gold nodes, starting positions, notation utils
  - `state.py` - GameState, Piece, Player, Move types (MoveStep, Promote, BombardAttack)
  - `rules.py` - Legal move generation, move application, check/checkmate, threefold repetition, 50-move rule
  - `notation.py` - DCN (DaveChess Notation) encoding/decoding
- `davechess/engine/` - AlphaZero training & MCTS
  - `network.py` - ResNet policy+value network (14 input planes, 4288 policy size)
  - `mcts.py` - Full PUCT MCTS with neural network evaluation
  - `mcts_lite.py` - Lightweight MCTS with random rollouts (no NN), used for opponents & validation
  - `mcts_worker.py` - Worker process for multiprocess self-play
  - `gpu_server.py` - GPU inference server batching requests from CPU workers
  - `selfplay.py` - ReplayBuffer, self-play game generation, training data extraction
  - `smart_seeds.py` - Heuristic seed games + endgame curriculum generator
  - `training.py` - Main trainer: iteration loop, adaptive sims, eval, checkpointing, W&B
  - `heuristic_player.py` - Position evaluation and strategic agents for seed generation
- `davechess/data/` - Game generation, ELO calibration
  - `elo.py` - Glicko-2 rating system
  - `generator.py` - Agent base class, RandomAgent, MCTSAgent, play_game()
  - `storage.py` - DCN file I/O, replay_game() state reconstruction
- `davechess/benchmark/` - LLM evaluation framework
  - `api/server.py` - FastAPI REST server (sessions, games, study, eval endpoints)
  - `api/session.py` - BenchmarkSession: phase-gated state machine (BASELINE→LEARNING→EVALUATION→COMPLETED)
  - `api/session_manager.py` - Creates/tracks sessions, holds shared OpponentPool/GameLibrary
  - `api/models.py` - Pydantic request/response models
  - `sdk.py` - BenchmarkClient HTTP wrapper + in-process session import
  - `agent_harness.py` - Autonomous learning loop with rolling message window
  - `agentic_protocol.py` - Full pipeline orchestrator (learning→eval→scoring)
  - `sequential_eval.py` - Adaptive Glicko-2 testing against calibrated opponents
  - `opponent_pool.py` - ELO-to-MCTS-sims mapping with log-space interpolation
  - `game_manager.py` - Concurrent practice/evaluation game management
  - `game_library.py` - GM game library (DCN), serves without replacement per session
  - `tools.py` - Tool definitions (study_games, start_practice_game, play_move, get_game_state)
  - `prompt.py` - RULES_TEXT, system prompts for agents
  - `scoring.py` - ELO computation, learning curves, GameBench AUC score
  - `token_tracker.py` - Token budget enforcement

### Scripts
- `scripts/train.py` - AlphaZero training orchestrator
- `scripts/play.py` - Interactive play (human vs human/MCTS)
- `scripts/agent_cli.py` - **Core benchmark CLI** — single-file interface with full docs
- `scripts/run_api_server.py` - FastAPI REST server startup
- `scripts/run_benchmark.py` - External harness driving LLM via subprocess
- `scripts/run_agentic_benchmark.py` - Multi-budget agentic benchmark (100K/1M/10M)
- `scripts/generate_and_save_seeds.py` - Smart seed generator (heuristic + endgame, `--append` mode)
- `scripts/calibrate_opponents.py` - Round-robin Glicko-2 calibration, saves calibration.json
- `scripts/validate_game.py` - Game health validation (win rate, draw rate, game length)
- `scripts/analyze_games.py` - Opening frequencies, move diversity, strategy patterns
- `scripts/endgame_analysis.py` - All static checkmate positions, minimax evaluation
- `scripts/endgame_barrier_analysis.py` - Tests R+C vs C forcing patterns
- `scripts/quick_balance_check.py` - Fast 100-game balance check
- `scripts/download_artifacts.py` - Download W&B artifacts (models, buffers, seeds)

### Configuration & Data
- `configs/training.yaml` - AlphaZero training (network, MCTS, self-play, buffer, optimizer)
- `configs/api_server.yaml` - REST server host/port, opponent calibration, game library config
- `configs/agentic_benchmark.yaml` - API provider, token budgets, rolling window, opponent pool
- `checkpoints/` - Model checkpoints, seed games, calibration (gitignored, use W&B artifacts)
- `data/gm_games/` - GM game library in DCN format (loaded at runtime for benchmark)
- `docs/index.html` - GitHub Pages leaderboard dashboard
- `wandb/` - Weights & Biases tracking (auto-generated, gitignored)

### Tests (149 tests across 6+ suites)
- `tests/test_game.py` - Board, state, moves, capture, promotion, check/checkmate, draw rules
- `tests/test_mcts.py` - MCTS correctness, UCB1, tree traversal, batched eval
- `tests/test_network.py` - Network inference, state encoding, policy/value shapes
- `tests/test_benchmark.py` - Legacy benchmark orchestration
- `tests/test_agentic_benchmark.py` - Agent harness, tools, token tracking, Glicko-2
- `tests/test_api_endpoints.py` - FastAPI endpoint tests (session lifecycle, game play)
- `tests/test_api_session.py` - BenchmarkSession phase transitions, baseline, eval
- `tests/test_sdk.py` - BenchmarkClient HTTP wrapper, round-trip correctness
- `tests/conftest.py` - Fixtures (short_games monkeypatch for fast API testing)

## Training Monitoring

**CRITICAL: NEVER start a training run without W&B connected! Verify W&B shows "logging enabled" in the logs before proceeding. If W&B fails with "Broken pipe", kill any stale wandb processes first (`pkill -9 wandb`).** All `wandb.log()` calls are wrapped in `_safe_wandb_log()` to prevent W&B service crashes from killing training.

Training logs to multiple destinations:
- **W&B Dashboard** - Primary monitoring at https://wandb.ai (MUST be connected)
- Console output
- `logs/training_log.jsonl` - Structured metrics
- TensorBoard at `logs/tensorboard/`

Large files (seeds, checkpoints) should be stored as W&B artifacts, not committed to git (GitHub 100MB limit).

W&B artifacts automatically backed up:
- **best-model-iter{N}** — best.pt uploaded on each ELO improvement
- **replay-buffer** — full replay buffer uploaded each iteration
- **game-log-iter{N}** — DCN game logs uploaded each iteration
- **game-logs-all** — one-time backfill of all historical DCN logs

Key metrics to watch:
- `selfplay/avg_game_length` - Should decrease over time; turn 100 = draw limit
- `selfplay/draw_rate` - Should decrease as model learns to checkmate before turn 100
- `selfplay/white_win_rate` - Should stay near 0.5 for balance
- `eval/win_rate` - Must exceed 0.51 to update best model
- `training/policy_loss` - Should decrease over iterations