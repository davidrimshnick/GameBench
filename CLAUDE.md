# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GameBench is a benchmark measuring how efficiently LLMs learn novel strategic reasoning from examples using DaveChess, a custom board game designed to be absent from training data. The system uses AlphaZero self-play training on a Jetson Orin Nano to produce an expert neural network that serves dual roles: generating synthetic grandmaster games for LLM study, and acting as the calibrated opponent that agents play against during evaluation.

## Current Status (Feb 2026)

**Training restarted from scratch with pure AlphaZero architecture (no best-network gatekeeper). Monitoring on W&B.**

Previous `best.pt` (claimed ELO 1982) was broken — see Known Issues #15-17. The model was only pre-trained on seeds (354 steps, iteration 0) with fictional ELO inherited across 28 crashed restarts. It could not beat random MCTS.

What's done:
- Training pipeline bugs fixed (see Known Issues #15-17)
- **Switched to pure AlphaZero** — single continuously-updated network, no best/training split, no eval gatekeeper (see Known Issue #19)
- **MCTSLite ELO probes** — lightweight, non-gating ELO estimation every 5 iterations against MCTSLite at 50 sims
- Sandbox setup and `agent_cli.py` working
- GitHub Pages leaderboard at `docs/index.html` (currently has mockup data)
- Benchmark automation scripts: `run_overnight_benchmark.sh` (runs all 3 agents sequentially), per-agent launchers (`_launch_codex_benchmark.py`, `_launch_gemini_benchmark.py`), and a heuristic baseline player (`_play_benchmark.py`)
- Placeholder results in `benchmark_results/` from heuristic baseline player (not real LLM runs — `tokens.total_used = 0` in all results). These serve as format examples for the leaderboard.
- Benchmark integrity rules added to `agent_cli.py` and `davechess/benchmark/prompt.py` — agents must reason about moves themselves, no writing game engines or search algorithms
- `DaveChessNetwork.from_checkpoint()` classmethod auto-infers architecture from weights (no more hardcoded num_res_blocks/num_filters)

What's needed:
1. **Delete old checkpoints and train from scratch on Jetson** — remove old `best.pt`, `step_*.pt`, and run `python scripts/train.py --config configs/training.yaml`. Monitor on W&B for real ELO progression with the fixed pipeline.
2. **Run NN-MCTS calibration on Jetson** once training produces a model that consistently beats random MCTS. Run `python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt`.
3. **Copy calibration.json to sandbox** and run real agent benchmarks (Claude Code, Codex CLI, Gemini CLI) with NN-MCTS opponents on Jetson GPU using `run_overnight_benchmark.sh`
4. **Update leaderboard** with real agent results (replace placeholder heuristic-player data)

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

### Benchmark Preparation (run on Jetson or any GPU/CPU machine)
```bash
# Full pipeline: download best model from W&B, extract GM games, calibrate
python scripts/prepare_benchmark.py --all

# Full pipeline on CPU (slower calibration, no CUDA needed)
python scripts/prepare_benchmark.py --all --device cpu

# Individual steps
python scripts/prepare_benchmark.py --download-model          # best.pt from W&B
python scripts/prepare_benchmark.py --download-games           # GM games from W&B
python scripts/prepare_benchmark.py --calibrate                # NN-MCTS ELO calibration
python scripts/prepare_benchmark.py --check                    # Readiness check

# Use specific W&B run instead of auto-detecting highest ELO
python scripts/prepare_benchmark.py --download-model --run-id xrjpggwn

# Force re-download/re-calibrate
python scripts/prepare_benchmark.py --all --force
```

### Legacy Harness (API-level token tracking)
```bash
# Run benchmark with direct API calls and token budget enforcement
python scripts/run_benchmark.py --provider anthropic --model claude-sonnet-4-20250514 --budget 500000

# Calibrate opponent pool against the trained NN (generates checkpoints/calibration.json)
# MUST use --checkpoint — calibration is only valid for the model agents will face
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

1. **Value Target Structure**: Uses tanh-aligned value targets (+1.0 for wins, configurable draw target, -1.0 for losses). Default training config sets `selfplay.draw_value_target: -0.1` to discourage draw-seeking local minima in draw-heavy self-play. Earlier versions incorrectly used 0/1 targets with a tanh head, causing the network to treat losses as neutral (tanh midpoint = 0 = "uncertain"). Another earlier version discarded draws entirely, but with repetition/turn-limit rules, draws still carry strategic signal.

2. **Smart Seed Generation**: Two types of seeds bootstrap learning:
   - **Heuristic games**: `CommanderHunter` and `HeuristicPlayer` provide strategic full-game seeds with middlegame patterns.
   - **Endgame curriculum**: MCTS-solved mating sequences teach the network what checkmate looks like. Endgame types: 2R+C, 2L+C, R+L+C, R+B+C, L+B+C, 2R+L+C, R+L+B+C, 2R+B+C — all vs lone Commander. Black Commander biased to edges/corners for realistic mating scenarios.

   Seeds are saved to `checkpoints/smart_seeds.pkl` (~8-10K positions, ~170MB, backed up as W&B artifact `smart-seeds:latest`). **Not committed to git** — download from W&B if missing. Training automatically loads these instead of regenerating. The model pre-trains on seed data before starting self-play. Located in:
   - `davechess/engine/heuristic_player.py` - Position evaluation and strategic play
   - `davechess/engine/smart_seeds.py` - Seed game generation + endgame position generator
   - `scripts/generate_and_save_seeds.py` - Standalone seed generator (`--append` to add endgame seeds to existing pickle)

3. **Warm-Start from W&B**: If `checkpoints/best.pt` exists when starting "fresh" training (no step checkpoint), the trainer loads those weights into both networks instead of starting from random. This enables restoring a model from W&B artifacts and injecting new seed data without losing learned knowledge.

4. **Adaptive MCTS Simulations**: The `adaptive_simulations()` function in `training.py` scales simulation count based on model ELO. Self-play uses a configurable minimum floor (`mcts.min_selfplay_simulations`, default 50) and a smoothed ELO signal (`training.adaptive_elo_smoothing`) rather than raw probe ELO to avoid search-depth jitter from noisy probes.

### Game State Representation

- **Board**: 8x8 grid stored as `state.board[row][col]` containing `Piece` objects
- **Nodes**: 4 Gold nodes (give resource income for promotion) at (3,3), (3,4), (4,3), (4,4)
- **Neural Network Input**: 14 planes (5 piece types × 2 players + gold nodes + player + 2 resources) via `state_to_planes()`
- **Move Encoding**: Policy size of 4288 (64×67 move slots per square) via `move_to_policy_index()`. Slots 0-55 = direction moves (8 dirs × 7 dist), 56-63 = bombard ranged, 64-66 = promotion targets (R, B, L).

### Critical Game Rules

**When changing game rules, update ALL three locations: `davechess/game/rules.py`, `davechess/benchmark/prompt.py` (RULES_TEXT), and `README.md` (DaveChess section).**

1. **Capture (Chess-style)**: Any piece can capture any piece by moving onto it. Attacker always takes the defender's square — no strength comparison. Enables sacrifices, forks, pins, and tactical depth.
2. **Commander Safety**: Must resolve check immediately (move/block/capture). Cannot make a move that leaves own Commander in check.
3. **Win Conditions**: Checkmate opponent's Commander (only way to win). Turn 100 with no checkmate = draw. Threefold repetition of the same position (board + player + resource affordability buckets) = draw. 50-move rule: 50 moves per side (100 halfmoves) with no capture or promotion = draw.
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

5. **Draw Data Flooding Replay Buffer**: When self-play is draw-heavy, the buffer fills with low-information positions that dilute decisive signal.
   - Solution: Keep draws but reduce their gradient dominance. Training supports configurable draw targets (`selfplay.draw_value_target`) and draw loss weighting (`training.draw_sample_weight`). Self-play now logs draw reasons (`repetition`, `turn_limit`, `fifty_move`, `stalemate_or_other`) so we can distinguish rule-pathology from normal stabilization.

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

13. **Threefold Repetition**: At high ELO, oscillation loops can dominate self-play.
   - Solution: `position_counts` tracks `get_position_key()` occurrences in both `apply_move()` and `apply_move_fast()`. The key now includes coarse resource buckets (promotion affordability) so materially different states are not collapsed into false repeats. MCTS sees repetition draws as terminal nodes, and `clone()` copies `position_counts` for correct loop tracking.

14. **Self-Play Overfitting**: At ELO 4000+, the network learned to exploit its own play patterns but lost to random MCTS rollouts. The policy became too narrow — all 20 games per iteration were homogeneous self-play with modest Dirichlet noise (alpha=0.3, epsilon=0.25).
   - Solution: Three changes to increase diversity: (1) Lowered `dirichlet_alpha` from 0.3→0.15 for spikier noise (DaveChess has ~30-80 legal moves, not Go's ~400). (2) Raised `dirichlet_epsilon` from 0.25→0.4 so 40% of the root policy comes from noise. (3) Added `random_opponent_fraction` (default 0.25): 25% of self-play games are played against a no-NN MCTS opponent (uniform policy + zero value), forcing the network to handle non-standard play. Training examples are only collected from the NN side in these games. The NN alternates White/Black across random-opponent games.

15. **Eval Win Rate Excluded Draws (CRITICAL)**: `evaluate_network()` computed `win_rate = wins / (wins + losses)`, completely ignoring draws. With 10 eval games where most are draws, even 2W/1L/7D gave win_rate=0.667 (+120 ELO). This inflated ELO by hundreds of points each iteration despite the model being unable to reliably checkmate.
   - Solution: Changed to standard chess/AlphaZero scoring: `win_rate = (wins + 0.5 * draws) / total_games`. Now 2W/1L/7D gives 0.55, not 0.667.

16. **ELO Inheritance Across Restarts**: When training crashed (28 times in 9 days) and restarted, `train()` warm-started from `best.pt` and inherited the old ELO value. Combined with the draw-excluding win rate bug, ELO ratcheted up through multiple restarts without the model actually improving. The "ELO 1982" was accumulated across 3 warm-start chains starting from ELO 800.
   - Solution: ELO is now reset to 0 on every fresh start. Inherited ELO is logged but not used. Only the running self-play ELO within a single training session is meaningful.

17. **save_best() After Pre-training Destroyed Warm-Started Weights**: On fresh start with existing `best.pt`, the code: (1) warm-started both networks from `best.pt`, (2) pre-trained on seeds modifying `self.network`, (3) called `save_best()` which saved the seed-pretrained weights as `best.pt`. On the next crash+restart, the warm-start loaded this seed-pretrained model instead of the self-play-trained one. Combined with ELO inheritance, this meant every restart effectively reset the model to "seed-pretrained only" while keeping the old ELO label.
   - Solution: Skip pre-training when warm-starting (the model already learned from seeds in a prior run). Only pre-train on truly fresh starts. Also save an initial step checkpoint so `load_checkpoint()` can resume properly after crashes instead of falling through to "Starting fresh training".

18. **Random Opponent Check Was Non-Gating**: `evaluate_network()` checked random opponent results but only logged warnings. A model losing to random MCTS (W:0 L:2 D:2) could still be "accepted" based on vs-best win rate.
   - Solution: If random losses > random wins, force win_rate=0 to veto promotion.

19. **Best/Training Network Split Was AlphaGo Zero, Not AlphaZero**: The codebase used an AlphaGo Zero-style architecture with separate best and training networks, an eval gatekeeper, and promotion logic. Real AlphaZero uses a single continuously-updated network — DeepMind dropped the best/training split because it wastes ~50% of compute on eval games that produce zero training data. The eval gatekeeper also created perverse dynamics: ELO was measured relative to best (not absolute), and combined with draw-counting bugs (#15) and crash inheritance (#16), ELO ratcheted up without real improvement.
   - Solution: Removed `best_network` entirely. Single network self-plays and continuously trains. No eval gatekeeper, no consecutive rejection logic, no promotion. ELO is estimated via periodic MCTSLite probes (every 5 iterations, non-gating). `best.pt` is saved every iteration for benchmark use.

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

Opponents use the trained neural network (`checkpoints/best.pt`) with MCTS at varying simulation counts. The NN is the same model being trained on the Jetson — it produces the GM games for study AND serves as the benchmark opponent. ELO calibration must be done against NN-backed MCTS (not MCTSLite), since the calibrated ELO values are only meaningful relative to the actual opponent the agents will face. The `agent_cli.py` hardcoded defaults are placeholders — real runs must use `--calibration checkpoints/calibration.json` with a calibration generated against the current best.pt.

### Running the Benchmark with a Coding Agent

To test the benchmark with a coding agent (Claude Code, Codex CLI, etc.), launch a separate instance in an isolated sandbox directory so it can't read the game engine source code.

**Sandbox setup:**
```bash
# Create minimal sandbox with CLI script, game data, model, and calibration
SANDBOX="/tmp/benchmark-sandbox"  # or any temp directory
mkdir -p "$SANDBOX/scripts" "$SANDBOX/data" "$SANDBOX/checkpoints/agent_sessions"
cp scripts/agent_cli.py "$SANDBOX/scripts/"
cp -r data/gm_games "$SANDBOX/data/"
cp checkpoints/best.pt "$SANDBOX/checkpoints/"
cp checkpoints/calibration.json "$SANDBOX/checkpoints/"

# davechess must be installed as a package: pip install -e . (from repo root)
```

**Launch the agent (with streaming output for monitoring):**
```bash
cd "$SANDBOX" && claude -p "You are in a sandbox directory. \
  Read scripts/agent_cli.py — it has all the docs you need. \
  Run the full DaveChess benchmark: create a session \
  (--budget 500000 --eval-min-games 5 --eval-max-games 15 \
   --checkpoint checkpoints/best.pt --calibration checkpoints/calibration.json), \
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

**Automated benchmark (all 3 agents sequentially):**
```bash
# Ensure sandbox exists, then run all agents overnight
bash scripts/run_overnight_benchmark.sh
```
This runs Claude Code (`claude -p`), Codex CLI (`codex exec`), and Gemini CLI (`gemini`) sequentially in the sandbox. Results go to `benchmark_results/`, logs to `benchmark_logs/`. Expected runtime: 4-12 hours total.

**Individual agent launchers:**
```bash
python scripts/_launch_codex_benchmark.py   # Just Codex CLI
python scripts/_launch_gemini_benchmark.py  # Just Gemini CLI
```

**Heuristic baseline (no LLM):**
```bash
python scripts/_play_benchmark.py [agent-name]
```
Plays through the full benchmark using simple Python heuristics (capture priority, center control, promotions). Useful for generating baseline results and testing the pipeline without burning API tokens. Not a real benchmark run — the agent doesn't reason about moves.

**Benchmark integrity:** Agents must reason about each move themselves. The rules (in both `agent_cli.py` and `prompt.py`) explicitly prohibit writing game engines, search algorithms (minimax, MCTS, alpha-beta), or any automated move-selection code. Violations invalidate the score.

### Benchmark Pipeline Prerequisites

The trained neural network is the foundation of the entire benchmark. It serves two roles:

1. **GM game generator** — Self-play games from the trained model are the "grandmaster" games that agents study during the LEARNING phase. These are saved as DCN files in `data/gm_games/`.
2. **Benchmark opponent** — During BASELINE, PRACTICE, and EVALUATION, the agent plays against NN-backed MCTS at calibrated simulation counts. The opponent's strength is controlled by varying the number of MCTS simulations.

**Pipeline order:**
1. **Train the model** on Jetson (W&B tracks progress). Wait for sufficient ELO.
2. **Run `prepare_benchmark.py --all`** — This single script handles steps 2-4:
   - Downloads `best.pt` from the highest-ELO W&B run to `checkpoints/`
   - Downloads game logs from high-ELO training runs, splits multi-game DCN files into individual files in `data/gm_games/`
   - Runs round-robin ELO calibration (NN-MCTS at varying sim counts) and saves `checkpoints/calibration.json`
3. **Run benchmark** — Launch agents in sandboxes with the CLI, GM games, and calibration data.

Calibration is model-specific: if the model is retrained, the calibration must be re-run. The hardcoded defaults in `agent_cli.py` and `api_server.yaml` are rough placeholders; real runs must use the calibration JSON.

### Agentic Benchmark Architecture (Legacy Harness)

The `run_benchmark.py` harness gives an LLM a token budget and tools, then measures ELO achieved through autonomous learning. It calls the LLM API directly and enforces token budgets.

Key design decisions:
- **Rolling context window** (20 messages): forces agents to use external memory at large budgets
- **Eval from same budget**: agent must budget tokens wisely between learning and evaluation
- **Opponent pool**: ELO-to-NN-MCTS-sims mapping with log-space interpolation between calibrated points
- **Move-by-move play**: agent plays each move individually via tool calls (not pre-committed strategies)

Key files:
- `scripts/prepare_benchmark.py` — Full pipeline: download model + games from W&B, calibrate
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
  - `mcts_lite.py` - Lightweight MCTS with random rollouts (no NN), used for validation & seed generation
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
  - `opponent_pool.py` - ELO-to-MCTS-sims mapping with log-space interpolation (uses NN-backed MCTS)
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
- `scripts/calibrate_opponents.py` - Round-robin Glicko-2 calibration against NN-backed MCTS, saves calibration.json
- `scripts/run_overnight_benchmark.sh` - Sequential runner for all 3 agents (Claude Code, Codex CLI, Gemini CLI)
- `scripts/_play_benchmark.py` - Heuristic baseline player (no LLM, uses simple Python heuristics)
- `scripts/_launch_codex_benchmark.py` - Standalone Codex CLI benchmark launcher
- `scripts/_launch_gemini_benchmark.py` - Standalone Gemini CLI benchmark launcher
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
- `benchmark_results/` - Agent benchmark results JSON (currently placeholder data from heuristic player)
- `benchmark_logs/` - Agent output logs from benchmark runs (gitignored)
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
