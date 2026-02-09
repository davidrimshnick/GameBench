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

### Agentic Benchmark
```bash
# Run agentic benchmark (single budget)
python scripts/run_agentic_benchmark.py --model gpt-4 --provider openai --budget 100000

# Run at multiple budget levels (100K, 1M, 10M)
python scripts/run_agentic_benchmark.py --model claude-3-5-sonnet-20241022 --provider anthropic --multi-budget

# Use config file
python scripts/run_agentic_benchmark.py --config configs/agentic_benchmark.yaml

# Calibrate opponent pool (generates checkpoints/calibration.json)
python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt
```

### Game Analysis & Seed Generation
```bash
# Validate game balance with MCTSLite
python scripts/validate_game.py --num-games 500 --sims 50

# Generate and save smart seed games (only needed once)
python scripts/generate_and_save_seeds.py --num-games 500 --output checkpoints/smart_seeds.pkl

# Check existing seed games
python scripts/generate_and_save_seeds.py  # Shows stats without regenerating

# Seeds are also stored as W&B artifact (smart-seeds:latest)

# Quick balance check
python scripts/quick_balance_check.py

# Detailed game analysis
python scripts/analyze_games.py
```

## Architecture & Key Design Decisions

### Training Pipeline

The AlphaZero implementation has several critical modifications for DaveChess:

1. **Value Target Structure**: Uses standard AlphaZero three-valued targets (+1.0 for wins, 0.0 for draws, -1.0 for losses) with tanh output head. Drawn games are included in training data so the model learns that draw-prone positions are worse than winning ones. Earlier versions incorrectly used 0/1 targets with a tanh head, causing the network to treat losses as neutral (tanh midpoint = 0 = "uncertain"). Another earlier version discarded draws entirely, but with the threefold repetition rule, draws carry meaningful signal.

2. **Smart Seed Generation**: Instead of MCTSLite's expensive random rollouts, uses heuristic players (`HeuristicPlayer`, `CommanderHunter`) that provide strategic seed games. Seeds are generated once and saved to `checkpoints/smart_seeds.pkl` (~20k positions, backed up as W&B artifact `smart-seeds:latest`). **Not committed to git** (233MB) — download from W&B if missing. Training automatically loads these instead of regenerating. All seed games end in checkmate (draws are discarded). The model pre-trains on seed data before starting self-play. Located in:
   - `davechess/engine/heuristic_player.py` - Position evaluation and strategic play
   - `davechess/engine/smart_seeds.py` - Seed game generation module
   - `scripts/generate_and_save_seeds.py` - Standalone seed generator

3. **Adaptive MCTS Simulations**: The `adaptive_simulations()` function in `training.py` scales simulation count based on model ELO (25 sims at ELO 0 → 100 sims at ELO 2000) to speed up early training. Min sims raised from 10→25 (selfplay) and 15→30 (eval) because too-shallow search produced low-quality data that couldn't learn from seed strategies.

### Game State Representation

- **Board**: 8x8 grid stored as `state.board[row][col]` containing `Piece` objects
- **Nodes**: 4 Gold nodes (give resource income for promotion) at (3,3), (3,4), (4,3), (4,4)
- **Neural Network Input**: 14 planes (5 piece types × 2 players + gold nodes + player + 2 resources) via `state_to_planes()`
- **Move Encoding**: Policy size of 2816 (64×44 move slots per square) via `move_to_policy_index()`. Slots 40-42 = promotion targets (R, B, L).

### Critical Game Rules

**When changing game rules, update ALL three locations: `davechess/game/rules.py`, `davechess/benchmark/prompt.py` (RULES_TEXT), and `README.md` (DaveChess section).**

1. **Capture (Chess-style)**: Any piece can capture any piece by moving onto it. Attacker always takes the defender's square — no strength comparison. Enables sacrifices, forks, pins, and tactical depth.
2. **Commander Safety**: Must resolve check immediately (move/block/capture). Cannot make a move that leaves own Commander in check.
3. **Win Conditions**: Checkmate opponent's Commander (only way to win). Turn 100 with no checkmate = draw. Threefold repetition of the same position (board + player, excluding resources) = draw. 50-move rule: 50 moves per side (100 halfmoves) with no capture or promotion = draw.
4. **Piece Types**: Commander (C), Warrior (W, pawn-like), Rider (R, up to 2 squares any direction), Bombard (B, 1 sq move + ranged attack), Lancer (L, diagonal up to 4 squares with jump)
5. **Promotion Costs**: R=5, B=7, L=9. No deployment — pieces promote in place by spending Gold resources.
6. **Starting Army**: 12 pieces per side — 1 Commander, 2 Riders, 1 Bombard, 8 Warriors. No new pieces are ever added.
7. **Warrior Movement**: Forward only (like chess pawns). Captures diagonal-forward only. White Warriors move toward row 8, Black toward row 1. No retreat, no sideways movement.
8. **Bombard**: 1 square movement any direction. Ranged attack at exactly 2 squares (straight line, clear path). Stays in place when attacking. Can't use ranged attack against Commander.
9. **Lancer**: Moves diagonally up to 4 squares, can jump over exactly one piece (any color)

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

8. **Replay Buffer OOM on Checkpoint Load/Save**: The buffer at 50K+ positions contains ~960MB of planes and ~563MB of policies. Stacking the full buffer into a contiguous array for save/load caused OOM on the 8GB Jetson.
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
- Replay buffer capped at 50K positions (~940MB). Save/load uses chunked I/O (5K chunks) to avoid temp array spikes
- All buffer data enforced as float32 on push — prevents accidental float64 doubling
- Always call `torch.cuda.empty_cache()` before switching between self-play, training, and eval phases
- Memory usage logged at phase transitions (`_log_memory()`) for OOM diagnosis
- MCTS trees explicitly freed after eval via `del` + `gc.collect()` (circular parent↔child refs)

### Agentic Benchmark Architecture

The agentic benchmark gives an LLM a token budget and tools, then measures ELO achieved through autonomous learning:

1. **Learning Phase**: Agent autonomously studies GM games and plays practice games using 4 tools:
   - `study_games(n)` — retrieve N grandmaster games in DCN notation
   - `start_practice_game(opponent_elo)` — start a game against calibrated opponent
   - `play_move(game_id, move_dcn)` — play a move, get opponent's response
   - `get_game_state(game_id)` — view board, legal moves, history

2. **Evaluation Phase**: Sequential Glicko-2 testing against calibrated opponents. Harness controls opponent selection (near estimated ELO for max info gain). Stops when rating deviation < 50 or max 200 games.

3. **Scoring**: Log-scale AUC across budget levels (100K/1M/10M tokens), normalized to 0-100.

Key design decisions:
- **Rolling context window** (20 messages): forces agents to use external memory at large budgets
- **Eval from same budget**: agent must budget tokens wisely between learning and evaluation
- **Opponent pool**: ELO-to-MCTS-sims mapping with log-space interpolation between calibrated points
- **Move-by-move play**: agent plays each move individually via tool calls (not pre-committed strategies)

Key files:
- `davechess/benchmark/agent_harness.py` — Core agentic loop with rolling window
- `davechess/benchmark/sequential_eval.py` — Glicko-2 adaptive ELO measurement
- `davechess/benchmark/agentic_protocol.py` — Orchestrator (learning → eval → scoring)
- `davechess/benchmark/tools.py` — Tool definitions and executor
- `davechess/benchmark/opponent_pool.py` — Calibrated opponent ELO interpolation
- `davechess/benchmark/token_tracker.py` — Token budget tracking
- `configs/agentic_benchmark.yaml` — Full configuration

## File Structure Notes

- `davechess/game/` - Core game engine, state representation, rules
- `davechess/engine/` - Neural network, MCTS, training loop, heuristic players
- `davechess/data/` - Game generation, ELO calibration
- `davechess/benchmark/` - LLM evaluation framework (legacy static + agentic)
- `configs/` - YAML configurations for all components
- `checkpoints/` - Model checkpoints and seed game storage (gitignored, use W&B artifacts)
- `wandb/` - Weights & Biases tracking (auto-generated, gitignored)

## Training Monitoring

**CRITICAL: NEVER start a training run without W&B connected! Verify W&B shows "logging enabled" in the logs before proceeding. If W&B fails with "Broken pipe", kill any stale wandb processes first (`pkill -9 wandb`).**

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