# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GameBench is a benchmark measuring how efficiently LLMs learn novel strategic reasoning from examples using DaveChess, a custom board game designed to be absent from training data. The system uses AlphaZero self-play training on a Jetson Orin Nano to create synthetic grandmaster games for LLM evaluation.

## Key Commands

### Development & Testing
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests (142 tests across 6 suites)
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

1. **Value Target Structure**: Uses binary rewards (1.0 for wins, 0.0 for losses/draws) to encourage aggressive play, as the game tends toward defensive stalemates.

2. **Smart Seed Generation**: Instead of MCTSLite's expensive random rollouts, uses heuristic players (`HeuristicPlayer`, `CommanderHunter`) that provide strategic seed games. Seeds are generated once and saved to `checkpoints/smart_seeds.pkl` (~20k positions, backed up as W&B artifact `smart-seeds:latest`). **Not committed to git** (233MB) — download from W&B if missing. Training automatically loads these instead of regenerating. All seed games end in checkmate (draws are discarded). The model pre-trains on seed data before starting self-play. Located in:
   - `davechess/engine/heuristic_player.py` - Position evaluation and strategic play
   - `davechess/engine/smart_seeds.py` - Seed game generation module
   - `scripts/generate_and_save_seeds.py` - Standalone seed generator

3. **Adaptive MCTS Simulations**: The `adaptive_simulations()` function in `training.py` scales simulation count based on model ELO (25 sims at ELO 0 → 100 sims at ELO 2000) to speed up early training. Min sims raised from 10→25 (selfplay) and 15→30 (eval) because too-shallow search produced low-quality data that couldn't learn from seed strategies.

### Game State Representation

- **Board**: 8x8 grid stored as `state.board[row][col]` containing `Piece` objects
- **Nodes**: 4 Gold nodes (give resource income) at (3,3), (3,4), (4,3), (4,4) + 4 Power nodes (give +1 strength to adjacent pieces) at (2,1), (2,6), (5,1), (5,6)
- **Neural Network Input**: 15 planes (5 piece types × 2 players + 2 node types + player + 2 resources) via `state_to_planes()`
- **Move Encoding**: Policy size of 2816 (64×44 move slots per square) via `move_to_policy_index()`

### Critical Game Rules

**When changing game rules, update ALL three locations: `davechess/game/rules.py`, `davechess/benchmark/prompt.py` (RULES_TEXT), and `README.md` (DaveChess section).**

1. **Commander Safety**: Must resolve check immediately (move/block/capture)
2. **Win Conditions**: Checkmate opponent's Commander (only way to win). Turn 100 with no checkmate = draw.
3. **Piece Types**: Commander (C, str 2), Warrior (W, str 1 + adjacency), Rider (R, str 2), Bombard (B, str 0), Lancer (L, str 3, diagonal up to 4 squares with jump)
4. **Deploy Costs**: W=2, R=4, B=5, L=6
5. **Warrior Strength**: Base 1 + 1 per adjacent friendly Warrior (clustering bonus)
6. **Power Node Bonus**: Any piece on or adjacent (8-directional) to a Power node gets +1 strength
7. **Bombard**: Can't use ranged attack against Commander (prevents cheese)
8. **Lancer**: Moves diagonally up to 4 squares, can jump over exactly one piece (any color)

### Known Issues & Solutions

1. **Defensive Stalemates**: Games naturally tend toward long defensive play ending in turn-100 draws
   - Solution: Commander hunting strategies, aggressive heuristics, binary value targets (draws = 0.0, same as losses)

2. **Slow Seed Generation**: MCTSLite with random rollouts takes minutes per game
   - Solution: Heuristic players provide 10x faster seed generation

3. **Training Instability**: Early iterations produce all draws/max-length games
   - Solution: Smart seeds with strategic play, adaptive simulation counts

4. **Replay Buffer OOM on Checkpoint Load**: The compressed NPZ buffer (~5MB on disk) decompresses to ~1.5GB (policies alone are 1.03GB at 96K×2816). Loading all arrays at once caused OOM on the 8GB Jetson.
   - Solution: `load_data()` loads/deletes arrays sequentially so peak memory is ~max(single_array) not sum(all_arrays)

5. **CUDA Fragmentation After Self-play**: Higher adaptive sims fragment GPU memory, causing `NVML_SUCCESS` assert failures when transitioning to training.
   - Solution: `torch.cuda.empty_cache()` before training phase

6. **Temperature Threshold Too Low**: At `temperature_threshold: 30`, the model played near-deterministically after move 30 with noisy policies, entrenching defensive patterns.
   - Solution: Raised to 60 to maintain exploration longer in self-play

### Hardware Constraints (Jetson Orin Nano)

- 8GB shared RAM (CPU + GPU)
- Sequential GPU inference only (no multiprocessing)
- Batch size 128, 5 ResBlocks, 64 filters optimized for memory
- Training uses mixed precision (FP16) when available
- Replay buffer at 100K positions uses ~1.1GB RAM — must load arrays sequentially to avoid OOM
- Always call `torch.cuda.empty_cache()` before switching between self-play and training phases

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

Key metrics to watch:
- `selfplay/avg_game_length` - Should decrease over time; turn 100 = draw limit
- `selfplay/draw_rate` - Should decrease as model learns to checkmate before turn 100
- `selfplay/white_win_rate` - Should stay near 0.5 for balance
- `eval/win_rate` - Must exceed 0.55 to update best model
- `training/policy_loss` - Should decrease over iterations