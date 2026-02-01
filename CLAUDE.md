# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GameBench is a benchmark measuring how efficiently LLMs learn novel strategic reasoning from examples using DaveChess, a custom board game designed to be absent from training data. The system uses AlphaZero self-play training on a Jetson Orin Nano to create synthetic grandmaster games for LLM evaluation.

## Key Commands

### Development & Testing
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests (94 tests across 5 suites)
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

2. **Smart Seed Generation**: Instead of MCTSLite's expensive random rollouts, uses heuristic players (`HeuristicPlayer`, `CommanderHunter`) that provide strategic seed games. Seeds are generated once and saved to `checkpoints/smart_seeds.pkl` (~20k positions, also backed up as W&B artifact `smart-seeds:latest`). Training automatically loads these instead of regenerating. At least one side must always be a `CommanderHunter` to ensure games end decisively. The model pre-trains on seed data before starting self-play. Located in:
   - `davechess/engine/heuristic_player.py` - Position evaluation and strategic play
   - `davechess/engine/smart_seeds.py` - Seed game generation module
   - `scripts/generate_and_save_seeds.py` - Standalone seed generator

3. **Adaptive MCTS Simulations**: The `adaptive_simulations()` function in `training.py` scales simulation count based on model ELO (2 sims at ELO 0 → 200 sims at ELO 2000) to speed up early training.

### Game State Representation

- **Board**: 8x8 grid stored as `state.board[row][col]` containing `Piece` objects
- **Resources**: 8 fixed nodes at positions (0,3), (0,4), (7,3), (7,4), (3,0), (4,0), (3,7), (4,7)
- **Neural Network Input**: 12 planes (6 piece types × 2 players) via `state_to_planes()`
- **Move Encoding**: Policy size of 576 (64×9 actions per square) via `move_to_policy_index()`

### Critical Game Rules

1. **Commander Safety**: Must resolve check immediately (move/block/capture)
2. **Win Conditions** (checked in order):
   - Checkmate opponent's Commander
   - Control 5+ resource nodes
   - Turn 100: most exclusive nodes
3. **Warrior Strength**: Base 1 + 1 per adjacent friendly Warrior (clustering bonus)
4. **Bombard**: Can't use ranged attack against Commander (prevents cheese)

### Known Issues & Solutions

1. **Defensive Stalemates**: Games naturally tend toward 200+ move defensive play
   - Solution: Commander hunting strategies, aggressive heuristics, binary value targets

2. **Slow Seed Generation**: MCTSLite with random rollouts takes minutes per game
   - Solution: Heuristic players provide 10x faster seed generation

3. **Training Instability**: Early iterations produce all draws/max-length games
   - Solution: Smart seeds with strategic play, adaptive simulation counts

### Hardware Constraints (Jetson Orin Nano)

- 8GB shared RAM (CPU + GPU)
- Sequential GPU inference only (no multiprocessing)
- Batch size 128, 5 ResBlocks, 64 filters optimized for memory
- Training uses mixed precision (FP16) when available

## File Structure Notes

- `davechess/game/` - Core game engine, state representation, rules
- `davechess/engine/` - Neural network, MCTS, training loop, heuristic players
- `davechess/data/` - Game generation, ELO calibration
- `davechess/benchmark/` - LLM evaluation framework
- `configs/` - YAML configurations for all components
- `checkpoints/` - Model checkpoints and seed game storage
- `wandb/` - Weights & Biases tracking (auto-generated)

## Training Monitoring

**CRITICAL: NEVER start a training run without W&B connected! Verify W&B shows "logging enabled" in the logs before proceeding. If W&B fails with "Broken pipe", kill any stale wandb processes first (`pkill -9 wandb`).**

Training logs to multiple destinations:
- **W&B Dashboard** - Primary monitoring at https://wandb.ai (MUST be connected)
- Console output
- `logs/training_log.jsonl` - Structured metrics
- TensorBoard at `logs/tensorboard/`

Large files (seeds, checkpoints) should be stored as W&B artifacts, not committed to git (GitHub 100MB limit).

Key metrics to watch:
- `selfplay/avg_game_length` - Should be 30-100 moves, not 200+
- `selfplay/white_win_rate` - Should stay near 0.5 for balance
- `eval/win_rate` - Must exceed 0.55 to update best model
- `training/policy_loss` - Should decrease over iterations