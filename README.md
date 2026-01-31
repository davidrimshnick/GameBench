# GameBench

Benchmark measuring how efficiently LLMs learn novel strategic reasoning from examples.

A custom board game ("DaveChess") is designed to be absent from training data. An AlphaZero agent masters it via self-play on a Jetson Orin Nano. Synthetic grandmaster games are generated. Models are scored by how well they learn to play from in-context examples.

**Headline metric:** GameBench Score (0-100) = normalized area under the ELO-vs-N learning curve, where N = number of example games shown in context.

## DaveChess

An original strategic board game on an 8x8 grid with 8 resource nodes.

**Pieces:**

| Piece | Symbol | Move | Strength | Deploy Cost |
|-------|--------|------|----------|-------------|
| Commander | C | 1 square, any direction | 2 | -- (starts on board) |
| Warrior | W | 1 square, orthogonal | 1 (+1 per adjacent friendly Warrior) | 2 |
| Rider | R | Up to 2 squares, straight line | 2 | 4 |
| Bombard | B | 1 square any dir; ranged attack at 2 squares (not vs Commander) | 0 (melee) | 5 |

**Check:** If your Commander is under attack, you must resolve it (move, block, or capture). Cannot leave your Commander in check.

**Win conditions (checked in order):**
1. Checkmate opponent's Commander (no legal escape from check)
2. Occupy 4+ of 8 resource nodes with your pieces
3. Turn 100: most exclusive nodes, tiebreak by piece count

**Turn structure:** Gain +1 resource per controlled node, then move a piece OR deploy a new piece.

## Project Structure

```
GameBench/
├── davechess/
│   ├── game/          # Game engine: state, rules, board, notation
│   ├── engine/        # MCTS (lite + full), neural network, self-play, training
│   ├── benchmark/     # LLM evaluation: protocol, scoring, prompt construction
│   └── data/          # Game generation, ELO/Glicko-2 ratings, DCN storage
├── configs/           # YAML configs for training, generation, benchmark
├── scripts/           # CLI entry points
│   ├── play.py        # Interactive play (human or MCTS)
│   ├── validate_game.py  # Game health metrics (Phase 1.5)
│   ├── train.py       # AlphaZero training
│   ├── generate_games.py
│   ├── calibrate_elo.py
│   ├── run_benchmark.py
│   └── compare_models.py
└── tests/             # 94 tests across 5 suites
```

## Phases

1. **Game Engine** -- DaveChess rules, state representation, DCN notation, CLI
2. **Game Validation** -- Lightweight MCTS stress-tests rules for degenerate strategies
3. **AlphaZero Training** -- ResNet policy+value network (670K params), PUCT MCTS, self-play
4. **Game Generation & ELO Calibration** -- Calibrated opponent ladder (random to 1600 sims)
5. **LLM Benchmark** -- Evaluate models across N=0..500 example games, compute GameBench Score

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests
pytest

# Play interactively
python scripts/play.py

# Play against MCTS bot (100 simulations/move)
python scripts/play.py --vs-mcts 100

# Watch MCTS vs MCTS
python scripts/play.py --mcts-vs-mcts 50 50

# Validate game health (Phase 1.5)
python scripts/validate_game.py --num-games 500 --sims 50

# Train AlphaZero (requires CUDA)
python scripts/train.py --config configs/training.yaml

# Run benchmark on an LLM
python scripts/run_benchmark.py --model gpt-4 --config configs/benchmark.yaml
```

## Hardware

Designed for Jetson Orin Nano Super (8GB shared RAM, 6-core ARM, CUDA 12.6).

## Dependencies

- `torch` (training/inference)
- `numpy`, `pyyaml`, `tqdm`
- `openai` (LLM benchmark)
- `matplotlib` (charts)
