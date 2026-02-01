# GameBench

A benchmark for **agentic learning** — measuring how well LLMs acquire novel strategic reasoning from experience, not memorization.

LLMs trained on internet-scale data have seen chess, Go, and every popular board game. To test whether a model can actually *learn* strategy rather than recall it, we need a game that doesn't exist in any training corpus. GameBench uses **DaveChess**, a custom board game designed from scratch, paired with an AlphaZero engine that generates expert-level games as training signal.

The core question: **given N example games of a game you've never seen, how quickly can you learn to play it well?**

**Headline metric:** GameBench Score (0-100) = normalized area under the ELO-vs-N learning curve, where N = number of example games shown in context.

## DaveChess

An original strategic board game on an 8x8 grid combining positional resource control with tactical combat.

### Board & Setup

Each side starts with 6 pieces on their back two rows: a Commander (king), 4 Warriors (infantry), and a Rider (cavalry). The board has **8 resource nodes** at fixed symmetric positions — 2 near each side, 4 in the center.

### Resources & Deployment

Each turn, you earn **+1 resource for each node you have a piece on or adjacent to** (orthogonally). You spend resources to **deploy new pieces** onto empty squares in your back two rows. This is the core economic loop: control nodes → earn resources → deploy reinforcements → control more nodes.

| Piece | Symbol | Deploy Cost | Move | Strength |
|-------|--------|-------------|------|----------|
| Commander | C | starts on board | 1 square, any direction | 2 |
| Warrior | W | 2 | 1 square, orthogonal | 1 (+1 per adjacent friendly Warrior) |
| Rider | R | 4 | Up to 2 squares, straight line | 2 |
| Bombard | B | 5 | 1 square, any direction | 0 (melee) |

### Combat

Move onto an enemy piece to attack. Compare total strength — higher wins, tie removes both. **Warrior clustering** is key: a lone Warrior has strength 1, but three adjacent Warriors each have strength 3.

**Bombard special ability:** Ranged attack at exactly 2 squares distance (straight line, clear path). Target is removed; Bombard stays put. Cannot target Commanders with ranged attacks.

### Check & Checkmate

If your Commander is under attack, you **must** resolve it (move, block, or capture). You cannot make a move that leaves your Commander in check. If no legal move resolves check, it's **checkmate** — you lose.

### Win Conditions

1. **Checkmate** opponent's Commander → you win
2. **Turn 100** with no checkmate → draw

### Turn Structure

1. Gain resources from controlled nodes
2. Take one action: **move** a piece OR **deploy** a new piece

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
