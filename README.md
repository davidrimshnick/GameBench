# GameBench

A benchmark for **agentic learning** — measuring how well LLMs acquire novel strategic reasoning from experience, not memorization.

LLMs trained on internet-scale data have seen chess, Go, and every popular board game. To test whether a model can actually *learn* strategy rather than recall it, we need a game that doesn't exist in any training corpus. GameBench uses **DaveChess**, a custom board game designed from scratch, paired with an AlphaZero engine that generates expert-level games as training signal.

The core question: **given a token budget and tools, how well can an LLM autonomously learn to play a game it has never seen?**

**Headline metric:** GameBench Score (0-100) = normalized area under the ELO-vs-budget learning curve (log-scale), measured across token budgets from 100K to 10M.

## DaveChess

An original strategic board game on an 8x8 grid combining positional resource control with tactical combat.

### Board & Setup

Each side starts with 6 pieces on their back two rows: a Commander (king), 4 Warriors (infantry), and a Rider (cavalry). The board has two types of nodes at fixed symmetric positions:

- **Gold nodes** ($) — 4 in the center. Control these for resource income.
- **Power nodes** (^) — 2 near each side. Pieces near these get +1 combat strength.

### Resources & Deployment

Each turn, you earn **+1 resource for each Gold node you have a piece on or adjacent to** (orthogonally). You spend resources to **deploy new pieces** onto empty squares in your back two rows. This is the core economic loop: control Gold nodes → earn resources → deploy reinforcements → control more territory.

| Piece | Symbol | Deploy Cost | Move | Strength |
|-------|--------|-------------|------|----------|
| Commander | C | starts on board | 1 square, any direction | 2 |
| Warrior | W | 2 | 1 square, orthogonal | 1 (+1 per adjacent friendly Warrior) |
| Rider | R | 4 | Up to 2 squares, straight line | 2 |
| Bombard | B | 5 | 1 square, any direction | 0 (melee) |
| Lancer | L | 6 | Up to 4 squares, diagonal only, can jump one piece | 3 |

### Node Types

**Gold nodes** provide economic power — each one you control gives +1 resource per turn.

**Power nodes** provide military power — any piece on or adjacent to a Power node gets **+1 strength** in combat. This creates a strategic tension: Gold nodes fuel your economy, Power nodes strengthen your army. You must decide where to invest your forces.

### Combat

Move onto an enemy piece to attack. Compare total strength (including Power node bonus) — higher wins, tie removes both. **Warrior clustering** is key: a lone Warrior has strength 1, but three adjacent Warriors each have strength 3.

**Bombard special ability:** Ranged attack at exactly 2 squares distance (straight line, clear path). Target is removed; Bombard stays put. Cannot target Commanders with ranged attacks.

**Lancer:** Diagonal-only piece that can jump over exactly one piece (friend or foe) in its path, similar to a limited bishop with jumping. At strength 3 and deploy cost 6, the Lancer is a powerful late-game attacker.

### Check & Checkmate

If your Commander is under attack, you **must** resolve it (move, block, or capture). You cannot make a move that leaves your Commander in check. If no legal move resolves check, it's **checkmate** — you lose.

### Win Conditions

1. **Checkmate** opponent's Commander → you win
2. **Turn 100** with no checkmate → draw
3. **Threefold repetition** of the same board position → draw
4. **50-move rule**: 50 moves per side with no capture or deploy → draw

### Turn Structure

1. Gain resources from controlled Gold nodes
2. Take one action: **move** a piece OR **deploy** a new piece

## Agentic Benchmark

The benchmark gives each model a **token budget** (100K / 1M / 10M) and four tools:

| Tool | Description |
|------|-------------|
| `study_games(n)` | Retrieve N grandmaster games in DCN notation |
| `start_practice_game(opponent_elo)` | Start a game against a calibrated opponent |
| `play_move(game_id, move_dcn)` | Play a move and receive opponent's response |
| `get_game_state(game_id)` | View current board, legal moves, and history |

The agent autonomously decides how to spend its budget: studying games, playing practice matches, reasoning about strategy. After the learning phase, ELO is measured via Glicko-2 sequential testing against calibrated opponents (AlphaZero + variable MCTS simulations).

Key design features:
- **Rolling 20-message context window** — at 10M tokens, the agent must use external memory
- **Evaluation from the same budget** — agents must balance learning vs evaluation
- **Log-scale AUC scoring** — each order of magnitude (100K→1M→10M) weighted equally

## Project Structure

```
GameBench/
├── davechess/
│   ├── game/          # Game engine: state, rules, board, notation
│   ├── engine/        # MCTS (lite + full), neural network, self-play, training
│   ├── benchmark/     # LLM evaluation: agentic + legacy static benchmark
│   │   ├── agent_harness.py     # Core agentic loop
│   │   ├── agentic_protocol.py  # Orchestrator (learn → eval → score)
│   │   ├── tools.py             # Tool definitions and executor
│   │   ├── sequential_eval.py   # Glicko-2 adaptive ELO testing
│   │   ├── opponent_pool.py     # Calibrated ELO → MCTS sims
│   │   └── token_tracker.py     # Budget tracking
│   └── data/          # Game generation, ELO/Glicko-2 ratings, DCN storage
├── configs/           # YAML configs for training, generation, benchmark
├── scripts/           # CLI entry points
│   ├── play.py                  # Interactive play (human or MCTS)
│   ├── validate_game.py         # Game health metrics
│   ├── train.py                 # AlphaZero training
│   ├── run_agentic_benchmark.py # Agentic benchmark CLI
│   ├── calibrate_opponents.py   # Generate opponent calibration data
│   ├── run_benchmark.py         # Legacy static benchmark
│   └── compare_models.py
└── tests/             # 142 tests across 6 suites
```

## Phases

1. **Game Engine** -- DaveChess rules, state representation, DCN notation, CLI
2. **Game Validation** -- Lightweight MCTS stress-tests rules for degenerate strategies
3. **AlphaZero Training** -- ResNet policy+value network (746K params), PUCT MCTS, self-play
4. **Game Generation & ELO Calibration** -- Calibrated opponent ladder (random to 1600 sims)
5. **Agentic Benchmark** -- Token budget + tools, autonomous learning, Glicko-2 ELO measurement

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

# Run agentic benchmark
python scripts/run_agentic_benchmark.py --model gpt-4 --provider openai --budget 100000

# Run at multiple budget levels (100K, 1M, 10M)
python scripts/run_agentic_benchmark.py --model gpt-4 --provider openai --multi-budget

# Legacy static benchmark
python scripts/run_benchmark.py --model gpt-4 --config configs/benchmark.yaml
```

## Hardware

Designed for Jetson Orin Nano Super (8GB shared RAM, 6-core ARM, CUDA 12.6).

## Dependencies

- `torch` (training/inference)
- `numpy`, `pyyaml`, `tqdm`
- `openai` (LLM benchmark, agentic benchmark)
- `anthropic` (agentic benchmark, optional)
- `matplotlib` (charts)
