# GameBench

A benchmark for **agentic learning** — measuring how well LLMs acquire novel strategic reasoning from experience, not memorization.

LLMs trained on internet-scale data have seen chess, Go, and every popular board game. To test whether a model can actually *learn* strategy rather than recall it, we need a game that doesn't exist in any training corpus. GameBench uses **DaveChess**, a custom board game designed from scratch, paired with an AlphaZero engine that generates expert-level games as training signal.

The core question: **given a token budget and tools, how well can an LLM autonomously learn to play a game it has never seen?**

**Headline metric:** GameBench Score (0-100) = normalized area under the ELO-vs-budget learning curve (log-scale), measured across token budgets from 100K to 10M.

## DaveChess

An original strategic board game on an 8x8 grid combining positional resource control with tactical combat.

### Board & Setup

Each side starts with 12 pieces on their back two rows: 1 Commander (king), 3 Riders (cavalry), 2 Bombards (artillery), and 6 Warriors (infantry). The board has 4 **Gold nodes** ($) in the center at fixed symmetric positions — control these for resource income.

### Resources & Promotion

Each turn, you earn **+1 resource for each Gold node you have a piece directly on**. You spend resources to **promote pieces** — upgrading them in place to a stronger type. This is the core economic loop: control Gold nodes → earn resources → promote pieces → strengthen your army.

| Piece | Symbol | Promotion Cost | Move | Capture |
|-------|--------|----------------|------|---------|
| Commander | C | cannot promote | 1 square, any direction | Same as move |
| Warrior | W | base piece | 1 square forward | 1 square diagonal-forward |
| Rider | R | 3 | Up to 3 squares, any straight line (no jumping) | Same as move |
| Bombard | B | 5 | 1 square, any direction | Melee: same as move. Ranged: exactly 2 squares, straight line, clear path (stays in place, cannot target Commanders) |
| Lancer | L | 7 | Up to 4 squares, diagonal only, can jump one piece | Same as move |

### Promotion

Spend resources to upgrade any non-Commander piece to a higher type, in place. The piece stays on its square and changes type. Cost = full price of the target type. Any piece can promote to Rider (3), Bombard (5), or Lancer (7). Commanders cannot promote. No new pieces are ever deployed — what you start with is all you get.

### Capture

**Chess-style:** move onto an enemy piece to capture it. The attacker always takes the defender's square — any piece can capture any piece. This enables sacrifices, forks, pins, and the full range of chess-like tactics.

**Warriors** move 1 square forward and capture 1 square diagonal-forward (like chess pawns). Warriors cannot move backward or sideways, creating irreversible pawn structure.

**Bombard ranged attack:** attacks at exactly 2 squares distance (straight line, clear path). Target is removed; Bombard stays put. Cannot target Commanders with ranged attacks.

**Lancer:** Diagonal-only piece that can jump over exactly one piece (friend or foe) in its path, similar to a limited bishop with jumping. At promotion cost 7, the Lancer is a powerful mid-game attacker.

### Check & Checkmate

If your Commander is under attack, you **must** resolve it (move, block, or capture). You cannot make a move that leaves your Commander in check. If no legal move resolves check, it's **checkmate** — you lose.

### Win Conditions

1. **Checkmate** opponent's Commander → you win
2. **Turn 100** with no checkmate → draw
3. **Threefold repetition** of the same board position → draw
4. **50-move rule**: 50 moves per side with no capture or promotion → draw

### Turn Structure

1. Gain resources from controlled Gold nodes
2. Take one action: **move** a piece OR **promote** a piece

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
└── tests/             # 149 tests across 6 suites
```

## Phases

1. **Game Engine** -- DaveChess rules, state representation, DCN notation, CLI
2. **Game Validation** -- Lightweight MCTS stress-tests rules for degenerate strategies
3. **AlphaZero Training** -- ResNet policy+value network, PUCT MCTS, self-play
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
