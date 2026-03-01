# CLAUDE.md

**Do NOT use plan mode (EnterPlanMode) — it is glitchy. Just explore and implement directly.**

**Before deleting checkpoints, buffers, or training artifacts:** Consider whether existing data can be reused (e.g., fix optimizer in-place rather than nuking buffer + retraining).

## Project Overview

GameBench measures how efficiently LLMs learn novel strategic reasoning from examples using DaveChess, a custom board game absent from training data. AlphaZero self-play training on a Jetson Orin Nano produces an expert NN that generates synthetic GM games for LLM study and acts as the calibrated benchmark opponent.

## Current Status (Feb 2026)

**Fresh restart (2026-02-24).** Previous run stalled at 39 iterations (vs-random ~44%, ELO ~300-358). Value head noise overrode correct policy priors via MCTS Q-values. Three fixes: (1) board flipping, (2) cpuct 1.5→4.0, (3) value_loss_weight 20→3. Seeds disabled. See Known Issue #26.

**Update (2026-03-01).** Added fail-closed guards for self-play/training so worker failures or non-finite losses cannot silently continue and overwrite `best.pt` (Known Issue #28).

### What's done
- All training pipeline bugs fixed (Known Issues #15-28)
- **Pure AlphaZero** — single continuously-updated network, no best/training split, no eval gatekeeper
- **Muon optimizer** — Newton-Schulz for trunk conv weights ONLY (21 params), SGD for heads/FC/biases/BN (54 params). **Never apply Muon to classification heads** (Known Issue #20).
- **Board flipping** — Black board flipped vertically so CNN always sees current player "moving up." Requires `flip=True` on `move_to_policy_index()` for Black. Incompatible with old checkpoints.
- **128 standard MCTS sims** — Gumbel disabled (Known Issue #25), standard PUCT explores more broadly
- **cpuct=4.0** — Trust policy prior over noisy value head. Reduce toward 2.0 once value head is reliable.
- **value_loss_weight=3.0** — Reduced from 20 (~7% of gradient vs 28%), letting policy lead training
- **Policy entropy reg** (`policy_entropy_weight: 0.1`), policy head freezing, policy target smoothing (configurable)
- **Reduced training intensity** — 200 steps/iter, Muon LR 0.01, head LR 0.001
- **Seeds disabled** (4:1 White bias). Pure self-play from random init.
- **40 games/iter**, MCTSLite ELO probes every 5 iters (800 NN sims vs MCTSLite-50 baseline ≈ 300 ELO)
- **Hot-reload config** — `training.yaml` re-read each iteration (architecture preserved)
- AMP overflow guard, hot-reload buffer resizing/LR, existing buffer carry-over, vs-random excluded from buffer, grad clipping scoped to SGD heads only
- Fail-closed training guards: abort on incomplete multiprocess self-play results, zero-example self-play batches, or non-finite losses (prevents silent checkpoint poisoning)
- Benchmark: sandbox + `agent_cli.py`, leaderboard at `docs/index.html`, automation scripts, integrity rules
- `DaveChessNetwork.from_checkpoint()` auto-infers architecture from weights

### What's needed
1. **Wait for ELO probe at iteration 20** — if ELO > 300, keep training; if ≤ 300, consider increasing value_loss_weight or smaller network
2. **Run calibration** once model beats random: `python scripts/calibrate_opponents.py --checkpoint checkpoints/best.pt`
3. **Run agent benchmarks** in sandbox with calibration.json
4. **Update leaderboard** with real results

## Key Commands

```bash
# Dev & Testing
pip install -e ".[dev]"
pytest                                        # All tests
pytest tests/test_game.py -v                  # Specific suite

# Interactive play
python scripts/play.py --vs-mcts 100         # Human vs MCTS
python scripts/play.py --mcts-vs-mcts 50 50  # Watch MCTS vs MCTS

# Training (ALWAYS use W&B — verify "logging enabled" before proceeding)
python scripts/train.py --config configs/training.yaml  # Auto-resumes from checkpoint
tensorboard --logdir logs/tensorboard

# Benchmark CLI (all docs in agent_cli.py docstring)
python scripts/agent_cli.py create --name "test" --budget 500000
python scripts/agent_cli.py study <session> 5
python scripts/agent_cli.py practice <session> 800
python scripts/agent_cli.py move <session> "Wd2-d3"
python scripts/agent_cli.py evaluate <session>
python scripts/agent_cli.py result <session>

# Benchmark preparation (download model + games from W&B, calibrate)
python scripts/prepare_benchmark.py --all
python scripts/prepare_benchmark.py --all --device cpu  # No CUDA

# REST API
pip install -e ".[api]"
python scripts/run_api_server.py --config configs/api_server.yaml

# Game analysis
python scripts/validate_game.py --num-games 500 --sims 50
python scripts/generate_and_save_seeds.py --num-games 100 --num-endgames 100 --endgame-sims 100 --force
```

## Architecture

### Game State & NN Encoding
- **Board**: 8x8 grid, `state.board[row][col]` → `Piece` objects
- **Gold nodes**: (3,3), (3,4), (4,3), (4,4) — resource income for promotion
- **NN input**: 14 planes (5 piece types × 2 players + gold + player + 2 resources) via `state_to_planes()`
- **Policy**: 4288 slots (64 squares × 67 move types) via `move_to_policy_index()`. Slots: 0-55 direction (8 dirs × 7 dist), 56-63 bombard ranged, 64-66 promotion (R, B, L).

### Critical Game Rules

**When changing rules, update ALL three: `davechess/game/rules.py`, `davechess/benchmark/prompt.py` (RULES_TEXT), `README.md`.**

1. **Chess-style capture**: Attacker always takes defender's square, no strength comparison
2. **Commander safety**: Must resolve check immediately; can't leave own Commander in check
3. **Win**: Checkmate only. Draw: turn 100, threefold repetition (includes resource affordability buckets), or 50-move rule
4. **Pieces**: Commander (C), Warrior (W, forward/diag-capture like pawn), Rider (R, 7sq ortho/3 diag), Bombard (B, 1sq move + 2sq ranged, no ranged vs Commander), Lancer (L, 7sq any dir, jumps one piece)
5. **Promotion**: R=3g, B=5g, L=7g. In-place, spending Gold. No deployment.
6. **Starting army**: 12/side — 1C, 3R, 2B, 6W

### Training Pipeline Key Points

- **Value targets**: +1 win, -1 loss, configurable draw target. Tanh head. All assignments must use +1/-1.
- **Seeds**: Heuristic games + MCTS endgame curriculum → `checkpoints/smart_seeds.pkl`. Currently disabled (White bias). Stored as W&B artifact.
- **Warm-start**: Loads `best.pt` if present on fresh start (no step checkpoint)
- **Adaptive sims**: `adaptive_simulations()` scales with ELO, floor at `mcts.min_selfplay_simulations` (128)
- **ELO probes**: NN (800 sims) vs MCTSLite-50 (≈300 ELO). Each sim doubling ≈ +180 ELO.
- **Muon optimizer**: `MuonSGD` — trunk convs get Newton-Schulz orthogonalization; heads/biases/BN get SGD. AMP overflow checked across all params.
- **Hot-reload**: `training.yaml` re-read each iteration. Network arch preserved, everything else updates live.

### Hardware (Jetson Orin Nano — 8GB shared RAM)
- 4 CPU MCTS workers + GPU inference server in main process
- Network: 10 ResBlocks, 256 filters (~12.4M params, ~350MB peak GPU)
- Mixed precision (FP16), buffer capped at 30K positions (~700MB), chunked save/load (5K chunks)
- **Always** `torch.cuda.empty_cache()` between phases. All buffer data float32. MCTS trees freed via `del` + `gc.collect()`.

### Benchmark

**`scripts/agent_cli.py` is the single source of truth.** All docs, commands, DCN notation, calibration tables in its docstring.

Session lifecycle: `BASELINE → LEARNING → EVALUATION → COMPLETED`

Opponents use `checkpoints/best.pt` with MCTS at varying sim counts. Calibration must match the model: `--calibration checkpoints/calibration.json`.

**Pipeline**: Train model → `prepare_benchmark.py --all` (downloads model + games from W&B, calibrates) → Launch agents in sandbox.

**Sandbox setup:**
```bash
SANDBOX="/tmp/benchmark-sandbox"
mkdir -p "$SANDBOX/scripts" "$SANDBOX/data" "$SANDBOX/checkpoints/agent_sessions"
cp scripts/agent_cli.py "$SANDBOX/scripts/"
cp -r data/gm_games "$SANDBOX/data/"
cp checkpoints/best.pt "$SANDBOX/checkpoints/"
cp checkpoints/calibration.json "$SANDBOX/checkpoints/"
```

**Launch agent:**
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
  --output-format stream-json 2>&1 | tee benchmark-output.log
```

Agents must reason about moves themselves — no writing game engines or search algorithms. Violations invalidate scores.

**Automation:** `bash scripts/run_overnight_benchmark.sh` runs Claude Code, Codex CLI, Gemini CLI sequentially. Individual: `_launch_codex_benchmark.py`, `_launch_gemini_benchmark.py`. Baseline: `_play_benchmark.py`.

## Known Issues (Condensed)

Issues #1-14 are historical fixes from v1/v2 game design and early training. Key lessons preserved:

| # | Issue | Solution | Status |
|---|-------|----------|--------|
| 1 | v1 strength-based capture → defensive stalemates | v2: chess-style capture, pawn-like Warriors | Fixed |
| 2 | 0/1 value targets with tanh head | +1/-1 targets matching tanh range | Fixed |
| 3 | Slow seed generation | Heuristic players (10x faster) | Fixed |
| 5 | Draw data flooding buffer | Configurable draw targets + weighting | Fixed |
| 6 | Consecutive rejection stalls | Auto-reset after N rejections (now N/A with pure AlphaZero) | N/A |
| 8 | Buffer OOM on save/load | Chunked I/O (5K/chunk), enforce float32 | Fixed |
| 9 | CUDA fragmentation after self-play | `empty_cache()` between phases | Fixed |
| 13 | Threefold repetition loops | Position key includes resource affordability buckets | Fixed |
| 14 | Self-play overfitting | Dirichlet α=0.15, ε=0.4, 25% random opponent games | Fixed |

### Critical Training Bugs (#15-28)

**#15 Eval excluded draws**: `wins/(wins+losses)` inflated ELO. Fix: `(wins + 0.5*draws) / total`.

**#16 ELO inheritance across restarts**: ELO ratcheted up through crash restarts. Fix: reset ELO on fresh start.

**#17 save_best() after pre-training**: Overwrote warm-started weights with seed-pretrained. Fix: skip pre-training on warm-start.

**#18 Random opponent check non-gating**: Model losing to random could still be promoted. Fix: veto if random losses > wins.

**#19 AlphaGo Zero architecture**: Best/training split wasted compute. Fix: single continuously-updated network.

**#20 Muon on classification heads (CRITICAL)**: Newton-Schulz on policy_fc (4288×128) is impossible — can't orthogonalize 4288 rows in 128 dims. Destroyed head weights silently. Fix: Muon restricted to trunk convs only via name filtering. **Required fresh restart.**

**#21 White-biased seeds**: 4:1 White win ratio poisoned value head. Fix: disabled seeds entirely (`seed_sample_weight_init: 0.0`).

**#22 vs-Random pessimistic loop**: Loss-heavy vs-random data skewed buffer + `clip_grad_norm_` across all params crushed head LR. Fix: exclude vs-random from buffer, clip only SGD params.

**#23 Gumbel truncation**: `max_num_considered_actions=16` hid 58-70% of legal moves. Fix: scale k with sim budget.

**#24 Gumbel improved policy collapse**: sigma_q scaling (11.2) overwhelmed logit range (~8), creating 99.97%-on-one-move targets. Fix: use visit count proportions as targets (standard AlphaZero).

**#25 Policy sharpening death spiral**: Training made network worse than random init. Root cause: value head overfitting to noisy data in 800 steps/iter → confidently wrong Q-values → peaked wrong policies. Fix: disable Gumbel, entropy reg, reduce to 200 steps/iter + lower LR.

**#26 Value noise overrides policy (CRITICAL)**: At cpuct=1.5, Q-noise of ±0.2 overrode 40%+ policy priors. value_loss_weight=20 gave 28% gradient to noisy targets. No board flipping wasted capacity. Fix: board flipping + cpuct=4.0 + value_loss_weight=3.0. **Required fresh restart.**

**#27 Value-head memorization/regression**: Value head fit replay targets too aggressively and produced brittle out-of-distribution estimates, destabilizing search over time. Fix: value-head dropout + reduced training intensity and tuned loss weighting to improve generalization.

**#28 Fail-open training pipeline after self-play/NaN failures (CRITICAL)**: Multiprocess self-play worker failures and non-finite losses could still allow training/checkpointing to proceed, including `best.pt` updates, silently poisoning model history. Fix: hard-fail on missing worker/game results, zero-example self-play batches, and non-finite per-step/average losses so bad iterations stop immediately.

## File Structure

- `davechess/game/` — Board, state, rules, notation (DCN)
- `davechess/engine/` — Network (`network.py`), MCTS (`mcts.py`, `mcts_lite.py`), self-play (`selfplay.py`), training (`training.py`), seeds, heuristic player, GPU server, multiprocess workers
- `davechess/data/` — Glicko-2 ELO, game generation, DCN storage
- `davechess/benchmark/` — Session lifecycle (`api/session.py`), opponent pool, game library, sequential eval (Glicko-2), token tracking, agent harness, tools, prompts, scoring
- `scripts/` — `train.py`, `play.py`, `agent_cli.py` (benchmark CLI), `prepare_benchmark.py`, `calibrate_opponents.py`, `run_overnight_benchmark.sh`, agent launchers
- `configs/` — `training.yaml`, `api_server.yaml`, `agentic_benchmark.yaml`
- `checkpoints/` — Models, seeds, calibration (gitignored, use W&B)
- `data/gm_games/` — GM game library (DCN)
- `tests/` — 149 tests: game, mcts, network, benchmark, api endpoints, api session, sdk

## Training Monitoring

**CRITICAL: NEVER start training without W&B connected!** If "Broken pipe", `pkill -9 wandb` first. All `wandb.log()` wrapped in `_safe_wandb_log()`.

Logs: W&B (primary), console, `logs/training_log.jsonl`, TensorBoard (`logs/tensorboard/`). Large files → W&B artifacts (not git).

Key metrics: `selfplay/avg_game_length` (↓), `selfplay/draw_rate` (↓), `selfplay/white_win_rate` (~0.5), `elo_probe/elo` (↑), `training/policy_loss` (↓)
