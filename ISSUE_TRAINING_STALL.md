# Training stall: policy loss drops but ELO flat, value head degrading

## Summary

AlphaZero training for DaveChess stalled across two runs. **Root cause found:** a broken loss spike detector halved the learning rate every iteration, reducing it from 0.001 to 9.5e-10 over 20 iterations. The model was effectively frozen after iteration ~3. Now restarting with the fix (run 3).

### Run history

1. **Run 1 (genial-sky-80):** Warm-started from old model, 7 iterations with fixed Gumbel subtree search but noisy move selection. Policy loss flat at ~2.45, ELO never probed.
2. **Run 2 (genial-plasma-81):** Fresh start with both fixes (subtree search + move selection fix from PR #3). 20 iterations. Policy loss drops steadily (7.2→5.8), but **ELO flat at 150-190** and **value head degrading**. Root cause: LR halved every iteration by broken spike detector.
3. **Run 3 (fiery-smoke-82):** Fresh start with spike detector removed. In progress.

## Root cause: loss spike detector destroyed the learning rate

The training code had a "spike detector" that halved the learning rate whenever `avg_total_loss > 10.0`:

```python
# REMOVED — this was the bug
if avg_losses["avg_total_loss"] > 10.0:
    logger.warning("Loss spike detected! Halving learning rate.")
    for param_group in self.optimizer.param_groups:
        param_group["lr"] *= 0.5
```

With a 4288-action policy space, the normal cross-entropy loss is ~5.8. The dynamic value scaling makes `total_loss = policy + scaled_value ≈ 2 * policy ≈ 11.6`. This **always** exceeds 10.0 — the detector fired on every single iteration:

```
Iter  1: 0.001  → 0.0005
Iter  2: → 0.00025
Iter  3: → 0.000125
...
Iter 20: → 9.5e-10  (1,048,576x too small)
```

The model effectively stopped learning after iteration 3-4. The policy loss appeared to drop because the network was still doing marginal updates at tiny LR, but the value head was getting conflicting targets it couldn't learn from at near-zero LR. ELO was flat because the model wasn't actually training.

**Fix:** Removed the spike detector entirely (commit 08f213a).

## Bug fixes applied (all merged to master)

1. **Batched Gumbel MCTS re-visit bug** (82a7f3f): Re-visits reused a constant stored value instead of doing deeper search. Fixed with subtree lookahead.
2. **Gumbel noise in move selection** (PR #3): Final action selection was using `gumbel + logits + sigma_q` — Gumbel noise leaked into the played move even at temp=0. Fixed to use `improved_logits` (no Gumbel).
3. **draw_sample_weight was a no-op** (82a7f3f): Draw mask checked `value == 0` but draws have `value = -0.1`. Fixed.
4. **Structured replay buffer** (5198d97): Three partitions — 20K seeds, 20K decisive, 10K draws.
5. **Dynamic value loss scaling** (0245337): Scales value loss to match policy gradient magnitude.
6. **Loss spike detector destroying LR** (08f213a): Threshold of 10.0 was below normal total loss (~11.6). Halved LR every iteration. Removed entirely.
7. **Sign error in non-batched GumbelMCTS._subtree_search** (2151b94): Terminal value signs were inverted. Only affects non-batched variant (not used in training).

## Run 2 metrics (genial-plasma-81) — with broken LR

W&B: https://wandb.ai/david-rimshnick-david-rimshnick/davechess/runs/ny4xis0k

| Iter | Policy | Value (raw) | Value Scale | LR | ELO |
|------|--------|-------------|-------------|-----|-----|
| 1 | 7.176 | 0.1553 | 52x | 5e-4 | |
| 2 | 6.744 | 0.1168 | 70x | 2.5e-4 | |
| 3 | 6.512 | 0.0893 | 95x | 1.25e-4 | |
| 4 | 6.308 | 0.0828 | 101x | 6.25e-5 | |
| 5 | 6.224 | 0.0668 | 132x | 3.1e-5 | **192** |
| 10 | 5.967 | 0.1197 | 57x | 9.8e-7 | **192** |
| 15 | 5.876 | 0.2485 | 25x | 3.1e-8 | **153** |
| 20 | 5.831 | 0.3743 | 16x | 9.5e-10 | **115** |

The LR column makes it clear — by iteration 5 the LR was already 30x too small, and by iteration 10 it was 1000x too small. The apparent "slow but steady" policy improvement was an illusion.

## Run 3 (fiery-smoke-82) — with fix, in progress

W&B: https://wandb.ai/david-rimshnick-david-rimshnick/davechess/runs/67bl3u8x

Just started. LR will remain at 0.001 throughout. First ELO probe at iteration 5.

## Remaining concerns (if run 3 still stalls)

These were identified during run 2 analysis. They may or may not matter now that LR is fixed:

### 1. Value head target conflict
The buffer has three value target populations: seeds (+1/-1), decisive self-play (+1/-1), draws (-0.1). Early-game positions from different games get different targets depending on eventual outcome. With ~50% draw rate, the value head sees conflicting targets for similar positions.

### 2. Seed domination (40% of buffer)
Seeds are permanent and make up 40% of training data. They teach heuristic play patterns (not MCTS-discovered). After pre-training, they may conflict with search-discovered strategies.

### 3. Dynamic value scaling
`value_scale = policy_loss / value_loss` makes the effective value gradient always equal to the policy gradient. This may over-train the value head early (when raw value loss is small) and under-train it later.

### 4. Shallow Gumbel batched search
The batched search only goes 2 plies deep (root → child → grandchild via subtree lookahead). Standard PUCT MCTS builds a full tree with 50+ sims. Shallower search = weaker policy targets.

## Architecture

- 10 ResBlocks, 256 filters (~12.4M params)
- Gumbel MCTS with k=16, 50 sims, batched search (subtree lookahead on re-visits)
- 20 games/iteration (5 vs random opponent, 15 self-play)
- ELO probe: standard PUCT MCTS at 64 sims vs MCTSLite at 50 sims
- 8x8 board, 4288 policy size, 14 input planes
- Jetson Orin Nano (8GB shared RAM), ~13 min/iteration
- SGD with momentum 0.9, LR 0.001, weight decay 1e-4

## Config

```yaml
# Network
num_res_blocks: 10
num_filters: 256

# MCTS
num_simulations: 50
min_selfplay_simulations: 50
cpuct: 1.5
dirichlet_alpha: 0.15
dirichlet_epsilon: 0.4
temperature_threshold: 60

# Self-play
games_per_iteration: 20
random_opponent_fraction: 0.25
draw_value_target: -0.1

# Training
learning_rate: 0.001
batch_size: 128
steps_per_iteration: 800
draw_sample_weight: 0.4

# Buffer
buffer_seed_size: 20000
buffer_decisive_size: 20000
buffer_draw_size: 10000

# Probes
elo_probe_interval: 5
elo_probe_games: 10
```
