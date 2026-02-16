# Training stall: policy loss plateau, high draw rate, ELO not improving

## Summary

AlphaZero training for DaveChess is stalling. After fixing a critical bug in the batched Gumbel MCTS (82a7f3f), we restarted training from scratch with proper subtree search. After 7+ iterations, the policy loss is flat at ~2.45 and self-play is 50-70% draws (turn-limit). ELO probe hasn't fired yet (runs at iteration 10), but the loss trajectory and draw rate suggest the model isn't learning to play decisively.

## What we've fixed so far

1. **Batched Gumbel MCTS re-visit bug** (82a7f3f): Re-visits to already-expanded actions were reusing a constant stored value (`v = -g["child_values"][a]`) instead of doing deeper search. With k=16 and 200 sims, each action got ~12 visits but all returned the same Q-value. Fixed with subtree lookahead — re-visits now pick a move from the child state using stored policy logits and batch-evaluate the grandchild.

2. **draw_sample_weight was a no-op** (82a7f3f): Draw mask checked `value == 0` but draws have `value = -0.1` (from `draw_value_target`), so the configured `draw_sample_weight: 0.4` did nothing. Fixed to check `abs(value) <= 0.5`.

3. **Structured replay buffer** (5198d97): Three partitions — 20K seeds (permanent), 20K decisive, 10K draws. Seeds never get evicted.

4. **Dynamic value loss scaling** (0245337): Scales value loss to match policy loss gradient magnitude.

## Current training metrics (run: genial-sky-80)

W&B: https://wandb.ai/david-rimshnick-david-rimshnick/davechess/runs/4ntz5n8k

**Loss (flat across 7 iterations):**
| Iter | Policy Loss | Value Loss (raw) | Scale |
|------|-----------|-----------------|-------|
| 1 | 2.391 | 0.0707 | 54x |
| 2 | 2.402 | 0.0631 | 58x |
| 3 | 2.455 | 0.0750 | 48x |
| 4 | 2.456 | 0.0599 | 66x |
| 5 | 2.512 | 0.0585 | 68x |
| 6 | 2.521 | 0.0650 | 66x |
| 7 | 2.447 | 0.0622 | 56x |

**Self-play (50-70% draws, mostly turn-limit):**
| Iter | W | B | D | Draw% | Avg Length |
|------|---|---|---|-------|-----------|
| 1 | 7 | 5 | 8 | 40% | 145 |
| 2 | 4 | 4 | 12 | 60% | 148 |
| 3 | 7 | 4 | 9 | 45% | 148 |
| 4 | 3 | 3 | 14 | 70% | 170 |
| 5 | 6 | 1 | 13 | 65% | 161 |
| 6 | 4 | 3 | 13 | 65% | 169 |
| 7 | 5 | 5 | 10 | 50% | 164 |

**Buffer at iter 7:** 34,645 total (seeds=19,118, decisive=5,527, draws=10,000)

## Symptoms

- **Policy loss not decreasing** — bouncing around 2.4-2.5 (random would be ~8.4 for 4288 action space, so it's learned something from seeds, but not improving)
- **High draw rate** — 50-70% of self-play games are turn-limit draws (200 moves). The model can't find checkmates.
- **Average game length not decreasing** — steady 145-170 moves, should trend down as the model learns to checkmate
- **ELO was 265 in previous run** (with broken MCTS) — unclear if the fix is helping yet since the ELO probe hasn't run in the new training

## Architecture

- 20 ResBlocks, 256 filters (~24M params)
- Gumbel MCTS with k=16, 50 sims, batched search
- 20 games/iteration (5 vs random opponent, 15 self-play)
- 8x8 board, 4288 policy size, 14 input planes
- Jetson Orin Nano (8GB shared RAM), ~13 min/iteration

## Config highlights

```yaml
num_simulations: 50
dirichlet_alpha: 0.15
dirichlet_epsilon: 0.4
temperature_threshold: 60
draw_value_target: -0.1
draw_sample_weight: 0.4
learning_rate: 0.001
batch_size: 256
buffer_capacity: 50000
```

## Questions for reviewers

1. **Is the subtree lookahead implementation correct?** Re-visits pick a stochastic move from child policy, evaluate the grandchild, and use `gc_val` directly (same player as root, 2 plies deep). See `gumbel_mcts.py:391-419` and `543-597`.

2. **Is the sign convention right?** First-visit: `v = -child_val` (child is opponent). Subtree: `v = gc_val` (grandchild is same player as root). This seems correct but worth double-checking.

3. **Should we use standard PUCT MCTS instead of Gumbel?** The non-batched `GumbelMCTS.search()` (which uses proper recursive tree search) works correctly. Gumbel's advantage is batching for GPU efficiency, but the depth-1 + subtree-lookahead approach may not give enough search depth.

4. **Is the network too large for the iteration speed?** 24M params with 50 sims and 20 games/iter means ~3K new positions per iteration. At ~500 training steps per iteration, the network sees each position multiple times but might not be getting enough diverse signal.

5. **Learning rate / optimizer?** Using Adam at 0.001 with no schedule. Standard AlphaZero uses SGD with momentum and a step schedule. Could the optimizer be a factor?

6. **Is the draw_value_target of -0.1 enough?** Draws are scored at -0.1 (slight penalty), but maybe the penalty needs to be stronger to push the model away from draw-seeking play.
