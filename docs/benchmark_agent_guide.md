# DaveChess Benchmark Agent Guide

How to run the DaveChess benchmark using `agent_cli.py`. This guide is written for both human operators and AI agents that drive the CLI.

## Prerequisites

- Python 3.9+ with `davechess` importable (either `pip install -e .` or set `PYTHONPATH=.`)
- Working directory: the GameBench repo root
- GM game files in `data/gm_games/` (`.dcn` format)

All commands below assume:
```bash
cd <repo_root>
PYTHONPATH=. python scripts/agent_cli.py <command> [args]
```

## Session Lifecycle

Every benchmark session progresses through phases:

```
BASELINE -> LEARNING -> EVALUATION -> COMPLETED
```

| Phase | What happens | Available commands |
|-------|-------------|--------------------|
| BASELINE | Play rated games with rules-only knowledge | `move`, `state`, `status` |
| LEARNING | Study GM games, play practice games | `study`, `practice`, `move`, `state`, `evaluate` |
| EVALUATION | Play rated games to measure final ELO | `move`, `state`, `status` |
| COMPLETED | Session done | `result`, `status` |

Transitions are automatic (BASELINE finishes after N games) or explicit (`evaluate` triggers LEARNING -> EVALUATION).

Use `--skip-baseline` on `create` to start directly in LEARNING (useful for split-session designs where baseline is measured separately).

## CLI Command Reference

### `create` - Create a new session

```bash
python scripts/agent_cli.py create --name "my-run" [options]
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `claude-code` | Session name (used as ID, spaces replaced with hyphens) |
| `--budget` | `1000000` | Token budget |
| `--baseline-games` | `10` | Number of baseline evaluation games |
| `--eval-min-games` | `10` | Minimum evaluation games before checking convergence |
| `--eval-max-games` | `200` | Maximum evaluation games |
| `--skip-baseline` | `false` | Start in LEARNING phase (skip baseline games) |

Output includes `session_file` path. **Save this path** - every subsequent command needs it.

Example output:
```json
{
  "session_file": "checkpoints/agent_sessions/my-run.pkl",
  "phase": "baseline",
  "library_games": 20,
  "game_id": "base_001",
  "game_state": { ... }
}
```

### `status` - Check session status

```bash
python scripts/agent_cli.py status <session_file>
```

Returns phase, ratings, and token usage. Use this to check progress.

### `resume` - Get state for continuation

```bash
python scripts/agent_cli.py resume <session_file>
```

Returns session status plus the current active game state. Designed for handing off to a continuation agent that needs to pick up where the previous agent stopped.

### `state` - Get current game state

```bash
python scripts/agent_cli.py state <session_file>
```

Returns the board, legal moves, resources, and move history for the active game. The response includes:
- `board`: ASCII board rendering
- `legal_moves`: list of valid moves in DCN notation
- `your_resources` / `opponent_resources`: current resource counts
- `agent_color`: whether you play "white" or "black"
- `move_history`: numbered move pairs

### `move` - Play a move

```bash
python scripts/agent_cli.py move <session_file> <move_dcn>
```

Play a move in DCN notation. The move must be in the `legal_moves` list. After your move, the opponent responds automatically. The response includes:
- `your_move`: the move you played
- `opponent_move`: the opponent's response (if game continues)
- `game_over`: true if the game ended
- `result`: "win", "loss", or "draw" (if game over)
- Updated board state and legal moves

If the game ends and there are more games to play, the response includes `next_game_id` and `next_game_state`.

If you send an illegal move, the response includes `error` and `legal_moves` so you can retry.

### `study` - Study GM games (LEARNING only)

```bash
python scripts/agent_cli.py study <session_file> <num_games>
```

Returns `num_games` grandmaster games in DCN notation. Games are not repeated within a session.

### `practice` - Start practice game (LEARNING only)

```bash
python scripts/agent_cli.py practice <session_file> <opponent_elo>
```

Starts a practice game against an MCTS opponent at the specified ELO. Use `move` to play moves in the practice game.

### `evaluate` - Start evaluation (LEARNING -> EVALUATION)

```bash
python scripts/agent_cli.py evaluate <session_file>
```

Transitions from LEARNING to EVALUATION phase. Creates the first rated evaluation game. Use `move` to play.

### `result` - Get final results (COMPLETED only)

```bash
python scripts/agent_cli.py result <session_file>
```

Returns baseline ELO, final ELO, and ELO gain.

### `rules` - Print game rules

```bash
python scripts/agent_cli.py rules <session_file>
```

Prints the full DaveChess rules text (plain text, not JSON).

## DCN Notation Quick Reference

| Action | Format | Example |
|--------|--------|---------|
| Move | `Xa1-b2` | `Wc1-c2` (Warrior moves c1 to c2) |
| Capture | `Xa1xb2` | `Rb1xd3` (Rider captures at d3) |
| Deploy | `+X@a1` | `+W@c2` (Deploy Warrior at c2) |
| Bombard ranged | `Xa1~b3` | `Bc3~e3` (Bombard ranged attack) |

## Running a Two-Session Benchmark

The recommended approach uses two separate sessions to cleanly measure ELO gain from learning:

### Session 1: Baseline (rules-only)

```bash
# Create baseline session
python scripts/agent_cli.py create --name baseline-run --baseline-games 5 --eval-min-games 5

# Play 5 rated games using only rules knowledge
# (agent plays moves via repeated `move` calls)

# Check result
python scripts/agent_cli.py status checkpoints/agent_sessions/baseline-run.pkl
# -> baseline_rating.elo = baseline ELO
```

### Session 2: Learning + Evaluation

```bash
# Create learning session (no baseline phase)
python scripts/agent_cli.py create --name learning-run --skip-baseline --eval-min-games 5 --eval-max-games 15

# Study GM games
python scripts/agent_cli.py study checkpoints/agent_sessions/learning-run.pkl 5

# Practice (optional)
python scripts/agent_cli.py practice checkpoints/agent_sessions/learning-run.pkl 800
# (play moves until practice game ends)

# Start evaluation
python scripts/agent_cli.py evaluate checkpoints/agent_sessions/learning-run.pkl

# Play rated games
# (agent plays moves via repeated `move` calls)

# Check result
python scripts/agent_cli.py status checkpoints/agent_sessions/learning-run.pkl
# -> final_rating.elo = final ELO
```

### Computing ELO Gain

```
elo_gain = final_elo - baseline_elo
```

A positive gain indicates the agent improved by studying GM games.

## Tips for AI Agent Drivers

1. **Always save the session_file path.** Every CLI output includes it, but if context is lost, the default path is `checkpoints/agent_sessions/<name>.pkl`.

2. **Pick moves from `legal_moves`.** If a move is illegal, the CLI returns the legal moves list. Just pick from it.

3. **Play fast.** Don't spend many turns analyzing a single move. The benchmark measures learning ability, not per-move computation.

4. **Check `game_over` after each move.** When a game ends, the next game may be auto-created (check for `next_game_id` in the response).

5. **Use `resume` for continuations.** If an agent runs out of context/turns, a new agent can use `resume` to get the current state and continue.

6. **Phases transition automatically.** BASELINE -> LEARNING happens after baseline games finish. EVALUATION -> COMPLETED happens after eval games finish. Only LEARNING -> EVALUATION requires explicit `evaluate` command.

## Opponent Calibration

The benchmark uses MCTSLite opponents with no neural network. Default calibration:

| MCTS Sims | Approximate ELO |
|-----------|-----------------|
| 0 (random) | 400 |
| 10 | 800 |
| 50 | 1200 |
| 200 | 1800 |
| 800 | 2400 |

Evaluation opponents are chosen near the agent's current estimated ELO for maximum information gain (Glicko-2 rating system).
