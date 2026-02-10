#!/usr/bin/env python3
"""
GameBench: DaveChess Benchmark CLI
===================================

This is the sole interface for running the DaveChess benchmark. Any coding
agent (Claude Code, Codex CLI, Gemini CLI, OpenCode, etc.) can drive this
script via sequential bash calls. All output is JSON. Session state persists
to a pickle file between calls, so each invocation is a fresh process.

Setup
-----
    cd <repo_root>
    pip install -e .          # or set PYTHONPATH=. before each command
    # GM games must exist in data/gm_games/ (.dcn format)

Session Lifecycle
-----------------
Every session progresses through phases:

    BASELINE  ->  LEARNING  ->  EVALUATION  ->  COMPLETED

    BASELINE:   Play rated games with rules-only knowledge (measures starting ELO).
    LEARNING:   Study GM games, play practice games to improve.
    EVALUATION: Play rated games to measure final ELO.
    COMPLETED:  Session done — retrieve results.

    Phase transitions:
      BASELINE -> LEARNING     automatic after baseline games finish
      LEARNING -> EVALUATION   explicit via `evaluate` command
      EVALUATION -> COMPLETED  automatic after convergence or budget exhausted

    --skip-baseline on `create` starts directly in LEARNING. Not recommended
    for real benchmarks — baseline is needed to measure ELO gain.

Token Budget
------------
Your token budget represents your learning opportunity. Every token you spend
studying GM games, practicing against opponents, and refining your strategy
makes you stronger. The benchmark measures how efficiently you convert tokens
into chess skill (ELO gain).

    - Every CLI response includes a "budget" section with tokens remaining.
    - Report your token usage with --tokens on any command (see below).
    - Don't rush to evaluation — exhaust your learning budget first!
    - More studying and practice = higher final ELO = better benchmark score.

Commands
--------
Every command prints JSON to stdout. Every response includes "session_file"
and a "budget" section with your token usage and remaining budget.

  create    Create a new benchmark session.
            python scripts/agent_cli.py create --name "my-run" [options]
              --budget N            Token budget (default: 1000000)
              --baseline-games N    Baseline rated games (default: 10)
              --eval-min-games N    Min eval games before convergence check (default: 10)
              --eval-max-games N    Max eval games (default: 200)
              --skip-baseline       Start in LEARNING (skip baseline — not recommended)
              --checkpoint PATH     Neural network checkpoint for MCTS opponents
              --calibration PATH    Calibration JSON (from calibrate_opponents.py)
            Output: {"session_file": "...", "phase": "...", "game_id": "...", ...}

  status    Show session status (phase, ratings, token usage).
            python scripts/agent_cli.py status <session_file>

  resume    Get status + current game state (for continuation agents).
            python scripts/agent_cli.py resume <session_file>

  rules     Print full DaveChess rules (plain text, not JSON).
            python scripts/agent_cli.py rules <session_file>

  state     Show current game: board, legal moves, resources, history.
            python scripts/agent_cli.py state <session_file>
            Output includes: board, legal_moves[], agent_color, turn,
            your_resources, opponent_resources, move_history

  move      Play a move in DCN notation. Opponent responds automatically.
            python scripts/agent_cli.py move <session_file> <move_dcn>
            Output: your_move, opponent_move, game_over, board, legal_moves
            If game ends: result ("win"/"loss"/"draw"), next_game_id
            If illegal move: error message + legal_moves list to retry

  study     Retrieve N grandmaster games for study (LEARNING only).
            python scripts/agent_cli.py study <session_file> <num_games>
            Games are in DCN notation. Not repeated within a session.

  practice  Start a practice game at target ELO (LEARNING only).
            python scripts/agent_cli.py practice <session_file> <opponent_elo>
            Then use `move` to play. Practice games don't affect rating.

  evaluate  Transition from LEARNING to EVALUATION phase.
            python scripts/agent_cli.py evaluate <session_file>
            Creates first rated evaluation game.

  result    Get final results (COMPLETED only).
            python scripts/agent_cli.py result <session_file>
            Output: baseline_elo, final_elo, elo_gain, game details

  report-tokens   Report token usage from an external harness.
            python scripts/agent_cli.py report-tokens <session_file> <prompt> <completion>
            Agents should self-report approximate token usage after each
            interaction to keep the budget tracker accurate.

Global Options (all commands except create)
-------------------------------------------
  --tokens PROMPT COMPLETION   Report token usage inline with any command.
    Example: python scripts/agent_cli.py move session.pkl "Wd2-d3" --tokens 5000 2000

DCN Notation
------------
    Move:     Xa1-b2     e.g. Wc1-c2  (Warrior c1 to c2)
    Capture:  Xa1xb2     e.g. Rb1xd3  (Rider captures at d3)
    Deploy:   +X@a1      e.g. +W@c2   (Deploy Warrior at c2, costs resources)
    Bombard:  Xa1~b3     e.g. Bc3~e3  (Bombard ranged attack from c3 to e3)

Opponent Calibration (MCTSLite, no neural network)
--------------------------------------------------
    MCTS Sims  |  Approx ELO
    -----------+-----------
    0 (random) |  400
    10         |  800
    50         |  1200
    200        |  1800
    800        |  2400

    Evaluation opponents are chosen near the agent's estimated ELO for
    maximum info gain (Glicko-2 rating system).

Example: Full Benchmark Run
----------------------------
    # 1. Create session (baseline measures your starting ELO)
    python scripts/agent_cli.py create --name "my-agent" --budget 500000

    # 2. Play baseline games (automatic — just play moves until LEARNING phase)
    python scripts/agent_cli.py move <session_file> "Wd2-d3" --tokens 3000 500
    # ... keep playing moves until phase transitions to LEARNING

    # 3. Read the rules
    python scripts/agent_cli.py rules <session_file>

    # 4. Study some GM games
    python scripts/agent_cli.py study <session_file> 5

    # 5. Play practice games to learn
    python scripts/agent_cli.py practice <session_file> 800
    python scripts/agent_cli.py move <session_file> "Wd2-d3" --tokens 3000 500
    # ... keep playing moves until game_over=true

    # 6. Transition to evaluation (only when you've used most of your budget!)
    python scripts/agent_cli.py evaluate <session_file>

    # 7. Play rated games (move until game_over, repeat for each new game)
    python scripts/agent_cli.py move <session_file> "Wd2-d3" --tokens 3000 500
    # ... phase auto-completes after convergence

    # 8. Get results
    python scripts/agent_cli.py result <session_file>

IMPORTANT: Benchmark Integrity Rules
--------------------------------------
    You MUST reason about each move yourself. Do NOT write scripts, game
    engines, search algorithms (minimax, MCTS, alpha-beta, etc.), or any
    automated move-selection code. The benchmark measures YOUR strategic
    reasoning, not your ability to build a chess engine. Violations
    invalidate your score.

Tips for AI Agents
------------------
    - Your token budget is your learning opportunity — use it all!
    - Report tokens with --tokens on every command to track your budget.
    - Every CLI response includes a "budget" section showing tokens remaining.
    - More studying and practice = higher final ELO = better benchmark score.
    - Don't rush to evaluation — exhaust your learning budget first.
    - Always pick moves from the legal_moves list in the state/move response.
    - Check game_over after each move. When a game ends, look for next_game_id.
    - Study GM games early to learn piece interactions and strategy.
    - Start practice at low ELO (800), increase as you improve.
    - When ready, call `evaluate` — you can't go back to studying after.
    - Use `resume` if context is lost — it returns full current state.
"""

import argparse
import json
import os
import pickle
import sys

# Fix Windows console encoding for Unicode
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Session + dependencies
from davechess.benchmark.api.session import BenchmarkSession
from davechess.benchmark.api.models import SessionPhase
from davechess.benchmark.opponent_pool import OpponentPool, CalibratedLevel
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.sequential_eval import EvalConfig

SCRATCHDIR = os.path.join("checkpoints", "agent_sessions")
GAMES_DIR = "data/gm_games"
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "best.pt")

# Default calibration for MCTSLite (no network)
DEFAULT_CALIBRATION = [
    CalibratedLevel(sim_count=0, measured_elo=400),
    CalibratedLevel(sim_count=10, measured_elo=800),
    CalibratedLevel(sim_count=50, measured_elo=1200),
    CalibratedLevel(sim_count=200, measured_elo=1800),
    CalibratedLevel(sim_count=800, measured_elo=2400),
]


def _load_network(checkpoint_path: str, device: str = "cpu"):
    """Load neural network from checkpoint, or return None if unavailable."""
    from davechess.engine.network import DaveChessNetwork
    net, ckpt = DaveChessNetwork.from_checkpoint(checkpoint_path, device=device)
    net.eval()
    elo = ckpt.get("elo_estimate", "?")
    iteration = ckpt.get("iteration", "?")
    print(json.dumps({"info": f"Loaded network: ELO {elo}, iteration {iteration}"}),
          file=sys.stderr)
    return net


def _save_session(session: BenchmarkSession, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(session, f)


def _load_session(path: str) -> BenchmarkSession:
    with open(path, "rb") as f:
        return pickle.load(f)


def _raw_print(data):
    """Print dict as indented JSON (no budget info — used before session exists)."""
    print(json.dumps(data, indent=2, default=str))


def _print_response(data, session: BenchmarkSession, session_file: str):
    """Print dict as indented JSON with budget info injected."""
    data["session_file"] = session_file
    tracker = session.token_tracker
    data["budget"] = {
        **tracker.summary(),
        "message": tracker.budget_message(phase=session.phase.value),
    }
    print(json.dumps(data, indent=2, default=str))


def cmd_create(args):
    network = None
    if args.checkpoint:
        network = _load_network(args.checkpoint)

    if args.calibration:
        pool = OpponentPool.from_calibration_file(
            args.calibration, network=network, device="cpu")
    else:
        pool = OpponentPool(network=network, device="cpu",
                            calibration=DEFAULT_CALIBRATION)
    library = GameLibrary(GAMES_DIR, max_games=200)
    loaded = library.load()

    eval_config = EvalConfig(
        initial_elo=1000, target_rd=50.0,
        max_games=args.eval_max_games, min_games=args.eval_min_games,
    )

    session = BenchmarkSession(
        session_id=args.name.replace(" ", "-"),
        agent_name=args.name,
        token_budget=args.budget,
        opponent_pool=pool,
        game_library=library,
        eval_config=eval_config,
        baseline_max_games=args.baseline_games,
        skip_baseline=args.skip_baseline,
    )

    session_file = os.path.join(SCRATCHDIR, f"{session.session_id}.pkl")
    _save_session(session, session_file)

    output = {
        "phase": session.phase.value,
        "library_games": loaded,
    }

    if not args.skip_baseline:
        evaluator = session._baseline_evaluator
        game = evaluator.current_game
        output["game_id"] = game.game_id
        output["game_state"] = evaluator.get_game_state()
    else:
        output["info"] = "Session started in LEARNING phase (baseline skipped)."

    _print_response(output, session, session_file)


def cmd_status(args):
    session = _load_session(args.session_file)
    status = session.get_status()
    _print_response(status, session, args.session_file)


def cmd_rules(args):
    session = _load_session(args.session_file)
    print(session.get_rules())
    # Append budget reminder as plain-text footer
    tracker = session.token_tracker
    print(f"\n---\nBudget: {tracker.budget_message(phase=session.phase.value)}")


def cmd_state(args):
    session = _load_session(args.session_file)

    # Find the current active game id
    if session.phase == SessionPhase.BASELINE:
        evaluator = session._baseline_evaluator
        if evaluator.current_game:
            result = evaluator.get_game_state()
        else:
            result = {"info": "No active baseline game. Phase may be transitioning."}
    elif session.phase == SessionPhase.EVALUATION:
        evaluator = session._final_evaluator
        if evaluator.current_game:
            result = evaluator.get_game_state()
        else:
            result = {"info": "No active eval game."}
    elif session.phase == SessionPhase.LEARNING:
        # Show active practice games
        active = session._game_manager.get_active_games()
        if active:
            game = active[0]
            result = session._game_manager.get_state(game.game_id)
        else:
            result = {"info": "No active practice games. Use 'practice' to start one."}
    else:
        result = {"phase": session.phase.value, "info": "Session completed."}

    _print_response(result, session, args.session_file)
    _save_session(session, args.session_file)


def cmd_move(args):
    session = _load_session(args.session_file)

    # Auto-detect game_id from current phase
    if session.phase == SessionPhase.BASELINE:
        game_id = session._baseline_evaluator.current_game.game_id
    elif session.phase == SessionPhase.EVALUATION:
        game_id = session._final_evaluator.current_game.game_id
    elif session.phase == SessionPhase.LEARNING:
        # Use explicitly provided game_id or first active game
        active = session._game_manager.get_active_games()
        if active:
            game_id = active[0].game_id
        else:
            _print_response({"error": "No active practice game"}, session, args.session_file)
            return
    else:
        _print_response({"error": f"Cannot play moves in {session.phase.value} phase"},
                        session, args.session_file)
        return

    result = session.play_move(game_id, args.move_dcn)
    _save_session(session, args.session_file)
    _print_response(result, session, args.session_file)


def cmd_study(args):
    session = _load_session(args.session_file)
    try:
        result = session.study_games(args.num_games)
    except Exception as e:
        _print_response({"error": str(e)}, session, args.session_file)
        return
    _save_session(session, args.session_file)
    _print_response({
        "num_returned": result["num_returned"],
        "remaining": result["remaining_in_library"],
        "games": result["games"],
    }, session, args.session_file)


def cmd_practice(args):
    session = _load_session(args.session_file)
    try:
        result = session.start_practice_game(args.opponent_elo)
    except Exception as e:
        _print_response({"error": str(e)}, session, args.session_file)
        return
    _save_session(session, args.session_file)
    _print_response(result, session, args.session_file)


def cmd_evaluate(args):
    session = _load_session(args.session_file)
    try:
        result = session.request_evaluation()
    except Exception as e:
        _print_response({"error": str(e)}, session, args.session_file)
        return
    _save_session(session, args.session_file)
    _print_response(result, session, args.session_file)


def cmd_resume(args):
    """Print session status + current game state for a continuation agent."""
    session = _load_session(args.session_file)
    status = session.get_status()
    output = {
        "phase": status["phase"],
        "baseline_rating": status.get("baseline_rating"),
        "final_rating": status.get("final_rating"),
    }

    # Include current game state if there's an active game
    if session.phase == SessionPhase.BASELINE:
        evaluator = session._baseline_evaluator
        if evaluator.current_game:
            output["current_game"] = evaluator.get_game_state()
            output["game_id"] = evaluator.current_game.game_id
    elif session.phase == SessionPhase.EVALUATION:
        evaluator = session._final_evaluator
        if evaluator and evaluator.current_game:
            output["current_game"] = evaluator.get_game_state()
            output["game_id"] = evaluator.current_game.game_id
        elif evaluator and not evaluator.is_complete:
            output["info"] = "No active game. Use 'state' to create next eval game."
    elif session.phase == SessionPhase.LEARNING:
        active = session._game_manager.get_active_games()
        if active:
            game = active[0]
            output["current_game"] = session._game_manager.get_state(game.game_id)
            output["game_id"] = game.game_id
        else:
            output["info"] = "In LEARNING phase. Use 'study', 'practice', or 'evaluate'."
    elif session.phase == SessionPhase.COMPLETED:
        output["info"] = "Session is COMPLETED. Use 'result' to get final results."

    _save_session(session, args.session_file)
    _print_response(output, session, args.session_file)


def cmd_report_tokens(args):
    """Report token usage from an external harness."""
    session = _load_session(args.session_file)
    try:
        result = session.report_tokens(args.prompt_tokens, args.completion_tokens)
    except Exception as e:
        _print_response({"error": str(e)}, session, args.session_file)
        return
    _save_session(session, args.session_file)
    _print_response(result, session, args.session_file)


def cmd_result(args):
    session = _load_session(args.session_file)
    try:
        result = session.get_result()
    except Exception as e:
        _print_response({"error": str(e)}, session, args.session_file)
        return
    _print_response(result, session, args.session_file)


def main():
    parser = argparse.ArgumentParser(description="Benchmark agent CLI")
    parser.add_argument("--tokens", nargs=2, type=int, metavar=("PROMPT", "COMPLETION"),
                        help="Report token usage inline (prompt_tokens completion_tokens)")
    sub = parser.add_subparsers(dest="command")

    # create
    p = sub.add_parser("create", help="Create a new session")
    p.add_argument("--budget", type=int, default=1_000_000)
    p.add_argument("--name", default="claude-code")
    p.add_argument("--baseline-games", type=int, default=10)
    p.add_argument("--eval-min-games", type=int, default=10)
    p.add_argument("--eval-max-games", type=int, default=200)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Start directly in LEARNING phase (no baseline games)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to neural network checkpoint for MCTS opponents")
    p.add_argument("--calibration", default=None,
                   help="Path to calibration JSON (from calibrate_opponents.py)")

    # status
    p = sub.add_parser("status", help="Show session status")
    p.add_argument("session_file")

    # rules
    p = sub.add_parser("rules", help="Print DaveChess rules")
    p.add_argument("session_file")

    # state
    p = sub.add_parser("state", help="Show current game state")
    p.add_argument("session_file")

    # move
    p = sub.add_parser("move", help="Play a move")
    p.add_argument("session_file")
    p.add_argument("move_dcn", help="Move in DCN notation")

    # study
    p = sub.add_parser("study", help="Study GM games")
    p.add_argument("session_file")
    p.add_argument("num_games", type=int)

    # practice
    p = sub.add_parser("practice", help="Start a practice game")
    p.add_argument("session_file")
    p.add_argument("opponent_elo", type=int)

    # evaluate
    p = sub.add_parser("evaluate", help="Request evaluation")
    p.add_argument("session_file")

    # resume
    p = sub.add_parser("resume", help="Get session state for continuation")
    p.add_argument("session_file")

    # report-tokens
    p = sub.add_parser("report-tokens", help="Report token usage from external harness")
    p.add_argument("session_file")
    p.add_argument("prompt_tokens", type=int)
    p.add_argument("completion_tokens", type=int)

    # result
    p = sub.add_parser("result", help="Show final results")
    p.add_argument("session_file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Handle global --tokens flag: record usage before running the command
    if args.tokens and args.command != "create":
        session = _load_session(args.session_file)
        session.token_tracker.record(args.tokens[0], args.tokens[1])
        _save_session(session, args.session_file)

    {
        "create": cmd_create,
        "status": cmd_status,
        "rules": cmd_rules,
        "state": cmd_state,
        "move": cmd_move,
        "study": cmd_study,
        "practice": cmd_practice,
        "evaluate": cmd_evaluate,
        "resume": cmd_resume,
        "report-tokens": cmd_report_tokens,
        "result": cmd_result,
    }[args.command](args)


if __name__ == "__main__":
    main()
