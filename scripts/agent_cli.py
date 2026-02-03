#!/usr/bin/env python3
"""CLI driver for an AI agent to play through the benchmark.

Wraps BenchmarkSession with simple single-command operations that
print JSON output. Designed to be driven by an LLM agent (e.g.
Claude Code) via sequential bash calls.

Session state is persisted to a pickle file between calls so each
invocation is a fresh process. Every command output includes
"session_file" so the caller never loses the path.

Commands:
    create   Create a new benchmark session
    status   Show session status (phase, ratings, tokens)
    resume   Get session state for continuation (status + active game)
    rules    Print full DaveChess rules text
    state    Show current game board, legal moves, resources
    move     Play a move in DCN notation
    study    Study N grandmaster games (LEARNING phase only)
    practice Start a practice game (LEARNING phase only)
    evaluate Transition to EVALUATION phase (from LEARNING)
    result   Show final results (COMPLETED phase only)

Usage:
    python scripts/agent_cli.py create --name "my-run" [--budget N]
                                       [--baseline-games N] [--skip-baseline]
                                       [--eval-min-games N] [--eval-max-games N]
    python scripts/agent_cli.py status <session_file>
    python scripts/agent_cli.py resume <session_file>
    python scripts/agent_cli.py rules <session_file>
    python scripts/agent_cli.py state <session_file>
    python scripts/agent_cli.py move <session_file> <move_dcn>
    python scripts/agent_cli.py study <session_file> <num_games>
    python scripts/agent_cli.py practice <session_file> <opponent_elo>
    python scripts/agent_cli.py evaluate <session_file>
    python scripts/agent_cli.py result <session_file>

Session Lifecycle:
    BASELINE -> LEARNING -> EVALUATION -> COMPLETED

    With --skip-baseline, starts directly at LEARNING.

See docs/benchmark_agent_guide.md for detailed usage instructions.
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

# Default calibration for MCTSLite (no network)
DEFAULT_CALIBRATION = [
    CalibratedLevel(sim_count=0, measured_elo=400),
    CalibratedLevel(sim_count=10, measured_elo=800),
    CalibratedLevel(sim_count=50, measured_elo=1200),
    CalibratedLevel(sim_count=200, measured_elo=1800),
    CalibratedLevel(sim_count=800, measured_elo=2400),
]


def _save_session(session: BenchmarkSession, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(session, f)


def _load_session(path: str) -> BenchmarkSession:
    with open(path, "rb") as f:
        return pickle.load(f)


def _print(data):
    """Print dict as indented JSON."""
    print(json.dumps(data, indent=2, default=str))


def cmd_create(args):
    pool = OpponentPool(network=None, device="cpu", calibration=DEFAULT_CALIBRATION)
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
        "session_file": session_file,
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

    _print(output)


def cmd_status(args):
    session = _load_session(args.session_file)
    status = session.get_status()
    status["session_file"] = args.session_file
    _print(status)


def cmd_rules(args):
    session = _load_session(args.session_file)
    print(session.get_rules())


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

    result["session_file"] = args.session_file
    _print(result)
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
            _print({"error": "No active practice game"})
            return
    else:
        _print({"error": f"Cannot play moves in {session.phase.value} phase"})
        return

    result = session.play_move(game_id, args.move_dcn)
    _save_session(session, args.session_file)
    result["session_file"] = args.session_file
    _print(result)


def cmd_study(args):
    session = _load_session(args.session_file)
    try:
        result = session.study_games(args.num_games)
    except Exception as e:
        _print({"error": str(e), "session_file": args.session_file})
        return
    _save_session(session, args.session_file)
    _print({
        "session_file": args.session_file,
        "num_returned": result["num_returned"],
        "remaining": result["remaining_in_library"],
        "games": result["games"],
    })


def cmd_practice(args):
    session = _load_session(args.session_file)
    try:
        result = session.start_practice_game(args.opponent_elo)
    except Exception as e:
        _print({"error": str(e), "session_file": args.session_file})
        return
    _save_session(session, args.session_file)
    result["session_file"] = args.session_file
    _print(result)


def cmd_evaluate(args):
    session = _load_session(args.session_file)
    try:
        result = session.request_evaluation()
    except Exception as e:
        _print({"error": str(e), "session_file": args.session_file})
        return
    _save_session(session, args.session_file)
    result["session_file"] = args.session_file
    _print(result)


def cmd_resume(args):
    """Print session status + current game state for a continuation agent."""
    session = _load_session(args.session_file)
    status = session.get_status()
    output = {
        "session_file": args.session_file,
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
    _print(output)


def cmd_result(args):
    session = _load_session(args.session_file)
    try:
        result = session.get_result()
    except Exception as e:
        _print({"error": str(e), "session_file": args.session_file})
        return
    result["session_file"] = args.session_file
    _print(result)


def main():
    parser = argparse.ArgumentParser(description="Benchmark agent CLI")
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

    # result
    p = sub.add_parser("result", help="Show final results")
    p.add_argument("session_file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

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
        "result": cmd_result,
    }[args.command](args)


if __name__ == "__main__":
    main()
