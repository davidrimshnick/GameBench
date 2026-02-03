"""FastAPI dependency injection setup."""

from __future__ import annotations

import logging
from typing import Optional

from davechess.benchmark.api.session_manager import SessionManager
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.opponent_pool import OpponentPool, CalibratedLevel
from davechess.benchmark.sequential_eval import EvalConfig

logger = logging.getLogger("davechess.benchmark.api")


def init_app(app, config: dict) -> None:
    """Initialize FastAPI app with shared resources from config.

    Loads OpponentPool, GameLibrary, and EvalConfig, then creates
    a SessionManager and stores it on app.state.

    Args:
        app: FastAPI application instance.
        config: Configuration dict with keys:
            - opponents.calibration: list of {sim_count, elo, rd} dicts
            - opponents.calibration_file: path to calibration JSON (alternative)
            - game_library.games_dir: path to GM games directory
            - game_library.max_games: max games to load
            - eval.initial_elo, eval.target_rd, eval.max_games, eval.min_games
            - baseline_max_games: max games for baseline evaluation
    """
    # --- Opponent Pool ---
    opp_cfg = config.get("opponents", {})
    calibration_file = opp_cfg.get("calibration_file")

    if calibration_file:
        pool = OpponentPool.from_calibration_file(
            calibration_file, network=None, device="cpu"
        )
    else:
        cal_list = opp_cfg.get("calibration", [
            {"sim_count": 0, "elo": 400, "rd": 50},
            {"sim_count": 10, "elo": 800, "rd": 50},
            {"sim_count": 50, "elo": 1200, "rd": 50},
            {"sim_count": 200, "elo": 1800, "rd": 50},
            {"sim_count": 800, "elo": 2400, "rd": 50},
        ])
        levels = [
            CalibratedLevel(
                sim_count=lvl["sim_count"],
                measured_elo=lvl["elo"],
                elo_rd=lvl.get("rd", 50.0),
            )
            for lvl in cal_list
        ]
        pool = OpponentPool(network=None, device="cpu", calibration=levels)

    logger.info(f"Opponent pool: ELO range [{pool.min_elo:.0f}, {pool.max_elo:.0f}]")

    # --- Game Library ---
    lib_cfg = config.get("game_library", {})
    games_dir = lib_cfg.get("games_dir", "data/gm_games")
    max_games = lib_cfg.get("max_games", 200)

    library = GameLibrary(games_dir, max_games=max_games)
    loaded = library.load()
    logger.info(f"Game library: {loaded} games loaded from {games_dir}")

    # --- Eval Config ---
    eval_cfg = config.get("eval", {})
    eval_config = EvalConfig(
        initial_elo=eval_cfg.get("initial_elo", 1000),
        target_rd=eval_cfg.get("target_rd", 50.0),
        max_games=eval_cfg.get("max_games", 200),
        min_games=eval_cfg.get("min_games", 10),
    )

    baseline_max_games = config.get("baseline_max_games", 30)

    # --- Session Manager ---
    manager = SessionManager(
        opponent_pool=pool,
        game_library=library,
        eval_config=eval_config,
        baseline_max_games=baseline_max_games,
    )

    app.state.session_manager = manager
    logger.info("Benchmark API initialized")


def get_session_manager(app) -> SessionManager:
    """Get SessionManager from app state."""
    return app.state.session_manager
