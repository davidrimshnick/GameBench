"""Shared test fixtures for benchmark API tests."""

import pytest

import davechess.game.rules as _rules_mod

# Store the real apply_move so we can wrap it
_real_apply_move = _rules_mod.apply_move


def _fast_apply_move(state, move):
    """apply_move wrapper that forces games to draw after 4 turns."""
    result = _real_apply_move(state, move)
    # Force draw at turn 4 instead of turn 100
    if state.turn > 4 and not state.done:
        state.done = True
        state.winner = None
    return result


@pytest.fixture
def short_games(monkeypatch):
    """Monkeypatch apply_move so games end in ~4 turns (draw).

    This makes baseline/eval games fast enough for HTTP-level tests
    while still exercising full Glicko-2 and phase-transition logic.
    """
    # Patch in every module that has already imported apply_move
    monkeypatch.setattr(_rules_mod, "apply_move", _fast_apply_move)
    monkeypatch.setattr(
        "davechess.benchmark.api.session.apply_move", _fast_apply_move
    )
    monkeypatch.setattr(
        "davechess.benchmark.game_manager.apply_move", _fast_apply_move
    )
    monkeypatch.setattr(
        "davechess.benchmark.sequential_eval.apply_move", _fast_apply_move
    )
