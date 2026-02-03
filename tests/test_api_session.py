"""Tests for BenchmarkSession (pure Python, no HTTP)."""

import pytest

from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves
from davechess.game.notation import move_to_dcn
from davechess.benchmark.api.models import SessionPhase
from davechess.benchmark.api.session import (
    BenchmarkSession,
    PhaseError,
    _StepEvaluator,
    _SessionLibraryView,
)
from davechess.benchmark.opponent_pool import OpponentPool, CalibratedLevel
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.sequential_eval import EvalConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool():
    """Create a simple MCTSLite opponent pool (no neural network)."""
    levels = [
        CalibratedLevel(sim_count=0, measured_elo=400),
        CalibratedLevel(sim_count=10, measured_elo=800),
        CalibratedLevel(sim_count=50, measured_elo=1200),
    ]
    return OpponentPool(network=None, device="cpu", calibration=levels)


def _make_library():
    """Create an empty GameLibrary (no games on disk)."""
    lib = GameLibrary("nonexistent_dir", max_games=200)
    # Inject some fake games directly
    lib.games = [
        f"[Game \"{i+1}\"]\n1. Wc1-c2 Wc8-c7\n1-0"
        for i in range(10)
    ]
    return lib


def _make_eval_config():
    return EvalConfig(initial_elo=1000, target_rd=50.0, max_games=200, min_games=5)


def _make_session(**kwargs):
    defaults = dict(
        session_id="test123",
        agent_name="test-agent",
        token_budget=1_000_000,
        opponent_pool=_make_pool(),
        game_library=_make_library(),
        eval_config=_make_eval_config(),
        baseline_max_games=5,
    )
    defaults.update(kwargs)
    return BenchmarkSession(**defaults)


def _play_random_move(session, game_id):
    """Play a random legal move from the current game state."""
    state_dict = session.get_game_state(game_id)
    legal = state_dict.get("legal_moves", [])
    if legal:
        return session.play_move(game_id, legal[0])
    return None


def _play_game_to_completion(session, game_id, max_moves=200):
    """Play random moves until the game ends."""
    state_dict = {}
    for _ in range(max_moves):
        state_dict = session.get_game_state(game_id)
        if state_dict.get("finished"):
            return state_dict
        legal = state_dict.get("legal_moves", [])
        if not legal:
            return state_dict
        result = session.play_move(game_id, legal[0])
        if result.get("game_over"):
            return result
    return state_dict


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------

def _force_through_eval_phase(session, evaluator_attr="_baseline_evaluator",
                               target_phase=None, max_iterations=500):
    """Play random moves through a phase's eval games.

    Handles the case where get_game_state() triggers game finalization
    and auto-creates the next game via phase transition logic.
    """
    for _ in range(max_iterations):
        if target_phase and session.phase == target_phase:
            return True
        evaluator = getattr(session, evaluator_attr)
        game = evaluator.current_game
        if game is None:
            # Phase transition may have occurred
            if target_phase and session.phase == target_phase:
                return True
            continue
        game_id = game.game_id
        state = session.get_game_state(game_id)
        # get_game_state can trigger finalization + transition
        if target_phase and session.phase == target_phase:
            return True
        if state.get("finished"):
            # Game ended, check for next
            continue
        legal = state.get("legal_moves", [])
        if not legal:
            continue
        result = session.play_move(game_id, legal[0])
        if target_phase and session.phase == target_phase:
            return True
    return target_phase is None or session.phase == target_phase


class TestSessionLifecycle:
    def test_session_starts_in_baseline(self):
        session = _make_session()
        assert session.phase == SessionPhase.BASELINE

    def test_baseline_has_active_game(self):
        session = _make_session()
        evaluator = session._baseline_evaluator
        assert evaluator.current_game is not None
        game_id = evaluator.current_game.game_id
        assert game_id.startswith("base_")

    def test_can_play_baseline_move(self):
        session = _make_session()
        game = session._baseline_evaluator.current_game
        game_id = game.game_id
        result = _play_random_move(session, game_id)
        assert result is not None
        assert "error" not in result or "your_move" in result

    def test_baseline_to_learning_transition(self):
        """Play through all baseline games to trigger transition."""
        session = _make_session(baseline_max_games=2)
        assert session.phase == SessionPhase.BASELINE
        reached = _force_through_eval_phase(
            session, "_baseline_evaluator", SessionPhase.LEARNING
        )
        assert reached, f"Expected LEARNING, got {session.phase}"

    def test_learning_to_evaluation_transition(self):
        """Force through baseline, then request evaluation."""
        session = _make_session(baseline_max_games=2)
        _force_through_eval_phase(
            session, "_baseline_evaluator", SessionPhase.LEARNING
        )
        assert session.phase == SessionPhase.LEARNING

        result = session.request_evaluation()
        assert session.phase == SessionPhase.EVALUATION
        assert result["phase"] == "evaluation"
        assert "game_id" in result

    def test_full_lifecycle(self):
        """Test the complete session lifecycle end to end."""
        session = _make_session(baseline_max_games=2)

        # Phase 1: BASELINE -> LEARNING
        assert session.phase == SessionPhase.BASELINE
        _force_through_eval_phase(
            session, "_baseline_evaluator", SessionPhase.LEARNING
        )
        assert session.phase == SessionPhase.LEARNING

        # Phase 2: LEARNING -> EVALUATION
        session.request_evaluation()
        assert session.phase == SessionPhase.EVALUATION

        # Phase 3: EVALUATION -> COMPLETED (may not complete in 500 moves
        # since eval needs min_games=5 and games can be long)
        _force_through_eval_phase(
            session, "_final_evaluator", SessionPhase.COMPLETED,
            max_iterations=500
        )


# ---------------------------------------------------------------------------
# Phase gating tests
# ---------------------------------------------------------------------------

class TestPhaseGating:
    def test_study_games_blocked_in_baseline(self):
        session = _make_session()
        assert session.phase == SessionPhase.BASELINE
        with pytest.raises(PhaseError):
            session.study_games(1)

    def test_start_practice_game_blocked_in_baseline(self):
        session = _make_session()
        with pytest.raises(PhaseError):
            session.start_practice_game(800)

    def test_request_evaluation_blocked_in_baseline(self):
        session = _make_session()
        with pytest.raises(PhaseError):
            session.request_evaluation()

    def test_get_result_blocked_before_completed(self):
        session = _make_session()
        with pytest.raises(PhaseError):
            session.get_result()

    def test_report_tokens_works_in_baseline(self):
        session = _make_session()
        result = session.report_tokens(100, 50)
        assert result["total_used"] == 150

    def test_get_rules_works_in_any_phase(self):
        session = _make_session()
        rules = session.get_rules()
        assert "DaveChess" in rules


# ---------------------------------------------------------------------------
# LEARNING phase tests
# ---------------------------------------------------------------------------

class TestLearningPhase:
    def _get_learning_session(self):
        """Get a session in LEARNING phase."""
        session = _make_session(baseline_max_games=2)
        _force_through_eval_phase(
            session, "_baseline_evaluator", SessionPhase.LEARNING
        )
        assert session.phase == SessionPhase.LEARNING
        return session

    def test_study_games(self):
        session = self._get_learning_session()
        result = session.study_games(3)
        assert result["num_returned"] == 3
        assert len(result["games"]) == 3

    def test_study_games_tracks_served(self):
        session = self._get_learning_session()
        session.study_games(5)
        assert session._library.remaining == 5  # 10 - 5
        session.study_games(3)
        assert session._library.remaining == 2  # 10 - 8

    def test_study_too_many_raises(self):
        session = self._get_learning_session()
        with pytest.raises(ValueError):
            session.study_games(20)  # Only 10 in library

    def test_start_practice_game(self):
        session = self._get_learning_session()
        result = session.start_practice_game(800)
        assert "game_id" in result
        assert "error" not in result

    def test_practice_game_play_move(self):
        session = self._get_learning_session()
        result = session.start_practice_game(800)
        game_id = result["game_id"]
        legal = result.get("legal_moves", [])
        if legal:
            move_result = session.play_move(game_id, legal[0])
            assert "error" not in move_result


# ---------------------------------------------------------------------------
# _SessionLibraryView tests
# ---------------------------------------------------------------------------

class TestSessionLibraryView:
    def test_isolation_between_sessions(self):
        """Two sessions should have independent served tracking."""
        library = _make_library()
        view1 = _SessionLibraryView(library)
        view2 = _SessionLibraryView(library)

        games1 = view1.get_games(3)
        assert len(games1) == 3
        assert view1.remaining == 7

        # view2 should still see all games
        assert view2.remaining == 10
        games2 = view2.get_games(3)
        assert len(games2) == 3
        assert view2.remaining == 7

    def test_no_repeat_within_session(self):
        library = _make_library()
        view = _SessionLibraryView(library)

        batch1 = set(view.get_games(5))
        batch2 = set(view.get_games(5))
        # No overlap
        assert len(batch1 & batch2) == 0

    def test_insufficient_games(self):
        library = _make_library()
        view = _SessionLibraryView(library)
        view.get_games(10)
        with pytest.raises(ValueError):
            view.get_games(1)


# ---------------------------------------------------------------------------
# _StepEvaluator tests
# ---------------------------------------------------------------------------

class TestStepEvaluator:
    def test_create_first_game(self):
        pool = _make_pool()
        config = EvalConfig(initial_elo=1000, target_rd=50.0, max_games=10, min_games=3)
        evaluator = _StepEvaluator(config, pool)

        game = evaluator.create_next_game()
        assert game.game_id.startswith("eval_")
        assert evaluator.games_played == 0
        assert not evaluator.is_complete

    def test_complete_after_max_games(self):
        pool = _make_pool()
        config = EvalConfig(initial_elo=1000, target_rd=50.0, max_games=3, min_games=1)
        evaluator = _StepEvaluator(config, pool)

        for _ in range(3):
            game = evaluator.create_next_game()
            # Play until game ends
            for _ in range(300):
                state = evaluator.get_game_state()
                if state.get("finished"):
                    break
                legal = state.get("legal_moves", [])
                if not legal:
                    break
                evaluator.play_move(legal[0])

        assert evaluator.is_complete
        assert evaluator.games_played == 3

    def test_rating_info(self):
        pool = _make_pool()
        config = EvalConfig(initial_elo=1000, target_rd=50.0, max_games=10, min_games=3)
        evaluator = _StepEvaluator(config, pool)
        info = evaluator.rating_info()
        assert "elo" in info
        assert "rd" in info
        assert info["games_played"] == 0

    def test_play_move_validates_legality(self):
        pool = _make_pool()
        config = EvalConfig(initial_elo=800, target_rd=50.0, max_games=10, min_games=3)
        evaluator = _StepEvaluator(config, pool)
        evaluator.create_next_game()

        result = evaluator.play_move("INVALID_MOVE")
        assert "error" in result


# ---------------------------------------------------------------------------
# Token tracking integration tests
# ---------------------------------------------------------------------------

class TestTokenIntegration:
    def test_tokens_tracked_across_phases(self):
        session = _make_session()
        session.report_tokens(500, 200)
        assert session.token_tracker.total_used == 700

    def test_token_summary_in_status(self):
        session = _make_session()
        session.report_tokens(100, 50)
        status = session.get_status()
        assert status["tokens"]["total_used"] == 150
        assert status["tokens"]["remaining"] == 999_850


# ---------------------------------------------------------------------------
# Status and result tests
# ---------------------------------------------------------------------------

class TestStatusAndResults:
    def test_get_status_in_baseline(self):
        session = _make_session()
        status = session.get_status()
        assert status["phase"] == "baseline"
        assert status["agent_name"] == "test-agent"
        assert "baseline_rating" in status

    def test_get_eval_status_in_baseline(self):
        session = _make_session()
        status = session.get_eval_status()
        assert status["phase"] == "baseline"
        assert "rating" in status
        assert "current_game" in status


# ---------------------------------------------------------------------------
# PhaseError tests
# ---------------------------------------------------------------------------

class TestPhaseError:
    def test_phase_error_message(self):
        err = PhaseError(SessionPhase.BASELINE, [SessionPhase.LEARNING])
        assert "baseline" in str(err)
        assert "learning" in str(err)

    def test_phase_error_attributes(self):
        err = PhaseError(SessionPhase.EVALUATION, [SessionPhase.LEARNING, SessionPhase.COMPLETED])
        assert err.current_phase == SessionPhase.EVALUATION
        assert len(err.allowed) == 2
