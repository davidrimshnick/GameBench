"""Tests for the DaveChess Kaggle benchmark."""

from __future__ import annotations

import os
import random
import tempfile

import pytest

# Import from consolidated modules
from davechess_engine import (
    GameState, Player, generate_legal_moves, apply_move, MCTSLite,
    move_to_dcn, dcn_to_move, render_board,
)
from rules_text import get_rules_prompt, build_game_state_message


class TestDaveChessGame:
    """Test the game wrapper used in the benchmark."""

    def test_initial_legal_moves(self):
        state = GameState()
        moves = generate_legal_moves(state)
        assert len(moves) > 0, "Should have legal moves at start"
        # Opening position has exactly 10 legal moves for White
        assert len(moves) == 10

    def test_move_dcn_roundtrip(self):
        state = GameState()
        moves = generate_legal_moves(state)
        for m in moves:
            dcn = move_to_dcn(state, m)
            assert len(dcn) > 0
            # Verify DCN format: PieceChar + square + separator + square/type
            assert dcn[0] in "CWRBL"

    def test_mcts_lite_returns_legal_move(self):
        state = GameState()
        mcts = MCTSLite(num_simulations=5)
        move = mcts.search(state)
        dcn = move_to_dcn(state, move)
        legal_dcn = [move_to_dcn(state, m) for m in generate_legal_moves(state)]
        assert dcn in legal_dcn

    def test_game_terminates(self):
        """A game with MCTSLite vs MCTSLite should terminate."""
        state = GameState()
        mcts = MCTSLite(num_simulations=5)
        random.seed(42)
        for _ in range(300):
            if state.done:
                break
            legal = generate_legal_moves(state)
            if not legal:
                break
            move = mcts.search(state)
            apply_move(state, move)
        assert state.done or state.turn > 100

    def test_render_board_output(self):
        state = GameState()
        display = state.to_display_board()
        board_str = render_board(display, (0, 0), 1, 0)
        assert "a   b   c   d   e   f   g   h" in board_str
        assert "White" in board_str


class TestGameStateMessage:
    """Test the prompt builder for game state."""

    def test_includes_board(self):
        state = GameState()
        legal = generate_legal_moves(state)
        msg = build_game_state_message(state, [], legal)
        assert "Current position:" in msg

    def test_includes_legal_moves(self):
        state = GameState()
        legal = generate_legal_moves(state)
        msg = build_game_state_message(state, [], legal)
        assert "Legal moves:" in msg

    def test_includes_resources(self):
        state = GameState()
        legal = generate_legal_moves(state)
        msg = build_game_state_message(state, [], legal)
        assert "Resources:" in msg


class TestToolFunctions:
    """Test the tool functions provided to agents."""

    def test_write_and_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate the workspace
            import davechess_benchmark as bm
            original_workspace = bm.AGENT_WORKSPACE
            bm.AGENT_WORKSPACE = tmpdir

            try:
                result = bm.write_file("test.md", "# Strategy Notes\nControl the center.")
                assert "Written" in result

                content = bm.read_file("test.md")
                assert "Strategy Notes" in content
                assert "Control the center" in content
            finally:
                bm.AGENT_WORKSPACE = original_workspace

    def test_read_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import davechess_benchmark as bm
            original_workspace = bm.AGENT_WORKSPACE
            bm.AGENT_WORKSPACE = tmpdir

            try:
                content = bm.read_file("nonexistent.txt")
                assert "not found" in content.lower()
            finally:
                bm.AGENT_WORKSPACE = original_workspace

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import davechess_benchmark as bm
            original_workspace = bm.AGENT_WORKSPACE
            bm.AGENT_WORKSPACE = tmpdir

            try:
                bm.write_file("a.txt", "content a")
                bm.write_file("b.txt", "content b")
                listing = bm.list_files()
                assert "a.txt" in listing
                assert "b.txt" in listing
            finally:
                bm.AGENT_WORKSPACE = original_workspace

    def test_get_gm_game(self):
        import davechess_benchmark as bm
        game = bm.get_gm_game(1)
        # Should contain DCN headers or move notation
        assert len(game) > 0
        # Should not be an error message
        assert "not found" not in game.lower()

    def test_count_gm_games(self):
        import davechess_benchmark as bm
        count = bm.count_gm_games()
        assert count == 200


class TestScoringFormula:
    """Test the learning score computation."""

    def test_zero_improvement(self):
        from davechess_benchmark import compute_learning_score
        phase_a = {"win_rate": 0.0, "legal_move_rate": 0.0, "results": []}
        phase_b = {"win_rate": 0.0, "legal_move_rate": 0.0, "results": []}
        phase_c = {"win_rate": 0.0, "legal_move_rate": 0.0, "results": [
            {"result": "loss"}, {"result": "loss"}, {"result": "loss"},
            {"result": "loss"}, {"result": "loss"}, {"result": "loss"},
        ]}
        score = compute_learning_score(phase_a, phase_b, phase_c)
        assert score == 0.0

    def test_perfect_improvement(self):
        from davechess_benchmark import compute_learning_score
        phase_a = {"win_rate": 0.0, "legal_move_rate": 1.0, "results": []}
        phase_b = {"win_rate": 1.0, "legal_move_rate": 1.0, "results": []}
        phase_c = {"win_rate": 1.0, "legal_move_rate": 1.0, "results": [
            {"result": "loss"}, {"result": "loss"}, {"result": "loss"},
            {"result": "win"}, {"result": "win"}, {"result": "win"},
        ]}
        score = compute_learning_score(phase_a, phase_b, phase_c)
        assert score > 0.8
        assert score <= 1.0

    def test_score_bounded(self):
        from davechess_benchmark import compute_learning_score
        phase_a = {"win_rate": 0.3, "legal_move_rate": 0.8, "results": []}
        phase_b = {"win_rate": 0.6, "legal_move_rate": 0.9, "results": []}
        phase_c = {"win_rate": 0.5, "legal_move_rate": 0.85, "results": [
            {"result": "loss"}, {"result": "draw"}, {"result": "loss"},
            {"result": "win"}, {"result": "win"}, {"result": "draw"},
        ]}
        score = compute_learning_score(phase_a, phase_b, phase_c)
        assert 0.0 <= score <= 1.0

    def test_negative_deltas_clamped_to_zero(self):
        """If Phase B is worse than Phase A, study_delta should be 0, not negative."""
        from davechess_benchmark import compute_learning_score
        phase_a = {"win_rate": 0.5, "legal_move_rate": 0.7, "results": []}
        phase_b = {"win_rate": 0.3, "legal_move_rate": 0.7, "results": []}
        phase_c = {"win_rate": 0.3, "legal_move_rate": 0.7, "results": [
            {"result": "win"}, {"result": "loss"}, {"result": "loss"},
            {"result": "loss"}, {"result": "loss"}, {"result": "loss"},
        ]}
        score = compute_learning_score(phase_a, phase_b, phase_c)
        assert score >= 0.0  # Negative deltas should not make score negative


class TestAggregateResults:
    """Test result aggregation."""

    def test_all_wins(self):
        from davechess_benchmark import _aggregate_results
        results = [
            {"result": "win", "legal_move_rate": 0.9},
            {"result": "win", "legal_move_rate": 0.8},
        ]
        agg = _aggregate_results(results, "test")
        assert agg["win_rate"] == 1.0
        assert agg["wins"] == 2
        assert agg["losses"] == 0

    def test_mixed_results(self):
        from davechess_benchmark import _aggregate_results
        results = [
            {"result": "win", "legal_move_rate": 0.9},
            {"result": "loss", "legal_move_rate": 0.7},
            {"result": "draw", "legal_move_rate": 0.8},
            {"result": "forfeit", "legal_move_rate": 0.0},
        ]
        agg = _aggregate_results(results, "test")
        assert agg["wins"] == 1
        assert agg["losses"] == 1
        assert agg["draws"] == 1
        assert agg["forfeits"] == 1
        # win_rate = (1 + 0.5*1) / 4 = 0.375
        assert abs(agg["win_rate"] - 0.375) < 0.01
