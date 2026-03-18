"""Tests for the Kaggle benchmark dataset modules (rules_text, elo)."""

from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# elo.py tests
# ---------------------------------------------------------------------------

class TestGlicko2Rating:
    def test_default_rating(self):
        from elo import Glicko2Rating
        r = Glicko2Rating()
        assert r.rating == pytest.approx(1500.0, abs=0.1)

    def test_default_rd(self):
        from elo import Glicko2Rating
        r = Glicko2Rating()
        assert r.rd == pytest.approx(350.0, abs=0.1)

    def test_from_rating_roundtrip(self):
        from elo import Glicko2Rating
        r = Glicko2Rating.from_rating(1800.0, rd=200.0)
        assert r.rating == pytest.approx(1800.0, abs=0.1)
        assert r.rd == pytest.approx(200.0, abs=0.1)


class TestEloFunctions:
    def test_elo_expected_equal(self):
        from elo import elo_expected
        assert elo_expected(1500, 1500) == pytest.approx(0.5)

    def test_elo_expected_stronger(self):
        from elo import elo_expected
        # Higher-rated player should expect > 0.5
        assert elo_expected(1700, 1500) > 0.5

    def test_elo_update_win(self):
        from elo import elo_update
        new = elo_update(1500.0, 0.5, 1.0, k=32.0)
        assert new > 1500.0

    def test_elo_update_loss(self):
        from elo import elo_update
        new = elo_update(1500.0, 0.5, 0.0, k=32.0)
        assert new < 1500.0


class TestEloFromWinrate:
    def test_fifty_percent(self):
        from elo import elo_from_winrate
        assert elo_from_winrate(0.5, 1500) == pytest.approx(1500.0)

    def test_zero_winrate(self):
        from elo import elo_from_winrate
        assert elo_from_winrate(0.0, 1500) == 700.0

    def test_full_winrate(self):
        from elo import elo_from_winrate
        assert elo_from_winrate(1.0, 1500) == 2300.0


class TestGlicko2Update:
    def test_win_increases_rating(self):
        from elo import Glicko2Rating, glicko2_update
        player = Glicko2Rating()
        opp = Glicko2Rating()
        updated = glicko2_update(player, [opp], [1.0])
        assert updated.rating > player.rating

    def test_loss_decreases_rating(self):
        from elo import Glicko2Rating, glicko2_update
        player = Glicko2Rating()
        opp = Glicko2Rating()
        updated = glicko2_update(player, [opp], [0.0])
        assert updated.rating < player.rating

    def test_no_games_increases_rd(self):
        from elo import Glicko2Rating, glicko2_update
        player = Glicko2Rating.from_rating(1500, rd=100)
        updated = glicko2_update(player, [], [])
        assert updated.rd > player.rd


class TestCalculateRatings:
    def test_calculate_elo_ratings(self):
        from elo import calculate_elo_ratings
        # Player 0 beats player 1 three times
        results = [(0, 1, 1.0), (0, 1, 1.0), (0, 1, 1.0)]
        ratings = calculate_elo_ratings(results, num_players=2)
        assert ratings[0] > ratings[1]

    def test_calculate_glicko2_ratings(self):
        from elo import calculate_glicko2_ratings
        results = [(0, 1, 1.0), (0, 1, 1.0), (0, 1, 1.0)]
        ratings = calculate_glicko2_ratings(results, num_players=2, iterations=5)
        assert ratings[0].rating > ratings[1].rating


# ---------------------------------------------------------------------------
# rules_text.py tests
# ---------------------------------------------------------------------------

class TestGetRulesPrompt:
    def test_returns_string(self):
        from rules_text import get_rules_prompt
        result = get_rules_prompt()
        assert isinstance(result, str)

    def test_contains_gold_positions(self):
        from rules_text import get_rules_prompt
        result = get_rules_prompt()
        # Gold nodes are d4, e4, d5, e5
        assert "d4" in result
        assert "e5" in result

    def test_no_format_placeholders(self):
        from rules_text import get_rules_prompt
        result = get_rules_prompt()
        assert "{gold_positions}" not in result

    def test_contains_rules_sections(self):
        from rules_text import get_rules_prompt
        result = get_rules_prompt()
        assert "## Board" in result
        assert "## Pieces" in result
        assert "## Check" in result
        assert "## Win Conditions" in result
        assert "## Notation (DCN)" in result
        assert "## Benchmark Rules" in result


class TestReplayGame:
    def test_empty_game(self):
        from rules_text import replay_game
        states, final = replay_game([])
        assert len(states) == 0
        assert final.turn == 1


class TestFormatExampleGames:
    def test_empty_list(self):
        from rules_text import format_example_games
        assert format_example_games([]) == ""

    def test_max_games_zero(self):
        from rules_text import format_example_games
        assert format_example_games([], max_games=0) == ""


class TestBuildGameStateMessage:
    def test_initial_position(self):
        from rules_text import build_game_state_message
        from davechess_engine import GameState, generate_legal_moves

        state = GameState()
        legal = generate_legal_moves(state)
        msg = build_game_state_message(state, [], legal)

        assert "White" in msg
        assert "Legal moves:" in msg
        assert "Your move:" in msg
        assert "Current position:" in msg

    def test_with_move_history(self):
        from rules_text import build_game_state_message
        from davechess_engine import GameState, generate_legal_moves

        state = GameState()
        legal = generate_legal_moves(state)
        msg = build_game_state_message(state, ["Wb2-b3", "Wb7-b6"], legal)

        assert "Game so far:" in msg
        assert "1. Wb2-b3 Wb7-b6" in msg


class TestBuildSystemPrompt:
    def test_no_examples(self):
        from rules_text import build_system_prompt
        prompt = build_system_prompt([], num_examples=0)
        assert "# DaveChess Rules" in prompt
        assert "# Instructions" in prompt

    def test_with_zero_examples_requested(self):
        from rules_text import build_system_prompt
        prompt = build_system_prompt([], num_examples=5)
        # No games available, so no example section
        assert "# Example Games" not in prompt
