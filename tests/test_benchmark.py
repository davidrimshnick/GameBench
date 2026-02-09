"""Tests for benchmark scoring, prompt construction, and move parsing."""

import pytest

from davechess.benchmark.scoring import (
    compute_auc, compute_gamebench_score, compute_llm_elo, compute_learning_curve,
)
from davechess.benchmark.prompt import get_rules_prompt, build_system_prompt
from davechess.benchmark.evaluator import _extract_move, _fuzzy_match_move


class TestScoring:
    def test_auc_flat_curve(self):
        """Flat curve at ELO 1000 over [0, 500]."""
        curve = [(0, 1000), (500, 1000)]
        auc = compute_auc(curve, max_n=500)
        assert auc == pytest.approx(500000.0)

    def test_auc_linear_curve(self):
        """Linearly increasing curve."""
        curve = [(0, 0), (500, 1000)]
        auc = compute_auc(curve, max_n=500)
        assert auc == pytest.approx(250000.0)

    def test_auc_single_point(self):
        """Single point returns point * max_n."""
        curve = [(0, 1000)]
        auc = compute_auc(curve, max_n=500)
        assert auc == pytest.approx(500000.0)

    def test_gamebench_score_random(self):
        """Random play should score 0."""
        curve = [(0, 400), (500, 400)]
        score = compute_gamebench_score(curve, random_elo=400, max_elo=2700)
        assert score == pytest.approx(0.0)

    def test_gamebench_score_perfect(self):
        """Perfect play should score 100."""
        curve = [(0, 2700), (500, 2700)]
        score = compute_gamebench_score(curve, random_elo=400, max_elo=2700)
        assert score == pytest.approx(100.0)

    def test_gamebench_score_middle(self):
        """Middle performance should score ~50."""
        mid_elo = (400 + 2700) / 2
        curve = [(0, mid_elo), (500, mid_elo)]
        score = compute_gamebench_score(curve, random_elo=400, max_elo=2700)
        assert abs(score - 50.0) < 1.0

    def test_gamebench_score_bounded(self):
        """Score should be clamped to [0, 100]."""
        curve = [(0, 0), (500, 0)]  # Below random
        score = compute_gamebench_score(curve, random_elo=400, max_elo=2700)
        assert score == 0.0

    def test_compute_llm_elo(self):
        level_elos = [400, 800, 1200, 1600, 2000]
        # 50% win rate at level 2 (ELO 1200) → should be near 1200
        results = {2: [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}
        elo = compute_llm_elo(results, level_elos)
        assert abs(elo - 1200) < 100

    def test_learning_curve(self):
        data = {0: 400, 10: 800, 100: 1500}
        curve = compute_learning_curve(data)
        assert curve == [(0, 400), (10, 800), (100, 1500)]


class TestPrompt:
    def test_rules_prompt_contains_key_info(self):
        prompt = get_rules_prompt()
        assert "Commander" in prompt
        assert "Warrior" in prompt
        assert "Rider" in prompt
        assert "Bombard" in prompt
        assert "resource" in prompt.lower()
        assert "DCN" in prompt

    def test_system_prompt_no_examples(self):
        prompt = build_system_prompt([], num_examples=0)
        assert "Instructions" in prompt
        assert "Example Games" not in prompt

    def test_system_prompt_with_examples(self):
        # Create a minimal game
        from davechess.game.state import GameState, MoveStep
        moves = [MoveStep((1, 2), (2, 2))]
        games = [(moves, "1-0")]
        prompt = build_system_prompt(games, num_examples=1)
        assert "Example Games" in prompt


class TestMoveExtraction:
    def test_extract_simple_move(self):
        assert _extract_move("Wc1-c2") == "Wc1-c2"

    def test_extract_move_from_text(self):
        assert _extract_move("I'll play Wc1-c2") == "Wc1-c2"

    def test_extract_move_in_backticks(self):
        assert _extract_move("My move: `Wc1-c2`") == "Wc1-c2"

    def test_extract_promote(self):
        assert _extract_move("Wa1>R") == "Wa1>R"

    def test_extract_bombard(self):
        assert _extract_move("Bc3~e3") == "Bc3~e3"

    def test_extract_capture(self):
        assert _extract_move("Rd4xe4") == "Rd4xe4"

    def test_no_move_found(self):
        assert _extract_move("I don't know what to do") is None

    def test_fuzzy_match(self):
        from davechess.game.state import MoveStep
        legal_dcn = {
            "Wc1-c2": MoveStep((0, 2), (1, 2)),
            "Rd1-d3": MoveStep((0, 3), (2, 3)),
        }
        result = _fuzzy_match_move("wc1-c2", legal_dcn)
        assert result is not None
        dcn, move = result
        assert dcn == "Wc1-c2"
