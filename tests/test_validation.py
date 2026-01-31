"""Tests for the validation metrics calculations."""

import pytest

from scripts.validate_game import analyze_results, check_health


def _make_result(winner=0, win_condition="commander_capture", num_moves=80,
                 deployed_types=None, opening_sig="0:MoveStep|1:MoveStep",
                 max_resource_diff=5, max_state_repeats=1):
    return {
        "game_id": 0,
        "winner": winner,
        "win_condition": win_condition,
        "num_moves": num_moves,
        "turn": num_moves // 2,
        "deployed_types": deployed_types or [],
        "opening_sig": opening_sig,
        "max_resource_diff": max_resource_diff,
        "max_state_repeats": max_state_repeats,
    }


class TestAnalyzeResults:
    def test_basic_analysis(self):
        results = [
            _make_result(0, "commander_capture", 80),
            _make_result(1, "resource_domination", 120),
            _make_result(None, "draw", 200),
        ]
        metrics = analyze_results(results)
        assert metrics["num_games"] == 3
        assert "commander_capture" in metrics["win_conditions"]
        assert metrics["avg_length"] == pytest.approx(133.33, abs=1)

    def test_win_rate_calculation(self):
        results = [_make_result(0)] * 6 + [_make_result(1)] * 4
        metrics = analyze_results(results)
        assert metrics["white_win_rate"] == pytest.approx(60.0)

    def test_draw_rate(self):
        results = [_make_result(0)] * 8 + [_make_result(None, "draw")] * 2
        metrics = analyze_results(results)
        assert metrics["draw_rate"] == pytest.approx(20.0)

    def test_piece_usage(self):
        results = [
            _make_result(deployed_types=[(0, 1), (0, 3)]),  # Warrior, Bombard
            _make_result(deployed_types=[(0, 2)]),            # Rider
        ]
        metrics = analyze_results(results)
        assert metrics["piece_usage_pct"]["WARRIOR"] == pytest.approx(50.0)
        assert metrics["piece_usage_pct"]["RIDER"] == pytest.approx(50.0)
        assert metrics["piece_usage_pct"]["BOMBARD"] == pytest.approx(50.0)


class TestHealthChecks:
    def test_healthy_game(self):
        results = []
        for i in range(100):
            if i < 40:
                results.append(_make_result(0, "commander_capture", 80,
                                            deployed_types=[(0, 1), (0, 2), (0, 3)]))
            elif i < 70:
                results.append(_make_result(1, "resource_domination", 100,
                                            deployed_types=[(1, 1), (1, 2), (1, 3)]))
            elif i < 90:
                results.append(_make_result(0, "turn_limit", 150,
                                            deployed_types=[(0, 1)]))
            else:
                results.append(_make_result(None, "draw", 200))

        metrics = analyze_results(results)
        checks = check_health(metrics)
        # Most checks should pass for this balanced distribution
        passed_count = sum(1 for _, p, _ in checks if p)
        assert passed_count > len(checks) // 2

    def test_degenerate_rush(self):
        """All short games should trigger rush check."""
        results = [_make_result(0, "commander_capture", 10)] * 100
        metrics = analyze_results(results)
        assert metrics["rush_rate"] == pytest.approx(100.0)
