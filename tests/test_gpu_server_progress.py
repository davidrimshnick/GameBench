"""Tests for mid-run GPU server progress aggregation."""

from davechess.engine.gpu_server import GameCompleted, ProgressTracker


def _msg(game_idx: int, game_type: str, winner: str, length: int = 100):
    return GameCompleted(
        worker_id=0,
        game_idx=game_idx,
        game_type=game_type,
        length=length,
        winner=winner,
    )


class TestProgressTracker:
    def test_vs_random_tracks_nn_perspective(self):
        tracker = ProgressTracker()

        # Even game_idx => NN plays White.
        tracker.record(_msg(0, "vs_random", "white", length=40))  # NN win
        # Odd game_idx => NN plays Black.
        tracker.record(_msg(1, "vs_random", "white", length=60))  # NN loss
        tracker.record(_msg(3, "vs_random", "draw", length=200))  # Draw

        assert tracker.vr_wins == 1
        assert tracker.vr_losses == 1
        assert tracker.vr_draws == 1
        assert tracker.games_completed == 3
        assert tracker.total_completed_length == 300

        summary = tracker.summary()
        assert "avg_len=100" in summary
        assert "vs_random W:1 L:1 D:1 score=50.0%" in summary

    def test_selfplay_tracks_white_black_draws(self):
        tracker = ProgressTracker()

        tracker.record(_msg(10, "selfplay", "white", length=80))
        tracker.record(_msg(11, "selfplay", "black", length=120))
        tracker.record(_msg(12, "selfplay", "draw", length=200))

        assert tracker.sp_white_wins == 1
        assert tracker.sp_black_wins == 1
        assert tracker.sp_draws == 1

        summary = tracker.summary()
        assert "selfplay W:1 B:1 D:1" in summary

    def test_mixed_summary_includes_both_sections(self):
        tracker = ProgressTracker()

        tracker.record(_msg(0, "vs_random", "white", length=50))
        tracker.record(_msg(10, "selfplay", "draw", length=200))

        summary = tracker.summary()
        assert "avg_len=125" in summary
        assert "vs_random W:1 L:0 D:0 score=100.0%" in summary
        assert "selfplay W:0 B:0 D:1" in summary
