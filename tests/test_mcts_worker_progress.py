"""Tests for live per-game progress notifications from MCTS workers."""

from queue import Empty, Queue

from davechess.engine import mcts_worker
from davechess.engine.gpu_server import GameCompleted, WorkerDone


def test_worker_entry_emits_live_game_completed_once(monkeypatch):
    def fake_play_wave(wave_games, nn_mcts, random_mcts, evaluator,
                       temperature_threshold, gumbel_search=None,
                       on_game_finished=None):
        assert on_game_finished is not None
        game = wave_games[0]
        game.finished = True
        game.winner_str = "white"
        game.move_count = 7
        game.draw_reason = None
        on_game_finished(game)

    monkeypatch.setattr(mcts_worker, "_play_wave", fake_play_wave)

    request_queue = Queue()
    response_queue = Queue()
    results_queue = Queue()

    mcts_worker.worker_entry(
        worker_id=3,
        request_queue=request_queue,
        response_queue=response_queue,
        results_queue=results_queue,
        game_assignments=[{
            "game_idx": 12,
            "is_random": False,
            "nn_plays_white": True,
        }],
        mcts_config={
            "num_simulations": 1,
            "cpuct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature_threshold": 30,
        },
    )

    msg = request_queue.get_nowait()
    assert isinstance(msg, GameCompleted)
    assert msg.worker_id == 3
    assert msg.game_idx == 12
    assert msg.game_type == "selfplay"
    assert msg.length == 7
    assert msg.winner == "white"

    msg = request_queue.get_nowait()
    assert isinstance(msg, WorkerDone)
    assert msg.worker_id == 3

    try:
        extra = request_queue.get_nowait()
    except Empty:
        extra = None
    assert extra is None

    worker_id, worker_results = results_queue.get_nowait()
    assert worker_id == 3
    assert len(worker_results) == 1
    assert worker_results[0]["game_record"]["winner"] == "white"
    assert worker_results[0]["game_record"]["length"] == 7
