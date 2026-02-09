"""Tests for BenchmarkClient SDK against a test server.

Uses the ``short_games`` fixture (conftest.py) to monkeypatch ``apply_move``
so every game ends in a draw after ~4 turns, making baseline / eval fast
enough for SDK round-trip tests.
"""

import pytest

from fastapi.testclient import TestClient

from davechess.benchmark.api.server import app as _shared_app
from davechess.benchmark.api.dependencies import init_app
from davechess.benchmark.sdk import BenchmarkClient, BenchmarkAPIError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TestTransport:
    """Adapter to make BenchmarkClient use FastAPI TestClient instead of requests."""

    def __init__(self, test_client: TestClient):
        self._client = test_client

    def post(self, url, json=None, timeout=None):
        return self._client.post(url, json=json)

    def get(self, url, timeout=None):
        return self._client.get(url)

    def delete(self, url, timeout=None):
        return self._client.delete(url)


_APP_CONFIG = {
    "opponents": {
        "calibration": [
            {"sim_count": 0, "elo": 400, "rd": 50},
            {"sim_count": 10, "elo": 800, "rd": 50},
            {"sim_count": 50, "elo": 1200, "rd": 50},
        ],
    },
    "game_library": {
        "games_dir": "nonexistent_dir",
        "max_games": 200,
    },
    "eval": {
        "initial_elo": 1000,
        "target_rd": 50.0,
        "max_games": 200,
        "min_games": 5,
    },
    "baseline_max_games": 3,
}


@pytest.fixture
def sdk_client(short_games):
    """BenchmarkClient backed by TestClient with short-game patching."""
    init_app(_shared_app, _APP_CONFIG)

    manager = _shared_app.state.session_manager
    manager.game_library.games = [
        f'[Game "{i+1}"]\n1. Wc1-c2 Wc8-c7\n1-0'
        for i in range(10)
    ]

    test_client = TestClient(_shared_app)

    client = BenchmarkClient(base_url="")
    client._session = _TestTransport(test_client)
    client._url = lambda path: path
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _play_through_baseline_sdk(client: BenchmarkClient, session_id: str):
    """Play random moves through baseline via SDK until LEARNING phase."""
    for _ in range(200):
        status = client.get_session_status(session_id)
        if status.phase.value != "baseline":
            return

        eval_status = client.get_eval_status(session_id)
        if eval_status.current_game is None:
            continue

        game_id = eval_status.current_game.game_id
        game_state = eval_status.current_game.game_state
        legal = game_state.get("legal_moves", [])
        if not legal:
            continue

        try:
            client.play_move(session_id, game_id, legal[0])
        except BenchmarkAPIError:
            continue


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSDKSessionManagement:
    def test_create_session(self, sdk_client):
        result = sdk_client.create_session(100000, "sdk-test-agent")
        assert result.session_id
        assert result.phase.value == "baseline"
        assert result.game_id

    def test_get_session_status(self, sdk_client):
        create = sdk_client.create_session(100000, "status-agent")
        status = sdk_client.get_session_status(create.session_id)
        assert status.session_id == create.session_id
        assert status.agent_name == "status-agent"

    def test_get_rules(self, sdk_client):
        create = sdk_client.create_session(100000, "rules-agent")
        rules = sdk_client.get_rules(create.session_id)
        assert "DaveChess" in rules

    def test_delete_session(self, sdk_client):
        create = sdk_client.create_session(100000, "delete-agent")
        assert sdk_client.delete_session(create.session_id)
        with pytest.raises(BenchmarkAPIError) as exc_info:
            sdk_client.get_session_status(create.session_id)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Game play
# ---------------------------------------------------------------------------

class TestSDKGamePlay:
    def test_play_move(self, sdk_client):
        create = sdk_client.create_session(100000, "move-agent")
        legal = create.game_state.get("legal_moves", [])
        assert legal, "Expected legal moves"

        result = sdk_client.play_move(create.session_id, create.game_id, legal[0])
        assert result.game_state is not None

    def test_play_illegal_move_raises(self, sdk_client):
        create = sdk_client.create_session(100000, "illegal-agent")
        with pytest.raises(BenchmarkAPIError) as exc_info:
            sdk_client.play_move(create.session_id, create.game_id, "INVALID")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Token reporting
# ---------------------------------------------------------------------------

class TestSDKTokens:
    def test_report_tokens(self, sdk_client):
        create = sdk_client.create_session(100000, "token-agent")
        result = sdk_client.report_tokens(create.session_id, 500, 200)
        assert result.tokens.total_used == 700
        assert result.tokens.remaining == 99300


# ---------------------------------------------------------------------------
# Learning phase
# ---------------------------------------------------------------------------

class TestSDKLearningPhase:
    def test_study_games(self, sdk_client):
        create = sdk_client.create_session(1_000_000, "study-agent")
        _play_through_baseline_sdk(sdk_client, create.session_id)

        status = sdk_client.get_session_status(create.session_id)
        assert status.phase.value == "learning", \
            f"Expected learning, got {status.phase.value}"

        result = sdk_client.study_games(create.session_id, 2)
        assert result.num_returned == 2

    def test_start_game(self, sdk_client):
        create = sdk_client.create_session(1_000_000, "practice-agent")
        _play_through_baseline_sdk(sdk_client, create.session_id)

        status = sdk_client.get_session_status(create.session_id)
        assert status.phase.value == "learning"

        result = sdk_client.start_game(create.session_id, 800)
        assert result.game_id


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestSDKEvaluation:
    def test_request_evaluation(self, sdk_client):
        create = sdk_client.create_session(1_000_000, "eval-agent")
        _play_through_baseline_sdk(sdk_client, create.session_id)

        status = sdk_client.get_session_status(create.session_id)
        assert status.phase.value == "learning"

        result = sdk_client.request_evaluation(create.session_id)
        assert result.phase.value == "evaluation"
        assert result.game_id

    def test_eval_status(self, sdk_client):
        create = sdk_client.create_session(1_000_000, "evalstatus-agent")
        _play_through_baseline_sdk(sdk_client, create.session_id)

        sdk_client.request_evaluation(create.session_id)
        eval_status = sdk_client.get_eval_status(create.session_id)
        assert eval_status.phase.value == "evaluation"
        assert not eval_status.is_complete


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestSDKErrorHandling:
    def test_not_found_error(self, sdk_client):
        with pytest.raises(BenchmarkAPIError) as exc_info:
            sdk_client.get_session_status("nonexistent")
        assert exc_info.value.status_code == 404

    def test_phase_error(self, sdk_client):
        create = sdk_client.create_session(100000, "phase-error-agent")
        with pytest.raises(BenchmarkAPIError) as exc_info:
            sdk_client.study_games(create.session_id, 1)
        assert exc_info.value.status_code == 409
