"""Tests for FastAPI endpoints via TestClient.

Uses the ``short_games`` fixture (conftest.py) to monkeypatch ``apply_move``
so every game ends in a draw after ~4 turns, making baseline / eval fast
enough for HTTP-level round-trip tests.
"""

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from davechess.benchmark.api.server import app as _shared_app
from davechess.benchmark.api.dependencies import init_app
from davechess.benchmark.api.models import SessionPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def client(short_games):
    """TestClient with short-game patching active."""
    init_app(_shared_app, _APP_CONFIG)

    # Inject fake games into the library
    manager = _shared_app.state.session_manager
    manager.game_library.games = [
        f'[Game "{i+1}"]\n1. Wc1-c2 Wc8-c7\n1-0'
        for i in range(10)
    ]

    return TestClient(_shared_app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_session(client) -> dict:
    resp = client.post("/sessions", json={
        "token_budget": 1_000_000,
        "agent_name": "test-bot",
    })
    assert resp.status_code == 201
    return resp.json()


def _play_through_baseline(client, session_id: str) -> None:
    """Play random moves through baseline games until LEARNING phase."""
    for _ in range(200):
        status = client.get(f"/sessions/{session_id}").json()
        if status["phase"] != "baseline":
            return

        eval_resp = client.get(f"/sessions/{session_id}/eval/status")
        eval_data = eval_resp.json()
        if eval_data.get("current_game") is None:
            # Game may have just finished; loop will re-check phase
            continue

        game_id = eval_data["current_game"]["game_id"]
        game_state = eval_data["current_game"]["game_state"]
        legal = game_state.get("legal_moves", [])
        if not legal:
            continue

        client.post(
            f"/sessions/{session_id}/games/{game_id}/move",
            json={"move_dcn": legal[0]},
        )


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_create_returns_201(self, client):
        resp = client.post("/sessions", json={
            "token_budget": 100000,
            "agent_name": "my-agent",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["phase"] == "baseline"
        assert "rules" in data
        assert "game_id" in data

    def test_create_requires_fields(self, client):
        resp = client.post("/sessions", json={})
        assert resp.status_code == 422

    def test_create_rejects_zero_budget(self, client):
        resp = client.post("/sessions", json={
            "token_budget": 0,
            "agent_name": "agent",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session status
# ---------------------------------------------------------------------------

class TestSessionStatus:
    def test_get_status(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        status = resp.json()
        assert status["session_id"] == sid
        assert status["phase"] == "baseline"
        assert "tokens" in status

    def test_not_found(self, client):
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Rules endpoint
# ---------------------------------------------------------------------------

class TestRulesEndpoint:
    def test_get_rules(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.get(f"/sessions/{sid}/rules")
        assert resp.status_code == 200
        assert "DaveChess" in resp.json()["rules"]


# ---------------------------------------------------------------------------
# Move endpoint
# ---------------------------------------------------------------------------

class TestPlayMove:
    def test_play_valid_move(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        game_id = data["game_id"]
        legal = data["game_state"].get("legal_moves", [])
        assert legal, "Expected legal moves in initial state"

        resp = client.post(
            f"/sessions/{sid}/games/{game_id}/move",
            json={"move_dcn": legal[0]},
        )
        assert resp.status_code == 200

    def test_play_illegal_move(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        game_id = data["game_id"]

        resp = client.post(
            f"/sessions/{sid}/games/{game_id}/move",
            json={"move_dcn": "INVALID"},
        )
        assert resp.status_code == 400

    def test_play_wrong_game_id(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.post(
            f"/sessions/{sid}/games/wrong_id/move",
            json={"move_dcn": "Wc1-c2"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Token reporting
# ---------------------------------------------------------------------------

class TestTokenReporting:
    def test_report_tokens(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.post(f"/sessions/{sid}/tokens", json={
            "prompt_tokens": 500,
            "completion_tokens": 200,
        })
        assert resp.status_code == 200
        tokens = resp.json()["tokens"]
        assert tokens["total_used"] == 700


# ---------------------------------------------------------------------------
# Study games (requires LEARNING phase)
# ---------------------------------------------------------------------------

class TestStudyGames:
    def test_study_blocked_in_baseline(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.post(f"/sessions/{sid}/study", json={"num_games": 1})
        assert resp.status_code == 409

    def test_study_works_in_learning(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        status = client.get(f"/sessions/{sid}").json()
        assert status["phase"] == "learning", \
            f"Expected learning phase, got {status['phase']}"

        resp = client.post(f"/sessions/{sid}/study", json={"num_games": 2})
        assert resp.status_code == 200
        assert resp.json()["num_returned"] == 2


# ---------------------------------------------------------------------------
# Practice games (requires LEARNING phase)
# ---------------------------------------------------------------------------

class TestPracticeGames:
    def test_start_practice_game(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        status = client.get(f"/sessions/{sid}").json()
        assert status["phase"] == "learning"

        resp = client.post(f"/sessions/{sid}/games",
                           json={"opponent_elo": 800})
        assert resp.status_code == 200
        assert "game_id" in resp.json()

    def test_practice_game_play_move(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        resp = client.post(f"/sessions/{sid}/games",
                           json={"opponent_elo": 600})
        game_id = resp.json()["game_id"]
        legal = resp.json()["game_state"].get("legal_moves", [])
        assert legal

        move_resp = client.post(
            f"/sessions/{sid}/games/{game_id}/move",
            json={"move_dcn": legal[0]},
        )
        assert move_resp.status_code == 200


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_blocked_in_baseline(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.post(f"/sessions/{sid}/evaluate")
        assert resp.status_code == 409

    def test_evaluate_transition(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        status = client.get(f"/sessions/{sid}").json()
        assert status["phase"] == "learning"

        resp = client.post(f"/sessions/{sid}/evaluate")
        assert resp.status_code == 200
        assert resp.json()["phase"] == "evaluation"

    def test_eval_status(self, client):
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        client.post(f"/sessions/{sid}/evaluate")
        resp = client.get(f"/sessions/{sid}/eval/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "evaluation"
        assert body["is_complete"] is False
        assert body["current_game"] is not None

    def test_eval_play_through(self, client):
        """Play moves through an eval game via HTTP."""
        data = _create_session(client)
        sid = data["session_id"]
        _play_through_baseline(client, sid)

        resp = client.post(f"/sessions/{sid}/evaluate")
        game_id = resp.json()["game_id"]
        game_state = resp.json()["game_state"]

        # Play a few moves in the eval game
        legal = game_state.get("legal_moves", [])
        if legal:
            move_resp = client.post(
                f"/sessions/{sid}/games/{game_id}/move",
                json={"move_dcn": legal[0]},
            )
            assert move_resp.status_code == 200


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_delete_existing(self, client):
        data = _create_session(client)
        sid = data["session_id"]

        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404
