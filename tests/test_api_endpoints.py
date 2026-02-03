"""Tests for FastAPI endpoints via TestClient."""

import pytest

from fastapi.testclient import TestClient

from davechess.benchmark.api.server import app
from davechess.benchmark.api.dependencies import init_app
from davechess.benchmark.api.models import SessionPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a test client with initialized app."""
    config = {
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
    init_app(app, config)

    # Inject fake games into the library
    manager = app.state.session_manager
    manager.game_library.games = [
        f'[Game "{i+1}"]\n1. Wc1-c2 Wc8-c7\n1-0'
        for i in range(10)
    ]

    return TestClient(app)


def _create_session(client) -> dict:
    """Helper to create a session and return response dict."""
    resp = client.post("/sessions", json={
        "token_budget": 1_000_000,
        "agent_name": "test-bot",
    })
    assert resp.status_code == 201
    return resp.json()


def _play_through_baseline(client, session_id: str) -> None:
    """Play random moves through all baseline games."""
    for _ in range(500):
        # Get session status to check phase
        status = client.get(f"/sessions/{session_id}").json()
        if status["phase"] != "baseline":
            return

        # Get eval status to find current game
        eval_resp = client.get(f"/sessions/{session_id}/eval/status")
        eval_data = eval_resp.json()
        if eval_data.get("current_game") is None:
            return

        game_id = eval_data["current_game"]["game_id"]
        game_state = eval_data["current_game"]["game_state"]
        legal = game_state.get("legal_moves", [])
        if not legal:
            continue

        move_resp = client.post(
            f"/sessions/{session_id}/games/{game_id}/move",
            json={"move_dcn": legal[0]},
        )
        if move_resp.status_code != 200:
            continue


# ---------------------------------------------------------------------------
# Session creation tests
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
# Session status tests
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
        game_state = data["game_state"]

        legal = game_state.get("legal_moves", [])
        if not legal:
            pytest.skip("No legal moves in initial state")

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

        # Check we're in learning
        status = client.get(f"/sessions/{sid}").json()
        if status["phase"] != "learning":
            pytest.skip("Could not reach learning phase")

        resp = client.post(f"/sessions/{sid}/study", json={"num_games": 2})
        assert resp.status_code == 200
        assert resp.json()["num_returned"] == 2


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
        if status["phase"] != "learning":
            pytest.skip("Could not reach learning phase")

        resp = client.post(f"/sessions/{sid}/evaluate")
        assert resp.status_code == 200
        assert resp.json()["phase"] == "evaluation"


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

        # Should be gone
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404
