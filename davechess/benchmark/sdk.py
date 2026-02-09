"""Benchmark SDK: HTTP client and in-process session access.

Usage (HTTP client):
    client = BenchmarkClient("http://localhost:8000")
    session = client.create_session(token_budget=100000, agent_name="my-agent")
    # Play baseline games, study, practice, evaluate...

Usage (in-process, no server needed):
    from davechess.benchmark.api import BenchmarkSession
    # Create directly with BenchmarkSession(...)
"""

from __future__ import annotations

from typing import Optional

import requests

from davechess.benchmark.api.models import (
    CreateSessionResponse,
    SessionStatus,
    StudyGamesResponse,
    StartGameResponse,
    PlayMoveResponse,
    GameStateResponse,
    ReportTokensResponse,
    RequestEvalResponse,
    EvalStatusResponse,
    SessionResultResponse,
    TokenInfo,
    RatingInfo,
    SessionPhase,
)

# Re-export for in-process usage
from davechess.benchmark.api.session import BenchmarkSession  # noqa: F401


class BenchmarkClient:
    """HTTP client for the GameBench benchmark API.

    Wraps all REST endpoints with typed Python methods.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _check(self, resp: requests.Response) -> dict:
        """Check response status and return JSON."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise BenchmarkAPIError(resp.status_code, detail)
        return resp.json()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, token_budget: int,
                       agent_name: str) -> CreateSessionResponse:
        """Create a new benchmark session."""
        resp = self._session.post(
            self._url("/sessions"),
            json={"token_budget": token_budget, "agent_name": agent_name},
            timeout=self.timeout,
        )
        return CreateSessionResponse(**self._check(resp))

    def get_session_status(self, session_id: str) -> SessionStatus:
        """Get session status."""
        resp = self._session.get(
            self._url(f"/sessions/{session_id}"),
            timeout=self.timeout,
        )
        return SessionStatus(**self._check(resp))

    def get_rules(self, session_id: str) -> str:
        """Get DaveChess rules text."""
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/rules"),
            timeout=self.timeout,
        )
        return self._check(resp)["rules"]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        resp = self._session.delete(
            self._url(f"/sessions/{session_id}"),
            timeout=self.timeout,
        )
        self._check(resp)
        return True

    # ------------------------------------------------------------------
    # Game play
    # ------------------------------------------------------------------

    def play_move(self, session_id: str, game_id: str,
                  move_dcn: str) -> PlayMoveResponse:
        """Play a move in a game."""
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/games/{game_id}/move"),
            json={"move_dcn": move_dcn},
            timeout=self.timeout,
        )
        return PlayMoveResponse(**self._check(resp))

    def get_game_state(self, session_id: str,
                       game_id: str) -> GameStateResponse:
        """Get game state."""
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/games/{game_id}"),
            timeout=self.timeout,
        )
        return GameStateResponse(**self._check(resp))

    # ------------------------------------------------------------------
    # Learning phase
    # ------------------------------------------------------------------

    def study_games(self, session_id: str,
                    num_games: int) -> StudyGamesResponse:
        """Study grandmaster games."""
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/study"),
            json={"num_games": num_games},
            timeout=self.timeout,
        )
        return StudyGamesResponse(**self._check(resp))

    def start_game(self, session_id: str,
                   opponent_elo: int) -> StartGameResponse:
        """Start a practice game."""
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/games"),
            json={"opponent_elo": opponent_elo},
            timeout=self.timeout,
        )
        return StartGameResponse(**self._check(resp))

    # ------------------------------------------------------------------
    # Token reporting
    # ------------------------------------------------------------------

    def report_tokens(self, session_id: str, prompt_tokens: int,
                      completion_tokens: int) -> ReportTokensResponse:
        """Report token usage."""
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/tokens"),
            json={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            timeout=self.timeout,
        )
        return ReportTokensResponse(**self._check(resp))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def request_evaluation(self, session_id: str) -> RequestEvalResponse:
        """Transition to evaluation phase."""
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/evaluate"),
            timeout=self.timeout,
        )
        return RequestEvalResponse(**self._check(resp))

    def get_eval_status(self, session_id: str) -> EvalStatusResponse:
        """Get evaluation progress."""
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/eval/status"),
            timeout=self.timeout,
        )
        return EvalStatusResponse(**self._check(resp))

    def get_result(self, session_id: str) -> SessionResultResponse:
        """Get final results."""
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/result"),
            timeout=self.timeout,
        )
        return SessionResultResponse(**self._check(resp))


class BenchmarkAPIError(Exception):
    """Error from the benchmark API."""

    def __init__(self, status_code: int, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")
