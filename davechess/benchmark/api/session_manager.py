"""Session manager: tracks active benchmark sessions."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Optional

from davechess.benchmark.api.session import BenchmarkSession
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.benchmark.sequential_eval import EvalConfig

logger = logging.getLogger("davechess.benchmark.api")


class SessionManager:
    """Manages active benchmark sessions.

    Holds shared resources (OpponentPool, GameLibrary, EvalConfig)
    and creates/tracks BenchmarkSession instances.
    """

    def __init__(
        self,
        opponent_pool: OpponentPool,
        game_library: GameLibrary,
        eval_config: EvalConfig,
        baseline_max_games: int = 30,
    ):
        self.opponent_pool = opponent_pool
        self.game_library = game_library
        self.eval_config = eval_config
        self.baseline_max_games = baseline_max_games

        self._sessions: dict[str, BenchmarkSession] = {}
        self._lock = threading.Lock()

    def create_session(self, token_budget: int, agent_name: str) -> BenchmarkSession:
        """Create a new benchmark session.

        Args:
            token_budget: Total token budget for the session.
            agent_name: Name of the AI agent.

        Returns:
            The created BenchmarkSession.
        """
        session_id = uuid.uuid4().hex[:12]

        session = BenchmarkSession(
            session_id=session_id,
            agent_name=agent_name,
            token_budget=token_budget,
            opponent_pool=self.opponent_pool,
            game_library=self.game_library,
            eval_config=self.eval_config,
            baseline_max_games=self.baseline_max_games,
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info(f"Created session {session_id} for agent '{agent_name}' "
                     f"(budget={token_budget:,})")
        return session

    def get_session(self, session_id: str) -> Optional[BenchmarkSession]:
        """Look up a session by ID. Returns None if not found."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if found and deleted."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
        return False

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)
