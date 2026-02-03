"""Core benchmark session: phase-gated game play and evaluation.

BenchmarkSession is the pure-Python core with no HTTP dependency.
It manages the full lifecycle: BASELINE -> LEARNING -> EVALUATION -> COMPLETED.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from davechess.benchmark.api.models import SessionPhase
from davechess.benchmark.game_manager import GameManager, ActiveGame
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.benchmark.token_tracker import TokenTracker
from davechess.benchmark.prompt import get_rules_prompt
from davechess.benchmark.sequential_eval import EvalConfig
from davechess.data.elo import Glicko2Rating, glicko2_update
from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.notation import move_to_dcn
from davechess.game.board import render_board, BOARD_SIZE

logger = logging.getLogger("davechess.benchmark.api")


class PhaseError(Exception):
    """Raised when an action is attempted in the wrong session phase."""

    def __init__(self, current_phase: SessionPhase, allowed: list[SessionPhase]):
        self.current_phase = current_phase
        self.allowed = allowed
        names = ", ".join(p.value for p in allowed)
        super().__init__(
            f"Action not allowed in {current_phase.value} phase. "
            f"Allowed in: {names}"
        )


# ---------------------------------------------------------------------------
# _SessionLibraryView: per-session wrapper around shared GameLibrary
# ---------------------------------------------------------------------------

class _SessionLibraryView:
    """Per-session view of the shared GameLibrary.

    Each session tracks its own set of served indices so games aren't
    repeated within a session, even though the underlying library is shared.
    """

    def __init__(self, library: GameLibrary):
        self._library = library
        self._served_indices: set[int] = set()

    def get_games(self, n: int) -> list[str]:
        """Return N unserved games for this session."""
        available = [
            i for i in range(len(self._library.games))
            if i not in self._served_indices
        ]
        if len(available) < n:
            raise ValueError(
                f"Requested {n} games but only {len(available)} remain "
                f"unserved in this session"
            )
        selected = random.sample(available, n)
        for idx in selected:
            self._served_indices.add(idx)
        return [self._library.games[i] for i in selected]

    @property
    def remaining(self) -> int:
        return len(self._library.games) - len(self._served_indices)

    @property
    def total_games(self) -> int:
        return len(self._library.games)


# ---------------------------------------------------------------------------
# _StepEvaluator: step-based Glicko-2 evaluation
# ---------------------------------------------------------------------------

@dataclass
class _EvalGame:
    """An evaluation game managed by _StepEvaluator."""
    game_id: str
    state: GameState
    opponent: object  # Agent
    opponent_elo: int
    agent_color: Player
    move_history_dcn: list[str] = field(default_factory=list)
    finished: bool = False
    score: Optional[float] = None  # 1.0/0.5/0.0 from agent's perspective


class _StepEvaluator:
    """Step-based adaptation of SequentialEvaluator.

    Instead of running a blocking evaluate() loop, this exposes
    create_next_game() / record_game_result() for incremental use.
    Reuses opponent selection logic and Glicko-2 updates.
    """

    def __init__(self, config: EvalConfig, opponent_pool: OpponentPool,
                 id_prefix: str = "eval"):
        self.config = config
        self.pool = opponent_pool
        self.rating = Glicko2Rating.from_rating(config.initial_elo, rd=350.0)
        self.results: list[dict] = []
        self._game_counter = 0
        self._id_prefix = id_prefix
        self.current_game: Optional[_EvalGame] = None

    @property
    def games_played(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r["score"] == 1.0)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.results if r["score"] == 0.0)

    @property
    def draws(self) -> int:
        return sum(1 for r in self.results if r["score"] == 0.5)

    @property
    def is_complete(self) -> bool:
        """Check if evaluation has reached target confidence or game limit."""
        if self.games_played >= self.config.max_games:
            return True
        if (self.games_played >= self.config.min_games
                and self.rating.rd < self.config.target_rd):
            return True
        return False

    def create_next_game(self) -> _EvalGame:
        """Create the next evaluation game with adaptively chosen opponent."""
        if self.is_complete:
            raise RuntimeError("Evaluation is already complete")

        self._game_counter += 1
        game_id = f"{self._id_prefix}_{self._game_counter:03d}"

        opponent_elo = self._select_opponent_elo()
        agent_plays_white = (self._game_counter % 2 == 1)
        agent_color = Player.WHITE if agent_plays_white else Player.BLACK

        opponent = self.pool.get_opponent(opponent_elo)
        state = GameState()

        game = _EvalGame(
            game_id=game_id,
            state=state,
            opponent=opponent,
            opponent_elo=opponent_elo,
            agent_color=agent_color,
        )

        # If agent is Black, opponent plays first
        if agent_color == Player.BLACK:
            opp_move = opponent.get_move(state)
            dcn = move_to_dcn(state, opp_move)
            apply_move(state, opp_move)
            game.move_history_dcn.append(dcn)

        self.current_game = game
        return game

    def play_move(self, move_dcn: str) -> dict:
        """Play agent's move in the current eval game. Returns state dict."""
        game = self.current_game
        if game is None:
            return {"error": "No active evaluation game"}
        if game.finished:
            return {"error": f"Game '{game.game_id}' is already finished",
                    "result": _score_to_result(game.score)}

        # Check if game ended before this move
        if game.state.done and not game.finished:
            self._finalize_game(game)
            return {**{"game_over": True, "result": _score_to_result(game.score)},
                    **self._state_dict(game)}

        # Verify it's agent's turn
        if game.state.current_player != game.agent_color:
            return {"error": "It's not your turn."}

        # Parse and validate
        legal_moves = generate_legal_moves(game.state)
        legal_dcn_map = {move_to_dcn(game.state, m): m for m in legal_moves}

        if move_dcn not in legal_dcn_map:
            # Case-insensitive match
            for dcn, move in legal_dcn_map.items():
                if dcn.lower() == move_dcn.lower():
                    move_dcn = dcn
                    break
            else:
                legal_list = list(legal_dcn_map.keys())
                return {
                    "error": f"'{move_dcn}' is not a legal move.",
                    "legal_moves": legal_list[:30],
                    "total_legal_moves": len(legal_list),
                }

        # Apply agent's move
        agent_move = legal_dcn_map[move_dcn]
        apply_move(game.state, agent_move)
        game.move_history_dcn.append(move_dcn)

        response: dict = {"your_move": move_dcn}

        # Trigger checkmate/stalemate detection for the opponent
        # (generate_legal_moves sets state.done if no legal moves exist)
        if not game.state.done:
            generate_legal_moves(game.state)

        # Check game over after agent move
        if game.state.done:
            self._finalize_game(game)
            response["game_over"] = True
            response["result"] = _score_to_result(game.score)
            return {**response, **self._state_dict(game)}

        # Opponent responds
        opp_move = game.opponent.get_move(game.state)
        opp_dcn = move_to_dcn(game.state, opp_move)
        apply_move(game.state, opp_move)
        game.move_history_dcn.append(opp_dcn)
        response["opponent_move"] = opp_dcn

        # Trigger checkmate/stalemate detection for the agent
        if not game.state.done:
            generate_legal_moves(game.state)

        # Check game over after opponent move
        if game.state.done:
            self._finalize_game(game)
            response["game_over"] = True
            response["result"] = _score_to_result(game.score)

        return {**response, **self._state_dict(game)}

    def get_game_state(self) -> dict:
        """Get state of the current eval game."""
        game = self.current_game
        if game is None:
            return {"error": "No active evaluation game"}
        if game.state.done and not game.finished:
            self._finalize_game(game)
        return self._state_dict(game)

    def _finalize_game(self, game: _EvalGame) -> None:
        """Finalize game, update Glicko-2 rating, record result."""
        game.finished = True
        if game.state.winner is None:
            game.score = 0.5
        elif game.state.winner == game.agent_color:
            game.score = 1.0
        else:
            game.score = 0.0

        # Glicko-2 update
        opp_rating = Glicko2Rating.from_rating(game.opponent_elo, rd=50.0)
        self.rating = glicko2_update(self.rating, [opp_rating], [game.score])

        self.results.append({
            "game": self.games_played + 1,
            "game_id": game.game_id,
            "opponent_elo": game.opponent_elo,
            "score": game.score,
            "agent_color": "white" if game.agent_color == Player.WHITE else "black",
            "estimated_elo": self.rating.rating,
            "rd": self.rating.rd,
        })

        logger.info(
            f"Eval game {game.game_id}: vs ELO {game.opponent_elo}, "
            f"score={game.score}, est={self.rating.rating:.0f} "
            f"(+/-{self.rating.rd:.0f})"
        )

        # Auto-create next game if eval not complete
        self.current_game = None

    def _select_opponent_elo(self) -> int:
        """Pick opponent ELO near current estimate for max info gain."""
        current = self.rating.rating
        spread = max(50, self.rating.rd * 0.5)
        target = current + random.gauss(0, spread)
        target = max(self.pool.min_elo, min(self.pool.max_elo, target))
        return round(target)

    def _state_dict(self, game: _EvalGame) -> dict:
        """Build state dict for an eval game."""
        from davechess.benchmark.game_manager import _board_to_tuples

        resp: dict = {
            "game_id": game.game_id,
            "opponent_elo": game.opponent_elo,
            "turn": game.state.turn,
            "board": render_board(
                _board_to_tuples(game.state),
                resource_counts=game.state.resources,
                turn=game.state.turn,
                current_player=int(game.state.current_player),
            ),
            "move_history": _format_move_history(game.move_history_dcn),
            "finished": game.finished,
            "agent_color": "white" if game.agent_color == Player.WHITE else "black",
        }

        if game.finished:
            resp["result"] = _score_to_result(game.score)
        else:
            if game.state.current_player == game.agent_color:
                legal = generate_legal_moves(game.state)
                legal_dcn = [move_to_dcn(game.state, m) for m in legal]
                resp["legal_moves"] = legal_dcn
                resp["num_legal_moves"] = len(legal_dcn)
            else:
                resp["waiting_for"] = "opponent"

        # Resources
        white_res = game.state.resources[0]
        black_res = game.state.resources[1]
        if game.agent_color == Player.WHITE:
            resp["your_resources"] = white_res
            resp["opponent_resources"] = black_res
        else:
            resp["your_resources"] = black_res
            resp["opponent_resources"] = white_res

        return resp

    def rating_info(self) -> dict:
        """Return current rating as a dict."""
        return {
            "elo": self.rating.rating,
            "rd": self.rating.rd,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }


# ---------------------------------------------------------------------------
# BenchmarkSession
# ---------------------------------------------------------------------------

class BenchmarkSession:
    """A single benchmark session for one AI agent.

    Lifecycle: BASELINE -> LEARNING -> EVALUATION -> COMPLETED.

    Pure Python — no HTTP dependency. The FastAPI server wraps this.
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        token_budget: int,
        opponent_pool: OpponentPool,
        game_library: GameLibrary,
        eval_config: EvalConfig,
        baseline_max_games: int = 30,
        skip_baseline: bool = False,
    ):
        self.session_id = session_id
        self.agent_name = agent_name
        self.token_tracker = TokenTracker(token_budget)

        self._opponent_pool = opponent_pool
        self._eval_config = eval_config

        # Per-session library view
        self._library = _SessionLibraryView(game_library)

        # Practice games (LEARNING phase)
        self._game_manager = GameManager(opponent_pool, max_concurrent=5)

        # Baseline evaluator — auto-created, smaller max_games
        baseline_config = EvalConfig(
            initial_elo=eval_config.initial_elo,
            target_rd=eval_config.target_rd,
            max_games=baseline_max_games,
            min_games=eval_config.min_games,
        )
        self._baseline_evaluator = _StepEvaluator(
            baseline_config, opponent_pool, id_prefix="base"
        )

        # Final evaluator — created on request_evaluation()
        self._final_evaluator: Optional[_StepEvaluator] = None

        # Phase management: skip baseline if requested
        if skip_baseline:
            self._phase = SessionPhase.LEARNING
        else:
            self._phase = SessionPhase.BASELINE
            # Auto-create first baseline game
            self._baseline_evaluator.create_next_game()

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    # ------------------------------------------------------------------
    # Phase guards
    # ------------------------------------------------------------------

    def _require_phase(self, *allowed: SessionPhase) -> None:
        if self._phase not in allowed:
            raise PhaseError(self._phase, list(allowed))

    # ------------------------------------------------------------------
    # Actions available in ALL game phases
    # ------------------------------------------------------------------

    def play_move(self, game_id: str, move_dcn: str) -> dict:
        """Play a move. Works in BASELINE, LEARNING, and EVALUATION phases."""
        self._require_phase(
            SessionPhase.BASELINE, SessionPhase.LEARNING, SessionPhase.EVALUATION
        )

        if self._phase == SessionPhase.BASELINE:
            return self._play_baseline_move(game_id, move_dcn)
        elif self._phase == SessionPhase.EVALUATION:
            return self._play_eval_move(game_id, move_dcn)
        else:
            # LEARNING — practice games via GameManager
            return self._game_manager.play_move(game_id, move_dcn)

    def get_game_state(self, game_id: str) -> dict:
        """Get game state. Works in BASELINE, LEARNING, and EVALUATION phases."""
        self._require_phase(
            SessionPhase.BASELINE, SessionPhase.LEARNING, SessionPhase.EVALUATION
        )

        if self._phase == SessionPhase.BASELINE:
            result = self._baseline_evaluator.get_game_state()
            # Check if game was finalized by this call and handle transition
            self._check_baseline_transition(result)
            return result
        elif self._phase == SessionPhase.EVALUATION:
            result = self._final_evaluator.get_game_state()
            self._check_eval_transition(result)
            return result
        else:
            return self._game_manager.get_state(game_id)

    def report_tokens(self, prompt_tokens: int, completion_tokens: int) -> dict:
        """Report token usage. Available in any non-completed phase."""
        self._require_phase(
            SessionPhase.BASELINE, SessionPhase.LEARNING, SessionPhase.EVALUATION
        )
        self.token_tracker.record(prompt_tokens, completion_tokens)
        return self.token_tracker.summary()

    def get_rules(self) -> str:
        """Get full DaveChess rules text. Available in any phase."""
        return get_rules_prompt()

    # ------------------------------------------------------------------
    # LEARNING-only actions
    # ------------------------------------------------------------------

    def study_games(self, num_games: int) -> dict:
        """Study grandmaster games. LEARNING phase only."""
        self._require_phase(SessionPhase.LEARNING)
        games = self._library.get_games(num_games)
        return {
            "games": games,
            "num_returned": len(games),
            "remaining_in_library": self._library.remaining,
        }

    def start_practice_game(self, opponent_elo: int) -> dict:
        """Start a practice game. LEARNING phase only."""
        self._require_phase(SessionPhase.LEARNING)
        return self._game_manager.start_game(opponent_elo)

    # ------------------------------------------------------------------
    # LEARNING -> EVALUATION transition
    # ------------------------------------------------------------------

    def request_evaluation(self) -> dict:
        """Transition from LEARNING to EVALUATION phase."""
        self._require_phase(SessionPhase.LEARNING)

        self._phase = SessionPhase.EVALUATION
        self._final_evaluator = _StepEvaluator(
            self._eval_config, self._opponent_pool, id_prefix="eval"
        )
        game = self._final_evaluator.create_next_game()

        logger.info(f"Session {self.session_id}: entering EVALUATION phase")
        return {
            "phase": self._phase.value,
            "game_id": game.game_id,
            "game_state": self._final_evaluator.get_game_state(),
        }

    # ------------------------------------------------------------------
    # Status / results
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current session status."""
        status: dict = {
            "session_id": self.session_id,
            "phase": self._phase.value,
            "agent_name": self.agent_name,
            "tokens": self.token_tracker.summary(),
        }
        status["baseline_rating"] = self._baseline_evaluator.rating_info()
        if self._final_evaluator is not None:
            status["final_rating"] = self._final_evaluator.rating_info()
        return status

    def get_eval_status(self) -> dict:
        """Get evaluation progress. Available in BASELINE and EVALUATION."""
        self._require_phase(SessionPhase.BASELINE, SessionPhase.EVALUATION)

        if self._phase == SessionPhase.BASELINE:
            evaluator = self._baseline_evaluator
        else:
            evaluator = self._final_evaluator

        result: dict = {
            "phase": self._phase.value,
            "rating": evaluator.rating_info(),
            "is_complete": evaluator.is_complete,
        }

        if evaluator.current_game is not None:
            game = evaluator.current_game
            result["current_game"] = {
                "game_id": game.game_id,
                "game_num": evaluator.games_played + 1,
                "opponent_elo": game.opponent_elo,
                "agent_color": "white" if game.agent_color == Player.WHITE else "black",
                "game_state": evaluator.get_game_state(),
            }

        return result

    def get_result(self) -> dict:
        """Get final results. COMPLETED phase only."""
        self._require_phase(SessionPhase.COMPLETED)

        baseline_info = self._baseline_evaluator.rating_info()
        final_info = self._final_evaluator.rating_info()

        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "baseline_elo": baseline_info["elo"],
            "baseline_rd": baseline_info["rd"],
            "final_elo": final_info["elo"],
            "final_rd": final_info["rd"],
            "elo_gain": final_info["elo"] - baseline_info["elo"],
            "baseline_games": baseline_info["games_played"],
            "eval_games": final_info["games_played"],
            "tokens": self.token_tracker.summary(),
            "baseline_details": self._baseline_evaluator.results,
            "eval_details": self._final_evaluator.results,
        }

    # ------------------------------------------------------------------
    # Internal: phase transition checks
    # ------------------------------------------------------------------

    def _check_baseline_transition(self, result: dict) -> None:
        """Check if baseline eval is done and transition to LEARNING."""
        evaluator = self._baseline_evaluator
        if evaluator.is_complete and self._phase == SessionPhase.BASELINE:
            self._phase = SessionPhase.LEARNING
            result["phase_transition"] = "learning"
            logger.info(
                f"Session {self.session_id}: baseline complete "
                f"(ELO={evaluator.rating.rating:.0f}, "
                f"RD={evaluator.rating.rd:.0f}), entering LEARNING"
            )
        elif (self._phase == SessionPhase.BASELINE
              and evaluator.current_game is None
              and not evaluator.is_complete
              and not self.token_tracker.exhausted):
            next_game = evaluator.create_next_game()
            result["next_game_id"] = next_game.game_id
            result["next_game_state"] = evaluator.get_game_state()

    def _check_eval_transition(self, result: dict) -> None:
        """Check if final eval is done and transition to COMPLETED."""
        evaluator = self._final_evaluator
        if ((evaluator.is_complete or self.token_tracker.exhausted)
                and self._phase == SessionPhase.EVALUATION):
            self._phase = SessionPhase.COMPLETED
            result["phase_transition"] = "completed"
            logger.info(
                f"Session {self.session_id}: evaluation complete "
                f"(ELO={evaluator.rating.rating:.0f}, "
                f"RD={evaluator.rating.rd:.0f})"
            )
        elif (self._phase == SessionPhase.EVALUATION
              and evaluator.current_game is None
              and not evaluator.is_complete
              and not self.token_tracker.exhausted):
            next_game = evaluator.create_next_game()
            result["next_game_id"] = next_game.game_id
            result["next_game_state"] = evaluator.get_game_state()

    # ------------------------------------------------------------------
    # Internal: baseline moves
    # ------------------------------------------------------------------

    def _play_baseline_move(self, game_id: str, move_dcn: str) -> dict:
        """Handle a move during the BASELINE phase."""
        evaluator = self._baseline_evaluator
        game = evaluator.current_game
        if game is None or game.game_id != game_id:
            return {"error": f"No active baseline game with id '{game_id}'"}

        result = evaluator.play_move(move_dcn)

        if result.get("game_over"):
            self._check_baseline_transition(result)

        return result

    def _play_eval_move(self, game_id: str, move_dcn: str) -> dict:
        """Handle a move during the EVALUATION phase."""
        evaluator = self._final_evaluator
        game = evaluator.current_game
        if game is None or game.game_id != game_id:
            return {"error": f"No active eval game with id '{game_id}'"}

        result = evaluator.play_move(move_dcn)

        if result.get("game_over"):
            self._check_eval_transition(result)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_result(score: Optional[float]) -> str:
    """Convert numeric score to result string."""
    if score is None:
        return "in_progress"
    if score == 1.0:
        return "win"
    if score == 0.0:
        return "loss"
    return "draw"


def _format_move_history(moves: list[str]) -> str:
    """Format move list as numbered pairs."""
    if not moves:
        return "(no moves yet)"
    lines = []
    for i in range(0, len(moves), 2):
        num = i // 2 + 1
        white = moves[i]
        if i + 1 < len(moves):
            lines.append(f"{num}. {white} {moves[i + 1]}")
        else:
            lines.append(f"{num}. {white}")
    return "\n".join(lines)
