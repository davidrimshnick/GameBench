"""Manages active practice and evaluation games for the agentic benchmark."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from davechess.game.state import GameState, Player, Move, PIECE_CHARS
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.board import render_board, BOARD_SIZE
from davechess.game.notation import move_to_dcn, dcn_to_move
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.data.generator import Agent

logger = logging.getLogger("davechess.benchmark")


@dataclass
class ActiveGame:
    """State for a single active game."""
    game_id: str
    state: GameState
    opponent: Agent
    opponent_elo: int
    agent_color: Player
    move_history_dcn: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished: bool = False
    result: Optional[str] = None  # "win", "loss", "draw"
    score: Optional[float] = None  # 1.0, 0.0, 0.5


class GameManager:
    """Manages concurrent practice games for the agent.

    Handles game creation, move application, opponent responses,
    and game state queries.
    """

    def __init__(self, opponent_pool: OpponentPool, max_concurrent: int = 5):
        self.opponent_pool = opponent_pool
        self.max_concurrent = max_concurrent
        self.games: dict[str, ActiveGame] = {}
        self._game_counter = 0

    def start_game(self, opponent_elo: int,
                   agent_color: Player = Player.WHITE) -> dict:
        """Create a new practice game.

        Args:
            opponent_elo: Target ELO for the opponent.
            agent_color: Which color the agent plays.

        Returns:
            Dict with game_id, initial state info, and legal moves.
        """
        active_count = sum(1 for g in self.games.values() if not g.finished)
        if active_count >= self.max_concurrent:
            return {"error": f"Maximum {self.max_concurrent} concurrent games. "
                    "Finish a game before starting a new one."}

        self._game_counter += 1
        game_id = f"game_{self._game_counter:03d}"

        opponent = self.opponent_pool.get_opponent(opponent_elo)
        state = GameState()

        game = ActiveGame(
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

        self.games[game_id] = game
        return self._state_response(game, extra={
            "message": f"Game started against ~ELO {opponent_elo} opponent.",
            "your_color": "white" if agent_color == Player.WHITE else "black",
        })

    def play_move(self, game_id: str, move_dcn: str) -> dict:
        """Apply agent's move and get opponent response.

        Args:
            game_id: The game identifier.
            move_dcn: Agent's move in DCN notation.

        Returns:
            Dict with move result, opponent response, updated state.
        """
        game = self.games.get(game_id)
        if game is None:
            return {"error": f"Game '{game_id}' not found."}
        if game.finished:
            return {"error": f"Game '{game_id}' is already finished.",
                    "result": game.result}

        # Check if game ended (e.g. turn limit reached)
        if game.state.done and not game.finished:
            self._finalize_game(game)
            return {**{"game_over": True, "result": game.result},
                    **self._state_response(game)}

        # Verify it's the agent's turn
        if game.state.current_player != game.agent_color:
            return {"error": "It's not your turn."}

        # Parse and validate the move
        legal_moves = generate_legal_moves(game.state)
        legal_dcn_map = {move_to_dcn(game.state, m): m for m in legal_moves}

        if move_dcn not in legal_dcn_map:
            # Try case-insensitive match
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

        response = {"your_move": move_dcn}

        # Check if game ended after agent's move
        if game.state.done:
            self._finalize_game(game)
            response["game_over"] = True
            response["result"] = game.result
            return {**response, **self._state_response(game)}

        # Opponent responds
        opp_move = game.opponent.get_move(game.state)
        opp_dcn = move_to_dcn(game.state, opp_move)
        apply_move(game.state, opp_move)
        game.move_history_dcn.append(opp_dcn)
        response["opponent_move"] = opp_dcn

        # Check if game ended after opponent's move
        if game.state.done:
            self._finalize_game(game)
            response["game_over"] = True
            response["result"] = game.result

        return {**response, **self._state_response(game)}

    def get_state(self, game_id: str) -> dict:
        """Get current state of a game.

        Args:
            game_id: The game identifier.

        Returns:
            Dict with board, legal moves, history, status.
        """
        game = self.games.get(game_id)
        if game is None:
            return {"error": f"Game '{game_id}' not found."}
        # Finalize if game engine says done but we haven't marked it yet
        if game.state.done and not game.finished:
            self._finalize_game(game)
        return self._state_response(game)

    def get_finished_games(self) -> list[ActiveGame]:
        """Return all completed games."""
        return [g for g in self.games.values() if g.finished]

    def get_active_games(self) -> list[ActiveGame]:
        """Return all in-progress games."""
        return [g for g in self.games.values() if not g.finished]

    def _finalize_game(self, game: ActiveGame) -> None:
        """Set game result from the agent's perspective."""
        game.finished = True
        if game.state.winner is None:
            game.result = "draw"
            game.score = 0.5
        elif game.state.winner == game.agent_color:
            game.result = "win"
            game.score = 1.0
        else:
            game.result = "loss"
            game.score = 0.0

    def _state_response(self, game: ActiveGame,
                        extra: Optional[dict] = None) -> dict:
        """Build a state response dict for the agent."""
        resp: dict = {
            "game_id": game.game_id,
            "opponent_elo": game.opponent_elo,
            "turn": game.state.turn,
            "board": render_board(_board_to_tuples(game.state),
                                  resource_counts=game.state.resources,
                                  turn=game.state.turn,
                                  current_player=int(game.state.current_player)),
            "move_history": _format_move_history(game.move_history_dcn),
            "finished": game.finished,
        }

        if game.finished:
            resp["result"] = game.result
        else:
            # Include legal moves if it's agent's turn
            if game.state.current_player == game.agent_color:
                legal = generate_legal_moves(game.state)
                legal_dcn = [move_to_dcn(game.state, m) for m in legal]
                resp["legal_moves"] = legal_dcn
                resp["num_legal_moves"] = len(legal_dcn)
            else:
                resp["waiting_for"] = "opponent"

        # Resource info
        white_res = game.state.resources[0]
        black_res = game.state.resources[1]
        if game.agent_color == Player.WHITE:
            resp["your_resources"] = white_res
            resp["opponent_resources"] = black_res
        else:
            resp["your_resources"] = black_res
            resp["opponent_resources"] = white_res

        if extra:
            resp.update(extra)
        return resp


def _board_to_tuples(state: GameState) -> list[list]:
    """Convert GameState board (Piece objects) to (char, player) tuples for render_board."""
    # Build reverse map: PieceType -> char
    char_map = {v: k for k, v in PIECE_CHARS.items()}
    board = []
    for row in range(BOARD_SIZE):
        row_data = []
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None:
                row_data.append(None)
            else:
                char = char_map.get(piece.piece_type, "?")
                row_data.append((char, int(piece.player)))
        board.append(row_data)
    return board


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
