"""Sequential evaluation: adaptively play games to determine agent ELO."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.notation import move_to_dcn
from davechess.game.board import render_board, BOARD_SIZE
from davechess.benchmark.opponent_pool import OpponentPool
from davechess.benchmark.token_tracker import TokenTracker
from davechess.data.elo import Glicko2Rating, glicko2_update
from davechess.data.generator import Agent

logger = logging.getLogger("davechess.benchmark")


@dataclass
class EvalConfig:
    """Configuration for sequential evaluation."""
    initial_elo: int = 1000
    target_rd: float = 50.0  # Stop when RD drops below this
    max_games: int = 200
    min_games: int = 10


@dataclass
class EvalResult:
    """Result of the sequential evaluation."""
    estimated_elo: float
    rd: float  # Rating deviation (uncertainty)
    games_played: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    tokens_used: int = 0
    game_results: list[dict] = field(default_factory=list)


class SequentialEvaluator:
    """Evaluates agent ELO using adaptive sequential testing.

    Plays games against calibrated opponents, updating the agent's
    Glicko-2 rating after each game. Selects opponents near the
    agent's estimated ELO for maximum information gain. Stops when
    the rating deviation is narrow enough or budget runs out.
    """

    def __init__(self, config: EvalConfig, opponent_pool: OpponentPool,
                 token_tracker: TokenTracker):
        self.config = config
        self.pool = opponent_pool
        self.tracker = token_tracker
        self.rating = Glicko2Rating.from_rating(config.initial_elo, rd=350.0)
        self.results: list[dict] = []

    def evaluate(self, agent_play_fn) -> EvalResult:
        """Run evaluation games until confidence is sufficient.

        Args:
            agent_play_fn: Callable(game_state_msg: str) -> Optional[str]
                Function that takes a game state message and returns
                the agent's move in DCN notation.

        Returns:
            EvalResult with estimated ELO and confidence.
        """
        logger.info(f"Starting evaluation. Target RD: {self.config.target_rd}, "
                     f"max games: {self.config.max_games}")

        wins = losses = draws = 0
        tokens_before = self.tracker.total_used

        for game_num in range(1, self.config.max_games + 1):
            if self.tracker.exhausted:
                logger.info("Budget exhausted during evaluation")
                break

            # Select opponent near current estimated ELO
            opponent_elo = self._select_opponent_elo()
            agent_plays_white = (game_num % 2 == 1)

            score = self._play_eval_game(
                agent_play_fn, opponent_elo, agent_plays_white
            )

            # Update Glicko-2 rating
            opp_rating = Glicko2Rating.from_rating(opponent_elo, rd=50.0)
            self.rating = glicko2_update(
                self.rating, [opp_rating], [score]
            )

            # Track results
            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1

            self.results.append({
                "game": game_num,
                "opponent_elo": opponent_elo,
                "score": score,
                "agent_white": agent_plays_white,
                "estimated_elo": self.rating.rating,
                "rd": self.rating.rd,
            })

            logger.info(f"Eval game {game_num}: vs ELO {opponent_elo}, "
                         f"score={score}, est={self.rating.rating:.0f} "
                         f"(+/-{self.rating.rd:.0f})")

            # Check if we can stop
            if game_num >= self.config.min_games and self.rating.rd < self.config.target_rd:
                logger.info(f"Target RD reached: {self.rating.rd:.1f} < {self.config.target_rd}")
                break

        tokens_used = self.tracker.total_used - tokens_before

        return EvalResult(
            estimated_elo=self.rating.rating,
            rd=self.rating.rd,
            games_played=len(self.results),
            wins=wins,
            losses=losses,
            draws=draws,
            tokens_used=tokens_used,
            game_results=self.results,
        )

    def _select_opponent_elo(self) -> int:
        """Pick opponent ELO that maximizes information gain.

        Play near the agent's current estimated ELO with some spread.
        """
        current = self.rating.rating
        # Add some noise to avoid always playing the same opponent
        spread = max(50, self.rating.rd * 0.5)
        target = current + random.gauss(0, spread)
        # Clamp to pool range
        target = max(self.pool.min_elo, min(self.pool.max_elo, target))
        return round(target)

    def _play_eval_game(self, agent_play_fn, opponent_elo: int,
                        agent_plays_white: bool) -> float:
        """Play one evaluation game.

        Args:
            agent_play_fn: Function to get agent's move from game state message.
            opponent_elo: ELO of the opponent.
            agent_plays_white: Whether agent plays White.

        Returns:
            Score from agent's perspective (1.0/0.5/0.0).
        """
        state = GameState()
        opponent = self.pool.get_opponent(opponent_elo)
        agent_color = Player.WHITE if agent_plays_white else Player.BLACK
        move_history: list[str] = []

        # If agent is Black, opponent plays first
        if not agent_plays_white:
            opp_move = opponent.get_move(state)
            dcn = move_to_dcn(state, opp_move)
            apply_move(state, opp_move)
            move_history.append(dcn)

        while not state.done:
            legal = generate_legal_moves(state)
            if not legal:
                break

            if state.current_player == agent_color:
                # Agent's turn -- build state message and get move
                msg = _build_eval_state_msg(state, move_history, legal,
                                            agent_plays_white, opponent_elo)
                move_dcn = agent_play_fn(msg)

                if move_dcn is None:
                    # Budget exhausted or error -- forfeit with random move
                    import random as rng
                    move = rng.choice(legal)
                    move_dcn = move_to_dcn(state, move)
                else:
                    # Validate and apply
                    legal_map = {move_to_dcn(state, m): m for m in legal}
                    if move_dcn in legal_map:
                        move = legal_map[move_dcn]
                    else:
                        # Try case-insensitive
                        matched = None
                        for d, m in legal_map.items():
                            if d.lower() == move_dcn.lower():
                                matched = m
                                break
                        if matched:
                            move = matched
                        else:
                            import random as rng
                            move = rng.choice(legal)
                            move_dcn = move_to_dcn(state, move)

                apply_move(state, move)
                move_history.append(move_dcn)
            else:
                # Opponent's turn
                move = opponent.get_move(state)
                dcn = move_to_dcn(state, move)
                apply_move(state, move)
                move_history.append(dcn)

        # Score from agent's perspective
        if state.winner is None:
            return 0.5
        elif state.winner == agent_color:
            return 1.0
        else:
            return 0.0


def _build_eval_state_msg(state: GameState, move_history: list[str],
                          legal_moves, agent_plays_white: bool,
                          opponent_elo: int) -> str:
    """Build a game state message for the evaluation game."""
    from davechess.game.state import PIECE_CHARS
    from davechess.benchmark.game_manager import _board_to_tuples

    parts = [f"**Evaluation game** (vs ~ELO {opponent_elo})"]
    parts.append("")

    # Move history
    if move_history:
        parts.append("Game so far:")
        for i in range(0, len(move_history), 2):
            num = i // 2 + 1
            white = move_history[i]
            if i + 1 < len(move_history):
                parts.append(f"{num}. {white} {move_history[i + 1]}")
            else:
                parts.append(f"{num}. {white}")
        parts.append("")

    # Board
    board_tuples = _board_to_tuples(state)
    parts.append(render_board(board_tuples,
                              resource_counts=state.resources,
                              turn=state.turn,
                              current_player=int(state.current_player)))
    parts.append("")

    # Legal moves
    color = "White" if agent_plays_white else "Black"
    legal_dcn = [move_to_dcn(state, m) for m in legal_moves]
    parts.append(f"You are {color}. Legal moves: {', '.join(legal_dcn[:30])}"
                 + (f"... ({len(legal_dcn)} total)" if len(legal_dcn) > 30 else ""))
    parts.append("Your move (use play_move tool or respond with DCN notation):")

    return "\n".join(parts)
