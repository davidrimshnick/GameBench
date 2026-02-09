"""Play LLM against calibrated opponents and collect results."""

from __future__ import annotations

import logging
import re
from typing import Optional

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.game.notation import move_to_dcn, dcn_to_move
from davechess.benchmark.llm_interface import LLMClient
from davechess.benchmark.prompt import build_system_prompt, build_game_state_message
from davechess.data.generator import Agent

logger = logging.getLogger("davechess.evaluator")


class LLMAgent(Agent):
    """Agent that uses an LLM to select moves."""

    def __init__(self, llm_client: LLMClient, system_prompt: str,
                 max_retries: int = 3):
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.conversation: list[dict] = []
        self.move_history_dcn: list[str] = []
        self.illegal_moves: int = 0
        self.forfeits: int = 0

    def reset(self):
        """Reset state for a new game."""
        self.conversation = []
        self.move_history_dcn = []

    def get_move(self, state: GameState) -> Move:
        """Get a move from the LLM."""
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves")

        # Build the state message
        msg = build_game_state_message(state, self.move_history_dcn, legal_moves)
        self.conversation.append({"role": "user", "content": msg})

        # Build legal move set for validation
        legal_dcn = {move_to_dcn(state, m): m for m in legal_moves}

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.get_move_response(self.system_prompt,
                                                       self.conversation)
            except Exception as e:
                logger.error(f"LLM API error: {e}")
                if attempt < self.max_retries:
                    continue
                # Forfeit
                self.forfeits += 1
                import random
                return random.choice(legal_moves)

            # Parse the response - extract DCN move
            move_text = _extract_move(response)

            if move_text and move_text in legal_dcn:
                self.conversation.append({"role": "assistant", "content": move_text})
                self.move_history_dcn.append(move_text)
                return legal_dcn[move_text]

            # Try fuzzy matching
            matched = _fuzzy_match_move(response, legal_dcn)
            if matched:
                dcn, move = matched
                self.conversation.append({"role": "assistant", "content": dcn})
                self.move_history_dcn.append(dcn)
                return move

            # Illegal move
            self.illegal_moves += 1
            if attempt < self.max_retries:
                retry_msg = (f"'{move_text or response}' is not a legal move. "
                             f"Legal moves are: {', '.join(list(legal_dcn.keys())[:20])}. "
                             f"Please respond with only a valid DCN move.")
                self.conversation.append({"role": "assistant", "content": response})
                self.conversation.append({"role": "user", "content": retry_msg})

        # All retries exhausted - forfeit, play random
        self.forfeits += 1
        logger.warning("LLM failed to produce legal move after retries, playing random")
        import random
        move = random.choice(legal_moves)
        dcn = move_to_dcn(state, move)
        self.move_history_dcn.append(dcn)
        return move

    def record_opponent_move(self, state: GameState, move: Move):
        """Record an opponent's move for conversation context."""
        dcn = move_to_dcn(state, move)
        self.move_history_dcn.append(dcn)


def _extract_move(text: str) -> Optional[str]:
    """Extract a DCN move from LLM response text."""
    text = text.strip()

    # Try exact match for common patterns
    patterns = [
        r'`([CWRBL][a-h][1-8][-x][a-h][1-8])`',  # Move/capture in backticks
        r'`([CWRBL][a-h][1-8]>[RBL])`',            # Promotion in backticks
        r'`(B[a-h][1-8]~[a-h][1-8])`',             # Bombard in backticks
        r'([CWRBL][a-h][1-8][-x][a-h][1-8])',       # Move/capture bare
        r'([CWRBL][a-h][1-8]>[RBL])',                # Promotion bare
        r'(B[a-h][1-8]~[a-h][1-8])',                 # Bombard bare
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    return None


def _fuzzy_match_move(text: str, legal_dcn: dict[str, Move]) -> Optional[tuple[str, Move]]:
    """Try to fuzzy-match a response to a legal move."""
    text = text.strip().upper()

    for dcn, move in legal_dcn.items():
        if dcn.upper() in text:
            return dcn, move

    return None


def play_llm_vs_opponent(llm_agent: LLMAgent, opponent: Agent,
                         llm_plays_white: bool = True) -> dict:
    """Play a single game of LLM vs calibrated opponent.

    Returns:
        Dict with game result and statistics.
    """
    state = GameState()
    llm_agent.reset()
    moves_played = []

    while not state.done:
        legal = generate_legal_moves(state)
        if not legal:
            break

        is_llm_turn = (
            (state.current_player == Player.WHITE and llm_plays_white) or
            (state.current_player == Player.BLACK and not llm_plays_white)
        )

        if is_llm_turn:
            move = llm_agent.get_move(state)
        else:
            move = opponent.get_move(state)
            # Record opponent move for LLM context
            llm_agent.record_opponent_move(state, move)

        dcn = move_to_dcn(state, move)
        moves_played.append(dcn)
        apply_move(state, move)

    # Determine result from LLM's perspective
    if state.winner is not None:
        llm_won = (
            (state.winner == Player.WHITE and llm_plays_white) or
            (state.winner == Player.BLACK and not llm_plays_white)
        )
        score = 1.0 if llm_won else 0.0
    else:
        score = 0.5

    return {
        "score": score,
        "num_moves": len(moves_played),
        "turns": state.turn,
        "winner": int(state.winner) if state.winner is not None else None,
        "llm_played_white": llm_plays_white,
        "illegal_moves": llm_agent.illegal_moves,
        "forfeits": llm_agent.forfeits,
    }
