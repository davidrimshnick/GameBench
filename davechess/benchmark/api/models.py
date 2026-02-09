"""Pydantic models for benchmark API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SessionPhase(str, Enum):
    """Current phase of a benchmark session."""
    BASELINE = "baseline"
    LEARNING = "learning"
    EVALUATION = "evaluation"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """Create a new benchmark session."""
    token_budget: int = Field(..., gt=0, description="Total token budget for the session")
    agent_name: str = Field(..., min_length=1, description="Name of the AI agent being evaluated")


class StudyGamesRequest(BaseModel):
    """Request to study grandmaster games."""
    num_games: int = Field(..., ge=1, le=20, description="Number of games to study (1-20)")


class StartGameRequest(BaseModel):
    """Start a practice game."""
    opponent_elo: int = Field(..., ge=0, le=3000, description="Target opponent ELO")


class PlayMoveRequest(BaseModel):
    """Play a move in DCN notation."""
    move_dcn: str = Field(..., min_length=1, description="Move in DaveChess notation")


class ReportTokensRequest(BaseModel):
    """Report token usage from an API call."""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class RatingInfo(BaseModel):
    """Glicko-2 rating information."""
    elo: float
    rd: float
    games_played: int
    wins: int = 0
    losses: int = 0
    draws: int = 0


class TokenInfo(BaseModel):
    """Token budget status."""
    budget: int
    total_used: int
    remaining: int
    prompt_tokens: int
    completion_tokens: int
    num_calls: int


class CreateSessionResponse(BaseModel):
    """Response from session creation."""
    session_id: str
    phase: SessionPhase
    rules: str
    token_budget: int
    # First baseline game info
    game_id: str
    game_state: dict


class SessionStatus(BaseModel):
    """Current session status."""
    session_id: str
    phase: SessionPhase
    agent_name: str
    tokens: TokenInfo
    baseline_rating: Optional[RatingInfo] = None
    final_rating: Optional[RatingInfo] = None


class StudyGamesResponse(BaseModel):
    """Response from studying games."""
    games: list[str]
    num_returned: int
    remaining_in_library: int


class StartGameResponse(BaseModel):
    """Response from starting a practice game."""
    game_id: str
    game_state: dict


class PlayMoveResponse(BaseModel):
    """Response from playing a move."""
    game_state: dict


class GameStateResponse(BaseModel):
    """Full game state."""
    game_state: dict


class ReportTokensResponse(BaseModel):
    """Response from reporting tokens."""
    tokens: TokenInfo


class EvalGameInfo(BaseModel):
    """Info about the current evaluation game."""
    game_id: str
    game_num: int
    opponent_elo: int
    agent_color: str
    game_state: dict


class EvalStatusResponse(BaseModel):
    """Evaluation progress."""
    phase: SessionPhase
    rating: RatingInfo
    current_game: Optional[EvalGameInfo] = None
    is_complete: bool


class RequestEvalResponse(BaseModel):
    """Response from requesting evaluation."""
    phase: SessionPhase
    game_id: str
    game_state: dict


class SessionResultResponse(BaseModel):
    """Final session results."""
    session_id: str
    agent_name: str
    baseline_elo: float
    baseline_rd: float
    final_elo: float
    final_rd: float
    elo_gain: float
    baseline_games: int
    eval_games: int
    tokens: TokenInfo
    baseline_details: list[dict]
    eval_details: list[dict]
