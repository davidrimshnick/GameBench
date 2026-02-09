"""FastAPI server for the benchmark API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from davechess.benchmark.api.models import (
    CreateSessionRequest,
    CreateSessionResponse,
    SessionStatus,
    StudyGamesRequest,
    StudyGamesResponse,
    StartGameRequest,
    StartGameResponse,
    PlayMoveRequest,
    PlayMoveResponse,
    GameStateResponse,
    ReportTokensRequest,
    ReportTokensResponse,
    RequestEvalResponse,
    EvalStatusResponse,
    SessionResultResponse,
    TokenInfo,
    RatingInfo,
    SessionPhase,
)
from davechess.benchmark.api.session import PhaseError

app = FastAPI(
    title="GameBench API",
    description="Benchmark API for evaluating AI agents on DaveChess",
    version="0.1.0",
)


def _manager(request: Request):
    """Get SessionManager from app state."""
    return request.app.state.session_manager


def _get_session(request: Request, session_id: str):
    """Look up session or raise 404."""
    session = _manager(request).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_session(body: CreateSessionRequest, request: Request):
    """Create a new benchmark session."""
    manager = _manager(request)
    session = manager.create_session(body.token_budget, body.agent_name)

    # First baseline game is auto-created
    evaluator = session._baseline_evaluator
    game = evaluator.current_game
    game_state = evaluator.get_game_state()

    return CreateSessionResponse(
        session_id=session.session_id,
        phase=session.phase,
        rules=session.get_rules(),
        token_budget=body.token_budget,
        game_id=game.game_id,
        game_state=game_state,
    )


@app.get("/sessions/{session_id}", response_model=SessionStatus)
def get_session_status(session_id: str, request: Request):
    """Get session status."""
    session = _get_session(request, session_id)
    status = session.get_status()

    return SessionStatus(
        session_id=status["session_id"],
        phase=SessionPhase(status["phase"]),
        agent_name=status["agent_name"],
        tokens=TokenInfo(**status["tokens"]),
        baseline_rating=RatingInfo(**status["baseline_rating"]) if status.get("baseline_rating") else None,
        final_rating=RatingInfo(**status.get("final_rating", {})) if status.get("final_rating") else None,
    )


@app.get("/sessions/{session_id}/rules")
def get_rules(session_id: str, request: Request):
    """Get full DaveChess rules text."""
    session = _get_session(request, session_id)
    return {"rules": session.get_rules()}


@app.post("/sessions/{session_id}/games/{game_id}/move", response_model=PlayMoveResponse)
def play_move(session_id: str, game_id: str, body: PlayMoveRequest, request: Request):
    """Play a move in a game."""
    session = _get_session(request, session_id)
    try:
        result = session.play_move(game_id, body.move_dcn)
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return PlayMoveResponse(game_state=result)


@app.get("/sessions/{session_id}/games/{game_id}", response_model=GameStateResponse)
def get_game_state(session_id: str, game_id: str, request: Request):
    """Get game state."""
    session = _get_session(request, session_id)
    try:
        result = session.get_game_state(game_id)
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return GameStateResponse(game_state=result)


@app.post("/sessions/{session_id}/study", response_model=StudyGamesResponse)
def study_games(session_id: str, body: StudyGamesRequest, request: Request):
    """Study grandmaster games."""
    session = _get_session(request, session_id)
    try:
        result = session.study_games(body.num_games)
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StudyGamesResponse(**result)


@app.post("/sessions/{session_id}/games", response_model=StartGameResponse)
def start_game(session_id: str, body: StartGameRequest, request: Request):
    """Start a practice game."""
    session = _get_session(request, session_id)
    try:
        result = session.start_practice_game(body.opponent_elo)
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return StartGameResponse(game_id=result["game_id"], game_state=result)


@app.post("/sessions/{session_id}/tokens", response_model=ReportTokensResponse)
def report_tokens(session_id: str, body: ReportTokensRequest, request: Request):
    """Report token usage."""
    session = _get_session(request, session_id)
    try:
        result = session.report_tokens(body.prompt_tokens, body.completion_tokens)
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return ReportTokensResponse(tokens=TokenInfo(**result))


@app.post("/sessions/{session_id}/evaluate", response_model=RequestEvalResponse)
def request_evaluation(session_id: str, request: Request):
    """Transition from LEARNING to EVALUATION phase."""
    session = _get_session(request, session_id)
    try:
        result = session.request_evaluation()
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return RequestEvalResponse(
        phase=SessionPhase(result["phase"]),
        game_id=result["game_id"],
        game_state=result["game_state"],
    )


@app.get("/sessions/{session_id}/eval/status", response_model=EvalStatusResponse)
def get_eval_status(session_id: str, request: Request):
    """Get evaluation progress."""
    session = _get_session(request, session_id)
    try:
        result = session.get_eval_status()
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    current_game = None
    if result.get("current_game"):
        from davechess.benchmark.api.models import EvalGameInfo
        cg = result["current_game"]
        current_game = EvalGameInfo(**cg)

    return EvalStatusResponse(
        phase=SessionPhase(result["phase"]),
        rating=RatingInfo(**result["rating"]),
        current_game=current_game,
        is_complete=result["is_complete"],
    )


@app.get("/sessions/{session_id}/result", response_model=SessionResultResponse)
def get_result(session_id: str, request: Request):
    """Get final session results."""
    session = _get_session(request, session_id)
    try:
        result = session.get_result()
    except PhaseError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return SessionResultResponse(
        session_id=result["session_id"],
        agent_name=result["agent_name"],
        baseline_elo=result["baseline_elo"],
        baseline_rd=result["baseline_rd"],
        final_elo=result["final_elo"],
        final_rd=result["final_rd"],
        elo_gain=result["elo_gain"],
        baseline_games=result["baseline_games"],
        eval_games=result["eval_games"],
        tokens=TokenInfo(**result["tokens"]),
        baseline_details=result["baseline_details"],
        eval_details=result["eval_details"],
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    """Clean up a session."""
    manager = _manager(request)
    if not manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"deleted": True, "session_id": session_id}
