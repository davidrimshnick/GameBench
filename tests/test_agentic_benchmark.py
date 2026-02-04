"""Tests for the agentic benchmark infrastructure."""

import json
import os
import tempfile

import pytest

from davechess.game.state import GameState, Player, PieceType, Piece
from davechess.game.rules import generate_legal_moves
from davechess.game.notation import move_to_dcn
from davechess.benchmark.token_tracker import TokenTracker, TokenUsage
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.opponent_pool import OpponentPool, CalibratedLevel
from davechess.benchmark.game_manager import GameManager
from davechess.benchmark.tools import ToolExecutor, ToolCall, TOOL_DEFINITIONS, tools_to_anthropic_format
from davechess.data.generator import RandomAgent


# ── TokenTracker ──────────────────────────────────────────────

class TestTokenTracker:
    def test_initial_state(self):
        tracker = TokenTracker(budget=1_000_000)
        assert tracker.total_used == 0
        assert tracker.remaining == 1_000_000
        assert not tracker.exhausted
        assert tracker.num_calls == 0

    def test_record_usage(self):
        tracker = TokenTracker(budget=10_000)
        tracker.record(prompt_tokens=500, completion_tokens=200)
        assert tracker.total_used == 700
        assert tracker.remaining == 9_300
        assert tracker.num_calls == 1

    def test_multiple_records(self):
        tracker = TokenTracker(budget=10_000)
        tracker.record(1000, 500)
        tracker.record(2000, 300)
        assert tracker.total_used == 3_800
        assert tracker.usage.prompt_tokens == 3_000
        assert tracker.usage.completion_tokens == 800
        assert tracker.num_calls == 2

    def test_budget_exhaustion(self):
        tracker = TokenTracker(budget=1_000)
        tracker.record(600, 300)
        assert not tracker.exhausted
        tracker.record(50, 50)
        assert tracker.exhausted
        assert tracker.remaining == 0

    def test_remaining_never_negative(self):
        tracker = TokenTracker(budget=100)
        tracker.record(200, 100)
        assert tracker.remaining == 0
        assert tracker.exhausted

    def test_has_budget_for(self):
        tracker = TokenTracker(budget=10_000)
        tracker.record(5_000, 3_000)
        assert tracker.has_budget_for(2_000)
        assert not tracker.has_budget_for(3_000)

    def test_summary(self):
        tracker = TokenTracker(budget=50_000)
        tracker.record(1000, 200)
        s = tracker.summary()
        assert s["budget"] == 50_000
        assert s["total_used"] == 1_200
        assert s["remaining"] == 48_800
        assert s["num_calls"] == 1

    def test_budget_message_no_usage(self):
        tracker = TokenTracker(budget=500_000)
        msg = tracker.budget_message(phase="learning")
        assert "500,000" in msg
        assert "--tokens" in msg

    def test_budget_message_learning_early(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(10_000, 5_000)  # 15% used, 85% remaining
        msg = tracker.budget_message(phase="learning")
        assert "85,000" in msg
        assert "Keep studying" in msg

    def test_budget_message_learning_mid(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(30_000, 20_000)  # 50% used, 50% remaining
        msg = tracker.budget_message(phase="learning")
        assert "50,000" in msg
        assert "Good progress" in msg

    def test_budget_message_learning_low(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(50_000, 25_000)  # 75% used, 25% remaining
        msg = tracker.budget_message(phase="learning")
        assert "25,000" in msg
        assert "evaluate" in msg.lower()

    def test_budget_message_nearly_exhausted(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(50_000, 47_000)  # 97% used, 3% remaining
        msg = tracker.budget_message(phase="learning")
        assert "nearly exhausted" in msg.lower()

    def test_budget_message_exhausted(self):
        tracker = TokenTracker(budget=1_000)
        tracker.record(600, 400)
        msg = tracker.budget_message(phase="evaluation")
        assert "exhausted" in msg.lower()

    def test_budget_message_evaluation(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(40_000, 10_000)
        msg = tracker.budget_message(phase="evaluation")
        assert "rated evaluation" in msg.lower()

    def test_budget_message_baseline(self):
        tracker = TokenTracker(budget=100_000)
        tracker.record(5_000, 1_000)
        msg = tracker.budget_message(phase="baseline")
        assert "starting ELO" in msg


# ── GameLibrary ───────────────────────────────────────────────

class TestGameLibrary:
    def test_load_empty_dir(self, tmp_path):
        lib = GameLibrary(str(tmp_path))
        count = lib.load()
        assert count == 0
        assert lib.total_games == 0

    def test_load_nonexistent_dir(self):
        lib = GameLibrary("/nonexistent/path")
        count = lib.load()
        assert count == 0

    def test_get_games_no_duplicates(self, tmp_path):
        # Create some fake DCN files
        for i in range(5):
            path = tmp_path / f"game_{i:03d}.dcn"
            path.write_text(
                f'[White "Player1"]\n[Black "Player2"]\n[Result "1-0"]\n\n'
                f'1. Wa2-a3 Wa7-a6\n1-0\n'
            )

        lib = GameLibrary(str(tmp_path), max_games=5)
        lib.load()
        assert lib.total_games == 5

        games1 = lib.get_games(3)
        assert len(games1) == 3
        assert lib.remaining == 2

        games2 = lib.get_games(2)
        assert len(games2) == 2
        assert lib.remaining == 0

    def test_get_games_exhaustion(self, tmp_path):
        path = tmp_path / "game_001.dcn"
        path.write_text(
            '[White "P1"]\n[Black "P2"]\n[Result "1-0"]\n\n'
            '1. Wa2-a3 Wa7-a6\n1-0\n'
        )

        lib = GameLibrary(str(tmp_path))
        lib.load()
        lib.get_games(1)

        with pytest.raises(ValueError, match="only 0 remain"):
            lib.get_games(1)

    def test_reset(self, tmp_path):
        for i in range(3):
            path = tmp_path / f"game_{i:03d}.dcn"
            path.write_text(
                f'[White "P1"]\n[Black "P2"]\n[Result "1-0"]\n\n'
                f'1. Wa2-a3 Wa7-a6\n1-0\n'
            )

        lib = GameLibrary(str(tmp_path))
        lib.load()
        lib.get_games(3)
        assert lib.remaining == 0

        lib.reset()
        assert lib.remaining == 3


# ── OpponentPool ──────────────────────────────────────────────

class TestOpponentPool:
    def _make_pool(self):
        """Create a pool with mock calibration data (no network)."""
        calibration = [
            CalibratedLevel(sim_count=0, measured_elo=400),
            CalibratedLevel(sim_count=1, measured_elo=600),
            CalibratedLevel(sim_count=10, measured_elo=900),
            CalibratedLevel(sim_count=50, measured_elo=1200),
            CalibratedLevel(sim_count=200, measured_elo=1600),
            CalibratedLevel(sim_count=800, measured_elo=2100),
        ]
        return OpponentPool(network=None, device="cpu", calibration=calibration)

    def test_min_max_elo(self):
        pool = self._make_pool()
        assert pool.min_elo == 400
        assert pool.max_elo == 2100

    def test_random_at_low_elo(self):
        pool = self._make_pool()
        agent = pool.get_opponent(400)
        assert isinstance(agent, RandomAgent)

    def test_agent_at_mid_elo(self):
        pool = self._make_pool()
        agent = pool.get_opponent(1000)
        # Should return an MCTSLiteAgent (no network) with interpolated sims
        assert agent is not None

    def test_clamped_above_max(self):
        pool = self._make_pool()
        agent = pool.get_opponent(3000)  # Above max
        assert agent is not None

    def test_clamped_below_min(self):
        pool = self._make_pool()
        agent = pool.get_opponent(100)  # Below min
        assert isinstance(agent, RandomAgent)

    def test_save_and_load_calibration(self, tmp_path):
        pool = self._make_pool()
        path = str(tmp_path / "calibration.json")
        pool.save_calibration(path)

        loaded = OpponentPool.from_calibration_file(path, network=None, device="cpu")
        assert len(loaded.calibration) == len(pool.calibration)
        assert loaded.min_elo == pool.min_elo
        assert loaded.max_elo == pool.max_elo


# ── GameManager ───────────────────────────────────────────────

class TestGameManager:
    def _make_manager(self):
        calibration = [
            CalibratedLevel(sim_count=0, measured_elo=400),
            CalibratedLevel(sim_count=1, measured_elo=800),
        ]
        pool = OpponentPool(network=None, device="cpu", calibration=calibration)
        return GameManager(pool, max_concurrent=3)

    def test_start_game(self):
        mgr = self._make_manager()
        result = mgr.start_game(400)
        assert "game_id" in result
        assert result["game_id"] == "game_001"
        assert "board" in result
        assert "legal_moves" in result
        assert not result["finished"]

    def test_play_legal_move(self):
        mgr = self._make_manager()
        start = mgr.start_game(400)
        game_id = start["game_id"]

        # Get a legal move
        legal = start["legal_moves"]
        assert len(legal) > 0
        move = legal[0]

        result = mgr.play_move(game_id, move)
        assert "error" not in result
        assert "your_move" in result
        # Opponent should have responded (unless game ended)
        if not result.get("game_over"):
            assert "opponent_move" in result

    def test_play_illegal_move(self):
        mgr = self._make_manager()
        start = mgr.start_game(400)
        game_id = start["game_id"]

        result = mgr.play_move(game_id, "Xa9-z9")
        assert "error" in result
        assert "not a legal move" in result["error"]

    def test_nonexistent_game(self):
        mgr = self._make_manager()
        result = mgr.play_move("game_999", "Wa2-a3")
        assert "error" in result

    def test_max_concurrent(self):
        mgr = self._make_manager()
        mgr.start_game(400)
        mgr.start_game(400)
        mgr.start_game(400)
        result = mgr.start_game(400)  # Should fail
        assert "error" in result
        assert "Maximum" in result["error"]

    def test_get_state(self):
        mgr = self._make_manager()
        start = mgr.start_game(400)
        game_id = start["game_id"]

        state = mgr.get_state(game_id)
        assert state["game_id"] == game_id
        assert "board" in state
        assert "turn" in state

    def test_game_completion(self):
        """Play a full game against random to verify completion tracking."""
        mgr = self._make_manager()
        start = mgr.start_game(400)
        game_id = start["game_id"]

        # Play moves until game ends (random vs agent picking first legal move)
        # Turn limit is 100 = 200 half-moves. Each play_move call does 2 half-moves.
        for _ in range(110):
            state = mgr.get_state(game_id)
            if state["finished"]:
                break
            if "legal_moves" not in state or not state["legal_moves"]:
                break
            mgr.play_move(game_id, state["legal_moves"][0])

        # Game should have ended (checkmate or turn 100 draw)
        final = mgr.get_state(game_id)
        assert final["finished"]
        finished = mgr.get_finished_games()
        assert len(finished) == 1
        assert finished[0].result in ("win", "loss", "draw")


# ── ToolExecutor ──────────────────────────────────────────────

class TestToolExecutor:
    def _make_executor(self, tmp_path):
        # Create a game library with a fake game
        lib_dir = tmp_path / "games"
        lib_dir.mkdir()
        (lib_dir / "game_001.dcn").write_text(
            '[White "P1"]\n[Black "P2"]\n[Result "1-0"]\n\n'
            '1. Wa2-a3 Wa7-a6\n1-0\n'
        )
        library = GameLibrary(str(lib_dir))
        library.load()

        calibration = [
            CalibratedLevel(sim_count=0, measured_elo=400),
            CalibratedLevel(sim_count=1, measured_elo=800),
        ]
        pool = OpponentPool(network=None, device="cpu", calibration=calibration)
        manager = GameManager(pool)

        return ToolExecutor(manager, library)

    def test_study_games(self, tmp_path):
        executor = self._make_executor(tmp_path)
        result = json.loads(executor.execute(
            ToolCall(id="1", name="study_games", arguments={"n": 1})
        ))
        assert "games" in result
        assert len(result["games"]) == 1
        assert executor.stats["games_studied"] == 1

    def test_start_practice_game(self, tmp_path):
        executor = self._make_executor(tmp_path)
        result = json.loads(executor.execute(
            ToolCall(id="1", name="start_practice_game",
                     arguments={"opponent_elo": 400})
        ))
        assert "game_id" in result
        assert executor.stats["practice_games_started"] == 1

    def test_play_move_via_tool(self, tmp_path):
        executor = self._make_executor(tmp_path)

        # Start a game
        start = json.loads(executor.execute(
            ToolCall(id="1", name="start_practice_game",
                     arguments={"opponent_elo": 400})
        ))
        game_id = start["game_id"]
        legal = start["legal_moves"]

        # Play a move
        result = json.loads(executor.execute(
            ToolCall(id="2", name="play_move",
                     arguments={"game_id": game_id, "move_dcn": legal[0]})
        ))
        assert "error" not in result
        assert executor.stats["moves_played"] == 1

    def test_get_game_state_via_tool(self, tmp_path):
        executor = self._make_executor(tmp_path)

        start = json.loads(executor.execute(
            ToolCall(id="1", name="start_practice_game",
                     arguments={"opponent_elo": 400})
        ))
        game_id = start["game_id"]

        result = json.loads(executor.execute(
            ToolCall(id="2", name="get_game_state",
                     arguments={"game_id": game_id})
        ))
        assert result["game_id"] == game_id
        assert executor.stats["state_queries"] == 1

    def test_unknown_tool(self, tmp_path):
        executor = self._make_executor(tmp_path)
        result = json.loads(executor.execute(
            ToolCall(id="1", name="nonexistent_tool", arguments={})
        ))
        assert "error" in result


# ── Tool Definitions ──────────────────────────────────────────

class TestToolDefinitions:
    def test_openai_format(self):
        assert len(TOOL_DEFINITIONS) == 4
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert names == {"study_games", "start_practice_game",
                         "play_move", "get_game_state"}

    def test_anthropic_format(self):
        tools = tools_to_anthropic_format()
        assert len(tools) == 4
        names = {t["name"] for t in tools}
        assert names == {"study_games", "start_practice_game",
                         "play_move", "get_game_state"}
        # Anthropic format should have input_schema
        for tool in tools:
            assert "input_schema" in tool
            assert "description" in tool
