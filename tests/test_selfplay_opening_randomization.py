"""Tests for self-play opening randomization."""

import pytest

from davechess.game.state import GameState
from davechess.engine.probe_openings import build_probe_start_state
from davechess.engine.selfplay import (
    _ActiveGame, play_selfplay_game, run_selfplay_batch,
)
from davechess.engine.mcts import MCTS
from davechess.engine.network import DaveChessNetwork


def _make_network():
    return DaveChessNetwork(num_res_blocks=1, num_filters=32)


class TestActiveGameStartState:
    def test_default_start_state(self):
        """_ActiveGame with no start_state uses standard starting position."""
        net = _make_network()
        mcts = MCTS(net, num_simulations=2, device="cpu")
        g = _ActiveGame(
            game_idx=0, nn_engine=mcts, opponent_engine=None,
            nn_plays_white=True, temperature_threshold=30, game_type="selfplay",
        )
        default = GameState()
        assert g.state.current_player == default.current_player
        assert g.state.turn == default.turn

    def test_custom_start_state(self):
        """_ActiveGame with start_state uses provided position."""
        net = _make_network()
        mcts = MCTS(net, num_simulations=2, device="cpu")
        start = build_probe_start_state(opening_seed=42, opening_plies=4)
        g = _ActiveGame(
            game_idx=0, nn_engine=mcts, opponent_engine=None,
            nn_plays_white=True, temperature_threshold=30, game_type="selfplay",
            start_state=start,
        )
        # After 4 plies, turn should have advanced (turn increments every 2 plies)
        assert g.state.turn > 1


class TestPlaySelfplayGameStartState:
    def test_start_state_used(self):
        """play_selfplay_game with start_state begins from given position."""
        net = _make_network()
        mcts = MCTS(net, num_simulations=2, cpuct=4.0, device="cpu")
        start = build_probe_start_state(opening_seed=7, opening_plies=2)

        _, record = play_selfplay_game(
            mcts, temperature_threshold=5, start_state=start,
        )
        # Game must have played at least 1 move beyond the opening plies
        assert record["length"] >= 1

    def test_no_start_state_default(self):
        """play_selfplay_game without start_state begins from standard position."""
        net = _make_network()
        mcts = MCTS(net, num_simulations=2, cpuct=4.0, device="cpu")

        _, record = play_selfplay_game(mcts, temperature_threshold=5)
        assert record["length"] >= 1


class TestRunSelfplayBatchOpeningPlies:
    def test_opening_plies_zero_is_default(self):
        """selfplay_opening_plies=0 should produce games from standard start."""
        net = _make_network()
        examples, stats = run_selfplay_batch(
            net, num_games=2, num_simulations=2, cpuct=4.0,
            device="cpu", selfplay_opening_plies=0,
        )
        assert len(examples) > 0
        assert stats["white_wins"] + stats["black_wins"] + stats["draws"] == 2

    def test_opening_plies_positive(self):
        """selfplay_opening_plies=4 should produce games with diverse starts."""
        net = _make_network()
        examples, stats = run_selfplay_batch(
            net, num_games=4, num_simulations=2, cpuct=4.0,
            device="cpu", selfplay_opening_plies=4,
        )
        assert len(examples) > 0
        assert stats["white_wins"] + stats["black_wins"] + stats["draws"] == 4

    def test_opening_plies_with_random_opponent(self):
        """Opening randomization works with vs-random games too."""
        net = _make_network()
        examples, stats = run_selfplay_batch(
            net, num_games=4, num_simulations=2, cpuct=4.0,
            random_opponent_fraction=0.5,
            device="cpu", selfplay_opening_plies=4,
        )
        assert len(examples) > 0
        assert stats["white_wins"] + stats["black_wins"] + stats["draws"] == 4


class TestOpeningDiversity:
    def test_different_seeds_produce_different_states(self):
        """Different opening seeds should produce different board positions."""
        boards = []
        for seed in range(10):
            s = build_probe_start_state(opening_seed=seed, opening_plies=4)
            board_repr = []
            for r in range(8):
                for c in range(8):
                    p = s.board[r][c]
                    board_repr.append(str(p) if p else ".")
            boards.append("".join(board_repr))

        unique_boards = set(boards)
        assert len(unique_boards) > 1, "Opening randomization should produce diverse positions"

    def test_same_seed_same_state(self):
        """Same seed should produce identical positions (deterministic)."""
        s1 = build_probe_start_state(opening_seed=42, opening_plies=4)
        s2 = build_probe_start_state(opening_seed=42, opening_plies=4)
        assert s1.turn == s2.turn
        assert s1.current_player == s2.current_player
        for r in range(8):
            for c in range(8):
                p1 = s1.board[r][c]
                p2 = s2.board[r][c]
                if p1 is None:
                    assert p2 is None
                else:
                    assert p2 is not None
                    assert p1.piece_type == p2.piece_type
                    assert p1.player == p2.player
