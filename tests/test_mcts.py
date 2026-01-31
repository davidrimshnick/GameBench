"""Tests for MCTS correctness."""

import pytest

from davechess.game.state import GameState, Player, Piece, PieceType, MoveStep
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.mcts_lite import MCTSLite, play_random_game


class TestMCTSLite:
    def test_returns_legal_move(self):
        """MCTS should always return a legal move."""
        state = GameState()
        mcts = MCTSLite(num_simulations=10)
        move = mcts.search(state)
        legal = generate_legal_moves(state)
        assert move in legal

    def test_finds_obvious_win(self):
        """MCTS should find an immediate winning capture."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        # White Rider can capture Black Commander
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[3][4] = Piece(PieceType.COMMANDER, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)

        mcts = MCTSLite(num_simulations=200)
        move = mcts.search(state)
        # Should capture the Commander
        assert isinstance(move, MoveStep)
        assert move.to_rc == (3, 4)
        assert move.is_capture

    def test_single_move(self):
        """With only one legal move, MCTS returns it immediately."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)
        # Commander in corner has limited moves
        # But should still return a valid one
        mcts = MCTSLite(num_simulations=10)
        move = mcts.search(state)
        legal = generate_legal_moves(state)
        assert move in legal

    def test_search_with_policy(self):
        state = GameState()
        mcts = MCTSLite(num_simulations=20)
        move, policy = mcts.search_with_policy(state)
        assert move in generate_legal_moves(state)
        assert isinstance(policy, dict)
        assert len(policy) > 0
        # Probabilities should sum to ~1
        assert abs(sum(policy.values()) - 1.0) < 0.01


class TestRandomGame:
    def test_random_game_terminates(self):
        """Random games should terminate within move limit."""
        state = play_random_game(max_moves=400)
        # Game should be done or we hit the move limit
        assert state.done or len(state.move_history) <= 400

    def test_random_game_multiple(self):
        """Run multiple random games to check no crashes."""
        for _ in range(10):
            state = play_random_game(max_moves=200)
            if state.done and state.winner is not None:
                assert state.winner in (Player.WHITE, Player.BLACK)


# Only run torch-dependent tests if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestMCTSWithNetwork:
    def test_mcts_with_network(self):
        from davechess.engine.mcts import MCTS
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        mcts = MCTS(net, num_simulations=10)

        state = GameState()
        move, info = mcts.get_move(state)
        legal = generate_legal_moves(state)
        assert move in legal
        assert "policy_target" in info
        assert "root_value" in info

    def test_mcts_finds_obvious_win(self):
        """With uniform priors (no network), MCTS should find a 1-move winning capture."""
        from davechess.engine.mcts import MCTS

        # Use None network -> uniform policy prior, so MCTS relies on
        # value propagation from terminal states
        mcts = MCTS(None, num_simulations=400, temperature=0.01)

        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[3][4] = Piece(PieceType.COMMANDER, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)

        move, _ = mcts.get_move(state, add_noise=False)
        # Should find the Commander capture
        assert isinstance(move, MoveStep)
        assert move.to_rc == (3, 4)
