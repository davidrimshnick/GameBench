"""Tests for neural network forward pass shape correctness."""

import pytest
import numpy as np

from davechess.game.state import GameState, Piece, Player, MoveStep, Promote, BombardAttack, PieceType
from davechess.engine.network import (
    state_to_planes, move_to_policy_index, policy_index_to_move,
    POLICY_SIZE, BOARD_SIZE, NUM_INPUT_PLANES,
)


class TestStateToPlanes:
    def test_output_shape(self):
        state = GameState()
        planes = state_to_planes(state)
        assert planes.shape == (NUM_INPUT_PLANES, 8, 8)

    def test_dtype(self):
        state = GameState()
        planes = state_to_planes(state)
        assert planes.dtype == np.float32

    def test_gold_nodes_plane(self):
        """Plane 10 should mark gold node positions."""
        state = GameState()
        planes = state_to_planes(state)
        from davechess.game.board import GOLD_NODES
        for r, c in GOLD_NODES:
            assert planes[10, r, c] == 1.0

    def test_current_player_plane(self):
        """Plane 11 should indicate current player."""
        state = GameState()
        planes = state_to_planes(state)
        # White to move: all 1s
        assert planes[11, 0, 0] == 1.0

    def test_piece_planes(self):
        """Current player's pieces should be on planes 0-4, opponent on 5-9."""
        state = GameState()
        planes = state_to_planes(state)
        # White Commander at (0, 4) - should be on plane 0 (Commander)
        assert planes[0, 0, 4] == 1.0
        # White Bombards at (0, 2) and (0, 6)
        assert planes[3, 0, 2] == 1.0
        assert planes[3, 0, 6] == 1.0
        # Black Commander at (7, 4) - opponent, plane 5
        assert planes[5, 7, 4] == 1.0


class TestMoveEncoding:
    def test_simple_move_encoding(self):
        """A simple move should produce a valid index."""
        move = MoveStep((3, 3), (4, 3))
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE

    def test_promote_encoding(self):
        move = Promote((0, 3), PieceType.RIDER)
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE

    def test_bombard_encoding(self):
        move = BombardAttack((3, 3), (5, 3))
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE

    def test_lancer_promote_encoding(self):
        move = Promote((0, 3), PieceType.LANCER)
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE

    def test_lancer_move_encoding(self):
        """Lancer diagonal move at distance 4."""
        move = MoveStep((0, 0), (4, 4))  # distance 4 diagonal
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE

    def test_lancer_move_roundtrip(self):
        """Lancer diagonal distance-4 move survives encode/decode."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[0][0] = Piece(PieceType.LANCER, Player.WHITE)

        move = MoveStep((0, 0), (4, 4))
        idx = move_to_policy_index(move)
        recovered = policy_index_to_move(idx, state)
        assert isinstance(recovered, MoveStep)
        assert recovered.from_rc == (0, 0)
        assert recovered.to_rc == (4, 4)

    def test_rider_long_range_roundtrip(self):
        """Rider orthogonal distance-7 move survives encode/decode."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[0][3] = Piece(PieceType.RIDER, Player.WHITE)

        move = MoveStep((0, 3), (7, 3))  # distance 7 orthogonal
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE
        recovered = policy_index_to_move(idx, state)
        assert isinstance(recovered, MoveStep)
        assert recovered.from_rc == (0, 3)
        assert recovered.to_rc == (7, 3)

    def test_lancer_orthogonal_roundtrip(self):
        """Lancer orthogonal distance-7 move survives encode/decode."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[0][0] = Piece(PieceType.LANCER, Player.WHITE)

        move = MoveStep((0, 0), (0, 7))  # distance 7 orthogonal
        idx = move_to_policy_index(move)
        assert 0 <= idx < POLICY_SIZE
        recovered = policy_index_to_move(idx, state)
        assert isinstance(recovered, MoveStep)
        assert recovered.from_rc == (0, 0)
        assert recovered.to_rc == (0, 7)

    def test_different_moves_different_indices(self):
        """Different moves should map to different indices."""
        m1 = MoveStep((3, 3), (4, 3))
        m2 = MoveStep((3, 3), (3, 4))
        m3 = Promote((0, 3), PieceType.RIDER)
        m4 = BombardAttack((3, 3), (5, 3))
        indices = {move_to_policy_index(m) for m in [m1, m2, m3, m4]}
        assert len(indices) == 4

    def test_roundtrip(self):
        """Encoding then decoding should recover the original move type."""
        state = GameState()

        # Simple move
        move = MoveStep((1, 2), (2, 2))
        idx = move_to_policy_index(move)
        recovered = policy_index_to_move(idx, state)
        assert isinstance(recovered, MoveStep)
        assert recovered.from_rc == move.from_rc
        assert recovered.to_rc == move.to_rc

        # Promotion
        move = Promote((0, 2), PieceType.RIDER)
        idx = move_to_policy_index(move)
        recovered = policy_index_to_move(idx, state)
        assert isinstance(recovered, Promote)
        assert recovered.to_type == move.to_type
        assert recovered.from_rc == move.from_rc


# Only run torch-dependent tests if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestNetworkForward:
    def test_forward_pass_shapes(self):
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        x = torch.randn(4, NUM_INPUT_PLANES, 8, 8)
        policy, value = net(x)
        assert policy.shape == (4, POLICY_SIZE)
        assert value.shape == (4, 1)

    def test_value_range(self):
        """Value head output should be in [-1, 1] (tanh)."""
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        x = torch.randn(8, NUM_INPUT_PLANES, 8, 8)
        _, value = net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_predict_method(self):
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        state = GameState()
        policy, value = net.predict(state)
        assert policy.shape == (POLICY_SIZE,)
        assert -1.0 <= value <= 1.0
        # Policy should sum to ~1 (softmax)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_parameter_count(self):
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=5, num_filters=64)
        params = sum(p.numel() for p in net.parameters())
        # Should be in the 500K-1M range
        assert 100_000 < params < 5_000_000
