"""Tests for board flipping in state encoding and move encoding.

Board flipping ensures the CNN always sees the current player's pieces
"moving up" — standard AlphaZero technique that doubles effective
network capacity for spatial patterns.
"""

import numpy as np
import pytest

from davechess.game.state import GameState, Player, PieceType, Piece, MoveStep, Promote, BombardAttack
from davechess.game.board import BOARD_SIZE, GOLD_NODES
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.network import (
    state_to_planes, move_to_policy_index, policy_index_to_move,
    _flip_row, NUM_INPUT_PLANES, POLICY_SIZE,
)


class TestFlipRow:
    def test_flip_row_edges(self):
        assert _flip_row(0) == 7
        assert _flip_row(7) == 0

    def test_flip_row_middle(self):
        assert _flip_row(3) == 4
        assert _flip_row(4) == 3

    def test_flip_row_double_flip_identity(self):
        for r in range(8):
            assert _flip_row(_flip_row(r)) == r


class TestBoardFlipPlanes:
    def test_white_not_flipped(self):
        """White's pieces should appear at their actual board positions."""
        state = GameState()  # White to move
        planes = state_to_planes(state)
        # White Commander starts at row 0 — should be at row 0 in planes
        # Commander is PieceType 0, current player planes are 0-4
        assert planes[0, 0, 4] == 1.0  # White Commander at (0, 4)

    def test_black_is_flipped(self):
        """Black's board should be flipped so pieces appear at flipped rows."""
        state = GameState()
        # Make a move so it's Black's turn
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        assert state.current_player == Player.BLACK

        planes = state_to_planes(state)
        # Black Commander is at row 7. After flipping, it should appear at row 0
        # in the current player's planes (planes 0-4).
        # PieceType.COMMANDER = 0
        assert planes[0, 0, 4] == 1.0  # Black Commander at flipped row 0

    def test_gold_nodes_symmetric(self):
        """Gold nodes are at (3,3),(3,4),(4,3),(4,4) — symmetric under flip."""
        state = GameState()
        planes_white = state_to_planes(state)

        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        planes_black = state_to_planes(state)

        # Gold node plane (10) should have same pattern after flip
        # since gold nodes are at rows 3,4 which flip to 4,3
        gold_white = set(zip(*np.where(planes_white[10] > 0)))
        gold_black = set(zip(*np.where(planes_black[10] > 0)))
        assert gold_white == gold_black

    def test_player_indicator_preserved(self):
        """Player indicator plane should be 1 for White, 0 for Black."""
        state = GameState()
        planes_w = state_to_planes(state)
        assert planes_w[11, 0, 0] == 1.0  # White

        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        planes_b = state_to_planes(state)
        assert planes_b[11, 0, 0] == 0.0  # Black

    def test_last_move_flipped_for_black(self):
        """Last move source/dest planes should be flipped for Black."""
        state = GameState()
        # White makes a move
        move = MoveStep((1, 0), (2, 0))  # Warrior a2-a3
        legal = generate_legal_moves(state)
        # Find a legal warrior move
        for m in legal:
            if isinstance(m, MoveStep) and m.from_rc[0] == 1:
                move = m
                break
        apply_move(state, move)

        # Now it's Black's turn, last_move was White's move
        planes = state_to_planes(state)
        # The last move source was at row=from_rc[0], after flip = 7-from_rc[0]
        flipped_src_row = 7 - move.from_rc[0]
        flipped_dst_row = 7 - move.to_rc[0]
        assert planes[16, flipped_src_row, move.from_rc[1]] == 1.0
        assert planes[17, flipped_dst_row, move.to_rc[1]] == 1.0


class TestMoveEncodingFlip:
    def test_white_move_no_flip(self):
        """White moves should encode the same with flip=False."""
        move = MoveStep((1, 3), (2, 3))  # Warrior d2-d3
        idx_no_flip = move_to_policy_index(move, flip=False)
        idx_default = move_to_policy_index(move)
        assert idx_no_flip == idx_default

    def test_black_move_different_with_flip(self):
        """Black moves should encode differently with flip=True."""
        move = MoveStep((6, 3), (5, 3))  # Black Warrior d7-d6
        idx_no_flip = move_to_policy_index(move, flip=False)
        idx_flip = move_to_policy_index(move, flip=True)
        assert idx_no_flip != idx_flip

    def test_symmetric_moves_same_index_after_flip(self):
        """A White move from row 1 and a Black move from row 6 (flipped to 1)
        should produce the same policy index if columns and distances match."""
        # White: Warrior at (1, 3) moves to (2, 3) — forward 1 square
        white_move = MoveStep((1, 3), (2, 3))
        white_idx = move_to_policy_index(white_move, flip=False)

        # Black: Warrior at (6, 3) moves to (5, 3) — forward 1 square from Black's perspective
        # After flip: (6,3) → (1,3), (5,3) → (2,3) — same as White!
        black_move = MoveStep((6, 3), (5, 3))
        black_idx = move_to_policy_index(black_move, flip=True)

        assert white_idx == black_idx

    def test_bombard_flip(self):
        """Bombard attacks should flip correctly."""
        # White bombard at (2, 4) attacks (4, 4) — direction (1, 0), dist 2
        white_attack = BombardAttack((2, 4), (4, 4))
        white_idx = move_to_policy_index(white_attack, flip=False)

        # Black bombard at (5, 4) attacks (3, 4) — same direction after flip
        # After flip: (5,4) → (2,4), (3,4) → (4,4)
        black_attack = BombardAttack((5, 4), (3, 4))
        black_idx = move_to_policy_index(black_attack, flip=True)

        assert white_idx == black_idx

    def test_promote_flip(self):
        """Promotion moves should flip the source square."""
        # White promotes at (3, 2) to Rider
        white_promote = Promote((3, 2), PieceType.RIDER)
        white_idx = move_to_policy_index(white_promote, flip=False)

        # Black promotes at (4, 2) to Rider — after flip (4,2) → (3,2)
        black_promote = Promote((4, 2), PieceType.RIDER)
        black_idx = move_to_policy_index(black_promote, flip=True)

        assert white_idx == black_idx

    def test_roundtrip_white(self):
        """move_to_policy_index → policy_index_to_move roundtrip for White."""
        state = GameState()
        legal = generate_legal_moves(state)
        for move in legal:
            idx = move_to_policy_index(move, flip=False)
            recovered = policy_index_to_move(idx, state)
            assert recovered is not None
            assert recovered.from_rc == move.from_rc
            if isinstance(move, MoveStep):
                assert recovered.to_rc == move.to_rc

    def test_roundtrip_black(self):
        """move_to_policy_index → policy_index_to_move roundtrip for Black."""
        state = GameState()
        legal_w = generate_legal_moves(state)
        apply_move(state, legal_w[0])
        assert state.current_player == Player.BLACK

        legal = generate_legal_moves(state)
        for move in legal:
            idx = move_to_policy_index(move, flip=True)
            recovered = policy_index_to_move(idx, state)
            assert recovered is not None, f"Failed to recover move {move} from index {idx}"
            assert recovered.from_rc == move.from_rc, (
                f"From mismatch: {recovered.from_rc} != {move.from_rc} for {move}"
            )
            if isinstance(move, MoveStep):
                assert recovered.to_rc == move.to_rc, (
                    f"To mismatch: {recovered.to_rc} != {move.to_rc} for {move}"
                )

    def test_all_legal_moves_unique_indices(self):
        """All legal moves should map to unique policy indices."""
        state = GameState()
        # Test White
        legal = generate_legal_moves(state)
        indices = [move_to_policy_index(m, flip=False) for m in legal]
        assert len(set(indices)) == len(indices), "White moves have duplicate indices"

        # Test Black
        apply_move(state, legal[0])
        legal_b = generate_legal_moves(state)
        indices_b = [move_to_policy_index(m, flip=True) for m in legal_b]
        assert len(set(indices_b)) == len(indices_b), "Black moves have duplicate indices"


class TestEndToEndFlip:
    def test_mcts_compatible_with_flip(self):
        """Verify MCTS expand works with flipped encoding for both colors."""
        try:
            from davechess.engine.mcts import MCTSNode, MCTS
        except ImportError:
            pytest.skip("torch not available")

        state = GameState()
        # Create a simple policy (uniform)
        policy = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE

        # White root
        root_w = MCTSNode(state=state)
        root_w.expand(policy)
        assert len(root_w.children) > 0

        # Make a move, create Black root
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        root_b = MCTSNode(state=state)
        root_b.expand(policy)
        assert len(root_b.children) > 0

    def test_planes_policy_consistency(self):
        """Verify that the network's policy output indices match the
        flipped move encoding — i.e., the policy slot for a move
        corresponds to the correct spatial position in the flipped planes."""
        state = GameState()
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])  # Now Black's turn

        planes = state_to_planes(state)
        legal_b = generate_legal_moves(state)

        # For each legal Black move, the source square in the flipped
        # encoding should have a piece in the current-player planes
        for move in legal_b:
            if isinstance(move, MoveStep):
                # Get the flipped source
                idx = move_to_policy_index(move, flip=True)
                sq_idx = idx // 67
                row = sq_idx // 8
                col = sq_idx % 8
                # There should be a current-player piece at this position in the planes
                has_piece = any(planes[p, row, col] > 0 for p in range(5))
                assert has_piece, (
                    f"No current-player piece at flipped ({row},{col}) "
                    f"for move {move} (original row {move.from_rc[0]})"
                )
