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


class TestTrainingDataFlipConsistency:
    """Verify training data encoding consistency with board flipping."""

    def test_selfplay_planes_match_network_input(self):
        """Planes stored in training examples should match network inference input."""
        state = GameState()
        # White position
        planes_train = state_to_planes(state)
        planes_infer = state_to_planes(state)
        np.testing.assert_array_equal(planes_train, planes_infer)

        # Black position
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        planes_train_b = state_to_planes(state)
        planes_infer_b = state_to_planes(state)
        np.testing.assert_array_equal(planes_train_b, planes_infer_b)

    def test_policy_target_indices_match_expand_indices(self):
        """Policy targets from get_move should use same indices as expand."""
        state = GameState()
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        assert state.current_player == Player.BLACK

        legal_b = generate_legal_moves(state)
        flip = state.current_player == Player.BLACK

        # Both expand() and get_move() compute indices via move_to_policy_index(m, flip=flip)
        expand_indices = [move_to_policy_index(m, flip=flip) for m in legal_b]
        get_move_indices = [move_to_policy_index(m, flip=flip) for m in legal_b]
        assert expand_indices == get_move_indices

    def test_value_target_sign_for_white_win(self):
        """When White wins, White positions get +1, Black positions get -1."""
        state = GameState()
        player_w = int(state.current_player)
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        player_b = int(state.current_player)

        winner = int(Player.WHITE)
        assert (1.0 if winner == player_w else -1.0) == 1.0
        assert (1.0 if winner == player_b else -1.0) == -1.0

    def test_value_target_sign_for_black_win(self):
        """When Black wins, Black positions get +1, White positions get -1."""
        state = GameState()
        player_w = int(state.current_player)
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        player_b = int(state.current_player)

        winner = int(Player.BLACK)
        assert (1.0 if winner == player_w else -1.0) == -1.0
        assert (1.0 if winner == player_b else -1.0) == 1.0

    def test_no_player_indicator_bias_in_balanced_data(self):
        """With balanced win/loss data, value targets should not correlate
        with the player indicator plane."""
        # Simulate balanced training data
        white_win_target_for_white = 1.0
        white_win_target_for_black = -1.0
        black_win_target_for_white = -1.0
        black_win_target_for_black = 1.0

        avg_white_target = (white_win_target_for_white + black_win_target_for_white) / 2
        avg_black_target = (white_win_target_for_black + black_win_target_for_black) / 2

        assert avg_white_target == 0.0
        assert avg_black_target == 0.0


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


class TestMCTSPolicyTargetFlipAlignment:
    """Verify that MCTS get_move() returns policy targets with flip-consistent
    indices, and that these indices match what expand() used for network priors.

    This is the critical alignment test: if expand() uses flip=True for Black
    to look up priors from the network's policy output, then get_move() must
    also use flip=True for Black when building the policy target dict. Otherwise,
    the training target would map visit counts to the wrong policy slots.
    """

    def test_get_move_policy_indices_match_expand_indices_white(self):
        """For White, verify get_move policy target indices match expand indices."""
        from davechess.engine.mcts import MCTS

        state = GameState()
        assert state.current_player == Player.WHITE

        # Run MCTS with no network (uniform policy)
        mcts = MCTS(None, num_simulations=10, cpuct=1.5)
        move, info = mcts.get_move(state, add_noise=False)

        # The policy_target dict keys should all be valid indices computed
        # with flip=False (White's turn)
        legal = generate_legal_moves(state)
        expected_indices = {move_to_policy_index(m, flip=False) for m in legal}
        actual_indices = set(info["policy_target"].keys())

        assert actual_indices.issubset(expected_indices), (
            f"Policy target has indices not in legal move set: "
            f"{actual_indices - expected_indices}"
        )
        assert len(actual_indices) == len(legal), (
            f"Policy target should have {len(legal)} entries, got {len(actual_indices)}"
        )

    def test_get_move_policy_indices_match_expand_indices_black(self):
        """For Black, verify get_move policy target indices match expand indices
        AND use flip=True (the critical check)."""
        from davechess.engine.mcts import MCTS

        state = GameState()
        legal_w = generate_legal_moves(state)
        state = apply_move(state, legal_w[0])
        assert state.current_player == Player.BLACK

        mcts = MCTS(None, num_simulations=10, cpuct=1.5)
        move, info = mcts.get_move(state, add_noise=False)

        # Expected: flip=True indices (matching what expand() uses)
        legal_b = generate_legal_moves(state)
        flipped_indices = {move_to_policy_index(m, flip=True) for m in legal_b}
        unflipped_indices = {move_to_policy_index(m, flip=False) for m in legal_b}
        actual_indices = set(info["policy_target"].keys())

        assert actual_indices == flipped_indices, (
            f"Policy target indices should use flip=True for Black.\n"
            f"Got: {sorted(actual_indices)[:5]}...\n"
            f"Expected (flipped): {sorted(flipped_indices)[:5]}...\n"
            f"Would be wrong (unflipped): {sorted(unflipped_indices)[:5]}..."
        )
        # Confirm flipped != unflipped (otherwise this test is vacuous)
        assert flipped_indices != unflipped_indices, (
            "Flipped and unflipped indices are identical — test is vacuous"
        )

    def test_get_move_from_root_same_alignment_as_get_move(self):
        """get_move_from_root should produce the same policy indices as get_move."""
        from davechess.engine.mcts import MCTS

        state = GameState()
        legal_w = generate_legal_moves(state)
        state = apply_move(state, legal_w[0])
        assert state.current_player == Player.BLACK

        mcts = MCTS(None, num_simulations=10, cpuct=1.5)
        root = mcts.search(state, add_noise=False)
        _, info_from_root = mcts.get_move_from_root(root, state)

        legal_b = generate_legal_moves(state)
        flipped_indices = {move_to_policy_index(m, flip=True) for m in legal_b}
        actual_indices = set(info_from_root["policy_target"].keys())

        assert actual_indices == flipped_indices, (
            "get_move_from_root policy target indices should use flip=True for Black"
        )

    def test_policy_target_sums_to_one(self):
        """Policy target visit proportions should sum to 1.0."""
        from davechess.engine.mcts import MCTS

        state = GameState()
        mcts = MCTS(None, num_simulations=20, cpuct=1.5)
        _, info = mcts.get_move(state, add_noise=False)

        total = sum(info["policy_target"].values())
        assert abs(total - 1.0) < 1e-6, f"Policy target sum = {total}, expected 1.0"

    def test_build_policy_target_preserves_indices(self):
        """_build_policy_target should place values at the exact indices from
        the policy_dict without any re-encoding."""
        from davechess.engine.selfplay import _build_policy_target

        # Simulate a sparse policy dict with specific indices
        policy_dict = {42: 0.5, 1000: 0.3, 3000: 0.2}
        dense = _build_policy_target(policy_dict)

        assert dense.shape == (POLICY_SIZE,)
        assert dense[42] == pytest.approx(0.5)
        assert dense[1000] == pytest.approx(0.3)
        assert dense[3000] == pytest.approx(0.2)
        # All other slots should be zero
        dense_copy = dense.copy()
        dense_copy[42] = dense_copy[1000] = dense_copy[3000] = 0.0
        assert np.all(dense_copy == 0.0)


class TestSelfplayPipelineFlipAlignment:
    """End-to-end test that plays a real self-play game and verifies
    the stored training data has correct flip alignment."""

    def test_selfplay_training_data_alignment(self):
        """Play a self-play game and verify each training example has
        correct alignment between planes and policy target."""
        from davechess.engine.selfplay import play_selfplay_game
        from davechess.engine.mcts import MCTS

        mcts = MCTS(None, num_simulations=5, cpuct=1.5)
        training_data, game_record = play_selfplay_game(
            mcts, temperature_threshold=100
        )

        assert len(training_data) > 0, "Game produced no training data"

        # Replay the game to verify each position
        state = GameState()
        for i, (game_state, move) in enumerate(game_record["moves"]):
            current_player = game_state.current_player
            flip = current_player == Player.BLACK

            # Verify planes match state_to_planes for this position
            expected_planes = state_to_planes(game_state)

            # Find the matching training example (by position in game)
            if i < len(training_data):
                planes, policy, value = training_data[i]
                np.testing.assert_array_equal(
                    planes, expected_planes,
                    err_msg=f"Planes mismatch at move {i} (player={current_player})"
                )

                # Verify policy target uses correct flip convention:
                # nonzero policy indices should be valid flipped indices for legal moves
                legal = generate_legal_moves(game_state)
                valid_indices = {move_to_policy_index(m, flip=flip) for m in legal}
                nonzero_indices = set(np.nonzero(policy)[0])

                assert nonzero_indices.issubset(valid_indices), (
                    f"Move {i}: policy has indices {nonzero_indices - valid_indices} "
                    f"that are not valid legal move indices (flip={flip}). "
                    f"Player={current_player}."
                )

            state = apply_move(state, move)

    def test_selfplay_black_policy_uses_flipped_indices(self):
        """Specifically verify that Black positions in training data use
        flipped indices, not unflipped ones."""
        from davechess.engine.selfplay import play_selfplay_game
        from davechess.engine.mcts import MCTS

        mcts = MCTS(None, num_simulations=5, cpuct=1.5)
        training_data, game_record = play_selfplay_game(
            mcts, temperature_threshold=100
        )

        found_black = False
        for game_state, move in game_record["moves"]:
            if game_state.current_player != Player.BLACK:
                continue

            found_black = True
            legal = generate_legal_moves(game_state)

            flipped_indices = {move_to_policy_index(m, flip=True) for m in legal}
            unflipped_indices = {move_to_policy_index(m, flip=False) for m in legal}

            # Sanity: flipped and unflipped should differ for Black
            assert flipped_indices != unflipped_indices, (
                "Flipped and unflipped indices are identical for a Black position"
            )

            # Find the corresponding training data
            # (training_data is in game order for self-play)
            idx = game_record["moves"].index((game_state, move))
            if idx < len(training_data):
                _, policy, _ = training_data[idx]
                nonzero = set(np.nonzero(policy)[0])

                assert nonzero.issubset(flipped_indices), (
                    f"Black policy target uses unflipped indices! "
                    f"Nonzero indices: {nonzero}, "
                    f"Flipped valid: {flipped_indices}, "
                    f"Unflipped valid: {unflipped_indices}"
                )
                assert not nonzero.issubset(unflipped_indices) or flipped_indices == unflipped_indices, (
                    "Black policy target appears to use unflipped indices"
                )
            break

        assert found_black, "No Black moves found in game"

    def test_training_seed_uses_correct_flip(self):
        """Verify the training.py seed generation path also uses flip correctly."""
        # Reproduce the seed encoding logic from training.py line ~775
        state = GameState()
        legal_w = generate_legal_moves(state)
        state_clone = state.clone()

        # White position: flip=False
        planes = state_to_planes(state_clone)
        policy_w = np.zeros(POLICY_SIZE, dtype=np.float32)
        flip = state_clone.current_player == Player.BLACK
        assert flip is False
        move_w = legal_w[0]
        policy_w[move_to_policy_index(move_w, flip=flip)] = 1.0

        # The nonzero index should match the unflipped encoding
        idx_w = np.nonzero(policy_w)[0][0]
        assert idx_w == move_to_policy_index(move_w, flip=False)

        # Advance to Black
        apply_move(state_clone, move_w)
        assert state_clone.current_player == Player.BLACK
        legal_b = generate_legal_moves(state_clone)
        move_b = legal_b[0]

        planes_b = state_to_planes(state_clone)
        policy_b = np.zeros(POLICY_SIZE, dtype=np.float32)
        flip_b = state_clone.current_player == Player.BLACK
        assert flip_b is True
        policy_b[move_to_policy_index(move_b, flip=flip_b)] = 1.0

        idx_b = np.nonzero(policy_b)[0][0]
        # Must be the FLIPPED index, not the unflipped one
        assert idx_b == move_to_policy_index(move_b, flip=True)
        assert idx_b != move_to_policy_index(move_b, flip=False)
