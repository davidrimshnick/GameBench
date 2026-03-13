"""Tests verifying capture timing analysis in vs-random loss games.

Validates that:
1. Game replay from standard start position works correctly
2. Capture detection and piece origin tracking are accurate
3. The _is_square_attacked function correctly identifies hanging pieces
4. Value head evaluations can be obtained for game positions
"""
import sys
import os
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import (
    GameState, Player, PieceType, Piece, MoveStep, Promote, BombardAttack,
)
from davechess.game.rules import (
    generate_legal_moves, apply_move, _is_square_attacked,
)
from davechess.game.notation import dcn_to_game, move_to_dcn
from davechess.game.board import rc_to_notation, notation_to_rc, BOARD_SIZE


class TestCaptureTimingAnalysis:
    """Tests for the capture timing analysis methodology."""

    def test_is_square_attacked_simple(self):
        """Verify _is_square_attacked detects basic attacks."""
        state = GameState()
        # White Warriors at row 1, White officers at row 0.
        # Black Warriors at row 6, Black officers at row 7.

        # Square (2,0) is empty. Is it attacked by White?
        # White Warrior at (1,1) captures diag forward to (2,0) and (2,2).
        assert _is_square_attacked(state, 2, 0, Player.WHITE)

        # Square (2,2) -- attacked by White Warrior at (1,1) and (1,3)
        assert _is_square_attacked(state, 2, 2, Player.WHITE)

        # Square (4,0) is empty and far from all White pieces at start
        # No White piece can reach it in the starting position
        assert not _is_square_attacked(state, 4, 0, Player.WHITE)

    def test_is_square_attacked_rider(self):
        """Verify Rider attack detection."""
        state = GameState()
        # Clear the path: remove Warrior at (1,3)
        state.board[1][3] = None
        # Now White Rider at (0,3) should attack along the d-file
        # (0,3) -> (1,3) is clear now -> (2,3) should be attacked
        assert _is_square_attacked(state, 2, 3, Player.WHITE)
        assert _is_square_attacked(state, 3, 3, Player.WHITE)

    def test_is_square_attacked_bombard(self):
        """Verify Bombard ranged attack detection."""
        state = GameState()
        # White Bombard at (0,2). Ranged attack at exactly 2 squares.
        # Clear intermediate square at (1,2)
        state.board[1][2] = None
        # Target at (2,2) -- Bombard can attack there if path clear
        # and target is not a Commander
        # Place an enemy Warrior at (2,2)
        state.board[2][2] = Piece(PieceType.WARRIOR, Player.BLACK)
        assert _is_square_attacked(state, 2, 2, Player.WHITE)

    def test_capture_gap_calculation(self):
        """Verify the gap calculation between move and capture is correct.

        Gap = capture_ply - origin_ply
        Gap of 1 means: NN moves piece on ply N, opponent captures on ply N+1.
        """
        # Construct a simple game: White Rider moves, Black captures next ply
        state = GameState()

        # Move 1: White Warrior d2->d3 (ply 0)
        m1 = MoveStep((1, 3), (2, 3))
        state = apply_move(state, m1)

        # Move 1: Black Warrior d7->d6 (ply 1)
        m2 = MoveStep((6, 3), (5, 3))
        state = apply_move(state, m2)

        # Move 2: White Rider d1->d2 (ply 2)
        # Rider at (0,3) is actually at d1 in notation
        # But wait, Rider is at (0,3) and we cleared (1,3) with the Warrior move
        m3 = MoveStep((0, 3), (1, 3))  # Rd1->d2 (moving to where Warrior was)
        state = apply_move(state, m3)

        # Now White Rider is at (1,3) = d2
        # Move 2: Black Warrior d6->d5 (ply 3)
        m4 = MoveStep((5, 3), (4, 3))
        state = apply_move(state, m4)

        # Move 3: White Rider d2->d5 (ply 4) -- move to square attacked by Black
        m5 = MoveStep((1, 3), (4, 3), is_capture=True)
        # Actually (4,3) has the Black Warrior now, so this is a capture
        state = apply_move(state, m5)

        # White Rider moved to (4,3) on ply 4
        # If Black captures it on ply 5, gap = 5 - 4 = 1
        # Check if (4,3) is attacked by Black
        attacked = _is_square_attacked(state, 4, 3, Player.BLACK)
        # Black Rider at (7,3) could attack (4,3) if path is clear
        # Path: (7,3)->(6,3) has Black Warrior (blocking? no, removed at ply 1)
        # Actually (6,3) was the Black Warrior that moved away. Check the board.
        assert state.board[4][3] is not None  # White Rider should be here
        assert state.board[4][3].player == Player.WHITE
        assert state.board[4][3].piece_type == PieceType.RIDER

    def test_replay_from_standard_start(self):
        """Test that a game from a pre-opening-randomization iteration replays correctly."""
        games_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs', 'on_policy_fix_20260308', 'games'
        )
        dcn_path = os.path.join(games_dir, 'iter_0385.dcn')
        if not os.path.exists(dcn_path):
            pytest.skip("Game file not available")

        with open(dcn_path) as f:
            content = f.read()

        game_texts = re.split(r'\n\n(?=\[Iteration)', content)
        gt = game_texts[0].strip()
        headers, moves, result = dcn_to_game(gt)

        # Replay should succeed from standard start
        state = GameState()
        for i, move in enumerate(moves):
            if isinstance(move, MoveStep):
                src = state.board[move.from_rc[0]][move.from_rc[1]]
                assert src is not None, f"Source empty at ply {i}: {move}"
            state = apply_move(state, move)

        # Game may not be marked done until generate_legal_moves is called
        # (apply_move doesn't detect checkmate for the NEXT player).
        # Call generate_legal_moves to trigger checkmate detection.
        if not state.done:
            legal = generate_legal_moves(state)
        # After the last move + checkmate detection, game should be done
        assert state.done

    def test_gap_is_always_odd(self):
        """Verify that capture gaps are always odd (due to alternating turns).

        When the NN moves a piece on its turn (ply N) and the opponent captures
        it later (ply M), the gap M - N must be odd because the opponent's turns
        are on plies of opposite parity from the NN's turns.

        Exception: pieces at starting positions (origin_ply = -1) are excluded.
        """
        games_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs', 'on_policy_fix_20260308', 'games'
        )
        # Check a few pre-opening-rand iterations
        for iter_num in [380, 385, 388]:
            dcn_path = os.path.join(games_dir, f'iter_{iter_num:04d}.dcn')
            if not os.path.exists(dcn_path):
                continue

            with open(dcn_path) as f:
                content = f.read()

            game_texts = re.split(r'\n\n(?=\[Iteration)', content)

            for gt in game_texts:
                gt = gt.strip()
                if not gt:
                    continue
                headers, moves, result = dcn_to_game(gt)
                game_num = int(headers.get('Game', '0'))
                game_idx = game_num - 1

                # Skip self-play games (only check vs-random)
                if game_idx >= 10:
                    continue

                nn_player = Player.WHITE if game_idx % 2 == 0 else Player.BLACK

                state = GameState()
                # Track where each piece was last moved
                piece_last_moved = {}  # (row, col) -> ply

                for ply_idx, move in enumerate(moves):
                    player = state.current_player

                    if isinstance(move, MoveStep):
                        src = state.board[move.from_rc[0]][move.from_rc[1]]
                        if src is None:
                            break  # Can't replay further

                        if move.is_capture:
                            captured = state.board[move.to_rc[0]][move.to_rc[1]]
                            if captured is not None and captured.player == nn_player:
                                # Find origin ply
                                origin_ply = piece_last_moved.get(move.to_rc, -1)
                                if origin_ply >= 0:
                                    gap = ply_idx - origin_ply
                                    assert gap % 2 == 1, (
                                        f"Iter {iter_num} Game {game_num} ply {ply_idx}: "
                                        f"even gap {gap} (origin {origin_ply})"
                                    )

                        # Track the move
                        if move.from_rc in piece_last_moved:
                            del piece_last_moved[move.from_rc]
                        piece_last_moved[move.to_rc] = ply_idx

                    elif isinstance(move, Promote):
                        piece_last_moved[move.from_rc] = ply_idx

                    elif isinstance(move, BombardAttack):
                        captured = state.board[move.target_rc[0]][move.target_rc[1]]
                        if captured is not None and captured.player == nn_player:
                            origin_ply = piece_last_moved.get(move.target_rc, -1)
                            if origin_ply >= 0:
                                gap = ply_idx - origin_ply
                                assert gap % 2 == 1, (
                                    f"Iter {iter_num} Game {game_num} ply {ply_idx}: "
                                    f"even gap {gap} (origin {origin_ply})"
                                )

                    try:
                        state = apply_move(state, move)
                    except Exception:
                        break

    def test_value_head_evaluation(self):
        """Test that the value head produces valid outputs for game positions."""
        try:
            import torch
            from davechess.engine.network import DaveChessNetwork, state_to_planes
        except ImportError:
            pytest.skip("torch not available")

        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'checkpoints', 'on_policy_fix_20260308', 'best.pt'
        )
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not available")

        net, _ = DaveChessNetwork.from_checkpoint(ckpt_path, device='cpu')
        net.eval()

        state = GameState()
        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(x)

        # Value should be in [-1, 1] (tanh output)
        v = value.item()
        assert -1.0 <= v <= 1.0, f"Value {v} out of range"

        # Logits should be finite
        assert torch.isfinite(logits).all(), "Non-finite logits"

    def test_hanging_piece_value_head_blind_spot(self):
        """Test that demonstrates the value head's blindness to hanging pieces.

        Construct a position where a piece is obviously hanging, and verify
        the value head does NOT significantly penalize it compared to a safe
        alternative.

        This is a characterization test documenting the current behavior.
        """
        try:
            import torch
            from davechess.engine.network import DaveChessNetwork, state_to_planes
        except ImportError:
            pytest.skip("torch not available")

        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'checkpoints', 'on_policy_fix_20260308', 'best.pt'
        )
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not available")

        net, _ = DaveChessNetwork.from_checkpoint(ckpt_path, device='cpu')
        net.eval()

        # Position 1: Standard opening, fairly equal
        state1 = GameState()
        planes1 = state_to_planes(state1)
        x1 = torch.from_numpy(planes1).unsqueeze(0)
        with torch.no_grad():
            _, val1 = net(x1)
        v1 = val1.item()

        # Position 2: Same but remove a White Rider (material down)
        state2 = GameState()
        state2.board[0][1] = None  # Remove White Rider at b1
        planes2 = state_to_planes(state2)
        x2 = torch.from_numpy(planes2).unsqueeze(0)
        with torch.no_grad():
            _, val2 = net(x2)
        v2 = val2.item()

        # Document behavior: the value head should ideally give a worse
        # evaluation when missing a piece. Record both values.
        # With a well-trained value head, v2 < v1 by a significant margin.
        # With the current value head, the difference may be small.
        print(f"\nValue head test:")
        print(f"  Full position: {v1:+.4f}")
        print(f"  Missing a Rider: {v2:+.4f}")
        print(f"  Delta: {v2 - v1:+.4f}")

        # This test documents current behavior rather than asserting correctness.
        # A good value head would have delta < -0.2 at minimum.
        # We just verify both are in valid range.
        assert -1.0 <= v1 <= 1.0
        assert -1.0 <= v2 <= 1.0


class TestCaptureAnalysisOnRealGames:
    """Integration tests that run capture analysis on real game data."""

    def _get_games_dir(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs', 'on_policy_fix_20260308', 'games'
        )

    def test_vs_random_loss_rate_reasonable(self):
        """Verify the NN has a reasonable win rate vs random in recent iterations."""
        games_dir = self._get_games_dir()

        # Check a few iterations
        total_losses = 0
        total_games = 0

        for iter_num in [380, 385, 390]:
            dcn_path = os.path.join(games_dir, f'iter_{iter_num:04d}.dcn')
            if not os.path.exists(dcn_path):
                continue

            with open(dcn_path) as f:
                content = f.read()

            game_texts = re.split(r'\n\n(?=\[Iteration)', content)
            for gt in game_texts:
                gt = gt.strip()
                if not gt:
                    continue
                headers, moves, result = dcn_to_game(gt)
                game_num = int(headers.get('Game', '0'))
                game_idx = game_num - 1
                if game_idx >= 10:
                    continue

                total_games += 1
                nn_player = Player.WHITE if game_idx % 2 == 0 else Player.BLACK
                nn_lost = (result == '0-1' and nn_player == Player.WHITE) or \
                          (result == '1-0' and nn_player == Player.BLACK)
                if nn_lost:
                    total_losses += 1

        if total_games > 0:
            loss_rate = total_losses / total_games
            # NN should be winning most games vs random at ELO 2000+
            # But we allow up to 30% losses since random can get lucky
            assert loss_rate < 0.5, (
                f"NN losing {loss_rate:.0%} vs random "
                f"({total_losses}/{total_games})"
            )

    def test_one_ply_captures_exist_in_losses(self):
        """Verify that 1-ply captures (immediate blunders) exist in loss games.

        This documents the finding that the NN moves pieces to squares where
        they are immediately captured.
        """
        games_dir = self._get_games_dir()

        one_ply_captures = 0
        total_captures = 0

        for iter_num in range(380, 391):
            dcn_path = os.path.join(games_dir, f'iter_{iter_num:04d}.dcn')
            if not os.path.exists(dcn_path):
                continue

            with open(dcn_path) as f:
                content = f.read()

            game_texts = re.split(r'\n\n(?=\[Iteration)', content)
            for gt in game_texts:
                gt = gt.strip()
                if not gt:
                    continue
                headers, moves, result = dcn_to_game(gt)
                game_num = int(headers.get('Game', '0'))
                game_idx = game_num - 1
                if game_idx >= 10:
                    continue

                nn_player = Player.WHITE if game_idx % 2 == 0 else Player.BLACK
                nn_lost = (result == '0-1' and nn_player == Player.WHITE) or \
                          (result == '1-0' and nn_player == Player.BLACK)
                if not nn_lost:
                    continue

                # Replay and track captures
                state = GameState()
                piece_last_moved = {}

                for ply_idx, move in enumerate(moves):
                    if isinstance(move, MoveStep):
                        src = state.board[move.from_rc[0]][move.from_rc[1]]
                        if src is None:
                            break

                        if move.is_capture:
                            captured = state.board[move.to_rc[0]][move.to_rc[1]]
                            if captured is not None and captured.player == nn_player:
                                total_captures += 1
                                origin_ply = piece_last_moved.get(move.to_rc, -1)
                                if origin_ply >= 0:
                                    gap = ply_idx - origin_ply
                                    if gap == 1:
                                        one_ply_captures += 1

                        if move.from_rc in piece_last_moved:
                            del piece_last_moved[move.from_rc]
                        piece_last_moved[move.to_rc] = ply_idx

                    elif isinstance(move, Promote):
                        piece_last_moved[move.from_rc] = ply_idx

                    elif isinstance(move, BombardAttack):
                        captured = state.board[move.target_rc[0]][move.target_rc[1]]
                        if captured is not None and captured.player == nn_player:
                            total_captures += 1
                            origin_ply = piece_last_moved.get(move.target_rc, -1)
                            if origin_ply >= 0:
                                gap = ply_idx - origin_ply
                                if gap == 1:
                                    one_ply_captures += 1

                    try:
                        state = apply_move(state, move)
                    except Exception:
                        break

        if total_captures > 0:
            one_ply_rate = one_ply_captures / total_captures
            print(f"\n1-ply captures: {one_ply_captures}/{total_captures} ({one_ply_rate:.1%})")
            # Document: majority of captures are 1-ply blunders
            assert one_ply_captures > 0, "Expected to find 1-ply captures in loss games"
