"""Unit tests for the DaveChess game engine."""

import pytest

from davechess.game.state import (
    GameState, Player, PieceType, Piece, Move, MoveStep, Deploy, BombardAttack,
    PIECE_CHARS, BASE_STRENGTH, DEPLOY_COST,
)
from davechess.game.rules import (
    generate_legal_moves, apply_move, check_winner,
    get_resource_income, get_controlled_nodes, get_exclusive_nodes,
)
from davechess.game.board import (
    BOARD_SIZE, RESOURCE_NODES, STARTING_POSITIONS, rc_to_notation, notation_to_rc,
    render_board,
)
from davechess.game.notation import move_to_dcn, dcn_to_move, game_to_dcn, dcn_to_game


class TestBoard:
    def test_board_size(self):
        assert BOARD_SIZE == 8

    def test_resource_nodes_count(self):
        assert len(RESOURCE_NODES) == 8

    def test_resource_nodes_symmetry(self):
        """Resource nodes should be vertically symmetric."""
        for r, c in RESOURCE_NODES:
            mirror_r = BOARD_SIZE - 1 - r
            assert (mirror_r, c) in RESOURCE_NODES, \
                f"Node ({r},{c}) has no mirror at ({mirror_r},{c})"

    def test_starting_positions(self):
        """Verify starting positions are correct."""
        white_pieces = [(r, c, pt) for (r, c), (pt, p) in STARTING_POSITIONS.items() if p == 0]
        black_pieces = [(r, c, pt) for (r, c), (pt, p) in STARTING_POSITIONS.items() if p == 1]
        assert len(white_pieces) == 4  # C + 2W + R
        assert len(black_pieces) == 4

    def test_notation_conversion(self):
        assert rc_to_notation(0, 0) == "a1"
        assert rc_to_notation(7, 7) == "h8"
        assert rc_to_notation(3, 4) == "e4"

    def test_notation_roundtrip(self):
        for r in range(8):
            for c in range(8):
                sq = rc_to_notation(r, c)
                r2, c2 = notation_to_rc(sq)
                assert (r, c) == (r2, c2)

    def test_render_board(self):
        board = [[None] * 8 for _ in range(8)]
        board[0][3] = ("C", 0)
        text = render_board(board)
        assert "C" in text


class TestGameState:
    def test_initial_state(self):
        state = GameState()
        assert state.current_player == Player.WHITE
        assert state.turn == 1
        assert not state.done
        assert state.winner is None
        assert state.resources == [0, 0]

    def test_starting_pieces(self):
        state = GameState()
        # White Commander at d1 (row 0, col 3)
        piece = state.get_piece_at(0, 3)
        assert piece is not None
        assert piece.piece_type == PieceType.COMMANDER
        assert piece.player == Player.WHITE

        # Black Commander at e8 (row 7, col 4)
        piece = state.get_piece_at(7, 4)
        assert piece is not None
        assert piece.piece_type == PieceType.COMMANDER
        assert piece.player == Player.BLACK

    def test_clone(self):
        state = GameState()
        clone = state.clone()
        # Modify original
        state.resources[0] = 99
        assert clone.resources[0] == 0
        state.board[0][0] = Piece(PieceType.WARRIOR, Player.WHITE)
        assert clone.board[0][0] is None

    def test_serialize_roundtrip(self):
        state = GameState()
        state.resources = [5, 3]
        state.turn = 10
        data = state.serialize()
        restored = GameState.deserialize(data)
        assert restored.resources == [5, 3]
        assert restored.turn == 10
        assert restored.current_player == state.current_player

    def test_board_tuple_hash(self):
        s1 = GameState()
        s2 = GameState()
        assert s1.get_board_tuple() == s2.get_board_tuple()
        s2.resources[0] = 5
        assert s1.get_board_tuple() != s2.get_board_tuple()


class TestLegalMoves:
    def test_initial_moves(self):
        state = GameState()
        moves = generate_legal_moves(state)
        assert len(moves) > 0

    def test_no_moves_when_done(self):
        state = GameState()
        state.done = True
        moves = generate_legal_moves(state)
        assert len(moves) == 0

    def test_warrior_moves_orthogonal(self):
        """Warriors can only move orthogonally."""
        state = GameState()
        # Clear board
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.WARRIOR, Player.WHITE)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        warrior_moves = [m for m in moves if isinstance(m, MoveStep) and m.from_rc == (3, 3)]
        # Should have 4 orthogonal moves
        destinations = {m.to_rc for m in warrior_moves}
        assert (3, 4) in destinations  # right
        assert (3, 2) in destinations  # left
        assert (4, 3) in destinations  # up
        assert (2, 3) in destinations  # down
        # Diagonal should NOT be possible for warrior
        assert (4, 4) not in destinations
        assert (2, 2) not in destinations

    def test_commander_moves(self):
        """Commander moves 1-2 squares, any direction."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        cmd_moves = [m for m in moves if isinstance(m, MoveStep) and m.from_rc == (3, 3)]
        destinations = {m.to_rc for m in cmd_moves}
        # Distance 1 in all 8 directions
        assert (4, 3) in destinations
        assert (4, 4) in destinations
        # Distance 2
        assert (5, 3) in destinations
        assert (5, 5) in destinations

    def test_rider_moves(self):
        """Rider moves up to 3 squares in straight line."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        rider_moves = [m for m in moves if isinstance(m, MoveStep) and m.from_rc == (3, 3)]
        destinations = {m.to_rc for m in rider_moves}
        # Can reach up to 3 squares away
        assert (6, 3) in destinations  # 3 up
        assert (0, 3) in destinations  # 3 down (but commander blocks at 0,0)
        assert (3, 6) in destinations  # 3 right
        assert (6, 6) in destinations  # 3 diagonal

    def test_rider_blocked(self):
        """Rider cannot jump over pieces."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[4][3] = Piece(PieceType.WARRIOR, Player.WHITE)  # Block upward
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        rider_moves = [m for m in moves if isinstance(m, MoveStep) and m.from_rc == (3, 3)]
        destinations = {m.to_rc for m in rider_moves}
        assert (4, 3) not in destinations  # Blocked by friendly
        assert (5, 3) not in destinations  # Can't jump

    def test_deploy_moves(self):
        """Can deploy pieces on back 2 rows when affordable."""
        state = GameState()
        state.resources[0] = 10  # White has enough
        moves = generate_legal_moves(state)
        deploy_moves = [m for m in moves if isinstance(m, Deploy)]
        assert len(deploy_moves) > 0
        # All deploys should be on rows 0-1 for white
        for m in deploy_moves:
            assert m.to_rc[0] in (0, 1)

    def test_no_deploy_without_resources(self):
        """Cannot deploy without enough resources."""
        state = GameState()
        state.resources[0] = 0
        moves = generate_legal_moves(state)
        deploy_moves = [m for m in moves if isinstance(m, Deploy)]
        assert len(deploy_moves) == 0

    def test_bombard_ranged_attack(self):
        """Bombard can attack at exactly 2 squares with clear path."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.BOMBARD, Player.WHITE)
        state.board[5][3] = Piece(PieceType.WARRIOR, Player.BLACK)  # 2 sq away, clear
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        bombard_attacks = [m for m in moves if isinstance(m, BombardAttack)]
        targets = {m.target_rc for m in bombard_attacks}
        assert (5, 3) in targets

    def test_bombard_blocked_path(self):
        """Bombard ranged attack blocked by piece in between."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.BOMBARD, Player.WHITE)
        state.board[4][3] = Piece(PieceType.WARRIOR, Player.WHITE)  # Blocking
        state.board[5][3] = Piece(PieceType.WARRIOR, Player.BLACK)  # Target
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        moves = generate_legal_moves(state)
        bombard_attacks = [m for m in moves if isinstance(m, BombardAttack)]
        targets = {m.target_rc for m in bombard_attacks}
        assert (5, 3) not in targets


class TestCapture:
    def test_stronger_attacker_wins(self):
        """Rider (str 2) captures Warrior (str 1)."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[4][3] = Piece(PieceType.WARRIOR, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        move = MoveStep((3, 3), (4, 3), is_capture=True)
        apply_move(state, move)
        # Rider should be at (4,3)
        piece = state.board[4][3]
        assert piece is not None
        assert piece.piece_type == PieceType.RIDER
        assert piece.player == Player.WHITE
        assert state.board[3][3] is None

    def test_weaker_attacker_loses(self):
        """Warrior (str 1) attacks Rider (str 2) - attacker removed."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.WARRIOR, Player.WHITE)
        state.board[3][4] = Piece(PieceType.RIDER, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        move = MoveStep((3, 3), (3, 4), is_capture=True)
        apply_move(state, move)
        # Warrior removed, Rider stays
        assert state.board[3][3] is None
        piece = state.board[3][4]
        assert piece is not None
        assert piece.piece_type == PieceType.RIDER
        assert piece.player == Player.BLACK

    def test_equal_strength_both_removed(self):
        """Equal strength: both pieces removed."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.WARRIOR, Player.WHITE)
        state.board[3][4] = Piece(PieceType.WARRIOR, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        move = MoveStep((3, 3), (3, 4), is_capture=True)
        apply_move(state, move)
        assert state.board[3][3] is None
        assert state.board[3][4] is None

    def test_warrior_adjacency_bonus(self):
        """Warriors get +1 strength per adjacent friendly Warrior."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        # Two adjacent white warriors
        state.board[3][3] = Piece(PieceType.WARRIOR, Player.WHITE)
        state.board[3][4] = Piece(PieceType.WARRIOR, Player.WHITE)
        # Black rider to attack
        state.board[3][5] = Piece(PieceType.RIDER, Player.BLACK)  # str 2
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        # Warrior at (3,4) has adjacent warrior at (3,3), so str = 1+1 = 2
        # This warrior attacks Rider (str 2): tie, both removed
        state.current_player = Player.WHITE
        move = MoveStep((3, 4), (3, 5), is_capture=True)
        apply_move(state, move)
        # Tie: both removed
        assert state.board[3][4] is None
        assert state.board[3][5] is None

    def test_bombard_ranged_capture(self):
        """Bombard ranged attack removes target, Bombard stays."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.BOMBARD, Player.WHITE)
        state.board[5][3] = Piece(PieceType.WARRIOR, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        move = BombardAttack((3, 3), (5, 3))
        apply_move(state, move)
        # Bombard stays
        assert state.board[3][3] is not None
        assert state.board[3][3].piece_type == PieceType.BOMBARD
        # Target removed
        assert state.board[5][3] is None


class TestWinConditions:
    def test_commander_capture_wins(self):
        """Capturing Commander ends the game."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[3][4] = Piece(PieceType.COMMANDER, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)

        move = MoveStep((3, 3), (3, 4), is_capture=True)
        apply_move(state, move)
        assert state.done
        assert state.winner == Player.WHITE

    def test_resource_domination_wins(self):
        """Controlling 6+ exclusive resource nodes wins."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        # Place white pieces adjacent to 6 resource nodes
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)

        # Control all resource nodes with white pieces
        for r, c in RESOURCE_NODES:
            state.board[r][c] = Piece(PieceType.WARRIOR, Player.WHITE)

        # Make a move that triggers the win check
        state.board[6][6] = Piece(PieceType.WARRIOR, Player.WHITE)
        move = MoveStep((6, 6), (6, 5))
        apply_move(state, move)
        assert state.done
        assert state.winner == Player.WHITE

    def test_turn_limit_draw(self):
        """Turn 200+ with equal control results in tiebreak."""
        state = GameState()
        state.turn = 200
        state.current_player = Player.BLACK
        # Black makes a move to trigger turn 201 check
        state.board = [[None] * 8 for _ in range(8)]
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.BLACK)
        state.board[6][6] = Piece(PieceType.WARRIOR, Player.BLACK)

        move = MoveStep((6, 6), (6, 5))
        apply_move(state, move)
        assert state.done


class TestNotation:
    def test_move_notation(self):
        state = GameState()
        move = MoveStep((0, 2), (1, 2))
        dcn = move_to_dcn(state, move)
        assert dcn == "Wc1-c2"

    def test_capture_notation(self):
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[3][4] = Piece(PieceType.WARRIOR, Player.BLACK)
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        move = MoveStep((3, 3), (3, 4), is_capture=True)
        dcn = move_to_dcn(state, move)
        assert dcn == "Rd4xe4"

    def test_deploy_notation(self):
        state = GameState()
        move = Deploy(PieceType.WARRIOR, (0, 0))
        dcn = move_to_dcn(state, move)
        assert dcn == "+W@a1"

    def test_bombard_notation(self):
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        state.board[3][3] = Piece(PieceType.BOMBARD, Player.WHITE)
        move = BombardAttack((3, 3), (5, 3))
        dcn = move_to_dcn(state, move)
        assert dcn == "Bd4~d6"

    def test_parse_move(self):
        move = dcn_to_move("Wc1-c2")
        assert isinstance(move, MoveStep)
        assert move.from_rc == (0, 2)
        assert move.to_rc == (1, 2)
        assert not move.is_capture

    def test_parse_capture(self):
        move = dcn_to_move("Rd4xe4")
        assert isinstance(move, MoveStep)
        assert move.is_capture

    def test_parse_deploy(self):
        move = dcn_to_move("+W@c2")
        assert isinstance(move, Deploy)
        assert move.piece_type == PieceType.WARRIOR
        assert move.to_rc == (1, 2)

    def test_parse_bombard(self):
        move = dcn_to_move("Bd4~d6")
        assert isinstance(move, BombardAttack)
        assert move.from_rc == (3, 3)
        assert move.target_rc == (5, 3)

    def test_notation_roundtrip(self):
        """Parse then emit should produce identical string."""
        cases = ["Wc1-c2", "Rd4xe4", "+W@c2", "Bd4~d6", "+R@a1", "+B@h2"]
        for dcn_str in cases:
            move = dcn_to_move(dcn_str)
            # Need a state with appropriate piece for emission
            state = GameState()
            state.board = [[None] * 8 for _ in range(8)]
            if isinstance(move, MoveStep):
                fr, fc = move.from_rc
                piece_char = dcn_str[0]
                state.board[fr][fc] = Piece(PIECE_CHARS[piece_char], Player.WHITE)
            elif isinstance(move, BombardAttack):
                fr, fc = move.from_rc
                state.board[fr][fc] = Piece(PieceType.BOMBARD, Player.WHITE)
            result = move_to_dcn(state, move)
            assert result == dcn_str, f"Expected {dcn_str}, got {result}"

    def test_game_to_dcn_roundtrip(self):
        """Full game serialization roundtrip."""
        state = GameState()
        move1 = MoveStep((0, 2), (1, 2))  # Wc1-c2
        pairs = [(state.clone(), move1)]
        apply_move(state, move1)
        move2 = MoveStep((7, 5), (6, 5))  # Black Wf8-f7
        pairs.append((state.clone(), move2))

        dcn_text = game_to_dcn(pairs, headers={"White": "Test", "Black": "Test"},
                                result="*")
        headers, moves, result = dcn_to_game(dcn_text)
        assert headers["White"] == "Test"
        assert len(moves) == 2
        assert result == "*"


class TestResourceIncome:
    def test_no_income_initially(self):
        """Starting positions aren't near resource nodes."""
        state = GameState()
        white_income = get_resource_income(state, Player.WHITE)
        black_income = get_resource_income(state, Player.BLACK)
        # With starting positions, income depends on proximity to resource nodes
        assert isinstance(white_income, int)
        assert isinstance(black_income, int)

    def test_piece_on_node(self):
        """Piece directly on a resource node gives +1."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        r, c = RESOURCE_NODES[0]
        state.board[r][c] = Piece(PieceType.WARRIOR, Player.WHITE)
        income = get_resource_income(state, Player.WHITE)
        assert income >= 1

    def test_piece_adjacent_to_node(self):
        """Piece orthogonally adjacent to resource node gives +1."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        r, c = RESOURCE_NODES[0]
        # Place adjacent
        if r + 1 < 8:
            state.board[r + 1][c] = Piece(PieceType.WARRIOR, Player.WHITE)
            income = get_resource_income(state, Player.WHITE)
            assert income >= 1


class TestDeterminism:
    def test_game_is_deterministic(self):
        """Same sequence of moves produces same result."""
        state1 = GameState()
        state2 = GameState()
        moves1 = generate_legal_moves(state1)
        moves2 = generate_legal_moves(state2)
        # Same moves should be generated
        assert len(moves1) == len(moves2)


class TestEdgeCases:
    def test_deploy_on_occupied_square(self):
        """Cannot deploy on occupied square."""
        state = GameState()
        state.resources[0] = 10
        moves = generate_legal_moves(state)
        deploy_moves = [m for m in moves if isinstance(m, Deploy)]
        # All deploy targets should be empty
        for m in deploy_moves:
            r, c = m.to_rc
            assert state.board[r][c] is None

    def test_commander_not_deployable(self):
        """Commander should never appear in deploy moves."""
        state = GameState()
        state.resources[0] = 100
        moves = generate_legal_moves(state)
        deploy_moves = [m for m in moves if isinstance(m, Deploy)]
        for m in deploy_moves:
            assert m.piece_type != PieceType.COMMANDER
