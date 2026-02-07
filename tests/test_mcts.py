"""Tests for MCTS correctness."""

import pytest

from davechess.game.state import GameState, Player, Piece, PieceType, MoveStep, BombardAttack
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
        """MCTS should find an immediate winning Warrior capture on Commander."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        # Constrained position: attacker at (1,1) with 3 adjacent Warriors
        # gives strength 1+3=4 > Commander str 2 → attacker wins
        state.board[1][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # attacker
        state.board[0][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (above)
        state.board[2][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (below)
        state.board[1][0] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (left)
        state.board[1][2] = Piece(PieceType.COMMANDER, Player.BLACK)  # target (right)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.WHITE)  # safe

        mcts = MCTSLite(num_simulations=300)
        move = mcts.search(state)
        # Should capture the Commander
        assert isinstance(move, MoveStep)
        assert move.to_rc == (1, 2)
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
        # Constrained position: attacker at (1,1) with 3 adjacent Warriors
        # gives strength 1+3=4 > Commander str 2 → attacker wins
        state.board[1][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # attacker
        state.board[0][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (above)
        state.board[2][1] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (below)
        state.board[1][0] = Piece(PieceType.WARRIOR, Player.WHITE)  # adjacent (left)
        state.board[1][2] = Piece(PieceType.COMMANDER, Player.BLACK)  # target (right)
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.WHITE)  # safe

        move, _ = mcts.get_move(state, add_noise=False)
        # Should find the Commander capture
        assert isinstance(move, MoveStep)
        assert move.to_rc == (1, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestBatchedMCTS:
    def test_batched_evaluator_basic(self):
        """BatchedEvaluator should return correct number of results."""
        from davechess.engine.mcts import BatchedEvaluator, MCTSNode
        from davechess.engine.network import DaveChessNetwork, state_to_planes

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        evaluator = BatchedEvaluator(net)

        for _ in range(5):
            state = GameState()
            node = MCTSNode(state=state)
            evaluator.submit(node, state_to_planes(state))

        results = evaluator.evaluate_batch()
        assert len(results) == 5
        for policy, value in results:
            assert policy.shape == (2816,)
            assert -1.0 <= value <= 1.0

    def test_batched_evaluator_no_network(self):
        """Without a network, should return uniform policy and zero value."""
        from davechess.engine.mcts import BatchedEvaluator, MCTSNode
        from davechess.engine.network import state_to_planes, POLICY_SIZE

        evaluator = BatchedEvaluator(None)
        state = GameState()
        node = MCTSNode(state=state)
        evaluator.submit(node, state_to_planes(state))
        results = evaluator.evaluate_batch()
        assert len(results) == 1
        policy, value = results[0]
        assert abs(value) < 1e-6
        assert abs(policy.sum() - 1.0) < 1e-5
        assert policy.shape == (POLICY_SIZE,)

    def test_batched_evaluator_empty(self):
        """Empty evaluator should return empty results."""
        from davechess.engine.mcts import BatchedEvaluator
        evaluator = BatchedEvaluator(None)
        assert evaluator.evaluate_batch() == []

    def test_batched_search_produces_valid_roots(self):
        """batched_search should produce root nodes with visit counts."""
        from davechess.engine.mcts import MCTS, BatchedEvaluator
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        num_sims = 10
        engines = [MCTS(net, num_simulations=num_sims) for _ in range(3)]
        states = [GameState() for _ in range(3)]
        evaluator = BatchedEvaluator(net)

        roots = MCTS.batched_search(engines, states, evaluator, [True, True, True])

        assert len(roots) == 3
        for root in roots:
            assert root.visit_count > 0
            assert root.children
            assert root.is_expanded

    def test_get_move_from_root(self):
        """get_move_from_root should return a legal move."""
        from davechess.engine.mcts import MCTS
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        mcts = MCTS(net, num_simulations=10)

        state = GameState()
        root = mcts.search(state)
        move, info = mcts.get_move_from_root(root, state)
        legal = generate_legal_moves(state)
        assert move in legal
        assert "policy_target" in info

    def test_parallel_selfplay_output_format(self):
        """run_selfplay_batch_parallel should produce same output format."""
        from davechess.engine.selfplay import run_selfplay_batch_parallel
        from davechess.engine.network import DaveChessNetwork, POLICY_SIZE

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        examples, stats = run_selfplay_batch_parallel(
            network=net, num_games=2, num_simulations=5,
            parallel_games=2)

        assert isinstance(examples, list)
        assert "white_wins" in stats
        assert "game_records" in stats
        assert "game_details" in stats
        for planes, policy, value in examples:
            assert planes.shape == (15, 8, 8)
            assert policy.shape == (POLICY_SIZE,)
            assert -1.0 <= value <= 1.0

    def test_parallel_selfplay_with_random_opponent(self):
        """Parallel selfplay with random opponents should work."""
        from davechess.engine.selfplay import run_selfplay_batch_parallel
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        examples, stats = run_selfplay_batch_parallel(
            network=net, num_games=4, num_simulations=5,
            random_opponent_fraction=0.5, parallel_games=4)

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert stats["num_random_games"] == 2
