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
        """MCTS should find an immediate winning capture on Commander."""
        state = GameState()
        state.board = [[None] * 8 for _ in range(8)]
        # Rider can capture Black Commander directly (chess-style capture)
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[4][3] = Piece(PieceType.COMMANDER, Player.BLACK)  # target
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.WHITE)  # safe

        mcts = MCTSLite(num_simulations=300)
        move = mcts.search(state)
        # Should capture the Commander
        assert isinstance(move, MoveStep)
        assert move.to_rc == (4, 3)
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
        # Rider can capture Commander directly (chess-style capture)
        state.board[3][3] = Piece(PieceType.RIDER, Player.WHITE)
        state.board[4][3] = Piece(PieceType.COMMANDER, Player.BLACK)  # target
        state.board[7][7] = Piece(PieceType.COMMANDER, Player.WHITE)  # safe

        move, _ = mcts.get_move(state, add_noise=False)
        # Should find the Commander capture
        assert isinstance(move, MoveStep)
        assert move.to_rc == (4, 3)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestBatchedMCTS:
    def test_batched_evaluator_basic(self):
        """BatchedEvaluator should return correct number of results."""
        from davechess.engine.mcts import BatchedEvaluator, MCTSNode
        from davechess.engine.network import DaveChessNetwork, state_to_planes, POLICY_SIZE

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        evaluator = BatchedEvaluator(net)

        for _ in range(5):
            state = GameState()
            node = MCTSNode(state=state)
            evaluator.submit(node, state_to_planes(state))

        results = evaluator.evaluate_batch()
        assert len(results) == 5
        for policy, value in results:
            assert policy.shape == (POLICY_SIZE,)
            assert -1.0 <= value <= 1.0

    def test_batched_evaluator_no_network(self):
        """Without a network, should return uniform logits and zero value."""
        from davechess.engine.mcts import BatchedEvaluator, MCTSNode
        from davechess.engine.network import state_to_planes, POLICY_SIZE

        evaluator = BatchedEvaluator(None)
        state = GameState()
        node = MCTSNode(state=state)
        evaluator.submit(node, state_to_planes(state))
        results = evaluator.evaluate_batch()
        assert len(results) == 1
        logits, value = results[0]
        assert abs(value) < 1e-6
        # Now returns uniform logits (all zeros), not softmax probs
        assert abs(logits.sum()) < 1e-5
        assert logits.shape == (POLICY_SIZE,)

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
            from davechess.engine.network import NUM_INPUT_PLANES; assert planes.shape == (NUM_INPUT_PLANES, 8, 8)
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



class TestMultiprocessMCTS:
    def test_remote_batched_evaluator_no_network(self):
        """RemoteBatchedEvaluator with use_network=False returns uniform logits."""
        import multiprocessing as mp
        from davechess.engine.mcts_worker import RemoteBatchedEvaluator
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import state_to_planes, POLICY_SIZE

        # No queues needed when use_network=False
        evaluator = RemoteBatchedEvaluator(
            worker_id=0, request_queue=None, response_queue=None,
            use_network=False)

        state = GameState()
        node = MCTSNode(state=state)
        evaluator.submit(node, state_to_planes(state))
        results = evaluator.evaluate_batch()

        assert len(results) == 1
        logits, value = results[0]
        assert abs(value) < 1e-6
        # Now returns uniform logits (all zeros), not softmax probs
        assert abs(logits.sum()) < 1e-5
        assert logits.shape == (POLICY_SIZE,)

    def test_distribute_games(self):
        """Games should be distributed evenly across workers."""
        from davechess.engine.selfplay import _distribute_games

        assignments = _distribute_games(num_games=20, num_workers=4,
                                         num_random_games=10)
        assert len(assignments) == 4
        total = sum(len(a) for a in assignments)
        assert total == 20

        # Each worker should get 5 games
        for a in assignments:
            assert len(a) == 5

        # First 10 games should be random
        all_games = []
        for a in assignments:
            all_games.extend(a)
        all_games.sort(key=lambda g: g["game_idx"])
        for g in all_games[:10]:
            assert g["is_random"]
        for g in all_games[10:]:
            assert not g["is_random"]

    def test_multiprocess_selfplay_output_format(self):
        """run_selfplay_multiprocess should produce same output format."""
        from davechess.engine.selfplay import run_selfplay_multiprocess
        from davechess.engine.network import DaveChessNetwork, POLICY_SIZE

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        examples, stats = run_selfplay_multiprocess(
            network=net, num_games=4, num_simulations=5,
            num_workers=2)

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert "white_wins" in stats
        assert "game_records" in stats
        assert "game_details" in stats
        for planes, policy, value in examples:
            from davechess.engine.network import NUM_INPUT_PLANES; assert planes.shape == (NUM_INPUT_PLANES, 8, 8)
            assert policy.shape == (POLICY_SIZE,)
            assert -1.0 <= value <= 1.0

    def test_multiprocess_selfplay_with_random_opponent(self):
        """Multiprocess selfplay with random opponents should work."""
        from davechess.engine.selfplay import run_selfplay_multiprocess
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        examples, stats = run_selfplay_multiprocess(
            network=net, num_games=4, num_simulations=5,
            random_opponent_fraction=0.5, num_workers=2)

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert stats["num_random_games"] == 2

    def test_aggregate_multiprocess_results_requires_complete_game_set(self):
        """Aggregation should fail loudly if any game result is missing."""
        from davechess.engine.selfplay import _aggregate_multiprocess_results

        partial_results = {
            0: [{
                "game_idx": 0,
                "game_type": "selfplay",
                "training_data": [],
                "game_record": {
                    "winner": "draw",
                    "length": 1,
                    "draw_reason": "turn_limit",
                    "moves": [],
                },
            }],
        }

        with pytest.raises(RuntimeError, match="incomplete game results"):
            _aggregate_multiprocess_results(
                partial_results, num_games=2, num_random_games=0
            )


class TestGumbelActionSelection:
    def test_gumbel_search_temp0_ignores_root_gumbel_for_played_move(self, monkeypatch):
        """At low temp, played move should come from improved policy, not gumbel."""
        from davechess.engine.gumbel_mcts import GumbelMCTS

        def fake_gumbel(size):
            import numpy as np
            arr = np.zeros(size, dtype=np.float64)
            if size > 1:
                arr[1] = 100.0
            return arr

        monkeypatch.setattr("numpy.random.gumbel", fake_gumbel)

        state = GameState()
        legal_moves = generate_legal_moves(state)
        assert len(legal_moves) > 1

        mcts = GumbelMCTS(network=None, num_simulations=8, temperature=0.0)
        move, _ = mcts.search(state)

        # With uniform logits/values (network=None), improved_logits tie and
        # argmax picks index 0. A gumbel-driven final argmax would pick index 1.
        assert move == legal_moves[0]

    def test_batched_gumbel_search_temp0_ignores_root_gumbel_for_played_move(self, monkeypatch):
        """Batched variant should also avoid gumbel noise in final action."""
        from davechess.engine.gumbel_mcts import GumbelBatchedSearch

        def fake_gumbel(size):
            import numpy as np
            arr = np.zeros(size, dtype=np.float64)
            if size > 1:
                arr[1] = 100.0
            return arr

        monkeypatch.setattr("numpy.random.gumbel", fake_gumbel)

        state = GameState()
        legal_moves = generate_legal_moves(state)
        assert len(legal_moves) > 1

        search = GumbelBatchedSearch(network=None, num_simulations=8, temperature=0.0)
        results = search.batched_search([state], [0.0])
        move, _ = results[0]

        assert move == legal_moves[0]


class TestGumbelConsideredActions:
    def test_effective_considered_actions_expands_with_budget(self):
        from davechess.engine.gumbel_mcts import _effective_considered_actions

        # If we can afford one root visit per action, don't hard-prune to max_k.
        assert _effective_considered_actions(50, 16, 128) == 50

    def test_effective_considered_actions_respects_sim_budget(self):
        from davechess.engine.gumbel_mcts import _effective_considered_actions

        # Can't cover more actions than total simulations.
        assert _effective_considered_actions(200, 16, 128) == 128
        assert _effective_considered_actions(10, 16, 128) == 10
        assert _effective_considered_actions(200, 0, 32) == 32


class TestTrainingConfigPassThrough:
    def test_selfplay_batch_forwards_cpuct(self, monkeypatch):
        """run_selfplay_batch should forward cpuct into MCTS constructor."""
        from davechess.engine import selfplay as sp

        created = []

        class DummyMCTS:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

        monkeypatch.setattr(sp, "MCTS", DummyMCTS)

        examples, stats = sp.run_selfplay_batch(
            network=None, num_games=0, num_simulations=5, cpuct=2.75,
        )

        assert examples == []
        assert stats["draws"] == 0
        assert created[0]["cpuct"] == 2.75

    def test_selfplay_batch_parallel_forwards_cpuct(self, monkeypatch):
        """run_selfplay_batch_parallel should forward cpuct into MCTS constructor."""
        from davechess.engine import selfplay as sp

        created = []

        class DummyMCTS:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

        monkeypatch.setattr(sp, "MCTS", DummyMCTS)

        examples, stats = sp.run_selfplay_batch_parallel(
            network=None, num_games=0, num_simulations=5, cpuct=3.25, parallel_games=2,
        )

        assert examples == []
        assert stats["draws"] == 0
        assert created[0]["cpuct"] == 3.25


class TestEloMapping:
    def test_win_rate_to_elo_diff_monotonic_at_top_end(self):
        """A perfect score should not map below near-perfect scores."""
        from davechess.engine.training import win_rate_to_elo_diff

        perfect = win_rate_to_elo_diff(1.0)
        near_perfect = win_rate_to_elo_diff((5.0 + 0.5) / 6.0)
        even = win_rate_to_elo_diff(0.5)
        zero = win_rate_to_elo_diff(0.0)

        assert zero <= even <= near_perfect <= perfect
        assert perfect <= 400.0
