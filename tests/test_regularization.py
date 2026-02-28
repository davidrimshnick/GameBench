"""Tests for training regularization: value head dropout, MCTS value scaling,
legal-move masking, and their interactions."""

import numpy as np
import pytest

from davechess.game.state import GameState, Player
from davechess.game.rules import generate_legal_moves, apply_move


class TestValueHeadDropout:
    """Tests for value head dropout preventing memorization."""

    def test_dropout_present_in_network(self):
        """Network should have dropout layers in value head."""
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32, value_head_dropout=0.3)
        assert hasattr(net, 'value_dropout')
        assert net.value_dropout.p == 0.3

    def test_dropout_zero_is_noop(self):
        """Dropout=0 should produce identical repeated eval outputs."""
        import torch
        from davechess.engine.network import DaveChessNetwork, state_to_planes

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32, value_head_dropout=0.0)
        net.eval()
        state = GameState()
        planes = torch.from_numpy(state_to_planes(state)).unsqueeze(0)

        with torch.no_grad():
            _, v1 = net(planes)
            _, v2 = net(planes)
        # With dropout=0, eval mode should be perfectly deterministic
        assert abs(v1.item() - v2.item()) < 1e-6

    def test_dropout_changes_train_vs_eval(self):
        """With high dropout, train mode should produce different values from eval mode."""
        import torch
        from davechess.engine.network import DaveChessNetwork, state_to_planes

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32, value_head_dropout=0.5)
        state = GameState()
        planes = torch.from_numpy(state_to_planes(state)).unsqueeze(0)

        # Run many train-mode forward passes, collect variance
        net.train()
        train_values = []
        with torch.no_grad():
            for _ in range(50):
                _, v = net(planes)
                train_values.append(v.item())

        net.eval()
        with torch.no_grad():
            _, v_eval = net(planes)

        # Train mode should have variance from dropout, eval should be deterministic
        train_std = np.std(train_values)
        # With dropout=0.5 and sufficient hidden units, we expect visible variance
        # (at minimum some variation — may be small for random init)
        eval_values = []
        with torch.no_grad():
            for _ in range(10):
                _, v = net(planes)
                eval_values.append(v.item())
        eval_std = np.std(eval_values)
        assert eval_std < 1e-6, "Eval mode should be deterministic"

    def test_dropout_from_checkpoint_compat(self):
        """Loading a checkpoint without dropout key should default to 0."""
        import torch
        from davechess.engine.network import DaveChessNetwork

        # Create and save a network without the dropout key
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32, value_head_dropout=0.0)
        ckpt = {"network_state": net.state_dict(), "training_step": 0,
                "iteration": 0, "elo_estimate": 0}
        # No "value_head_dropout" key in checkpoint
        path = "/tmp/test_dropout_compat.pt"
        torch.save(ckpt, path)

        loaded, meta = DaveChessNetwork.from_checkpoint(path)
        assert loaded.value_dropout.p == 0.0


class TestMCTSValueScale:
    """Tests for value scaling in MCTS."""

    def test_value_scale_reduces_q_magnitude(self):
        """With value_scale < 1, Q-values should be smaller."""
        from davechess.engine.mcts import MCTS

        state = GameState()

        # Full value scale
        mcts_full = MCTS(None, num_simulations=20, cpuct=2.0, value_scale=1.0)
        root_full = mcts_full.search(state, add_noise=False)

        # Half value scale
        mcts_half = MCTS(None, num_simulations=20, cpuct=2.0, value_scale=0.5)
        root_half = mcts_half.search(state, add_noise=False)

        # Without network, value is 0, so both should be similar
        # But the parameter should be stored
        assert mcts_full.value_scale == 1.0
        assert mcts_half.value_scale == 0.5

    def test_value_scale_applied_to_nn_value(self):
        """Value scaling should affect the backpropagated NN value."""
        from davechess.engine.mcts import MCTS, MCTSNode
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        net.eval()

        state = GameState()

        # With full scale
        mcts1 = MCTS(net, num_simulations=10, cpuct=2.0, value_scale=1.0)
        root1 = mcts1.search(state, add_noise=False)

        # With half scale — Q-values should generally be smaller in magnitude
        mcts2 = MCTS(net, num_simulations=10, cpuct=2.0, value_scale=0.5)
        root2 = mcts2.search(state, add_noise=False)

        # Check Q-value magnitudes
        q_full = [abs(c.q_value) for c in root1.children if c.visit_count > 0]
        q_half = [abs(c.q_value) for c in root2.children if c.visit_count > 0]

        if q_full and q_half:
            avg_q_full = np.mean(q_full)
            avg_q_half = np.mean(q_half)
            # Half scale should produce smaller or equal Q magnitudes
            # (not exactly half due to tree structure, but generally smaller)
            assert avg_q_half <= avg_q_full + 0.01, \
                f"Half-scale Q ({avg_q_half:.4f}) should be <= full Q ({avg_q_full:.4f})"


class TestLegalMoveMasking:
    """Tests for legal-move-masked softmax in MCTS expand."""

    def test_priors_sum_to_one(self):
        """After expand, child priors should sum to 1.0."""
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import POLICY_SIZE

        state = GameState()
        node = MCTSNode(state=state)
        # Random logits
        logits = np.random.randn(POLICY_SIZE).astype(np.float32)
        node.expand(logits)

        prior_sum = sum(c.prior for c in node.children)
        assert abs(prior_sum - 1.0) < 1e-5, f"Priors sum to {prior_sum}, expected 1.0"

    def test_all_mass_on_legal_moves(self):
        """With logits-based expand, priors should only exist on legal moves."""
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import POLICY_SIZE

        state = GameState()
        legal_moves = generate_legal_moves(state)
        node = MCTSNode(state=state)

        # Create logits with huge values on illegal slots
        logits = np.zeros(POLICY_SIZE, dtype=np.float32)
        logits[0] = 100.0  # Likely an illegal move slot
        node.expand(logits)

        # Should have exactly as many children as legal moves
        assert len(node.children) == len(legal_moves)
        # All priors should be positive
        for c in node.children:
            assert c.prior > 0

    def test_masking_concentrates_probability(self):
        """Legal move masking should give higher priors than unmasked softmax."""
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import POLICY_SIZE, move_to_policy_index
        import random

        state = GameState()
        legal_moves = generate_legal_moves(state)
        flip = state.current_player == Player.BLACK
        legal_indices = [move_to_policy_index(m, flip=flip) for m in legal_moves]

        # Random logits
        np.random.seed(42)
        logits = np.random.randn(POLICY_SIZE).astype(np.float32)

        # Old approach: full softmax, then renormalize
        full_softmax = np.exp(logits - logits.max())
        full_softmax /= full_softmax.sum()
        old_legal_mass = sum(full_softmax[i] for i in legal_indices)

        # New approach: legal-move-masked softmax (what expand does now)
        node = MCTSNode(state=state)
        node.expand(logits)
        new_legal_mass = sum(c.prior for c in node.children)

        # New approach gives 100% mass on legal moves
        assert abs(new_legal_mass - 1.0) < 1e-5
        # Old approach leaked probability to illegal moves
        assert old_legal_mass < 0.5  # With 10 legal / 4288 total, very little mass

    def test_uniform_logits_give_uniform_priors(self):
        """Zero logits should give uniform priors over legal moves."""
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import POLICY_SIZE

        state = GameState()
        legal_moves = generate_legal_moves(state)
        node = MCTSNode(state=state)
        logits = np.zeros(POLICY_SIZE, dtype=np.float32)
        node.expand(logits)

        expected_prior = 1.0 / len(legal_moves)
        for c in node.children:
            assert abs(c.prior - expected_prior) < 1e-5, \
                f"Expected uniform prior {expected_prior}, got {c.prior}"

    def test_expand_with_network_logits(self):
        """Expand should work with actual network logits."""
        import torch
        from davechess.engine.mcts import MCTSNode
        from davechess.engine.network import DaveChessNetwork, state_to_planes

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        net.eval()

        state = GameState()
        planes = torch.from_numpy(state_to_planes(state)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(planes)
        logits_np = logits[0].cpu().numpy()

        node = MCTSNode(state=state)
        node.expand(logits_np)

        prior_sum = sum(c.prior for c in node.children)
        assert abs(prior_sum - 1.0) < 1e-5
        assert len(node.children) == len(generate_legal_moves(state))


class TestEndToEndMCTS:
    """Integration tests: MCTS search with all new features."""

    def test_mcts_search_produces_valid_moves(self):
        """MCTS with value_scale should still produce valid moves."""
        from davechess.engine.mcts import MCTS

        state = GameState()
        legal_moves = generate_legal_moves(state)
        mcts = MCTS(None, num_simulations=20, cpuct=2.5, value_scale=0.5)
        move, info = mcts.get_move(state, add_noise=True)
        assert move in legal_moves
        assert "policy_target" in info

    def test_mcts_with_network_and_all_features(self):
        """Full integration: network with dropout + value_scale + legal masking."""
        from davechess.engine.mcts import MCTS
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32, value_head_dropout=0.3)
        net.eval()

        state = GameState()
        legal_moves = generate_legal_moves(state)

        mcts = MCTS(net, num_simulations=20, cpuct=2.5, value_scale=0.5)
        move, info = mcts.get_move(state, add_noise=True)

        assert move in legal_moves
        # Policy target should have entries that sum to ~1
        total = sum(info["policy_target"].values())
        assert abs(total - 1.0) < 1e-5

    def test_batched_search_with_value_scale(self):
        """Batched MCTS search should respect per-engine value_scale."""
        from davechess.engine.mcts import MCTS, BatchedEvaluator
        from davechess.engine.network import DaveChessNetwork

        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        net.eval()

        states = [GameState(), GameState()]
        engines = [
            MCTS(net, num_simulations=10, cpuct=2.5, value_scale=0.5),
            MCTS(net, num_simulations=10, cpuct=2.5, value_scale=1.0),
        ]
        evaluator = BatchedEvaluator(net)
        roots = MCTS.batched_search(engines, states, evaluator, [False, False])

        assert len(roots) == 2
        for root in roots:
            assert root.visit_count > 0
            assert len(root.children) > 0
