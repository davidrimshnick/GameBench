"""Tests for legal-move masking in the policy loss computation.

The policy loss should compute cross-entropy only over legal moves, not all
4288 action slots. Without masking, the network wastes gradient learning to
suppress ~4260 illegal move logits instead of learning which legal move is best.
"""

import numpy as np
import pytest
import torch

from davechess.engine.network import (
    DaveChessNetwork, state_to_planes, NUM_INPUT_PLANES, move_to_policy_index,
)
from davechess.engine.selfplay import build_legal_move_mask
from davechess.game.state import GameState
from davechess.game.rules import generate_legal_moves


def _masked_cross_entropy(logits, target, legal_mask):
    """Compute legal-move-masked cross-entropy, matching training.py logic."""
    neg_inf = torch.tensor(float("-inf"), device=logits.device)
    masked_logits = torch.where(legal_mask, logits, neg_inf)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
    return -torch.sum(target * log_probs, dim=1)


class TestLegalMoveMaskingInLoss:
    """Verify policy loss is computed only over legal moves."""

    def _make_policy_target(self, legal_indices, num_actions=4288):
        """Create a policy target with uniform distribution over legal moves."""
        target = np.zeros(num_actions, dtype=np.float32)
        target[legal_indices] = 1.0 / len(legal_indices)
        return target

    def test_masked_loss_lower_than_unmasked(self):
        """Masked policy loss should be much lower than unmasked for a random network.

        With masking, baseline is ln(num_legal) ≈ 2.3-3.6.
        Without masking, baseline is ln(4288) ≈ 8.4.
        """
        net = DaveChessNetwork(num_res_blocks=2, num_filters=32)
        net.eval()

        state = GameState()
        planes = torch.from_numpy(state_to_planes(state)).unsqueeze(0)
        legal_moves = generate_legal_moves(state)
        num_legal = len(legal_moves)

        with torch.no_grad():
            logits, _ = net(planes)

        # Create uniform target over legal moves
        legal_indices = [move_to_policy_index(m) for m in legal_moves]
        target = torch.zeros(1, 4288)
        target[0, legal_indices] = 1.0 / num_legal
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, legal_indices] = True

        # Unmasked loss: softmax over all 4288
        unmasked_loss = -torch.sum(
            target * torch.log_softmax(logits, dim=1), dim=1
        ).item()

        # Masked loss: softmax only over legal moves
        masked_loss = _masked_cross_entropy(logits, target, legal_mask).item()

        # Masked loss should be much smaller (near ln(num_legal) ≈ 2.3)
        # Unmasked loss should be near ln(4288) ≈ 8.4
        assert masked_loss < unmasked_loss
        assert masked_loss < np.log(num_legal) + 1.0  # within 1 nat of optimal
        assert unmasked_loss > np.log(num_legal) + 2.0  # substantially higher

    def test_masked_loss_equals_legal_only_crossentropy(self):
        """Masked loss should equal cross-entropy computed only over legal moves."""
        logits = torch.randn(1, 4288)
        legal_indices = [0, 10, 50, 100, 200]
        target = torch.zeros(1, 4288)
        target[0, legal_indices] = 0.2  # uniform over 5 legal moves
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, legal_indices] = True

        # Method 1: mask + full softmax
        loss_masked = _masked_cross_entropy(logits, target, legal_mask).item()

        # Method 2: extract legal logits and compute directly
        legal_logits = logits[0, legal_indices]
        legal_probs = torch.softmax(legal_logits, dim=0)
        loss_direct = -torch.sum(
            torch.tensor([0.2] * 5) * torch.log(legal_probs)
        ).item()

        assert loss_masked == pytest.approx(loss_direct, abs=1e-5)

    def test_gradient_only_flows_to_legal_moves(self):
        """Gradients should only affect logits for legal moves, not illegal ones."""
        logits = torch.randn(1, 4288, requires_grad=True)
        legal_indices = [5, 15, 25, 35, 45]
        target = torch.zeros(1, 4288)
        target[0, legal_indices] = 0.2
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, legal_indices] = True

        loss = _masked_cross_entropy(logits, target, legal_mask)
        loss.backward()

        grad = logits.grad[0]
        # Illegal move gradients should be zero
        illegal_indices = [i for i in range(4288) if i not in legal_indices]
        assert torch.all(grad[illegal_indices] == 0.0)
        # Legal move gradients should be non-zero
        assert torch.any(grad[legal_indices] != 0.0)

    def test_perfect_prediction_gives_target_entropy(self):
        """If network perfectly matches the target over legal moves, loss = target entropy."""
        target = torch.zeros(1, 4288)
        target[0, 10] = 0.8
        target[0, 20] = 0.15
        target[0, 30] = 0.05
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, [10, 20, 30]] = True

        # Logits that produce exactly the target distribution over legal moves
        logits = torch.full((1, 4288), -100.0)
        logits[0, 10] = np.log(0.8)
        logits[0, 20] = np.log(0.15)
        logits[0, 30] = np.log(0.05)

        loss = _masked_cross_entropy(logits, target, legal_mask).item()

        # Loss should equal entropy of the target distribution
        expected_entropy = -(0.8 * np.log(0.8) + 0.15 * np.log(0.15) + 0.05 * np.log(0.05))
        assert loss == pytest.approx(expected_entropy, abs=1e-4)

    def test_train_step_uses_masking(self):
        """Integration test: actual train_step should produce reasonable masked loss."""
        from davechess.engine.training import Trainer

        config = {
            "network": {"num_res_blocks": 2, "num_filters": 32, "input_planes": NUM_INPUT_PLANES},
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "value_loss_weight": 1.0,
                "steps_per_iteration": 1,
                "batch_size": 4,
            },
            "mcts": {"num_simulations": 8},
            "selfplay": {
                "replay_buffer_size": 100,
                "buffer_seed_size": 0,
                "buffer_decisive_size": 50,
                "buffer_draw_size": 50,
            },
            "paths": {
                "checkpoint_dir": "/tmp/test_masking_ckpt",
                "log_dir": "/tmp/test_masking_log",
                "training_log": "test_masking.jsonl",
            },
        }

        trainer = Trainer(config)

        # Add some training data with known legal moves
        state = GameState()
        legal_moves = generate_legal_moves(state)
        legal_indices = [move_to_policy_index(m) for m in legal_moves]
        num_legal = len(legal_indices)

        for i in range(10):
            planes_np = state_to_planes(state)
            policy = np.zeros(4288, dtype=np.float32)
            policy[legal_indices] = 1.0 / num_legal
            value = 1.0 if i % 2 == 0 else -1.0
            trainer.replay_buffer.push(
                planes_np, policy, value,
                legal_mask=build_legal_move_mask(state),
            )

        # Run one training step and check loss is reasonable
        losses = trainer.train_step(batch_size=4)
        policy_loss = losses["policy_loss"]

        # With masking, policy loss should be near ln(num_legal) ≈ 2.3
        # Without masking it would be near ln(4288) ≈ 8.4
        assert policy_loss < np.log(num_legal) + 2.0, (
            f"Policy loss {policy_loss:.2f} too high — masking may not be working. "
            f"Expected near ln({num_legal})={np.log(num_legal):.2f}"
        )
        assert policy_loss > 0.0  # sanity: loss should be positive

    def test_single_legal_move_gives_zero_loss(self):
        """When there's only one legal move, policy loss should be ~0."""
        logits = torch.randn(1, 4288)
        target = torch.zeros(1, 4288)
        target[0, 42] = 1.0  # single legal move
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, 42] = True

        loss = _masked_cross_entropy(logits, target, legal_mask).item()

        # softmax of a single element is always 1.0, so log(1.0) = 0
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_illegal_logits_dont_affect_masked_loss(self):
        """Changing illegal move logits should not change the masked loss."""
        legal_indices = [5, 15, 25]
        target = torch.zeros(1, 4288)
        target[0, legal_indices] = 1.0 / 3.0
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, legal_indices] = True

        logits1 = torch.randn(1, 4288)
        logits2 = logits1.clone()
        # Change illegal move logits
        logits2[0, 100] = 999.0
        logits2[0, 200] = -999.0

        loss1 = _masked_cross_entropy(logits1, target, legal_mask).item()
        loss2 = _masked_cross_entropy(logits2, target, legal_mask).item()

        assert loss1 == pytest.approx(loss2, abs=1e-6)

    def test_no_nan_in_loss(self):
        """Masked loss should never produce NaN, even with extreme logits."""
        target = torch.zeros(1, 4288)
        target[0, [0, 1, 2]] = 1.0 / 3.0
        legal_mask = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask[0, [0, 1, 2]] = True

        # Test with extreme logit values
        logits = torch.randn(1, 4288) * 100
        loss = _masked_cross_entropy(logits, target, legal_mask)
        assert torch.isfinite(loss).all()

        # Test with very small number of legal moves
        target2 = torch.zeros(1, 4288)
        target2[0, 0] = 1.0
        legal_mask2 = torch.zeros(1, 4288, dtype=torch.bool)
        legal_mask2[0, 0] = True
        loss2 = _masked_cross_entropy(torch.randn(1, 4288), target2, legal_mask2)
        assert torch.isfinite(loss2).all()

    def test_batch_consistency(self):
        """Masked loss should work correctly across a batch with varying legal moves."""
        batch_size = 4
        logits = torch.randn(batch_size, 4288)
        target = torch.zeros(batch_size, 4288)
        legal_mask = torch.zeros(batch_size, 4288, dtype=torch.bool)

        # Different legal move counts per example
        target[0, [0, 1, 2]] = 1.0 / 3.0          # 3 legal moves
        target[1, [10, 20, 30, 40, 50]] = 0.2      # 5 legal moves
        target[2, [100]] = 1.0                      # 1 legal move
        target[3, list(range(50))] = 1.0 / 50.0     # 50 legal moves
        legal_mask[0, [0, 1, 2]] = True
        legal_mask[1, [10, 20, 30, 40, 50]] = True
        legal_mask[2, [100]] = True
        legal_mask[3, list(range(50))] = True

        losses = _masked_cross_entropy(logits, target, legal_mask)
        assert losses.shape == (batch_size,)
        assert torch.isfinite(losses).all()

        # Single legal move should have lowest loss (= 0)
        assert losses[2].item() == pytest.approx(0.0, abs=1e-5)
        # More legal moves → higher baseline loss
        assert losses[3].item() > losses[0].item()  # 50 moves > 3 moves

    def test_zero_visit_legal_moves_remain_in_mask(self):
        """A legal move with zero visits must stay in the normalization set."""
        logits = torch.tensor([[5.0, 4.0, -5.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])

        # Move 1 is legal but got zero visits; move 2 is illegal.
        legal_mask = torch.tensor([[True, True, False]])

        loss = _masked_cross_entropy(logits, target, legal_mask).item()
        assert loss == pytest.approx(0.31326166, abs=1e-6)

        support_only_mask = target > 0
        bad_loss = _masked_cross_entropy(logits, target, support_only_mask).item()
        assert bad_loss == pytest.approx(0.0, abs=1e-6)
        assert loss > bad_loss
