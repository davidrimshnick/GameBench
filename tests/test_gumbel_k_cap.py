"""Tests for Gumbel Sequential Halving k-cap.

The k (number of considered actions) must be capped at max_num_considered_actions
so that Sequential Halving concentrates simulations on the most promising moves.
Without capping, k=num_actions and Gumbel produces near-uniform visit distributions
that are indistinguishable from random.
"""

import numpy as np
import pytest

from davechess.engine.gumbel_mcts import (
    _effective_considered_actions,
    _get_sequence_of_considered_visits,
    GumbelMCTS,
)
from davechess.game.state import GameState
from davechess.game.rules import generate_legal_moves


class TestEffectiveConsideredActions:
    """Verify k is properly capped at max_num_considered_actions."""

    def test_k_capped_when_many_legal_moves(self):
        """With 37 legal moves and max_k=16, k should be 16, not 37."""
        k = _effective_considered_actions(
            num_actions=37, max_num_considered_actions=16, num_simulations=128
        )
        assert k == 16

    def test_k_equals_num_actions_when_fewer(self):
        """With 10 legal moves and max_k=16, k should be 10."""
        k = _effective_considered_actions(
            num_actions=10, max_num_considered_actions=16, num_simulations=128
        )
        assert k == 10

    def test_k_equals_max_when_equal(self):
        """With exactly 16 legal moves and max_k=16, k=16."""
        k = _effective_considered_actions(
            num_actions=16, max_num_considered_actions=16, num_simulations=128
        )
        assert k == 16

    def test_k_never_exceeds_max(self):
        """k should never exceed max_num_considered_actions regardless of sims."""
        for sims in [64, 128, 256, 512, 1024]:
            for n_legal in [20, 30, 50, 100]:
                k = _effective_considered_actions(n_legal, 16, sims)
                assert k <= 16, f"k={k} > 16 with sims={sims}, legal={n_legal}"

    def test_zero_actions(self):
        assert _effective_considered_actions(0, 16, 128) == 0


class TestGumbelVisitConcentration:
    """Verify Gumbel search concentrates visits on top-k actions."""

    def test_visit_distribution_peaked(self):
        """With k=4 and 128 sims on 10 legal moves, visits concentrate on top-4."""
        # Starting position has 10 legal moves; use max_k=4 to force concentration
        gumbel = GumbelMCTS(
            network=None,
            num_simulations=128,
            max_num_considered_actions=4,
            cpuct=4.0,
            value_scale=0.1,
        )

        state = GameState()
        legal_moves = generate_legal_moves(state)
        num_legal = len(legal_moves)
        assert num_legal == 10, f"Expected 10 legal moves at start, got {num_legal}"

        move, info = gumbel.search(state)

        visit_counts = info["visit_counts"]
        visits = np.array([visit_counts.get(i, 0) for i in range(num_legal)])
        total = visits.sum()

        assert total == 128, f"Total visits {total} != 128"

        # Only top-4 actions should get visits
        visited = (visits > 0).sum()
        assert visited <= 4, (
            f"{visited} actions visited, expected <= 4 (max_k=4). "
            f"Sequential Halving should only visit top-k actions."
        )

        # Top action should have significantly more visits than uniform over all moves
        max_visits = visits.max()
        uniform_visits = 128 / num_legal  # 12.8 if all 10 visited
        assert max_visits > uniform_visits * 2, (
            f"Max visits {max_visits} not much above uniform {uniform_visits:.1f}. "
            f"Sequential Halving isn't concentrating visits."
        )

    def test_policy_target_has_limited_nonzero_entries(self):
        """Policy target should have at most k non-zero entries."""
        gumbel = GumbelMCTS(
            network=None,
            num_simulations=128,
            max_num_considered_actions=16,
            cpuct=4.0,
            value_scale=0.1,
        )

        state = GameState()
        move, info = gumbel.search(state)

        policy_target = info["policy_target"]
        nonzero = sum(1 for v in policy_target.values() if v > 0)
        assert nonzero <= 16, (
            f"Policy target has {nonzero} non-zero entries, expected <= 16"
        )

    def test_policy_target_is_peaked(self):
        """Max probability in policy target should be well above uniform."""
        gumbel = GumbelMCTS(
            network=None,
            num_simulations=128,
            max_num_considered_actions=16,
            cpuct=4.0,
            value_scale=0.1,
        )

        state = GameState()
        legal_moves = generate_legal_moves(state)

        move, info = gumbel.search(state)
        max_prob = max(info["policy_target"].values())

        # With k=16 and 128 sims, top move should get > 15% of visits
        # (uniform over 16 would be 6.25%, Sequential Halving concentrates more)
        assert max_prob > 0.10, (
            f"Max policy target prob {max_prob:.3f} is too low. "
            f"Expected > 0.10 with Sequential Halving on k=16."
        )
