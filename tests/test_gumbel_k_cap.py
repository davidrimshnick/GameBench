"""Tests for Gumbel Sequential Halving k-cap and visit concentration.

The k (number of considered actions) must be capped at max_num_considered_actions
so that Sequential Halving concentrates simulations on the most promising moves.
Without capping, k=num_actions and Gumbel produces near-uniform visit distributions
that are indistinguishable from random.

Additionally, Sequential Halving must produce a schedule where visit levels
stay in sync with actual MCTS visit counts, so that later phases properly
halve the considered set and concentrate visits on fewer actions.
"""

import math
import numpy as np
import pytest

from davechess.engine.gumbel_mcts import (
    _effective_considered_actions,
    _get_sequence_of_considered_visits,
    GumbelMCTS,
    GumbelBatchedSearch,
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


class TestSequentialHalvingSchedule:
    """Verify the visit schedule produces proper halving concentration."""

    def test_schedule_length_matches_sims(self):
        """Schedule length should equal num_simulations."""
        for k in [2, 4, 8, 16]:
            for sims in [32, 64, 128]:
                seq = _get_sequence_of_considered_visits(k, sims)
                assert len(seq) == sims, f"k={k}, sims={sims}: len={len(seq)}"

    def test_schedule_visit_levels_increment(self):
        """Visit levels should increment after each sweep of k actions.

        With k=16 and 128 sims, the first 16 entries should be cv=0
        (one per action), then next 16 should be cv=1, etc.
        """
        seq = _get_sequence_of_considered_visits(16, 128)
        # First sweep: cv=0 for 16 entries
        assert all(v == 0 for v in seq[:16]), (
            f"First 16 entries should all be 0, got {seq[:16]}"
        )
        # Second sweep: cv=1 for 16 entries
        assert all(v == 1 for v in seq[16:32]), (
            f"Entries 16-31 should all be 1, got {seq[16:32]}"
        )

    def test_schedule_halves_considered_set(self):
        """After each phase, entries per visit level should halve."""
        seq = _get_sequence_of_considered_visits(16, 128)
        from collections import Counter
        counts = Counter(seq)
        # Phase 1 (nc=16): cv=0 and cv=1 each have 16 entries
        assert counts[0] == 16, f"cv=0 count={counts[0]}, expected 16"
        assert counts[1] == 16, f"cv=1 count={counts[1]}, expected 16"
        # Phase 2 (nc=8): cv=2..5 each have 8 entries
        for cv in range(2, 6):
            assert counts[cv] == 8, f"cv={cv} count={counts[cv]}, expected 8"
        # Phase 3 (nc=4): cv=6..13 each have 4 entries
        for cv in range(6, 14):
            assert counts[cv] == 4, f"cv={cv} count={counts[cv]}, expected 4"
        # Phase 4 (nc=2): cv=14+ each have 2 entries
        for cv in range(14, 30):
            assert counts[cv] == 2, f"cv={cv} count={counts[cv]}, expected 2"

    def test_schedule_not_uniform(self):
        """Schedule should have many distinct visit levels, not just 4.

        Old broken schedule had only 4 levels (0,1,2,3), each repeated 32 times,
        which produced uniform visits. Fixed schedule has 30 distinct levels.
        """
        seq = _get_sequence_of_considered_visits(16, 128)
        unique_levels = len(set(seq))
        assert unique_levels >= 10, (
            f"Only {unique_levels} unique visit levels — schedule may be broken. "
            f"Expected 30 levels with k=16, sims=128."
        )

    def test_k1_schedule(self):
        """With k=1, schedule should be simple ascending."""
        seq = _get_sequence_of_considered_visits(1, 10)
        assert seq == list(range(10))


class TestGumbelVisitConcentration:
    """Verify Gumbel search concentrates visits on top-k actions."""

    def test_visit_distribution_peaked_small_k(self):
        """With k=4 and 128 sims on 10 legal moves, visits concentrate on top-4."""
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

        # Top action should have significantly more visits than uniform
        max_visits = visits.max()
        uniform_visits = 128 / num_legal  # 12.8
        assert max_visits > uniform_visits * 2, (
            f"Max visits {max_visits} not much above uniform {uniform_visits:.1f}."
        )

    def test_visit_distribution_not_uniform_k16(self):
        """With k=16 (all 10 legal moves), visits should NOT be uniform.

        This is the critical test. With the old broken schedule, all k actions
        got exactly sims/k visits each (perfectly uniform). With the fixed
        schedule, Sequential Halving concentrates visits on top actions.
        """
        gumbel = GumbelMCTS(
            network=None,
            num_simulations=128,
            max_num_considered_actions=16,
            cpuct=4.0,
            value_scale=0.1,
        )

        state = GameState()
        legal_moves = generate_legal_moves(state)
        num_legal = len(legal_moves)

        move, info = gumbel.search(state)

        visit_counts = info["visit_counts"]
        visits = np.array([visit_counts.get(i, 0) for i in range(num_legal)])

        # Visit distribution should NOT be perfectly uniform
        # With 10 actions and 128 sims, uniform would be 12.8 each
        max_v = visits.max()
        min_v = visits[visits > 0].min() if (visits > 0).any() else 0

        # Max should be significantly above min (concentration)
        assert max_v >= min_v * 2, (
            f"Visits nearly uniform: max={max_v}, min={min_v}. "
            f"Sequential Halving should concentrate visits."
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

        # With proper Sequential Halving, top move should get >>6.25% of visits
        # (uniform over 16 would be 6.25%)
        assert max_prob > 0.15, (
            f"Max policy target prob {max_prob:.3f} is too low. "
            f"Expected > 0.15 with Sequential Halving."
        )


class TestBackpropValueScale:
    """Verify backprop_value_scale dampens NN value predictions in subtrees.

    Standard MCTS uses value_scale to dampen hallucinated NN predictions
    during backpropagation. Gumbel MCTS must do the same via backprop_value_scale,
    otherwise confident-but-wrong ±1.0 predictions dominate Q-values.
    """

    def test_default_backprop_scale_is_one(self):
        """Default backprop_value_scale should be 1.0 (no dampening)."""
        gumbel = GumbelMCTS(
            network=None, num_simulations=32,
            max_num_considered_actions=4,
        )
        assert gumbel.backprop_value_scale == 1.0

    def test_custom_backprop_scale(self):
        """backprop_value_scale should be settable."""
        gumbel = GumbelMCTS(
            network=None, num_simulations=32,
            max_num_considered_actions=4,
            backprop_value_scale=0.5,
        )
        assert gumbel.backprop_value_scale == 0.5

    def test_batched_default_backprop_scale(self):
        """GumbelBatchedSearch default backprop_value_scale should be 1.0."""
        gbs = GumbelBatchedSearch(
            network=None, num_simulations=32,
            max_num_considered_actions=4,
        )
        assert gbs.backprop_value_scale == 1.0

    def test_batched_custom_backprop_scale(self):
        """GumbelBatchedSearch backprop_value_scale should be settable."""
        gbs = GumbelBatchedSearch(
            network=None, num_simulations=32,
            max_num_considered_actions=4,
            backprop_value_scale=0.5,
        )
        assert gbs.backprop_value_scale == 0.5

    def test_search_works_with_half_scale(self):
        """Gumbel search should work correctly with backprop_value_scale=0.5."""
        gumbel = GumbelMCTS(
            network=None, num_simulations=64,
            max_num_considered_actions=4,
            backprop_value_scale=0.5,
        )

        state = GameState()
        move, info = gumbel.search(state)

        # Should produce valid results
        assert move is not None
        assert "policy_target" in info
        visits = info["visit_counts"]
        total = sum(visits.values())
        assert total == 64, f"Total visits {total} != 64"
