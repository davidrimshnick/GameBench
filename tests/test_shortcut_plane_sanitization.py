"""Tests for sanitizing reserved shortcut planes in training data."""

import numpy as np

from davechess.engine.network import NUM_INPUT_PLANES, POLICY_SIZE
from davechess.engine.selfplay import ReplayBuffer


class TestShortcutPlaneSanitization:
    def test_replay_buffer_push_zeroes_reserved_planes(self):
        buf = ReplayBuffer(max_size=4)
        planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
        planes[14:18] = 1.0
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[0] = 1.0

        buf.push(planes, policy, 1.0)

        stored = buf.planes[0]
        assert np.count_nonzero(stored[14:18]) == 0

    def test_replay_buffer_load_zeroes_reserved_planes(self, tmp_path):
        path = tmp_path / "legacy_buffer.npz"
        planes = np.zeros((1, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
        planes[:, 14:18] = 1.0
        policies = np.zeros((1, POLICY_SIZE), dtype=np.float32)
        policies[:, 0] = 1.0
        values = np.array([1.0], dtype=np.float32)
        legal_masks = np.zeros((1, POLICY_SIZE), dtype=np.bool_)
        legal_masks[:, 0] = True
        np.savez_compressed(
            path,
            planes=planes,
            policies=policies,
            values=values,
            legal_masks=legal_masks,
        )

        buf = ReplayBuffer(max_size=4)
        buf.load_data(str(path))

        stored = buf.planes[0]
        assert np.count_nonzero(stored[14:18]) == 0
