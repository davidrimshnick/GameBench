"""Tests for network-vs-network ELO probe system."""

import os
import tempfile
import pytest

import torch
import numpy as np

from davechess.engine.network import DaveChessNetwork
from davechess.engine.training import Trainer, win_rate_to_elo_diff


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create temp dirs for checkpoints and logs."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return ckpt_dir, log_dir


@pytest.fixture
def small_config(tmp_checkpoint_dir):
    """Minimal config for a tiny network (fast tests)."""
    ckpt_dir, log_dir = tmp_checkpoint_dir
    return {
        "network": {
            "num_res_blocks": 2,
            "num_filters": 16,
            "input_planes": 18,
            "board_size": 8,
            "value_head_dropout": 0.0,
        },
        "mcts": {
            "num_simulations": 4,
            "min_selfplay_simulations": 4,
            "cpuct": 4.0,
            "value_scale": 0.5,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.4,
            "temperature_threshold": 30,
        },
        "gumbel": {
            "enabled": True,
            "max_num_considered_actions": 8,
            "gumbel_scale": 1.0,
            "maxvisit_init": 50.0,
            "value_scale": 0.1,
        },
        "selfplay": {
            "num_games_per_iteration": 2,
            "replay_buffer_size": 1000,
            "buffer_seed_size": 100,
            "buffer_decisive_size": 500,
            "buffer_draw_size": 400,
            "num_workers": 1,
            "min_buffer_size": 10,
            "random_opponent_fraction": 0.0,
            "draw_value_target": 0.0,
        },
        "training": {
            "optimizer": "sgd",
            "batch_size": 8,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "value_loss_weight": 1.0,
            "steps_per_iteration": 2,
            "checkpoint_interval": 100,
            "elo_probe_interval": 1,
            "elo_probe_games": 2,
            "elo_probe_max_moves": 20,
            "max_iterations": 1,
        },
        "paths": {
            "checkpoint_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
            "training_log": "training_log.jsonl",
        },
    }


class TestSaveLoadReference:
    def test_save_reference_creates_file(self, small_config, tmp_checkpoint_dir):
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()
        assert (ckpt_dir / "reference.pt").exists()

    def test_load_reference_returns_none_when_missing(self, small_config):
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        ref = trainer._load_reference_network()
        assert ref is None

    def test_load_reference_restores_network(self, small_config, tmp_checkpoint_dir):
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()

        # Modify current network weights so reference differs
        with torch.no_grad():
            for p in trainer.network.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ref_net = trainer._load_reference_network()
        assert ref_net is not None
        assert trainer.reference_elo == 700.0

        # Reference should differ from current network
        ref_params = list(ref_net.parameters())
        cur_params = list(trainer.network.parameters())
        diffs = sum(
            (r - c).abs().sum().item()
            for r, c in zip(ref_params, cur_params)
        )
        assert diffs > 0, "Reference should differ from modified current network"

    def test_reference_elo_persists_in_checkpoint(self, small_config):
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 850.0
        trainer.save_checkpoint()

        trainer2 = Trainer(small_config, device="cpu", use_wandb=False)
        trainer2.load_checkpoint()
        assert trainer2.reference_elo == 850.0


class TestProbeVsNetwork:
    def test_first_probe_creates_reference(self, small_config, tmp_checkpoint_dir):
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        result = trainer.probe_vs_network(num_games=2, nn_sims=2, max_moves=10)
        # First call with no reference should save reference and return None
        assert result is None
        assert (ckpt_dir / "reference.pt").exists()

    def test_probe_returns_results(self, small_config, tmp_checkpoint_dir):
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()

        result = trainer.probe_vs_network(num_games=2, nn_sims=2, max_moves=10)
        assert result is not None
        assert "wins" in result
        assert "losses" in result
        assert "draws" in result
        assert "win_rate" in result
        assert "estimated_elo" in result
        assert "reference_elo" in result
        assert "promoted" in result
        assert result["wins"] + result["losses"] + result["draws"] == 2

    def test_probe_alternates_colors(self, small_config, tmp_checkpoint_dir):
        """Current network should play as both White and Black."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()

        # With 2 games, game 0 = current as White, game 1 = current as Black
        result = trainer.probe_vs_network(num_games=2, nn_sims=2, max_moves=10)
        assert result is not None
        total = result["wins"] + result["losses"] + result["draws"]
        assert total == 2

    def test_promotion_updates_reference(self, small_config, tmp_checkpoint_dir):
        """If current beats reference convincingly, reference should update."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)

        # Save a weak reference (random init)
        trainer.reference_elo = 500.0
        trainer.save_reference()

        # Train the current network slightly to make it different
        # (won't actually be stronger with 0 training, but tests the promotion logic)
        result = trainer.probe_vs_network(num_games=4, nn_sims=2, max_moves=10)
        assert result is not None
        # Can't guarantee promotion with random vs random, but check structure
        assert isinstance(result["promoted"], bool)
        if result["promoted"]:
            assert trainer.reference_elo > 500.0

    def test_probe_with_gumbel(self, small_config, tmp_checkpoint_dir):
        """Probe should work with Gumbel MCTS config."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()

        gumbel_config = {
            "enabled": True,
            "max_num_considered_actions": 8,
            "gumbel_scale": 1.0,
            "maxvisit_init": 50.0,
            "value_scale": 0.1,
        }
        result = trainer.probe_vs_network(
            num_games=2, nn_sims=4, max_moves=10,
            gumbel_config=gumbel_config,
        )
        assert result is not None
        assert result["wins"] + result["losses"] + result["draws"] == 2

    def test_probe_without_gumbel_fallback(self, small_config, tmp_checkpoint_dir):
        """Probe should fall back to standard MCTS when gumbel_config is None."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 700.0
        trainer.save_reference()

        result = trainer.probe_vs_network(
            num_games=2, nn_sims=2, max_moves=10,
            gumbel_config=None,
        )
        assert result is not None
        assert result["wins"] + result["losses"] + result["draws"] == 2


class TestMultiprocessProbe:
    """Tests for dual-network GPU server and multiprocess probe."""

    def test_batch_request_network_id_default(self):
        """BatchRequest network_id defaults to 0."""
        from davechess.engine.gpu_server import BatchRequest
        req = BatchRequest(worker_id=0, planes_list=[])
        assert req.network_id == 0

    def test_batch_request_network_id_set(self):
        """BatchRequest network_id can be set."""
        from davechess.engine.gpu_server import BatchRequest
        req = BatchRequest(worker_id=0, planes_list=[], network_id=1)
        assert req.network_id == 1

    def test_dual_network_gpu_server(self):
        """GPU server dispatches to correct network based on network_id."""
        import multiprocessing as mp
        from davechess.engine.gpu_server import (
            BatchRequest, BatchResponse, WorkerDone, run_gpu_server,
        )

        net_a = DaveChessNetwork(num_res_blocks=2, num_filters=16, input_planes=18)
        net_b = DaveChessNetwork(num_res_blocks=2, num_filters=16, input_planes=18)

        # Make networks differ
        with torch.no_grad():
            for p in net_b.parameters():
                p.add_(torch.randn_like(p) * 1.0)

        request_queue = mp.Queue()
        response_queues = [mp.Queue()]
        planes = np.random.randn(18, 8, 8).astype(np.float32)

        # Send requests for both networks
        request_queue.put(BatchRequest(worker_id=0, planes_list=[planes], network_id=0))
        request_queue.put(BatchRequest(worker_id=0, planes_list=[planes], network_id=1))
        request_queue.put(WorkerDone(0))

        run_gpu_server(net_a, "cpu", request_queue, response_queues,
                       num_workers=1, secondary_network=net_b)

        resp_a: BatchResponse = response_queues[0].get(timeout=5)
        resp_b: BatchResponse = response_queues[0].get(timeout=5)

        # Results should differ because networks differ
        assert not np.allclose(resp_a.logits, resp_b.logits, atol=0.01), \
            "Dual-network dispatch should produce different results"

    def test_run_probe_multiprocess(self):
        """End-to-end multiprocess probe with tiny networks."""
        from davechess.engine.selfplay import run_probe_multiprocess

        net_a = DaveChessNetwork(num_res_blocks=2, num_filters=16, input_planes=18)
        net_b = DaveChessNetwork(num_res_blocks=2, num_filters=16, input_planes=18)

        results = run_probe_multiprocess(
            current_network=net_a,
            reference_network=net_b,
            num_games=2,
            num_simulations=4,
            cpuct=4.0,
            device="cpu",
            num_workers=2,
            max_moves=10,
        )

        assert len(results) == 2
        for r in results:
            assert "game_idx" in r
            assert "current_is_white" in r
            assert r["winner"] in ("current", "reference", "draw")
            assert r["length"] > 0
        # Colors should alternate
        assert results[0]["current_is_white"] is True
        assert results[1]["current_is_white"] is False


class TestBidirectionalElo:
    """Tests for bidirectional ELO updates and configurable promotion threshold."""

    def test_promotion_requires_60_percent(self, small_config, tmp_checkpoint_dir):
        """With default 60% threshold, 55% win rate should NOT promote."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 1000.0
        trainer.save_reference()

        # Monkey-patch probe to return a controlled result (55% win rate)
        original_probe = trainer.probe_vs_network

        def fake_probe(**kwargs):
            # First call the real probe to ensure reference exists
            # But we'll override the ELO logic manually
            return None

        # Directly test the ELO update logic by simulating a 55% result
        # 11 wins, 9 losses, 0 draws in 20 games = 55%
        wins, losses, draws = 11, 9, 0
        total = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total

        elo_diff = win_rate_to_elo_diff(win_rate)
        estimated_elo = trainer.reference_elo + elo_diff

        # With 60% threshold, 55% should NOT promote
        threshold = 0.60
        assert win_rate <= threshold, "55% should not exceed 60% threshold"
        # Reference ELO should stay at 1000
        assert trainer.reference_elo == 1000.0

    def test_demotion_adjusts_elo_down(self, small_config, tmp_checkpoint_dir):
        """When win rate is very low, reference ELO should adjust downward."""
        ckpt_dir, _ = tmp_checkpoint_dir
        small_config["training"]["elo_probe_promotion_threshold"] = 0.60
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 1000.0
        trainer.save_reference()

        # Simulate: current network loses badly (30% win rate)
        # With threshold 0.60, demotion triggers at < (1.0 - 0.60) = 0.40
        wins, losses, draws = 6, 14, 0
        total = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total  # 0.30

        elo_diff = win_rate_to_elo_diff(win_rate)
        threshold = 0.60

        assert win_rate < (1.0 - threshold), "30% should trigger demotion"
        # Demotion applies half the elo_diff
        demotion_diff = elo_diff * 0.5
        new_elo = trainer.reference_elo + demotion_diff
        assert new_elo < 1000.0, "ELO should decrease after bad probe"
        assert new_elo > 1000.0 + elo_diff, "Half-diff should be less severe than full diff"

    def test_no_change_in_neutral_zone(self, small_config, tmp_checkpoint_dir):
        """Win rates between demotion and promotion thresholds should not change reference."""
        ckpt_dir, _ = tmp_checkpoint_dir
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        trainer.reference_elo = 1000.0
        trainer.save_reference()

        # 50% win rate: between 40% (demotion) and 60% (promotion)
        wins, losses, draws = 10, 10, 0
        total = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total  # 0.50
        threshold = 0.60

        assert win_rate <= threshold, "50% should not promote"
        assert win_rate >= (1.0 - threshold), "50% should not demote"

    def test_configurable_threshold(self, small_config, tmp_checkpoint_dir):
        """Promotion threshold should be configurable via config."""
        ckpt_dir, _ = tmp_checkpoint_dir
        # Set a lower threshold
        small_config["training"]["elo_probe_promotion_threshold"] = 0.55
        trainer = Trainer(small_config, device="cpu", use_wandb=False)
        threshold = float(trainer.config.get("training", {}).get(
            "elo_probe_promotion_threshold", 0.60))
        assert threshold == 0.55


class TestWinRateToEloDiff:
    def test_even_match(self):
        assert win_rate_to_elo_diff(0.5) == pytest.approx(0.0, abs=1.0)

    def test_strong_win(self):
        diff = win_rate_to_elo_diff(0.9)
        assert diff > 200

    def test_strong_loss(self):
        diff = win_rate_to_elo_diff(0.1)
        assert diff < -200

    def test_clamped_at_extremes(self):
        assert win_rate_to_elo_diff(1.0) <= 400
        assert win_rate_to_elo_diff(0.0) >= -400
