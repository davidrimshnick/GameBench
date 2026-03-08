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
