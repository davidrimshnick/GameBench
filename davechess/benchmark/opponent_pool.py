"""Calibrated opponent pool with ELO-to-agent mapping."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Optional

from davechess.data.generator import Agent, RandomAgent, MCTSAgent, MCTSLiteAgent

logger = logging.getLogger("davechess.benchmark")


@dataclass
class CalibratedLevel:
    """A calibrated opponent level from a tournament."""
    sim_count: int
    measured_elo: float
    elo_rd: float = 50.0  # Rating deviation (uncertainty)


class OpponentPool:
    """Creates opponents at arbitrary ELO by interpolating between calibrated levels.

    Uses log-space interpolation on simulation count, since ELO scales
    roughly logarithmically with MCTS sim count.
    """

    def __init__(self, network, device: str,
                 calibration: list[CalibratedLevel]):
        self.network = network
        self.device = device
        self.calibration = sorted(calibration, key=lambda c: c.measured_elo)

        if not self.calibration:
            raise ValueError("Calibration must have at least one level")

    def get_opponent(self, target_elo: int) -> Agent:
        """Create an agent at approximately the target ELO.

        Interpolates in log-sim-space between calibrated levels.
        """
        # Clamp to calibrated range
        min_elo = self.calibration[0].measured_elo
        max_elo = self.calibration[-1].measured_elo
        clamped_elo = max(min_elo, min(max_elo, target_elo))

        # Find the bracketing levels
        lower = self.calibration[0]
        upper = self.calibration[-1]
        for i in range(len(self.calibration) - 1):
            if self.calibration[i].measured_elo <= clamped_elo <= self.calibration[i + 1].measured_elo:
                lower = self.calibration[i]
                upper = self.calibration[i + 1]
                break

        # Special case: random agent at the bottom
        if lower.sim_count == 0 and upper.sim_count == 0:
            return RandomAgent()
        if lower.sim_count == 0:
            if clamped_elo <= lower.measured_elo:
                return RandomAgent()
            # Interpolate between 0 (random) and upper
            # Use linear in sim-space since log(0) is undefined
            elo_range = upper.measured_elo - lower.measured_elo
            if elo_range <= 0:
                return self._make_mcts_agent(upper.sim_count)
            fraction = (clamped_elo - lower.measured_elo) / elo_range
            sims = max(1, round(upper.sim_count * fraction))
            return self._make_mcts_agent(sims)

        # Log-space interpolation
        elo_range = upper.measured_elo - lower.measured_elo
        if elo_range <= 0:
            return self._make_mcts_agent(lower.sim_count)

        fraction = (clamped_elo - lower.measured_elo) / elo_range
        log_lower = math.log(max(1, lower.sim_count))
        log_upper = math.log(max(1, upper.sim_count))
        log_sims = log_lower + fraction * (log_upper - log_lower)
        sims = max(1, round(math.exp(log_sims)))

        return self._make_mcts_agent(sims)

    def _make_mcts_agent(self, sim_count: int) -> Agent:
        """Create an MCTS agent with the given simulation count."""
        if self.network is not None:
            return MCTSAgent(self.network, self.device,
                             num_simulations=sim_count, temperature=0.1)
        else:
            return MCTSLiteAgent(num_simulations=sim_count)

    @property
    def min_elo(self) -> float:
        return self.calibration[0].measured_elo

    @property
    def max_elo(self) -> float:
        return self.calibration[-1].measured_elo

    @classmethod
    def from_calibration_file(cls, path: str, network, device: str) -> OpponentPool:
        """Load calibration data from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        levels = [
            CalibratedLevel(
                sim_count=lvl["sim_count"],
                measured_elo=lvl["elo"],
                elo_rd=lvl.get("rd", 50.0),
            )
            for lvl in data["levels"]
        ]
        return cls(network=network, device=device, calibration=levels)

    def save_calibration(self, path: str) -> None:
        """Save calibration data to a JSON file."""
        data = {
            "levels": [
                {
                    "sim_count": lvl.sim_count,
                    "elo": lvl.measured_elo,
                    "rd": lvl.elo_rd,
                }
                for lvl in self.calibration
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
