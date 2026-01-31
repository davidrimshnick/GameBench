"""ELO and Glicko-2 rating calculation from game results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EloRating:
    """Simple ELO rating."""
    rating: float = 1500.0
    games: int = 0


@dataclass
class Glicko2Rating:
    """Glicko-2 rating with rating deviation and volatility."""
    mu: float = 0.0       # Rating on Glicko-2 internal scale
    phi: float = 350.0 / 173.7178  # Rating deviation
    sigma: float = 0.06   # Volatility

    @property
    def rating(self) -> float:
        """Convert to traditional rating scale."""
        return self.mu * 173.7178 + 1500.0

    @property
    def rd(self) -> float:
        """Rating deviation on traditional scale."""
        return self.phi * 173.7178

    @classmethod
    def from_rating(cls, rating: float, rd: float = 350.0,
                    sigma: float = 0.06) -> Glicko2Rating:
        return cls(
            mu=(rating - 1500.0) / 173.7178,
            phi=rd / 173.7178,
            sigma=sigma,
        )


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating: float, expected: float, actual: float,
               k: float = 32.0) -> float:
    """Update ELO rating given expected and actual score."""
    return rating + k * (actual - expected)


def calculate_elo_ratings(results: list[tuple[int, int, float]],
                          num_players: int,
                          initial_rating: float = 1500.0,
                          k: float = 32.0,
                          iterations: int = 50) -> list[float]:
    """Calculate ELO ratings from a list of game results.

    Args:
        results: List of (player_a_id, player_b_id, score) where score is
                 1.0 for A wins, 0.0 for B wins, 0.5 for draw.
        num_players: Total number of players.
        initial_rating: Starting rating for all players.
        k: K-factor for ELO updates.
        iterations: Number of passes through results for convergence.

    Returns:
        List of final ratings indexed by player ID.
    """
    ratings = [initial_rating] * num_players

    for _ in range(iterations):
        for a, b, score in results:
            expected_a = elo_expected(ratings[a], ratings[b])
            ratings[a] = elo_update(ratings[a], expected_a, score, k)
            ratings[b] = elo_update(ratings[b], 1.0 - expected_a, 1.0 - score, k)

    return ratings


# Glicko-2 implementation
TAU = 0.5  # System constant constraining volatility change


def _g(phi: float) -> float:
    """Glicko-2 g function."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Glicko-2 expected score."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_update(player: Glicko2Rating,
                   opponents: list[Glicko2Rating],
                   scores: list[float]) -> Glicko2Rating:
    """Update a single Glicko-2 rating given opponents and scores.

    Args:
        player: Current player rating.
        opponents: List of opponent ratings.
        scores: List of scores (1.0=win, 0.5=draw, 0.0=loss).

    Returns:
        Updated Glicko-2 rating.
    """
    if not opponents:
        # No games: increase uncertainty
        new_phi = math.sqrt(player.phi ** 2 + player.sigma ** 2)
        return Glicko2Rating(player.mu, new_phi, player.sigma)

    # Step 3: Compute v (estimated variance)
    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in zip(opponents, scores):
        g_val = _g(opp.phi)
        e_val = _E(player.mu, opp.mu, opp.phi)
        v_inv += g_val ** 2 * e_val * (1.0 - e_val)
        delta_sum += g_val * (score - e_val)

    v = 1.0 / v_inv if v_inv > 0 else 1e6
    delta = v * delta_sum

    # Step 4: Update volatility using Illinois algorithm
    a = math.log(player.sigma ** 2)

    def f(x):
        ex = math.exp(x)
        d2 = delta ** 2
        p2 = player.phi ** 2
        return (ex * (d2 - p2 - v - ex)) / (2.0 * (p2 + v + ex) ** 2) - \
               (x - a) / TAU ** 2

    A = a
    if delta ** 2 > player.phi ** 2 + v:
        B = math.log(delta ** 2 - player.phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    f_A = f(A)
    f_B = f(B)

    for _ in range(100):
        if abs(B - A) < 1e-6:
            break
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C)
        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A /= 2.0
        B = C
        f_B = f_C

    new_sigma = math.exp(A / 2.0)

    # Step 5: Update rating deviation
    phi_star = math.sqrt(player.phi ** 2 + new_sigma ** 2)

    # Step 6: Update rating and RD
    new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    new_mu = player.mu + new_phi ** 2 * delta_sum

    return Glicko2Rating(new_mu, new_phi, new_sigma)


def calculate_glicko2_ratings(results: list[tuple[int, int, float]],
                              num_players: int,
                              iterations: int = 20) -> list[Glicko2Rating]:
    """Calculate Glicko-2 ratings from game results.

    Args:
        results: List of (player_a_id, player_b_id, score).
        num_players: Total number of players.
        iterations: Number of rating periods to simulate.

    Returns:
        List of Glicko2Rating objects indexed by player ID.
    """
    ratings = [Glicko2Rating() for _ in range(num_players)]

    # Group results by player
    for _ in range(iterations):
        player_games: dict[int, tuple[list[Glicko2Rating], list[float]]] = {
            i: ([], []) for i in range(num_players)
        }

        for a, b, score in results:
            player_games[a][0].append(ratings[b])
            player_games[a][1].append(score)
            player_games[b][0].append(ratings[a])
            player_games[b][1].append(1.0 - score)

        new_ratings = []
        for i in range(num_players):
            opps, scores = player_games[i]
            new_ratings.append(glicko2_update(ratings[i], opps, scores))
        ratings = new_ratings

    return ratings


def elo_from_winrate(winrate: float, opponent_elo: float) -> float:
    """Estimate ELO from win rate against a known-strength opponent."""
    if winrate <= 0.0:
        return opponent_elo - 800
    if winrate >= 1.0:
        return opponent_elo + 800
    return opponent_elo - 400 * math.log10(1.0 / winrate - 1.0)
