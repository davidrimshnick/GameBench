"""Compute ELO from game results and GameBench Score (AUC)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from davechess.data.elo import elo_from_winrate


def compute_llm_elo(results_by_level: dict[int, list[float]],
                    level_elos: list[float]) -> float:
    """Compute LLM's ELO from game results against calibrated opponents.

    Args:
        results_by_level: Dict mapping level index to list of scores (0/0.5/1).
        level_elos: Known ELO for each calibrated level.

    Returns:
        Estimated ELO for the LLM.
    """
    if not results_by_level:
        return level_elos[0] if level_elos else 400.0

    # Weight by number of games at each level
    total_weight = 0
    weighted_elo = 0

    for level_idx, scores in results_by_level.items():
        if not scores or level_idx >= len(level_elos):
            continue
        winrate = sum(scores) / len(scores)
        estimated = elo_from_winrate(winrate, level_elos[level_idx])
        weight = len(scores)
        weighted_elo += estimated * weight
        total_weight += weight

    if total_weight == 0:
        return level_elos[0]

    return weighted_elo / total_weight


def compute_learning_curve(results_by_n: dict[int, float]) -> list[tuple[int, float]]:
    """Convert raw results to a learning curve.

    Args:
        results_by_n: Dict mapping N (number of examples) to LLM ELO.

    Returns:
        Sorted list of (N, ELO) points.
    """
    curve = sorted(results_by_n.items())
    return curve


def compute_auc(curve: list[tuple[int, float]], max_n: int = 500) -> float:
    """Compute area under the learning curve using trapezoidal rule.

    Args:
        curve: List of (N, ELO) sorted by N.
        max_n: Maximum N value for integration.

    Returns:
        AUC value.
    """
    if len(curve) < 2:
        if curve:
            return curve[0][1] * max_n
        return 0.0

    # Clip to max_n
    curve = [(n, e) for n, e in curve if n <= max_n]

    auc = 0.0
    for i in range(len(curve) - 1):
        n1, e1 = curve[i]
        n2, e2 = curve[i + 1]
        auc += (n2 - n1) * (e1 + e2) / 2.0

    return auc


def compute_gamebench_score(curve: list[tuple[int, float]],
                            random_elo: float = 400.0,
                            max_elo: float = 2700.0,
                            max_n: int = 500) -> float:
    """Compute the GameBench Score (0-100).

    Score = (observed_AUC - random_AUC) / (perfect_AUC - random_AUC) * 100

    Args:
        curve: Learning curve as list of (N, ELO).
        random_elo: ELO of random play (baseline).
        max_elo: Maximum calibrated ELO (ceiling).
        max_n: Maximum N for integration.

    Returns:
        GameBench Score between 0 and 100.
    """
    observed_auc = compute_auc(curve, max_n)

    # Random AUC: flat line at random_elo
    random_auc = random_elo * max_n

    # Perfect AUC: flat line at max_elo
    perfect_auc = max_elo * max_n

    if perfect_auc <= random_auc:
        return 0.0

    score = (observed_auc - random_auc) / (perfect_auc - random_auc) * 100.0
    return max(0.0, min(100.0, score))


def compute_budget_learning_curve(
    results_by_budget: dict[int, float]
) -> list[tuple[int, float]]:
    """Convert {token_budget: elo} to a sorted learning curve.

    Args:
        results_by_budget: Dict mapping token budget to achieved ELO.

    Returns:
        Sorted list of (budget, ELO) points.
    """
    return sorted(results_by_budget.items())


def compute_agentic_score(curve: list[tuple[int, float]],
                           random_elo: float = 400.0,
                           max_elo: float = 2700.0,
                           max_budget: int = 10_000_000) -> float:
    """Compute agentic GameBench score using log-scale AUC.

    Uses log-scale for the x-axis since budgets span orders of magnitude
    (100K to 10M). Each order of magnitude is weighted equally.

    Score = (observed_AUC - random_AUC) / (perfect_AUC - random_AUC) * 100

    Args:
        curve: List of (budget, ELO) sorted by budget.
        random_elo: ELO of random play (baseline).
        max_elo: Maximum calibrated ELO (ceiling).
        max_budget: Maximum budget for integration.

    Returns:
        Score between 0 and 100.
    """
    if len(curve) < 2:
        if curve:
            return max(0.0, min(100.0,
                (curve[0][1] - random_elo) / (max_elo - random_elo) * 100.0
            ))
        return 0.0

    # Clip to max_budget
    curve = [(b, e) for b, e in curve if b <= max_budget]
    if len(curve) < 2:
        return 0.0

    # Log-scale AUC using trapezoidal rule on log(budget)
    auc = 0.0
    for i in range(len(curve) - 1):
        b1, e1 = curve[i]
        b2, e2 = curve[i + 1]
        if b1 <= 0 or b2 <= 0:
            continue
        log_width = math.log(b2) - math.log(b1)
        auc += log_width * (e1 + e2) / 2.0

    # Total log range for normalization
    min_budget = curve[0][0]
    capped_max = min(curve[-1][0], max_budget)
    if min_budget <= 0 or capped_max <= min_budget:
        return 0.0

    log_range = math.log(capped_max) - math.log(min_budget)
    random_auc = random_elo * log_range
    perfect_auc = max_elo * log_range

    if perfect_auc <= random_auc:
        return 0.0

    score = (auc - random_auc) / (perfect_auc - random_auc) * 100.0
    return max(0.0, min(100.0, score))


def format_results(model_name: str, curve: list[tuple[int, float]],
                   score: float) -> str:
    """Format benchmark results as a readable string."""
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"GameBench Score: {score:.1f}/100")
    lines.append("")
    lines.append("Learning Curve:")
    lines.append(f"  {'N':>6s}  {'ELO':>8s}")
    lines.append(f"  {'---':>6s}  {'---':>8s}")
    for n, elo in curve:
        lines.append(f"  {n:>6d}  {elo:>8.1f}")
    return "\n".join(lines)
