"""Gumbel AlphaZero MCTS — policy improvement with fewer simulations.

Implements "Policy improvement by planning with Gumbel" (Danihelka et al., 2022).
Key differences from standard AlphaZero MCTS:
  - Sequential Halving allocates simulations to the most promising actions
  - Gumbel noise provides principled exploration (replaces Dirichlet)
  - Completed Q-values impute values for unvisited actions
  - Policy target = softmax(logits + sigma(completed_q)), not visit counts
  - Guaranteed policy improvement even with very few simulations

Reference: https://openreview.net/forum?id=bERaNdoegnO
Reference impl: https://github.com/google-deepmind/mctx
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move_fast
from davechess.engine.network import (
    state_to_planes, move_to_policy_index, POLICY_SIZE,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _get_sequence_of_considered_visits(max_k: int, num_simulations: int) -> list[int]:
    """Compute the Sequential Halving visit schedule.

    Returns a list of length num_simulations. Each entry is the target visit
    count for the "considered" set at that simulation step. Actions whose visit
    count equals the considered_visit are eligible for selection.

    Algorithm from Appendix A of the Gumbel MuZero paper.
    """
    if max_k <= 1:
        return list(range(num_simulations))

    log2k = max(1, int(math.ceil(math.log2(max_k))))
    sequence: list[int] = []
    # visits[i] = how many visits action at rank i has received
    visits = [0] * max_k
    num_considered = max_k

    while len(sequence) < num_simulations:
        num_extra = max(1, num_simulations // (log2k * num_considered))
        for _ in range(num_extra):
            for i in range(num_considered):
                sequence.append(visits[i])
                if len(sequence) >= num_simulations:
                    break
            if len(sequence) >= num_simulations:
                break
        for i in range(num_considered):
            visits[i] += 1
        num_considered = max(2, num_considered // 2)

    return sequence[:num_simulations]


def _qtransform(qvalues: np.ndarray, visit_counts: np.ndarray,
                value_score: float,
                maxvisit_init: float = 50.0,
                value_scale: float = 0.1,
                epsilon: float = 1e-8) -> np.ndarray:
    """Transform Q-values into the sigma(q) scale for combining with logits.

    1. Complete Q-values: use actual Q where visited, parent value where not.
    2. Normalize to [0, 1] using min/max across all actions.
    3. Scale by (maxvisit_init + max_visits) * value_scale.

    This puts Q-values on a comparable scale to logits.
    """
    # Complete Q-values: impute unvisited with parent's value
    completed_q = np.where(visit_counts > 0, qvalues, value_score)

    # Normalize to [0, 1]
    q_min = completed_q.min()
    q_max = completed_q.max()
    q_range = max(q_max - q_min, epsilon)
    normalized_q = (completed_q - q_min) / q_range

    # Scale
    max_visits = max(visit_counts.max(), 1)
    scale = (maxvisit_init + max_visits) * value_scale

    return scale * normalized_q


class GumbelMCTS:
    """Gumbel AlphaZero MCTS with Sequential Halving.

    Args:
        network: Neural network for evaluation (policy logits + value).
        num_simulations: Total simulation budget per move.
        max_num_considered_actions: Top-k actions to consider (k in paper).
            Should be <= num_simulations. Paper uses 16 for Go.
        cpuct: PUCT exploration constant (used for non-root tree traversal).
        gumbel_scale: Scale for Gumbel noise (default 1.0).
        maxvisit_init: Q-transform parameter (default 50.0).
        value_scale: Q-transform parameter (default 0.1).
        temperature: Temperature for action selection (1.0 = sample, 0 = greedy).
        device: Torch device string.
    """

    def __init__(self, network, num_simulations: int = 50,
                 max_num_considered_actions: int = 16,
                 cpuct: float = 1.5,
                 gumbel_scale: float = 1.0,
                 maxvisit_init: float = 50.0,
                 value_scale: float = 0.1,
                 temperature: float = 1.0,
                 device: str = "cpu"):
        self.network = network
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.cpuct = cpuct
        self.gumbel_scale = gumbel_scale
        self.maxvisit_init = maxvisit_init
        self.value_scale = value_scale
        self.temperature = temperature
        self.device = device

    def _evaluate(self, state: GameState) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate state with neural network.

        Returns (policy_probs, policy_logits, value).
        """
        if not HAS_TORCH or self.network is None:
            logits = np.zeros(POLICY_SIZE, dtype=np.float32)
            return np.ones(POLICY_SIZE) / POLICY_SIZE, logits, 0.0

        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits_t, value_t = self.network(x)

        logits = logits_t[0].cpu().numpy()
        policy = np.exp(logits - logits.max())
        policy = policy / policy.sum()
        return policy, logits, value_t.item()

    def search(self, state: GameState) -> tuple[Move, dict]:
        """Run Gumbel MCTS search from state.

        Returns (selected_move, info_dict) where info_dict contains:
        - policy_target: improved policy from completed Q-values
        - root_value: value estimate at root
        """
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves")

        if len(legal_moves) == 1:
            # Only one legal move — skip search
            idx = move_to_policy_index(legal_moves[0])
            policy_target = {idx: 1.0}
            return legal_moves[0], {
                "policy_target": policy_target,
                "root_value": 0.0,
            }

        # Evaluate root
        root_state = state.clone()
        policy_probs, raw_logits, root_value = self._evaluate(root_state)

        # Extract logits and priors for legal moves only
        num_actions = len(legal_moves)
        move_indices = [move_to_policy_index(m) for m in legal_moves]
        logits = np.array([raw_logits[idx] for idx in move_indices], dtype=np.float64)
        # Normalize logits (subtract max for numerical stability)
        logits = logits - logits.max()

        # Initialize per-action tracking
        visit_counts = np.zeros(num_actions, dtype=np.int32)
        total_values = np.zeros(num_actions, dtype=np.float64)
        # Children states (lazily created)
        child_states: list[Optional[GameState]] = [None] * num_actions
        # Children are expanded (have been evaluated by NN)
        child_expanded: list[bool] = [False] * num_actions
        # Child policies and values (filled on expansion)
        child_policies: list[Optional[np.ndarray]] = [None] * num_actions
        child_values: list[float] = [0.0] * num_actions

        # Sample Gumbel noise for root actions
        gumbel = self.gumbel_scale * np.random.gumbel(size=num_actions)

        # Determine k (number of considered actions)
        k = min(self.max_num_considered_actions, num_actions)

        # Get Sequential Halving schedule
        seq_schedule = _get_sequence_of_considered_visits(k, self.num_simulations)

        # Phase 1: Select top-k actions using Gumbel-Top-k
        # Score = gumbel + logits (no Q-values yet, all unvisited)
        scores = gumbel + logits
        # Get indices of top-k actions
        if k < num_actions:
            top_k_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_k_indices = np.arange(num_actions)

        # Phase 2: Sequential Halving — allocate simulations
        for sim_idx in range(self.num_simulations):
            considered_visit = seq_schedule[sim_idx]

            # Compute completed Q-values for scoring
            qvalues = np.where(
                visit_counts > 0,
                total_values / np.maximum(visit_counts, 1),
                0.0
            )
            sigma_q = _qtransform(
                qvalues, visit_counts, root_value,
                self.maxvisit_init, self.value_scale,
            )

            # Score considered actions: gumbel + logits + sigma(q)
            # Only consider actions in top_k whose visit_count == considered_visit
            best_score = -np.inf
            best_action = -1
            for a in top_k_indices:
                if visit_counts[a] == considered_visit:
                    s = gumbel[a] + logits[a] + sigma_q[a]
                    if s > best_score:
                        best_score = s
                        best_action = a

            if best_action == -1:
                # No action matches — pick least-visited among top-k
                min_v = visit_counts[top_k_indices].min()
                for a in top_k_indices:
                    if visit_counts[a] == min_v:
                        s = gumbel[a] + logits[a] + sigma_q[a]
                        if s > best_score:
                            best_score = s
                            best_action = a

            if best_action == -1:
                best_action = top_k_indices[0]

            # Simulate: expand child if needed, then get value
            a = best_action
            if child_states[a] is None:
                child_states[a] = root_state.clone()
                apply_move_fast(child_states[a], legal_moves[a])

            if child_states[a].done:
                # Terminal node
                if child_states[a].winner is not None:
                    # Value from root's perspective
                    v = 1.0 if child_states[a].winner == root_state.current_player else -1.0
                else:
                    v = 0.0  # draw
            elif not child_expanded[a]:
                # First visit — evaluate with NN
                _, child_logits, child_val = self._evaluate(child_states[a])
                child_expanded[a] = True
                child_policies[a] = child_logits
                child_values[a] = child_val
                # Value is from child's current_player perspective
                # We need it from root's perspective (opponent), so negate
                v = -child_val
            else:
                # Already expanded — do a deeper PUCT search from this child
                v = self._subtree_search(child_states[a], child_policies[a],
                                         child_values[a])
                v = -v  # Negate to root's perspective

            # Update stats
            visit_counts[a] += 1
            total_values[a] += v

        # Compute improved policy target: softmax(logits + sigma(completed_q))
        qvalues = np.where(
            visit_counts > 0,
            total_values / np.maximum(visit_counts, 1),
            0.0
        )
        sigma_q = _qtransform(
            qvalues, visit_counts, root_value,
            self.maxvisit_init, self.value_scale,
        )
        improved_logits = logits + sigma_q
        # Softmax
        improved_logits = improved_logits - improved_logits.max()
        improved_policy = np.exp(improved_logits)
        improved_policy = improved_policy / improved_policy.sum()

        # Build policy target dict
        policy_target = {move_indices[i]: float(improved_policy[i])
                         for i in range(num_actions)}

        # Select action to play
        # Final action: argmax(gumbel + logits + sigma(completed_q))
        final_scores = gumbel + logits + sigma_q
        if self.temperature == 0:
            selected_idx = np.argmax(final_scores)
        else:
            # Use improved policy for sampling
            selected_idx = np.random.choice(num_actions, p=improved_policy)

        root_q = np.sum(total_values) / max(np.sum(visit_counts), 1)

        return legal_moves[selected_idx], {
            "policy_target": policy_target,
            "root_value": float(root_q),
            "visit_counts": {i: int(visit_counts[i]) for i in range(num_actions)},
        }

    def get_move(self, state: GameState, add_noise: bool = True) -> tuple[Move, dict]:
        """Drop-in compatible interface with MCTS.get_move()."""
        return self.search(state)

    def _subtree_search(self, state: GameState, parent_logits: np.ndarray,
                        parent_value: float) -> float:
        """Do a single PUCT-style deeper search from an already-expanded child.

        This provides better Q-value estimates for repeatedly-visited children.
        Returns value from the state's current_player perspective.
        """
        legal_moves = generate_legal_moves(state)
        if not legal_moves or state.done:
            if state.winner is not None:
                return 1.0 if state.winner == state.current_player else -1.0
            return 0.0

        # Use the child's policy to pick the most promising move via PUCT
        move_indices = [move_to_policy_index(m) for m in legal_moves]
        priors = np.array([max(parent_logits[idx], 0.0) for idx in move_indices])
        prior_sum = priors.sum()
        if prior_sum > 0:
            priors = priors / prior_sum
        else:
            priors = np.ones(len(legal_moves)) / len(legal_moves)

        # Pick the highest-prior action (simple one-step lookahead)
        best = np.argmax(priors)
        child_state = state.clone()
        apply_move_fast(child_state, legal_moves[best])

        if child_state.done:
            if child_state.winner is not None:
                return -1.0 if child_state.winner == state.current_player else 1.0
            return 0.0

        # Evaluate the grandchild with NN
        _, _, value = self._evaluate(child_state)
        # value is from grandchild's current_player perspective
        # grandchild's current_player = state.current_player (same as parent of this fn)
        # so we negate to get value from state.current_player's perspective... wait
        # Actually: child_state has current_player = opponent of state.current_player
        # After apply_move_fast, it's the opponent's turn
        # value is from child_state.current_player (opponent) perspective
        # We want it from state.current_player perspective, so negate
        return -value


class GumbelBatchedSearch:
    """Batched Gumbel MCTS for multiple simultaneous games.

    Runs Sequential Halving across all games, batching NN evaluations
    for GPU efficiency.
    """

    def __init__(self, network, num_simulations: int = 50,
                 max_num_considered_actions: int = 16,
                 gumbel_scale: float = 1.0,
                 maxvisit_init: float = 50.0,
                 value_scale: float = 0.1,
                 temperature: float = 1.0,
                 device: str = "cpu"):
        self.network = network
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.gumbel_scale = gumbel_scale
        self.maxvisit_init = maxvisit_init
        self.value_scale = value_scale
        self.temperature = temperature
        self.device = device

    def _batch_evaluate(self, states: list[GameState]) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Batch evaluate states. Returns list of (policy, logits, value)."""
        if not states:
            return []

        if not HAS_TORCH or self.network is None:
            return [(np.ones(POLICY_SIZE) / POLICY_SIZE,
                     np.zeros(POLICY_SIZE, dtype=np.float32), 0.0)
                    for _ in states]

        planes_batch = np.stack([state_to_planes(s) for s in states])
        x = torch.from_numpy(planes_batch).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits_t, values_t = self.network(x)

        logits_np = logits_t.cpu().numpy()
        values_np = values_t.cpu().numpy().flatten()

        results = []
        for i in range(len(states)):
            logits = logits_np[i]
            policy = np.exp(logits - logits.max())
            policy = policy / policy.sum()
            results.append((policy, logits, float(values_np[i])))
        return results

    def batched_search(self, states: list[GameState],
                       temperatures: list[float]) -> list[tuple[Move, dict]]:
        """Run Gumbel MCTS on multiple states with batched NN evaluation.

        Args:
            states: List of game states to search from.
            temperatures: Per-game temperature for action selection.

        Returns:
            List of (move, info_dict) per game.
        """
        n = len(states)
        if n == 0:
            return []

        # Batch evaluate all root states
        root_results = self._batch_evaluate(states)

        # Per-game search state
        games = []
        for i in range(n):
            state = states[i]
            legal_moves = generate_legal_moves(state)
            if not legal_moves:
                games.append(None)
                continue

            policy, raw_logits, root_value = root_results[i]
            num_actions = len(legal_moves)
            move_indices = [move_to_policy_index(m) for m in legal_moves]
            logits = np.array([raw_logits[idx] for idx in move_indices], dtype=np.float64)
            logits = logits - logits.max()

            k = min(self.max_num_considered_actions, num_actions)
            gumbel = self.gumbel_scale * np.random.gumbel(size=num_actions)
            scores = gumbel + logits
            if k < num_actions:
                top_k = np.argpartition(scores, -k)[-k:]
            else:
                top_k = np.arange(num_actions)

            seq_schedule = _get_sequence_of_considered_visits(k, self.num_simulations)

            games.append({
                "state": state.clone(),
                "legal_moves": legal_moves,
                "move_indices": move_indices,
                "num_actions": num_actions,
                "logits": logits,
                "root_value": root_value,
                "gumbel": gumbel,
                "top_k": top_k,
                "seq_schedule": seq_schedule,
                "visit_counts": np.zeros(num_actions, dtype=np.int32),
                "total_values": np.zeros(num_actions, dtype=np.float64),
                "child_states": [None] * num_actions,
                "child_expanded": [False] * num_actions,
                "child_values": [0.0] * num_actions,
                "temperature": temperatures[i],
            })

        # Run simulations in lockstep across all games
        for sim_idx in range(self.num_simulations):
            # For each game, select action to simulate
            to_evaluate: list[tuple[int, int, GameState]] = []  # (game_idx, action_idx, state)

            for gi in range(n):
                g = games[gi]
                if g is None:
                    continue

                considered_visit = g["seq_schedule"][sim_idx]
                qvalues = np.where(
                    g["visit_counts"] > 0,
                    g["total_values"] / np.maximum(g["visit_counts"], 1),
                    0.0
                )
                sigma_q = _qtransform(
                    qvalues, g["visit_counts"], g["root_value"],
                    self.maxvisit_init, self.value_scale,
                )

                # Select action
                best_score = -np.inf
                best_action = -1
                for a in g["top_k"]:
                    if g["visit_counts"][a] == considered_visit:
                        s = g["gumbel"][a] + g["logits"][a] + sigma_q[a]
                        if s > best_score:
                            best_score = s
                            best_action = a

                if best_action == -1:
                    min_v = g["visit_counts"][g["top_k"]].min()
                    for a in g["top_k"]:
                        if g["visit_counts"][a] == min_v:
                            s = g["gumbel"][a] + g["logits"][a] + sigma_q[a]
                            if s > best_score:
                                best_score = s
                                best_action = a

                if best_action == -1:
                    best_action = g["top_k"][0]

                a = best_action

                # Create child state if needed
                if g["child_states"][a] is None:
                    g["child_states"][a] = g["state"].clone()
                    apply_move_fast(g["child_states"][a], g["legal_moves"][a])

                child_state = g["child_states"][a]

                if child_state.done:
                    # Terminal — update immediately
                    if child_state.winner is not None:
                        v = 1.0 if child_state.winner == g["state"].current_player else -1.0
                    else:
                        v = 0.0
                    g["visit_counts"][a] += 1
                    g["total_values"][a] += v
                elif not g["child_expanded"][a]:
                    # Need NN evaluation
                    to_evaluate.append((gi, a, child_state))
                else:
                    # Already expanded — use stored value (simple approach)
                    v = -g["child_values"][a]
                    g["visit_counts"][a] += 1
                    g["total_values"][a] += v

            # Batch evaluate all pending states
            if to_evaluate:
                eval_states = [s for _, _, s in to_evaluate]
                eval_results = self._batch_evaluate(eval_states)

                for (gi, a, _), (_, child_logits, child_val) in zip(to_evaluate, eval_results):
                    g = games[gi]
                    g["child_expanded"][a] = True
                    g["child_values"][a] = child_val
                    v = -child_val  # Negate to root's perspective
                    g["visit_counts"][a] += 1
                    g["total_values"][a] += v

        # Collect results
        results = []
        for gi in range(n):
            g = games[gi]
            if g is None:
                results.append((None, {"policy_target": {}, "root_value": 0.0}))
                continue

            # Compute improved policy
            qvalues = np.where(
                g["visit_counts"] > 0,
                g["total_values"] / np.maximum(g["visit_counts"], 1),
                0.0
            )
            sigma_q = _qtransform(
                qvalues, g["visit_counts"], g["root_value"],
                self.maxvisit_init, self.value_scale,
            )
            improved_logits = g["logits"] + sigma_q
            improved_logits = improved_logits - improved_logits.max()
            improved_policy = np.exp(improved_logits)
            improved_policy = improved_policy / improved_policy.sum()

            policy_target = {g["move_indices"][i]: float(improved_policy[i])
                             for i in range(g["num_actions"])}

            # Select action
            final_scores = g["gumbel"] + g["logits"] + sigma_q
            if g["temperature"] == 0 or g["temperature"] < 0.2:
                selected_idx = np.argmax(final_scores)
            else:
                selected_idx = np.random.choice(g["num_actions"], p=improved_policy)

            root_q = (np.sum(g["total_values"]) /
                      max(np.sum(g["visit_counts"]), 1))

            results.append((g["legal_moves"][selected_idx], {
                "policy_target": policy_target,
                "root_value": float(root_q),
                "visit_counts": {i: int(g["visit_counts"][i])
                                 for i in range(g["num_actions"])},
            }))

        return results
