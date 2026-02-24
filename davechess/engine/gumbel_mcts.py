"""Gumbel AlphaZero MCTS — policy improvement with fewer simulations.

Implements "Policy improvement by planning with Gumbel" (Danihelka et al., 2022).
Key differences from standard AlphaZero MCTS:
  - Sequential Halving allocates simulations to the most promising actions
  - Gumbel noise provides principled exploration (replaces Dirichlet)
  - Completed Q-values impute values for unvisited actions
  - Policy target = softmax(logits + sigma(completed_q)), not visit counts
  - Guaranteed policy improvement even with very few simulations

Each simulation does full MCTS tree traversal (PUCT select → expand →
evaluate → backpropagate) within the selected action's subtree.
Sequential Halving only controls which ROOT action gets the next sim.

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
from davechess.engine.mcts import MCTSNode

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


def _effective_considered_actions(num_actions: int,
                                  max_num_considered_actions: int,
                                  num_simulations: int) -> int:
    """Choose how many root actions Sequential Halving may consider.

    In high-branching games, a fixed small k (e.g., 16) can permanently
    hide tactical actions as soon as priors become slightly biased. When
    the simulation budget can cover more actions, expand k accordingly.
    """
    if num_actions <= 0:
        return 0

    sims_cap = min(num_actions, max(1, int(num_simulations)))
    base_k = max(1, int(max_num_considered_actions))
    return min(num_actions, max(base_k, sims_cap))


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


def _select_gumbel_action(top_k_indices, visit_counts, gumbel, logits,
                           sigma_q, considered_visit):
    """Select root action via Gumbel + Sequential Halving scoring."""
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

    return best_action


def _puct_select_leaf(subtree_root: MCTSNode, cpuct: float) -> MCTSNode:
    """PUCT tree traversal from subtree root to an unexpanded leaf."""
    node = subtree_root
    while node.is_expanded and node.children:
        node = node.select_child(cpuct)
    node.ensure_state()
    return node


def _terminal_value(state: GameState, root_player) -> float:
    """Value of a terminal state from root_player's perspective."""
    if state.winner is not None:
        return 1.0 if state.winner == root_player else -1.0
    return 0.0


def _terminal_backprop_value(node: MCTSNode) -> float:
    """Value for backpropagating a terminal node (from parent's perspective)."""
    if node.state.winner is not None:
        parent_player = (node.parent.state.current_player
                         if node.parent else node.state.current_player)
        return 1.0 if node.state.winner == parent_player else -1.0
    return 0.0


class GumbelMCTS:
    """Gumbel AlphaZero MCTS with Sequential Halving and deep tree search.

    Each simulation: Sequential Halving picks root action → full PUCT tree
    search within that action's subtree → NN evaluation at leaf →
    backpropagation through subtree.
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
        - policy_target: visit count proportions (not Gumbel improved policy)
        - root_value: value estimate at root
        """
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves")

        flip = state.current_player == Player.BLACK
        if len(legal_moves) == 1:
            idx = move_to_policy_index(legal_moves[0], flip=flip)
            policy_target = {idx: 1.0}
            return legal_moves[0], {
                "policy_target": policy_target,
                "root_value": 0.0,
            }

        # Evaluate root
        root_state = state.clone()
        policy_probs, raw_logits, root_value = self._evaluate(root_state)

        # Extract logits for legal moves only
        num_actions = len(legal_moves)
        move_indices = [move_to_policy_index(m, flip=flip) for m in legal_moves]
        logits = np.array([raw_logits[idx] for idx in move_indices], dtype=np.float64)
        logits = logits - logits.max()

        # Per-action tracking
        visit_counts = np.zeros(num_actions, dtype=np.int32)
        total_values = np.zeros(num_actions, dtype=np.float64)
        subtrees: list[Optional[MCTSNode]] = [None] * num_actions

        # Sample Gumbel noise and select top-k
        gumbel = self.gumbel_scale * np.random.gumbel(size=num_actions)
        k = _effective_considered_actions(
            num_actions=num_actions,
            max_num_considered_actions=self.max_num_considered_actions,
            num_simulations=self.num_simulations,
        )
        scores = gumbel + logits
        if k < num_actions:
            top_k_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_k_indices = np.arange(num_actions)

        seq_schedule = _get_sequence_of_considered_visits(k, self.num_simulations)

        for sim_idx in range(self.num_simulations):
            considered_visit = seq_schedule[sim_idx]
            qvalues = np.where(
                visit_counts > 0,
                total_values / np.maximum(visit_counts, 1),
                0.0
            )
            sigma_q = _qtransform(
                qvalues, visit_counts, root_value,
                self.maxvisit_init, self.value_scale,
            )

            a = _select_gumbel_action(top_k_indices, visit_counts, gumbel,
                                       logits, sigma_q, considered_visit)

            if subtrees[a] is None:
                # First visit: create child state
                child_state = root_state.clone()
                apply_move_fast(child_state, legal_moves[a])

                if child_state.done:
                    v = _terminal_value(child_state, root_state.current_player)
                    visit_counts[a] += 1
                    total_values[a] += v
                else:
                    # Evaluate and create subtree root
                    policy, _, child_val = self._evaluate(child_state)
                    subtree_root = MCTSNode(state=child_state)
                    subtree_root.expand(policy)
                    subtrees[a] = subtree_root
                    # child_val from child's perspective; negate for root
                    visit_counts[a] += 1
                    total_values[a] += -child_val
            else:
                # Re-visit: PUCT simulation in subtree
                subtree = subtrees[a]
                old_total = subtree.total_value

                leaf = _puct_select_leaf(subtree, self.cpuct)

                if leaf.state.done:
                    v = _terminal_backprop_value(leaf)
                    leaf.backpropagate(v)
                else:
                    policy, _, value = self._evaluate(leaf.state)
                    leaf.expand(policy)
                    leaf.backpropagate(-value)

                # Delta at subtree root = value from root's perspective
                delta = subtree.total_value - old_total
                visit_counts[a] += 1
                total_values[a] += delta

        # Move selection and policy target use improved policy (Gumbel's strength)
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
        improved_logits = improved_logits - improved_logits.max()
        improved_policy = np.exp(improved_logits)
        improved_policy = improved_policy / improved_policy.sum()

        policy_target = {move_indices[i]: float(improved_policy[i])
                         for i in range(num_actions)}

        if self.temperature == 0:
            selected_idx = np.argmax(improved_logits)
        else:
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


class GumbelBatchedSearch:
    """Batched Gumbel MCTS for multiple simultaneous games.

    Runs Sequential Halving across all games, batching NN evaluations
    for GPU efficiency. Each simulation does full PUCT tree search within
    the selected action's subtree (not shallow 1-2 ply evaluation).

    Args:
        evaluator: Optional callable replacing _batch_evaluate for remote GPU.
            Signature: (list[GameState]) -> list[tuple[np.ndarray, np.ndarray, float]]
            Each tuple is (policy_probs, logits, value). When provided, network
            and device are ignored and all NN calls go through the evaluator
            (used by multiprocess workers to route through GPU server).
    """

    def __init__(self, network, num_simulations: int = 50,
                 max_num_considered_actions: int = 16,
                 cpuct: float = 1.5,
                 gumbel_scale: float = 1.0,
                 maxvisit_init: float = 50.0,
                 value_scale: float = 0.1,
                 temperature: float = 1.0,
                 device: str = "cpu",
                 evaluator=None):
        self.network = network
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.cpuct = cpuct
        self.gumbel_scale = gumbel_scale
        self.maxvisit_init = maxvisit_init
        self.value_scale = value_scale
        self.temperature = temperature
        self.device = device
        self._evaluator = evaluator

    def _batch_evaluate(self, states: list[GameState]) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Batch evaluate states. Returns list of (policy, logits, value)."""
        if not states:
            return []

        # Delegate to external evaluator (used by multiprocess workers)
        if self._evaluator is not None:
            return self._evaluator(states)

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

        Uses proper deep tree search: Sequential Halving selects which root
        action gets the next simulation, then each simulation does full PUCT
        tree traversal within that action's MCTSNode subtree.

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
            flip = state.current_player == Player.BLACK
            move_indices = [move_to_policy_index(m, flip=flip) for m in legal_moves]
            logits = np.array([raw_logits[idx] for idx in move_indices], dtype=np.float64)
            logits = logits - logits.max()

            k = _effective_considered_actions(
                num_actions=num_actions,
                max_num_considered_actions=self.max_num_considered_actions,
                num_simulations=self.num_simulations,
            )
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
                "subtrees": [None] * num_actions,
                "temperature": temperatures[i],
            })

        # Run simulations in lockstep across all games
        for sim_idx in range(self.num_simulations):
            # Phase A: select actions and traverse subtrees to find leaves
            new_roots = []     # (gi, a, child_state) — first visit
            tree_leaves = []   # (gi, a, old_total, leaf_node) — PUCT leaf

            for gi in range(n):
                g = games[gi]
                if g is None:
                    continue

                # Sequential Halving: select root action
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
                a = _select_gumbel_action(
                    g["top_k"], g["visit_counts"], g["gumbel"],
                    g["logits"], sigma_q, considered_visit,
                )

                if g["subtrees"][a] is None:
                    # First visit: create child state
                    child_state = g["state"].clone()
                    apply_move_fast(child_state, g["legal_moves"][a])

                    if child_state.done:
                        v = _terminal_value(child_state, g["state"].current_player)
                        g["visit_counts"][a] += 1
                        g["total_values"][a] += v
                    else:
                        new_roots.append((gi, a, child_state))
                else:
                    # Re-visit: PUCT simulation in existing subtree
                    subtree = g["subtrees"][a]
                    old_total = subtree.total_value

                    leaf = _puct_select_leaf(subtree, self.cpuct)

                    if leaf.state.done:
                        v = _terminal_backprop_value(leaf)
                        leaf.backpropagate(v)
                        delta = subtree.total_value - old_total
                        g["visit_counts"][a] += 1
                        g["total_values"][a] += delta
                    elif leaf.is_expanded and not leaf.children:
                        # Stalemate: no legal moves
                        leaf.backpropagate(0.0)
                        delta = subtree.total_value - old_total
                        g["visit_counts"][a] += 1
                        g["total_values"][a] += delta
                    else:
                        tree_leaves.append((gi, a, old_total, leaf))

            # Phase B: batch evaluate all pending states
            eval_states = []
            for gi, a, child_state in new_roots:
                eval_states.append(child_state)
            n_new = len(new_roots)
            for gi, a, old_total, leaf in tree_leaves:
                eval_states.append(leaf.state)

            if eval_states:
                all_results = self._batch_evaluate(eval_states)

                # Process new subtree roots (first visits)
                for idx, (gi, a, child_state) in enumerate(new_roots):
                    policy, _, value = all_results[idx]
                    g = games[gi]
                    subtree_root = MCTSNode(state=child_state)
                    subtree_root.expand(policy)
                    g["subtrees"][a] = subtree_root
                    # value from child's perspective → negate for root
                    g["visit_counts"][a] += 1
                    g["total_values"][a] += -value

                # Process subtree PUCT leaves (re-visits)
                for idx, (gi, a, old_total, leaf) in enumerate(tree_leaves):
                    policy, _, value = all_results[n_new + idx]
                    g = games[gi]
                    leaf.expand(policy)
                    leaf.backpropagate(-value)
                    delta = g["subtrees"][a].total_value - old_total
                    g["visit_counts"][a] += 1
                    g["total_values"][a] += delta

        # Collect results
        results = []
        for gi in range(n):
            g = games[gi]
            if g is None:
                results.append((None, {"policy_target": {}, "root_value": 0.0}))
                continue

            # Move selection and policy target use improved policy
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

            if g["temperature"] == 0 or g["temperature"] < 0.2:
                selected_idx = np.argmax(improved_logits)
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
