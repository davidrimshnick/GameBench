"""MCTS with PUCT and neural network evaluation for AlphaZero.

Uses lazy child expansion: child states are only materialized when first visited,
avoiding ~30 unnecessary state clones per node expansion.
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


class MCTSNode:
    """Node in the MCTS search tree with PUCT and lazy state creation."""

    __slots__ = [
        "state", "parent", "move", "children", "visit_count",
        "total_value", "prior", "is_expanded",
    ]

    def __init__(self, state: Optional[GameState] = None,
                 parent: Optional[MCTSNode] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.state = state  # None until first visit (lazy)
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, cpuct: float) -> float:
        parent_visits = max(1, self.parent.visit_count if self.parent else 1)
        u = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self, cpuct: float) -> MCTSNode:
        return max(self.children, key=lambda c: c.puct_score(cpuct))

    def ensure_state(self):
        """Lazily create state by cloning parent and applying move."""
        if self.state is None and self.parent is not None:
            self.state = self.parent.state.clone()
            apply_move_fast(self.state, self.move)

    def expand(self, policy: np.ndarray):
        """Expand node: create children with moves and priors but NO states.

        States are materialized lazily on first visit via ensure_state().
        """
        if self.is_expanded or self.state.done:
            return

        legal_moves = generate_legal_moves(self.state)
        if not legal_moves:
            self.is_expanded = True
            return

        move_indices = [move_to_policy_index(m) for m in legal_moves]
        priors = np.array([policy[idx] for idx in move_indices])
        prior_sum = priors.sum()
        if prior_sum > 0:
            priors = priors / prior_sum
        else:
            priors = np.ones(len(legal_moves)) / len(legal_moves)

        for move, prior in zip(legal_moves, priors):
            child = MCTSNode(state=None, parent=self, move=move, prior=prior)
            self.children.append(child)

        self.is_expanded = True

    def backpropagate(self, value: float):
        """Backpropagate a value up the tree.

        Value is from the perspective of the node's parent's current_player.
        We flip sign at each level.
        """
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.total_value += v
            v = -v
            node = node.parent


class BatchedEvaluator:
    """Collects leaf states from multiple MCTS trees and evaluates in one batch."""

    def __init__(self, network, device: str = "cpu"):
        self.network = network
        self.device = device
        self._pending: list[tuple[MCTSNode, np.ndarray]] = []

    def submit(self, node: MCTSNode, planes: np.ndarray):
        """Queue a leaf node for batched evaluation."""
        self._pending.append((node, planes))

    def evaluate_batch(self) -> list[tuple[np.ndarray, float]]:
        """Evaluate all pending leaves in a single forward pass.

        Returns list of (policy, value) tuples in submission order.
        """
        if not self._pending:
            return []

        if not HAS_TORCH or self.network is None:
            results = [(np.ones(POLICY_SIZE) / POLICY_SIZE, 0.0)
                       for _ in self._pending]
            self._pending.clear()
            return results

        planes_batch = np.stack([p for _, p in self._pending])
        x = torch.from_numpy(planes_batch).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits, values = self.network(x)

        policies = torch.softmax(logits, dim=1).cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        results = [(policies[i], float(values_np[i]))
                    for i in range(len(self._pending))]
        self._pending.clear()
        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class MCTS:
    """Monte Carlo Tree Search with neural network evaluation."""

    def __init__(self, network, num_simulations: int = 200, cpuct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0, device: str = "cpu"):
        self.network = network
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.device = device

    def _evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Evaluate state with neural network."""
        if not HAS_TORCH or self.network is None:
            return np.ones(POLICY_SIZE) / POLICY_SIZE, 0.0

        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(x)

        policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        return policy, value.item()

    def search(self, state: GameState, add_noise: bool = True) -> MCTSNode:
        """Run MCTS search from state. Returns root node."""
        root = MCTSNode(state=state.clone())

        # Evaluate and expand root
        policy, value = self._evaluate(root.state)
        root.expand(policy)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                              self.dirichlet_epsilon * n

        for _ in range(self.num_simulations):
            node = root

            # Select: traverse tree using PUCT
            while node.is_expanded and node.children:
                node = node.select_child(self.cpuct)

            # Lazy state materialization
            node.ensure_state()

            # Terminal node
            if node.state.done:
                # Value from parent's perspective
                if node.state.winner is not None:
                    parent_player = node.parent.state.current_player if node.parent else node.state.current_player
                    v = 1.0 if node.state.winner == parent_player else -1.0
                else:
                    v = 0.0  # draw
                node.backpropagate(v)
                continue

            # Evaluate with NN and expand
            policy, value = self._evaluate(node.state)
            node.expand(policy)

            # value is from current_player's perspective
            # backpropagate expects value from parent's perspective
            # parent's current_player is the opponent of node's current_player
            # so we negate
            node.backpropagate(-value)

        return root

    def get_move(self, state: GameState, add_noise: bool = True) -> tuple[Move, dict]:
        """Get best move and search statistics."""
        root = self.search(state, add_noise=add_noise)

        if not root.children:
            legal = generate_legal_moves(state)
            if legal:
                import random
                return random.choice(legal), {"visits": {}, "value": 0.0}
            raise ValueError("No legal moves")

        visits = np.array([c.visit_count for c in root.children], dtype=np.float64)
        moves = [c.move for c in root.children]
        total_visits = visits.sum()

        if total_visits == 0:
            # No simulations ran — uniform random fallback
            best_idx = np.random.randint(len(moves))
            policy_target = np.ones(len(moves)) / len(moves)
        elif self.temperature == 0:
            best_idx = np.argmax(visits)
            policy_target = visits / total_visits
        elif self.temperature == float("inf"):
            best_idx = np.random.randint(len(moves))
            policy_target = visits / total_visits
        else:
            visits_temp = visits ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(moves), p=probs)
            policy_target = visits / total_visits
        move_policies = {move_to_policy_index(m): p
                         for m, p in zip(moves, policy_target)}

        return moves[best_idx], {
            "visit_counts": dict(zip(range(len(moves)), visits.tolist())),
            "policy_target": move_policies,
            "root_value": root.q_value,
        }

    def get_move_from_root(self, root: MCTSNode, state: GameState) -> tuple[Move, dict]:
        """Select a move from a completed search tree root.

        Same logic as get_move() but works with a pre-computed root node.
        """
        if not root.children:
            legal = generate_legal_moves(state)
            if legal:
                import random
                return random.choice(legal), {"visits": {}, "value": 0.0}
            raise ValueError("No legal moves")

        visits = np.array([c.visit_count for c in root.children], dtype=np.float64)
        moves = [c.move for c in root.children]
        total_visits = visits.sum()

        if total_visits == 0:
            best_idx = np.random.randint(len(moves))
            policy_target = np.ones(len(moves)) / len(moves)
        elif self.temperature == 0:
            best_idx = np.argmax(visits)
            policy_target = visits / total_visits
        elif self.temperature == float("inf"):
            best_idx = np.random.randint(len(moves))
            policy_target = visits / total_visits
        else:
            visits_temp = visits ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(moves), p=probs)
            policy_target = visits / total_visits
        move_policies = {move_to_policy_index(m): p
                         for m, p in zip(moves, policy_target)}

        return moves[best_idx], {
            "visit_counts": dict(zip(range(len(moves)), visits.tolist())),
            "policy_target": move_policies,
            "root_value": root.q_value,
        }

    def search_one_iteration_batched(self, root: MCTSNode,
                                      evaluator: BatchedEvaluator) -> Optional[MCTSNode]:
        """Run one MCTS iteration: select leaf, optionally submit to evaluator.

        Returns the leaf node submitted for evaluation, or None if terminal.
        """
        node = root

        while node.is_expanded and node.children:
            node = node.select_child(self.cpuct)

        node.ensure_state()

        if node.state.done:
            if node.state.winner is not None:
                parent_player = node.parent.state.current_player if node.parent else node.state.current_player
                v = 1.0 if node.state.winner == parent_player else -1.0
            else:
                v = 0.0
            node.backpropagate(v)
            return None

        planes = state_to_planes(node.state)
        evaluator.submit(node, planes)
        return node

    @staticmethod
    def batched_search(engines: list[MCTS], states: list[GameState],
                       evaluator: BatchedEvaluator,
                       add_noise_flags: list[bool]) -> list[MCTSNode]:
        """Run MCTS search on multiple states with batched NN evaluation.

        Args:
            engines: MCTS instance per game (may differ in temperature etc).
            states: GameState per game.
            evaluator: Shared BatchedEvaluator.
            add_noise_flags: Per-game Dirichlet noise flag.

        Returns:
            List of root MCTSNode, one per game.
        """
        n = len(engines)

        # Phase 1: create and batch-evaluate root nodes
        roots: list[MCTSNode] = []
        for i in range(n):
            root = MCTSNode(state=states[i].clone())
            roots.append(root)
            evaluator.submit(root, state_to_planes(root.state))

        root_results = evaluator.evaluate_batch()

        for i in range(n):
            policy, value = root_results[i]
            roots[i].expand(policy)
            if add_noise_flags[i] and roots[i].children:
                eng = engines[i]
                noise = np.random.dirichlet(
                    [eng.dirichlet_alpha] * len(roots[i].children))
                for child, ns in zip(roots[i].children, noise):
                    child.prior = ((1 - eng.dirichlet_epsilon) * child.prior
                                   + eng.dirichlet_epsilon * ns)

        # Phase 2: run simulations in lockstep
        num_sims = engines[0].num_simulations
        for _ in range(num_sims):
            pending_nodes: list[MCTSNode] = []

            for i in range(n):
                leaf = engines[i].search_one_iteration_batched(roots[i], evaluator)
                if leaf is not None:
                    pending_nodes.append(leaf)

            if pending_nodes:
                results = evaluator.evaluate_batch()
                for node, (policy, value) in zip(pending_nodes, results):
                    node.expand(policy)
                    node.backpropagate(-value)

        return roots
