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
        parent_visits = self.parent.visit_count if self.parent else 1
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
