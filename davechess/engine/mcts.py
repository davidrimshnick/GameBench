"""MCTS with PUCT and neural network evaluation for AlphaZero.

Supports batched neural network evaluation for efficiency.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.network import (
    state_to_planes, move_to_policy_index, POLICY_SIZE,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MCTSNode:
    """Node in the MCTS search tree with PUCT."""

    __slots__ = [
        "state", "parent", "move", "children", "visit_count",
        "total_value", "prior", "is_expanded",
    ]

    def __init__(self, state: GameState, parent: Optional[MCTSNode] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, cpuct: float) -> float:
        """PUCT selection score."""
        parent_visits = self.parent.visit_count if self.parent else 1
        u = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self, cpuct: float) -> MCTSNode:
        """Select child with highest PUCT score."""
        return max(self.children, key=lambda c: c.puct_score(cpuct))

    def expand(self, policy: np.ndarray):
        """Expand node with children, using policy prior from neural network."""
        if self.is_expanded or self.state.done:
            return

        legal_moves = generate_legal_moves(self.state)
        if not legal_moves:
            self.is_expanded = True
            return

        # Mask and renormalize policy over legal moves
        move_indices = [move_to_policy_index(m) for m in legal_moves]
        priors = np.array([policy[idx] for idx in move_indices])
        prior_sum = priors.sum()
        if prior_sum > 0:
            priors = priors / prior_sum
        else:
            priors = np.ones(len(legal_moves)) / len(legal_moves)

        for move, prior in zip(legal_moves, priors):
            child_state = self.state.clone()
            apply_move(child_state, move)
            child = MCTSNode(child_state, parent=self, move=move, prior=prior)
            self.children.append(child)

        self.is_expanded = True

    def backpropagate_winner(self, winner: Optional[Player], value_if_no_winner: float = 0.0):
        """Backpropagate using the game outcome.

        Each node stores total_value from the perspective of the player who
        CHOSE to visit this node (the parent's current_player). This way,
        select_child maximizing Q+U picks the best move for the selecting player.

        Args:
            winner: The winning player, or None for draw/evaluation.
            value_if_no_winner: Value to use if winner is None (e.g., NN evaluation).
        """
        node = self
        while node is not None:
            node.visit_count += 1
            if node.parent is not None:
                parent_player = node.parent.state.current_player
                if winner is not None:
                    node.total_value += 1.0 if winner == parent_player else -1.0
                else:
                    node.total_value += value_if_no_winner
                    value_if_no_winner = -value_if_no_winner  # Flip for next level
            else:
                # Root node
                if winner is not None:
                    node.total_value += 1.0 if winner == node.state.current_player else -1.0
                else:
                    node.total_value += value_if_no_winner
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
            # Fallback to uniform policy
            return np.ones(POLICY_SIZE) / POLICY_SIZE, 0.0

        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(x)

        policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        return policy, value.item()

    def _evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """Batch evaluate states with neural network."""
        if not HAS_TORCH or self.network is None or not states:
            return [(np.ones(POLICY_SIZE) / POLICY_SIZE, 0.0) for _ in states]

        planes_list = [state_to_planes(s) for s in states]
        batch = torch.from_numpy(np.stack(planes_list)).to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits, values = self.network(batch)

        policies = torch.softmax(logits, dim=1).cpu().numpy()
        values = values.cpu().numpy().flatten()

        return list(zip(policies, values))

    def search(self, state: GameState, add_noise: bool = True) -> MCTSNode:
        """Run MCTS search from state. Returns root node."""
        root = MCTSNode(state=state.clone())

        # Evaluate root
        policy, value = self._evaluate(root.state)
        root.expand(policy)

        # Add Dirichlet noise to root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                              self.dirichlet_epsilon * n

        for _ in range(self.num_simulations):
            node = root

            # Select
            while node.is_expanded and node.children:
                node = node.select_child(self.cpuct)

            # If terminal, backpropagate
            if node.state.done:
                node.backpropagate_winner(node.state.winner)
                continue

            # Expand and evaluate
            policy, value = self._evaluate(node.state)
            node.expand(policy)

            # NN returns value from state.current_player's perspective
            # For non-terminal nodes, convert NN value to a "virtual winner" framework
            # Value > 0 means current_player is likely to win
            # We use backpropagate_winner with None winner and pass the NN value
            # adjusted to root player's perspective
            root_player = root.state.current_player
            if node.state.current_player == root_player:
                nn_value = value
            else:
                nn_value = -value
            node.backpropagate_winner(None, value_if_no_winner=nn_value)

        return root

    def get_move(self, state: GameState, add_noise: bool = True) -> tuple[Move, dict]:
        """Get best move and search statistics.

        Returns:
            (best_move, info_dict) where info_dict contains visit counts and value.
        """
        root = self.search(state, add_noise=add_noise)

        if not root.children:
            legal = generate_legal_moves(state)
            if legal:
                import random
                return random.choice(legal), {"visits": {}, "value": 0.0}
            raise ValueError("No legal moves")

        # Build visit count distribution
        visits = np.array([c.visit_count for c in root.children], dtype=np.float64)
        moves = [c.move for c in root.children]

        if self.temperature == 0:
            # Select best
            best_idx = np.argmax(visits)
        elif self.temperature == float("inf"):
            # Uniform random
            best_idx = np.random.randint(len(moves))
        else:
            # Temperature-scaled sampling
            visits_temp = visits ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(moves), p=probs)

        # Policy target (normalized visit counts)
        policy_target = visits / visits.sum()
        move_policies = {move_to_policy_index(m): p
                         for m, p in zip(moves, policy_target)}

        return moves[best_idx], {
            "visit_counts": dict(zip(range(len(moves)), visits.tolist())),
            "policy_target": move_policies,
            "root_value": root.q_value,
        }
