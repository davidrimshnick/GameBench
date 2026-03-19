"""Neural network opponent using numpy-only inference + MCTS.

Uses the AlphaZero-trained network with PUCT search.
Sim count controls strength: more sims = stronger play.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

from davechess_engine import (
    GameState, Player, Move,
    generate_legal_moves, apply_move,
)
from nn_numpy import NumpyNetwork
from nn_network import state_to_planes, move_to_policy_index, policy_index_to_move


class NNNode:
    """MCTS node with neural network evaluation."""
    __slots__ = ['state', 'parent', 'move', 'children', 'visit_count',
                 'total_value', 'prior', 'is_expanded']

    def __init__(self, state: GameState, parent: Optional[NNNode] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[NNNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, cpuct: float = 4.0) -> float:
        """PUCT selection formula."""
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def backpropagate(self, value: float):
        """Backpropagate value up the tree, alternating sign."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip for opponent's perspective
            node = node.parent


class NNOpponent:
    """Neural network opponent with MCTS search.

    Args:
        network: NumpyNetwork loaded from checkpoint
        num_simulations: MCTS search budget (more = stronger)
        cpuct: Exploration constant
        value_scale: Scale NN value predictions (0.5 = dampen)
    """

    def __init__(self, network: NumpyNetwork, num_simulations: int = 20,
                 cpuct: float = 4.0, value_scale: float = 0.5):
        self.network = network
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.value_scale = value_scale

    def search(self, state: GameState) -> Move:
        """Run MCTS and return the best move."""
        root = NNNode(state=state.clone())
        self._expand(root)

        if not root.children:
            raise ValueError("No legal moves")

        if len(root.children) == 1:
            return root.children[0].move

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_expanded and not node.state.done:
                value = self._expand(node)
            else:
                # Terminal or already expanded leaf
                value = self._evaluate_terminal(node)
            node.backpropagate(-value * self.value_scale)

        # Pick move with most visits
        best = max(root.children, key=lambda c: c.visit_count)
        return best.move

    def _select(self, node: NNNode) -> NNNode:
        """Select leaf node using PUCT."""
        while node.is_expanded and node.children and not node.state.done:
            node = max(node.children, key=lambda c: c.puct_score(self.cpuct))
        return node

    def _expand(self, node: NNNode) -> float:
        """Expand node: evaluate with NN, create children with priors.

        Returns the value estimate for the current player.
        """
        state = node.state
        if state.done:
            return self._evaluate_terminal(node)

        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            return 0.0

        # Neural network evaluation
        planes = state_to_planes(state)
        policy_logits, value = self.network.predict(planes)

        # Mask illegal moves and softmax
        legal_indices = []
        for move in legal_moves:
            idx = move_to_policy_index(move, state)
            if idx is not None:
                legal_indices.append((move, idx))

        if not legal_indices:
            # Fallback: uniform priors
            uniform = 1.0 / len(legal_moves)
            for move in legal_moves:
                child_state = state.clone()
                apply_move(child_state, move)
                child = NNNode(child_state, parent=node, move=move, prior=uniform)
                node.children.append(child)
        else:
            # Softmax over legal moves
            logits = np.array([policy_logits[idx] for _, idx in legal_indices])
            logits = logits - logits.max()  # Numerical stability
            probs = np.exp(logits) / np.exp(logits).sum()

            for (move, _), prior in zip(legal_indices, probs):
                child_state = state.clone()
                apply_move(child_state, move)
                child = NNNode(child_state, parent=node, move=move, prior=float(prior))
                node.children.append(child)

        node.is_expanded = True
        return float(value)

    def _evaluate_terminal(self, node: NNNode) -> float:
        """Evaluate a terminal state."""
        if not node.state.done:
            # Not terminal — evaluate with NN
            planes = state_to_planes(node.state)
            _, value = self.network.predict(planes)
            return float(value)

        if node.state.winner is None:
            return 0.0  # Draw

        # Win/loss from current player's perspective
        current = node.state.current_player
        if node.state.winner == current:
            return 1.0
        return -1.0


def load_nn_opponent(weights_path: str = "model_weights.npz",
                     num_simulations: int = 20,
                     cpuct: float = 4.0) -> NNOpponent:
    """Load neural network opponent from weights file."""
    network = NumpyNetwork.from_npz(weights_path)
    return NNOpponent(network, num_simulations=num_simulations, cpuct=cpuct)
