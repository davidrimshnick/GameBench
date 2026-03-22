"""Neural network opponent with configurable strength.

Strength is controlled by temperature applied to the value head's
evaluation of each legal move. Lower temperature = stronger play
(always picks the best move). Higher temperature = weaker play
(more likely to pick suboptimal moves).

This mirrors how real chess engines create lower-ELO bots:
for each legal move, evaluate the resulting position with the NN
value head, then sample from softmax(Q / temperature).

Temperature-to-ELO mapping is calibrated empirically.
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
from nn_network import state_to_planes, move_to_policy_index


class NNOpponent:
    """Neural network opponent with temperature-controlled strength.

    Args:
        network: NumpyNetwork loaded from checkpoint
        temperature: Controls playing strength.
            0.0  = always pick the best move (strongest)
            0.1  = very strong, occasional suboptimal moves
            0.5  = moderate, mix of good and okay moves
            1.0  = weak, often picks mediocre moves
            2.0+ = very weak, nearly random
        num_simulations: MCTS search budget. 0 = raw policy + value (fastest).
        cpuct: MCTS exploration constant (only used if num_simulations > 0)
        value_scale: Scale for NN value predictions in MCTS backprop
    """

    def __init__(self, network: NumpyNetwork, temperature: float = 0.5,
                 blunder_rate: float = 0.0,
                 num_simulations: int = 0, cpuct: float = 4.0,
                 value_scale: float = 0.5):
        self.network = network
        self.temperature = max(temperature, 0.001)
        self.blunder_rate = blunder_rate  # Probability of playing random move
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.value_scale = value_scale

    def search(self, state: GameState) -> Move:
        """Pick a move using policy temperature + optional blunders."""
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves")
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Blunder: play random move X% of the time
        if self.blunder_rate > 0 and np.random.random() < self.blunder_rate:
            return legal_moves[np.random.randint(len(legal_moves))]

        if self.num_simulations > 0:
            return self._mcts_search(state, legal_moves)
        else:
            return self._policy_sample(state, legal_moves)

    def _policy_sample(self, state: GameState, legal_moves: list[Move]) -> Move:
        """Sample move from NN policy with temperature. Single forward pass.

        Temperature controls strength:
        - Low temp (0.1): almost always picks the NN's top move
        - Med temp (1.0): samples proportional to NN policy
        - High temp (3.0+): nearly uniform over legal moves
        """
        planes = state_to_planes(state)
        policy_logits, _ = self.network.predict(planes)

        # Get logits for legal moves
        indices = []
        valid_moves = []
        for move in legal_moves:
            idx = move_to_policy_index(move, state)
            if idx is not None:
                indices.append(idx)
                valid_moves.append(move)

        if not valid_moves:
            return legal_moves[np.random.randint(len(legal_moves))]

        logits = np.array([policy_logits[i] for i in indices])

        # Apply temperature
        scaled = logits / self.temperature
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs = probs / probs.sum()

        idx = np.random.choice(len(valid_moves), p=probs)
        return valid_moves[idx]

    def _mcts_search(self, state: GameState, legal_moves: list[Move]) -> Move:
        """MCTS search with temperature-based move selection."""
        root = _MCTSNode(state=state.clone())
        self._expand(root)

        if not root.children:
            raise ValueError("No legal moves after expansion")

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_expanded and not node.state.done:
                value = self._expand(node)
            else:
                value = self._evaluate_terminal(node)
            node.backpropagate(-value * self.value_scale)

        # Apply temperature to visit counts for move selection
        visits = np.array([c.visit_count for c in root.children], dtype=float)
        if self.temperature < 0.01:
            # Near-zero temp: pick best
            idx = np.argmax(visits)
        else:
            # Temperature-weighted sampling on visit counts
            log_visits = np.log(visits + 1)
            scaled = log_visits / self.temperature
            scaled = scaled - scaled.max()
            probs = np.exp(scaled)
            probs = probs / probs.sum()
            idx = np.random.choice(len(root.children), p=probs)

        return root.children[idx].move

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while node.is_expanded and node.children and not node.state.done:
            node = max(node.children, key=lambda c: c.puct_score(self.cpuct))
        return node

    def _expand(self, node: _MCTSNode) -> float:
        state = node.state
        if state.done:
            return self._evaluate_terminal(node)

        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            return 0.0

        planes = state_to_planes(state)
        policy_logits, value = self.network.predict(planes)

        legal_indices = []
        for move in legal_moves:
            idx = move_to_policy_index(move, state)
            if idx is not None:
                legal_indices.append((move, idx))

        if not legal_indices:
            uniform = 1.0 / len(legal_moves)
            for move in legal_moves:
                child_state = state.clone()
                apply_move(child_state, move)
                child = _MCTSNode(child_state, parent=node, move=move, prior=uniform)
                node.children.append(child)
        else:
            logits = np.array([policy_logits[idx] for _, idx in legal_indices])
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            for (move, _), prior in zip(legal_indices, probs):
                child_state = state.clone()
                apply_move(child_state, move)
                child = _MCTSNode(child_state, parent=node, move=move, prior=float(prior))
                node.children.append(child)

        node.is_expanded = True
        return float(value)

    def _evaluate_terminal(self, node: _MCTSNode) -> float:
        if not node.state.done:
            planes = state_to_planes(node.state)
            _, value = self.network.predict(planes)
            return float(value)
        if node.state.winner is None:
            return 0.0
        current = node.state.current_player
        return 1.0 if node.state.winner == current else -1.0


class _MCTSNode:
    """Internal MCTS node."""
    __slots__ = ['state', 'parent', 'move', 'children', 'visit_count',
                 'total_value', 'prior', 'is_expanded']

    def __init__(self, state: GameState, parent: Optional[_MCTSNode] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[_MCTSNode] = []
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
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent


def load_nn_opponent(weights_path: str = "model_weights.npz",
                     temperature: float = 0.5,
                     blunder_rate: float = 0.0,
                     num_simulations: int = 0,
                     cpuct: float = 4.0) -> NNOpponent:
    """Load neural network opponent from weights file.

    Args:
        weights_path: Path to model_weights.npz
        temperature: Policy temperature (0.1=strong, 1.0=moderate, 3.0+=weak)
        blunder_rate: Probability of playing a random legal move (0.0-1.0)
        num_simulations: MCTS sims (0=policy only, faster)
        cpuct: MCTS exploration constant
    """
    network = NumpyNetwork.from_npz(weights_path)
    return NNOpponent(network, temperature=temperature,
                      blunder_rate=blunder_rate,
                      num_simulations=num_simulations, cpuct=cpuct)
