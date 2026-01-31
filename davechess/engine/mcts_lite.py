"""Lightweight MCTS with random rollouts (no neural network).

Used for game validation in Phase 1.5 and as a baseline opponent.

Value convention: each node stores cumulative wins from the perspective of the
player whose turn it is at that node's *parent*. This way, the parent can simply
pick the child with the highest win rate (UCB1 always maximizes).

For the root node (no parent), value is stored from the root player's perspective.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from davechess.game.state import GameState, Player, Move
from davechess.game.rules import generate_legal_moves, apply_move, check_winner


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""
    state: GameState
    parent: Optional[MCTSNode] = None
    move: Optional[Move] = None  # Move that led to this node
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0  # From the perspective of the node's parent's player
    untried_moves: Optional[list[Move]] = None

    @property
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.done

    def ucb1(self, exploration: float = 1.41) -> float:
        """Upper confidence bound for trees."""
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSLite:
    """Lightweight MCTS engine using random rollouts."""

    def __init__(self, num_simulations: int = 100, max_rollout_depth: int = 100,
                 exploration: float = 1.41):
        self.num_simulations = num_simulations
        self.max_rollout_depth = max_rollout_depth
        self.exploration = exploration

    def search(self, state: GameState) -> Move:
        """Run MCTS and return the best move."""
        root = MCTSNode(state=state.clone())
        root.untried_moves = generate_legal_moves(root.state)

        if not root.untried_moves:
            raise ValueError("No legal moves available")

        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            winner = self._rollout(node)
            self._backpropagate(node, winner)

        # Select child with most visits
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def search_with_policy(self, state: GameState) -> tuple[Move, dict[Move, float]]:
        """Run MCTS and return best move plus visit-count policy."""
        root = MCTSNode(state=state.clone())
        root.untried_moves = generate_legal_moves(root.state)

        if not root.untried_moves:
            raise ValueError("No legal moves available")

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            winner = self._rollout(node)
            self._backpropagate(node, winner)

        total_visits = sum(c.visits for c in root.children)
        policy = {}
        for child in root.children:
            policy[child.move] = child.visits / total_visits if total_visits > 0 else 0

        best = max(root.children, key=lambda c: c.visits)
        return best.move, policy

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand one untried move."""
        if node.untried_moves is None:
            node.untried_moves = generate_legal_moves(node.state)

        if not node.untried_moves:
            return node

        move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
        new_state = node.state.clone()
        apply_move(new_state, move)

        child = MCTSNode(state=new_state, parent=node, move=move)
        child.untried_moves = generate_legal_moves(child.state)
        node.children.append(child)
        return child

    def _rollout(self, node: MCTSNode) -> Optional[Player]:
        """Random rollout from node's state. Returns the winner (or None for draw)."""
        state = node.state.clone()
        depth = 0

        while not state.done and depth < self.max_rollout_depth:
            moves = generate_legal_moves(state)
            if not moves:
                break
            move = random.choice(moves)
            apply_move(state, move)
            depth += 1

        return state.winner  # Player enum or None

    def _backpropagate(self, node: MCTSNode, winner: Optional[Player]):
        """Backpropagate the rollout result up the tree.

        Each node stores wins from the perspective of the player who chose
        to go to this node (i.e., the node's parent's current player).
        """
        while node is not None:
            node.visits += 1
            if winner is not None:
                if node.parent is not None:
                    # Store from parent's player's perspective
                    parent_player = node.parent.state.current_player
                    if winner == parent_player:
                        node.wins += 1.0
                    # If opponent won, wins stays 0 (loss)
                    # Draw = 0 (already handled by winner being None)
                else:
                    # Root node: store from root player's perspective
                    if winner == node.state.current_player:
                        node.wins += 1.0
            else:
                # Draw: half point
                node.wins += 0.5
            node = node.parent


def play_random_game(state: Optional[GameState] = None,
                     max_moves: int = 400) -> GameState:
    """Play a game with random moves. Useful for testing."""
    if state is None:
        state = GameState()

    move_count = 0
    while not state.done and move_count < max_moves:
        moves = generate_legal_moves(state)
        if not moves:
            break
        move = random.choice(moves)
        apply_move(state, move)
        move_count += 1

    return state
