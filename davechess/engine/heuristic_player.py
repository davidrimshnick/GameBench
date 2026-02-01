"""Heuristic-based player for generating quality seed games.

Instead of random rollouts, uses game knowledge to evaluate positions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import random
import math

from davechess.game.state import GameState, Player, PieceType, Move, MoveStep, Deploy
from davechess.game.rules import generate_legal_moves, apply_move, check_winner


@dataclass
class PositionEval:
    """Evaluation of a game position."""
    score: float  # From current player's perspective (-1 to 1)
    material: float
    position: float
    safety: float
    resources: float


class HeuristicPlayer:
    """Plays using hand-crafted heuristics instead of random rollouts."""

    # Piece values based on game understanding
    PIECE_VALUES = {
        PieceType.COMMANDER: 10.0,  # Losing it loses the game
        PieceType.WARRIOR: 3.0,     # Strong attacker
        PieceType.RIDER: 2.5,       # Mobile
        PieceType.BOMBARD: 2.0,     # Ranged but vulnerable
    }

    # Resource node locations
    RESOURCE_NODES = [
        (0, 3), (0, 4),  # White's home nodes
        (7, 3), (7, 4),  # Black's home nodes
        (3, 0), (4, 0),  # Left side nodes
        (3, 7), (4, 7),  # Right side nodes
    ]

    def __init__(self, exploration: float = 0.2, aggression: float = 0.5):
        """
        Args:
            exploration: Chance of making random move (0-1)
            aggression: Preference for captures vs positional play (0-1)
        """
        self.exploration = exploration
        self.aggression = aggression

    def get_move(self, state: GameState) -> Move:
        """Select a move using heuristic evaluation."""
        moves = generate_legal_moves(state)
        if not moves:
            return None  # No legal moves

        # Occasionally play random for diversity
        if random.random() < self.exploration:
            return random.choice(moves)

        # Evaluate each move
        best_move = None
        best_score = -float('inf')

        for move in moves:
            # Try the move
            test_state = state.clone()
            apply_move(test_state, move)

            # Check for immediate win
            if test_state.done and test_state.winner == state.current_player:
                return move  # Always take winning moves

            # Evaluate resulting position (from opponent's perspective)
            opp_eval = self.evaluate_position(test_state)
            # Negate because we want to minimize opponent's score
            score = -opp_eval.score

            # Add capture bonus if aggressive
            if isinstance(move, MoveStep):
                to_r, to_c = move.to_rc
                target = state.board[to_r][to_c]
                if target is not None:
                    score += self.aggression * self.PIECE_VALUES.get(target.piece_type, 0) / 10

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else moves[0]

    def evaluate_position(self, state: GameState) -> PositionEval:
        """Evaluate a position from current player's perspective."""
        if state.done:
            if state.winner == state.current_player:
                return PositionEval(1.0, 0, 0, 0, 0)
            elif state.winner is None:
                return PositionEval(0.0, 0, 0, 0, 0)
            else:
                return PositionEval(-1.0, 0, 0, 0, 0)

        material = self._eval_material(state)
        position = self._eval_position(state)
        safety = self._eval_commander_safety(state)
        resources = self._eval_resources(state)

        # Weighted combination
        total = (
            material * 0.4 +
            position * 0.2 +
            safety * 0.25 +
            resources * 0.15
        )

        # Clamp to [-1, 1]
        total = max(-1.0, min(1.0, total))

        return PositionEval(total, material, position, safety, resources)

    def _eval_material(self, state: GameState) -> float:
        """Evaluate material balance (-1 to 1)."""
        my_value = 0.0
        opp_value = 0.0

        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece is None:
                    continue

                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.player == state.current_player:
                    my_value += value
                else:
                    opp_value += value

        # Normalize
        total = my_value + opp_value
        if total > 0:
            return (my_value - opp_value) / total
        return 0.0

    def _eval_position(self, state: GameState) -> float:
        """Evaluate piece positioning (-1 to 1)."""
        score = 0.0

        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece is None:
                    continue

                # Center control is good
                center_dist = abs(r - 3.5) + abs(c - 3.5)
                center_bonus = (7 - center_dist) / 7  # 0 to 1

                # Forward progress is good
                if piece.player == Player.WHITE:
                    forward_bonus = r / 7
                else:
                    forward_bonus = (7 - r) / 7

                piece_score = (center_bonus * 0.5 + forward_bonus * 0.5)

                if piece.player == state.current_player:
                    score += piece_score
                else:
                    score -= piece_score

        # Normalize to [-1, 1]
        return score / 10  # Rough normalization

    def _eval_commander_safety(self, state: GameState) -> float:
        """Evaluate commander safety (-1 to 1)."""
        my_commander_pos = None
        opp_commander_pos = None

        # Find commanders
        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece and piece.piece_type == PieceType.COMMANDER:
                    if piece.player == state.current_player:
                        my_commander_pos = (r, c)
                    else:
                        opp_commander_pos = (r, c)

        my_safety = 0.0
        opp_safety = 0.0

        if my_commander_pos:
            # Check threats around my commander
            r, c = my_commander_pos
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        piece = state.board[nr][nc]
                        if piece:
                            if piece.player == state.current_player:
                                my_safety += 0.1  # Friendly piece nearby
                            else:
                                my_safety -= 0.2  # Enemy threat
        else:
            return -1.0  # Commander dead!

        if opp_commander_pos:
            # Check threats around opponent commander
            r, c = opp_commander_pos
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        piece = state.board[nr][nc]
                        if piece:
                            if piece.player != state.current_player:
                                opp_safety += 0.1
                            else:
                                opp_safety -= 0.2  # We threaten them
        else:
            return 1.0  # Opponent commander dead!

        return my_safety - opp_safety

    def _eval_resources(self, state: GameState) -> float:
        """Evaluate resource control (-1 to 1)."""
        my_resources = 0
        opp_resources = 0

        for r, c in self.RESOURCE_NODES:
            piece = state.board[r][c]
            if piece:
                if piece.player == state.current_player:
                    my_resources += 1
                else:
                    opp_resources += 1

        # Check for resource domination win
        if my_resources >= 5:
            return 1.0
        if opp_resources >= 5:
            return -1.0

        # Otherwise normalize
        total = len(self.RESOURCE_NODES)
        return (my_resources - opp_resources) / total


class SmartMCTS:
    """MCTS that uses heuristic evaluation instead of random rollouts."""

    def __init__(self, num_simulations: int = 50, exploration: float = 1.4):
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.evaluator = HeuristicPlayer(exploration=0.0, aggression=0.5)

    def get_move(self, state: GameState) -> Move:
        """Get best move using MCTS with heuristic evaluation."""
        return self.search(state)

    def search(self, state: GameState) -> Move:
        """Run MCTS with heuristic evaluation."""
        from davechess.engine.mcts_lite import MCTSNode

        root = MCTSNode(state=state.clone())
        root.untried_moves = generate_legal_moves(root.state)

        if not root.untried_moves:
            # No moves available - return None to let game end
            return None

        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            value = self._evaluate(node)  # Use heuristic instead of rollout
            self._backpropagate(node, value)

        # Select child with most visits
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _select(self, node):
        """Select a leaf node using UCB1."""
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            # Add small random to break ties
            node = max(node.children,
                      key=lambda c: c.ucb1(self.exploration) + random.random() * 0.001)
        return node

    def _expand(self, node):
        """Expand one untried move."""
        if node.untried_moves is None:
            node.untried_moves = generate_legal_moves(node.state)

        if not node.untried_moves:
            return node

        move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
        new_state = node.state.clone()
        apply_move(new_state, move)

        from davechess.engine.mcts_lite import MCTSNode
        child = MCTSNode(state=new_state, parent=node, move=move)
        child.untried_moves = generate_legal_moves(child.state)
        node.children.append(child)
        return child

    def _evaluate(self, node) -> float:
        """Evaluate position using heuristics instead of rollout.
        Returns value from perspective of node's parent's player."""
        eval_result = self.evaluator.evaluate_position(node.state)

        # Convert [-1, 1] score to [0, 1] for backprop
        # Score is from current player's perspective
        value = (eval_result.score + 1.0) / 2.0

        # Need to return from parent's perspective
        if node.parent:
            # If parent is opponent, invert value
            if node.parent.state.current_player != node.state.current_player:
                value = 1.0 - value

        return value

    def _backpropagate(self, node, value: float):
        """Backpropagate the evaluation value up the tree."""
        while node is not None:
            node.visits += 1
            node.wins += value
            # Flip value for opponent
            value = 1.0 - value
            node = node.parent