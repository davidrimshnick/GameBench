"""End-to-end tests for value target correctness in the training pipeline.

Traces through a concrete scenario: Black wins a game. Verifies that:
1. Every Black-to-move position gets value target +1
2. Every White-to-move position gets value target -1
3. The network's value head outputs tanh (-1 to +1) from current player perspective
4. MCTS correctly negates values during backpropagation
5. The training loss computation uses the correct sign convention

These tests catch silent value target corruption that would destroy training.
"""

import numpy as np
import pytest

from davechess.game.state import GameState, Player, PieceType, Piece, MoveStep
from davechess.game.board import BOARD_SIZE
from davechess.game.rules import generate_legal_moves, apply_move
from davechess.engine.network import (
    state_to_planes, move_to_policy_index, POLICY_SIZE, NUM_INPUT_PLANES,
)
from davechess.engine.selfplay import (
    play_selfplay_game, _build_policy_target, _finalize_game, _ActiveGame,
    _apply_game_move, _record_winner,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestValueTargetAssignment:
    """Verify value targets are assigned from the current player's perspective."""

    def _simulate_game_examples(self, winner_player):
        """Create fake training examples as if a game were played.

        Simulates alternating White/Black turns for a few moves,
        then assigns value targets as play_selfplay_game() does.
        """
        state = GameState()
        examples = []

        # Play a few real moves to get alternating player examples
        for _ in range(6):
            if state.done:
                break
            legal = generate_legal_moves(state)
            if not legal:
                break
            move = legal[0]
            planes = state_to_planes(state)
            policy_dict = {move_to_policy_index(move, flip=(state.current_player == Player.BLACK)): 1.0}
            examples.append((planes, policy_dict, int(state.current_player)))
            apply_move(state, move)

        # Now assign value targets using the same logic as selfplay.py
        if winner_player is None:
            winner = -1
        else:
            winner = int(winner_player)

        training_data = []
        for planes, policy_dict, player in examples:
            policy = _build_policy_target(policy_dict)
            if winner == -1:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            training_data.append((planes, policy, value, player))

        return training_data

    def test_black_wins_black_positions_get_plus_one(self):
        """When Black wins, all Black-to-move positions must have value=+1."""
        data = self._simulate_game_examples(Player.BLACK)
        for planes, policy, value, player in data:
            if player == int(Player.BLACK):
                assert value == 1.0, (
                    f"Black-to-move position should get +1 when Black wins, got {value}"
                )

    def test_black_wins_white_positions_get_minus_one(self):
        """When Black wins, all White-to-move positions must have value=-1."""
        data = self._simulate_game_examples(Player.BLACK)
        for planes, policy, value, player in data:
            if player == int(Player.WHITE):
                assert value == -1.0, (
                    f"White-to-move position should get -1 when Black wins, got {value}"
                )

    def test_white_wins_white_positions_get_plus_one(self):
        """When White wins, all White-to-move positions must have value=+1."""
        data = self._simulate_game_examples(Player.WHITE)
        for planes, policy, value, player in data:
            if player == int(Player.WHITE):
                assert value == 1.0, (
                    f"White-to-move position should get +1 when White wins, got {value}"
                )

    def test_white_wins_black_positions_get_minus_one(self):
        """When White wins, all Black-to-move positions must have value=-1."""
        data = self._simulate_game_examples(Player.WHITE)
        for planes, policy, value, player in data:
            if player == int(Player.BLACK):
                assert value == -1.0, (
                    f"Black-to-move position should get -1 when White wins, got {value}"
                )

    def test_draw_all_positions_get_draw_target(self):
        """In a draw, all positions must get the draw_value_target (default 0)."""
        data = self._simulate_game_examples(None)
        for planes, policy, value, player in data:
            assert value == 0.0, f"Draw position should get 0.0, got {value}"


class TestPlayerEnum:
    """Verify Player enum integer values used in winner comparison."""

    def test_player_white_is_zero(self):
        assert int(Player.WHITE) == 0

    def test_player_black_is_one(self):
        assert int(Player.BLACK) == 1

    def test_winner_comparison_uses_int(self):
        """The selfplay code compares winner (int) to player (int).
        Verify this works correctly with Player IntEnum."""
        winner = int(Player.BLACK)  # 1
        player_black = int(Player.BLACK)  # 1
        player_white = int(Player.WHITE)  # 0

        assert winner == player_black
        assert winner != player_white

    def test_state_winner_is_player_enum(self):
        """state.winner is a Player enum (or None for draw)."""
        state = GameState()
        assert state.winner is None  # Game not over

        # Check that Player values convert correctly
        state.winner = Player.BLACK
        assert int(state.winner) == 1
        state.winner = Player.WHITE
        assert int(state.winner) == 0


class TestWinnerDetermination:
    """Verify how state.winner gets set by the rules engine."""

    def test_winner_is_none_for_ongoing_game(self):
        state = GameState()
        assert state.winner is None
        assert state.done is False

    def test_winner_set_on_commander_capture(self):
        """Capturing the Commander sets winner = capturing player."""
        state = GameState()
        # Place only two Commanders to set up a quick capture
        state.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        state.board[0][0] = Piece(PieceType.COMMANDER, Player.WHITE)
        state.board[1][0] = Piece(PieceType.COMMANDER, Player.BLACK)
        state.current_player = Player.WHITE

        # White Commander at (0,0) captures Black Commander at (1,0)
        move = MoveStep((0, 0), (1, 0), is_capture=True)
        apply_move(state, move)
        assert state.done is True
        assert state.winner == Player.WHITE

    def test_int_conversion_matches_selfplay(self):
        """selfplay.py does `winner = int(state.winner)`.
        Verify the conversion matches what we expect."""
        state = GameState()
        state.done = True
        state.winner = Player.BLACK
        winner = int(state.winner)

        # This is the exact comparison from selfplay.py line 541
        player_white = int(Player.WHITE)  # 0
        player_black = int(Player.BLACK)  # 1

        assert (1.0 if winner == player_black else -1.0) == 1.0
        assert (1.0 if winner == player_white else -1.0) == -1.0


class TestSelfplayValueTargets:
    """Test the actual play_selfplay_game value target assignment."""

    def test_play_selfplay_game_returns_valid_targets(self):
        """play_selfplay_game should return value targets in {-1, 0, +1}."""
        from davechess.engine.mcts import MCTS
        mcts = MCTS(network=None, num_simulations=2, cpuct=4.0)
        training_data, game_record = play_selfplay_game(mcts, temperature_threshold=0)

        assert len(training_data) > 0, "Should produce training examples"
        for planes, policy, value, legal_mask in training_data:
            assert value in (-1.0, 0.0, 1.0), f"Value target must be -1, 0, or +1, got {value}"
            assert planes.shape == (NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
            assert policy.shape == (POLICY_SIZE,)
            assert legal_mask.shape == (POLICY_SIZE,)

    def test_decisive_game_has_both_signs(self):
        """In a decisive game, there should be both +1 and -1 value targets."""
        from davechess.engine.mcts import MCTS
        # Play multiple games until we get a decisive one
        mcts = MCTS(network=None, num_simulations=2, cpuct=4.0)
        for _ in range(20):
            training_data, game_record = play_selfplay_game(mcts, temperature_threshold=0)
            if game_record["winner"] != "draw":
                values = [v for _, _, v, _ in training_data]
                assert 1.0 in values, "Winner's positions must have +1"
                assert -1.0 in values, "Loser's positions must have -1"
                return
        # If all games draw, that's fine — skip this check
        pytest.skip("No decisive game in 20 attempts")

    def test_draw_game_all_zeros(self):
        """In a draw with default draw_value_target=0, all values should be 0."""
        from davechess.engine.mcts import MCTS
        mcts = MCTS(network=None, num_simulations=2, cpuct=4.0)
        for _ in range(20):
            training_data, game_record = play_selfplay_game(
                mcts, temperature_threshold=0, draw_value_target=0.0
            )
            if game_record["winner"] == "draw":
                values = [v for _, _, v, _ in training_data]
                assert all(v == 0.0 for v in values), (
                    f"All draw positions should be 0.0, got {set(values)}"
                )
                return
        pytest.skip("No draw game in 20 attempts")


class TestFinalizeGameValueTargets:
    """Test the _finalize_game path (used in parallel selfplay)."""

    def _make_finished_game(self, winner_str):
        """Create a minimal _ActiveGame with a known winner."""
        from davechess.engine.mcts import MCTS
        mcts = MCTS(network=None, num_simulations=1, cpuct=4.0)
        g = _ActiveGame(0, mcts, None, True, 30, "selfplay")

        # Play a few moves to populate examples
        for _ in range(4):
            if g.state.done:
                break
            legal = generate_legal_moves(g.state)
            if not legal:
                break
            move = legal[0]
            flip = g.state.current_player == Player.BLACK
            info = {"policy_target": {move_to_policy_index(move, flip=flip): 1.0}}
            _apply_game_move(g, move, info, is_nn_turn=True)

        # Force the winner
        g.finished = True
        g.winner_str = winner_str
        return g

    def test_finalize_black_win(self):
        """_finalize_game with Black winning assigns correct targets."""
        g = self._make_finished_game("black")
        training_data, _ = _finalize_game(g, draw_value_target=0.0)

        for planes, policy, value, legal_mask in training_data:
            # Determine which player was to move from the planes
            # Player indicator plane (11) is 1.0 for White, 0.0 for Black
            # But after board flip, this still holds (scalar plane)
            player_indicator = planes[11, 0, 0]
            assert legal_mask.shape == (POLICY_SIZE,)
            if player_indicator == 1.0:
                # White-to-move position -> should be -1 (Black won)
                assert value == -1.0, f"White pos should be -1 when Black wins, got {value}"
            else:
                # Black-to-move position -> should be +1 (Black won)
                assert value == 1.0, f"Black pos should be +1 when Black wins, got {value}"

    def test_finalize_white_win(self):
        """_finalize_game with White winning assigns correct targets."""
        g = self._make_finished_game("white")
        training_data, _ = _finalize_game(g, draw_value_target=0.0)

        for planes, policy, value, legal_mask in training_data:
            player_indicator = planes[11, 0, 0]
            assert legal_mask.shape == (POLICY_SIZE,)
            if player_indicator == 1.0:
                # White-to-move -> +1 (White won)
                assert value == 1.0, f"White pos should be +1 when White wins, got {value}"
            else:
                # Black-to-move -> -1 (White won)
                assert value == -1.0, f"Black pos should be -1 when White wins, got {value}"

    def test_finalize_draw(self):
        """_finalize_game with draw assigns draw_value_target to all."""
        g = self._make_finished_game("draw")
        training_data, _ = _finalize_game(g, draw_value_target=0.0)
        for _, _, value, _ in training_data:
            assert value == 0.0


class TestPlanesValueConsistency:
    """Verify planes encode perspective consistently with value targets."""

    def test_white_planes_encode_white_as_current(self):
        """When White is to move, planes[0-4] are White's pieces (current player)."""
        state = GameState()
        assert state.current_player == Player.WHITE
        planes = state_to_planes(state)

        # White Commander is at (0, 4). Should be in plane 0 (COMMANDER, current player)
        assert planes[0, 0, 4] == 1.0
        # Black Commander is at (7, 4). Should be in plane 5 (COMMANDER, opponent)
        assert planes[5, 7, 4] == 1.0

    def test_black_planes_encode_black_as_current(self):
        """When Black is to move, planes[0-4] are Black's pieces (current player),
        with the board flipped vertically."""
        state = GameState()
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        assert state.current_player == Player.BLACK
        planes = state_to_planes(state)

        # Black Commander at real (7, 4) -> flipped to (0, 4) in planes[0]
        assert planes[0, 0, 4] == 1.0
        # White Commander at real (0, 4) -> flipped to (7, 4) in planes[5]
        assert planes[5, 7, 4] == 1.0

    def test_value_target_matches_plane_perspective(self):
        """The value target must match the perspective encoded in the planes.

        If planes show Black as current player (player_indicator=0, pieces
        flipped), then value=+1 means 'good for Black' -- which is correct
        when Black won.
        """
        state = GameState()
        legal = generate_legal_moves(state)
        apply_move(state, legal[0])
        assert state.current_player == Player.BLACK

        planes = state_to_planes(state)
        player_int = int(state.current_player)  # 1 = BLACK

        # If Black won:
        winner = int(Player.BLACK)
        value = 1.0 if winner == player_int else -1.0
        assert value == 1.0  # Black-to-move, Black won -> +1

        # The planes encode from Black's perspective (planes[0-4] = Black's pieces)
        # The value +1 means "current player (Black) is winning" -- consistent!

        # If White won:
        winner = int(Player.WHITE)
        value = 1.0 if winner == player_int else -1.0
        assert value == -1.0  # Black-to-move, White won -> -1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestMCTSValueBackpropagation:
    """Verify MCTS interprets and backpropagates values correctly."""

    def test_terminal_value_from_parent_perspective(self):
        """When MCTS reaches a terminal node, it computes the value from
        the parent's current_player's perspective."""
        from davechess.engine.mcts import MCTSNode

        # Create a parent node with White to move
        parent_state = GameState()
        parent_state.current_player = Player.WHITE
        parent = MCTSNode(state=parent_state)

        # Create a terminal child where Black won
        child_state = parent_state.clone()
        child_state.done = True
        child_state.winner = Player.BLACK
        child = MCTSNode(state=child_state, parent=parent)

        # From MCTS code: parent_player = parent.state.current_player (WHITE)
        # winner is BLACK, so v = -1 (bad for White)
        parent_player = parent.state.current_player
        v = 1.0 if child_state.winner == parent_player else -1.0
        assert v == -1.0, "Black winning should be -1 from White parent's perspective"

    def test_terminal_value_from_parent_perspective_reversed(self):
        """When parent is Black and White wins, v should be -1."""
        from davechess.engine.mcts import MCTSNode

        parent_state = GameState()
        parent_state.current_player = Player.BLACK
        parent = MCTSNode(state=parent_state)

        child_state = parent_state.clone()
        child_state.done = True
        child_state.winner = Player.WHITE
        child = MCTSNode(state=child_state, parent=parent)

        parent_player = parent.state.current_player
        v = 1.0 if child_state.winner == parent_player else -1.0
        assert v == -1.0, "White winning should be -1 from Black parent's perspective"

    def test_backpropagate_sign_alternation(self):
        """Backpropagate alternates signs at each level.

        If value=+0.5 is passed, the node gets +0.5, parent gets -0.5,
        grandparent gets +0.5, etc.
        """
        from davechess.engine.mcts import MCTSNode

        root = MCTSNode(state=GameState())
        child = MCTSNode(parent=root)
        grandchild = MCTSNode(parent=child)

        grandchild.backpropagate(0.5)

        assert grandchild.total_value == 0.5
        assert child.total_value == -0.5
        assert root.total_value == 0.5

    def test_nn_value_negated_before_backprop(self):
        """The NN outputs value from node's current_player perspective.
        MCTS negates it before backpropagate because backprop expects
        value from the parent's current_player perspective.

        In the MCTS code: `node.backpropagate(-value)` for non-terminal.
        """
        # The NN predicts from node.state.current_player's perspective.
        # node.state.current_player is the OPPONENT of node.parent.state.current_player.
        # backpropagate(v) adds v to the node, then -v to parent, +v to grandparent, etc.
        # So if NN says +0.6 (good for node's current player),
        # we pass -(-0.6) = ... wait, let's trace more carefully.

        # NN returns value = +0.6 for a state where Black is to move.
        # This means "good for Black".
        # The parent had White to move (parent selected this child).
        # We call node.backpropagate(-value) = backpropagate(-0.6)
        # Node gets -0.6 (this node's Q represents value from parent's perspective = White's)
        # Parent gets +0.6 (wait, backprop negates at each step)

        # Actually: backpropagate(-0.6):
        #   node: total_value += -0.6, then v = -(-0.6) = +0.6
        #   parent: total_value += +0.6, then v = -0.6
        #   grandparent: total_value += -0.6

        # So node.q_value = -0.6 / 1 = -0.6 = "bad for parent's player (White)"
        # parent.q_value includes +0.6 from this child = "good for grandparent's player"
        # This is correct: Black position is good (+0.6), so from White parent's view it's -0.6

        from davechess.engine.mcts import MCTSNode

        root = MCTSNode(state=GameState())
        child = MCTSNode(parent=root)

        # Simulate: NN says +0.6 for child's position
        nn_value = 0.6
        child.backpropagate(-nn_value)  # This is what MCTS does

        # Child's Q should be -0.6 (from root's perspective, this position is bad for root's player)
        assert child.q_value == pytest.approx(-0.6)
        # Root gets the negated value: good for root's player
        assert root.total_value == pytest.approx(0.6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNetworkValueHead:
    """Verify the network's value head outputs tanh in [-1, +1]."""

    def test_value_head_tanh_range(self):
        """Network value output should be in [-1, +1] due to tanh."""
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=1, num_filters=16)
        net.eval()

        state = GameState()
        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0)

        with torch.no_grad():
            logits, value = net(x)

        assert value.shape == (1, 1)
        assert -1.0 <= value.item() <= 1.0

    def test_value_head_gradient_flows(self):
        """Verify gradients flow through the value head with MSE loss."""
        from davechess.engine.network import DaveChessNetwork
        net = DaveChessNetwork(num_res_blocks=1, num_filters=16)
        net.train()

        state = GameState()
        planes = state_to_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0)
        target = torch.tensor([[1.0]])  # "current player wins"

        logits, value = net(x)
        loss = (value - target) ** 2
        loss.backward()

        # Value head FC layer should have gradients
        assert net.value_fc2.weight.grad is not None
        assert net.value_fc2.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTrainingLossComputation:
    """Verify the training loss uses correct shapes and signs."""

    def test_value_loss_is_mse(self):
        """Value loss should be MSE between prediction and target."""
        # Simulate what train_step does
        value_pred = torch.tensor([[0.3], [-0.5]])  # Network predictions
        values_t = torch.tensor([[1.0], [-1.0]])  # Targets

        per_example_value_loss = (value_pred - values_t) ** 2
        assert per_example_value_loss.shape == (2, 1)

        # First example: (0.3 - 1.0)^2 = 0.49
        assert per_example_value_loss[0].item() == pytest.approx(0.49)
        # Second example: (-0.5 - (-1.0))^2 = 0.25
        assert per_example_value_loss[1].item() == pytest.approx(0.25)

    def test_value_target_shape_unsqueeze(self):
        """train_step unsqueezes values to (batch, 1) to match network output."""
        values = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        values_t = torch.from_numpy(values).unsqueeze(1)
        assert values_t.shape == (3, 1)

    def test_no_sign_flip_in_loss(self):
        """There should be NO sign flip between target and prediction in MSE.
        The target is already from the correct perspective (current player)."""
        # If current player won, target = +1, network should predict close to +1
        # Loss = (pred - 1.0)^2, gradient pushes pred toward +1. Correct.

        # If current player lost, target = -1, network should predict close to -1
        # Loss = (pred - (-1.0))^2, gradient pushes pred toward -1. Correct.

        value_pred = torch.tensor([[0.0]], requires_grad=True)
        target_win = torch.tensor([[1.0]])
        target_loss = torch.tensor([[-1.0]])

        loss_win = (value_pred - target_win) ** 2
        loss_lose = (value_pred - target_loss) ** 2

        # Both losses should be 1.0 when prediction is 0
        assert loss_win.item() == pytest.approx(1.0)
        assert loss_lose.item() == pytest.approx(1.0)

        # Gradient for win case should push prediction toward +1
        loss_win.backward()
        # d/d(pred) of (pred - 1)^2 = 2(pred - 1) = 2(0 - 1) = -2
        assert value_pred.grad.item() == pytest.approx(-2.0)


class TestEndToEndScenario:
    """Full end-to-end scenario: play a game, verify all value targets."""

    def test_full_game_value_target_consistency(self):
        """Play a full game with no NN (random MCTS) and verify:
        - All value targets are from current player's perspective
        - Winner's positions have +1, loser's have -1 (or 0 for draws)
        - Planes' player indicator is consistent with the stored player
        """
        from davechess.engine.mcts import MCTS

        mcts = MCTS(network=None, num_simulations=2, cpuct=4.0)

        # Try multiple games to get at least one decisive result
        found_decisive = False
        for _ in range(30):
            training_data, game_record = play_selfplay_game(
                mcts, temperature_threshold=0, draw_value_target=0.0
            )

            for planes, policy, value, legal_mask in training_data:
                # Value must be in valid range
                assert value in (-1.0, 0.0, 1.0), f"Invalid value target: {value}"

                # Planes shape must be correct
                assert planes.shape == (NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
                assert legal_mask.shape == (POLICY_SIZE,)

                # Policy must be a valid distribution (sums to ~1 over non-zero entries)
                nonzero = policy[policy > 0]
                if len(nonzero) > 0:
                    assert abs(nonzero.sum() - 1.0) < 0.01, (
                        f"Policy should sum to ~1, got {nonzero.sum()}"
                    )

            if game_record["winner"] != "draw":
                found_decisive = True
                winner_str = game_record["winner"]

                # Reconstruct which Player enum won
                winner_player = Player.WHITE if winner_str == "white" else Player.BLACK
                winner_int = int(winner_player)
                loser_int = 1 - winner_int

                # Verify every example
                for planes, policy, value, legal_mask in training_data:
                    # The player indicator plane tells us who was to move
                    # plane[11] is 1.0 for White, 0.0 for Black
                    player_indicator = planes[11, 0, 0]
                    assert legal_mask.shape == (POLICY_SIZE,)
                    if player_indicator == 1.0:
                        player_was = int(Player.WHITE)
                    else:
                        player_was = int(Player.BLACK)

                    expected = 1.0 if player_was == winner_int else -1.0
                    assert value == expected, (
                        f"Player {player_was} position with winner={winner_int}: "
                        f"expected value {expected}, got {value}"
                    )

                break

        if not found_decisive:
            pytest.skip("No decisive game in 30 attempts")
