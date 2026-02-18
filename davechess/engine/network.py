"""ResNet policy+value network for DaveChess AlphaZero.

Input: 18 planes of 8x8
  Planes 0-4: Current player's C/W/R/B/L positions (binary)
  Planes 5-9: Opponent's C/W/R/B/L positions (binary)
  Plane 10: Gold node positions (binary)
  Plane 11: Current player indicator (all 1s or 0s)
  Plane 12: Current player resources (scalar broadcast, normalized)
  Plane 13: Opponent resources (scalar broadcast, normalized)
  Plane 14: Turn progress (turn/100, broadcast — urgency signal)
  Plane 15: Position repetition count (0/0.5/1.0 for 1x/2x/3x+)
  Plane 16: Last move source square (binary one-hot)
  Plane 17: Last move destination square (binary one-hot)

Output:
  Policy: flat logit vector over all possible moves (4288 logits)
  Value: single tanh scalar (-1 to +1)
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from davechess.game.board import BOARD_SIZE, GOLD_NODES
from davechess.game.state import GameState, Player, PieceType, Move, MoveStep, Promote, BombardAttack

# Move encoding per source square:
#   Slots 0-55:  direction moves (8 dirs x 7 max distances)
#   Slots 56-63: bombard ranged attacks (8 dirs, always dist 2)
#   Slots 64-66: promotion target types (R=0, B=1, L=2)
# Total: 67 slots per square, 64 squares = 4288 total
MOVES_PER_SQUARE = 67
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * MOVES_PER_SQUARE  # 4288

# Direction encoding
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(ALL_DIRS)}

# Resource normalization constant
RESOURCE_NORM = 50.0


NUM_INPUT_PLANES = 18


def state_to_planes(state: GameState) -> np.ndarray:
    """Convert game state to 18x8x8 input planes.

    The planes are always from the perspective of the current player.
    """
    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    current = state.current_player
    opponent = Player(1 - current)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None:
                continue
            pt = int(piece.piece_type)  # 0-4
            if piece.player == current:
                planes[pt, row, col] = 1.0
            else:
                planes[5 + pt, row, col] = 1.0

    # Gold nodes
    for r, c in GOLD_NODES:
        planes[10, r, c] = 1.0

    # Current player indicator
    planes[11, :, :] = 1.0 if current == Player.WHITE else 0.0

    # Resources (normalized, broadcast)
    planes[12, :, :] = min(state.resources[current] / RESOURCE_NORM, 1.0)
    planes[13, :, :] = min(state.resources[opponent] / RESOURCE_NORM, 1.0)

    # Turn progress: urgency signal (0.0 at turn 1, 1.0 at turn 100)
    planes[14, :, :] = min(state.turn / 100.0, 1.0)

    # Repetition count for current position
    pos_key = state.get_position_key()
    rep_count = state.position_counts.get(pos_key, 1)
    # Encode: 1x=0.0, 2x=0.5, 3x+=1.0
    planes[15, :, :] = min((rep_count - 1) * 0.5, 1.0)

    # Last move source and destination squares
    if state.last_move is not None:
        move = state.last_move
        if isinstance(move, MoveStep):
            fr, fc = move.from_rc
            tr, tc = move.to_rc
            planes[16, fr, fc] = 1.0
            planes[17, tr, tc] = 1.0
        elif isinstance(move, Promote):
            # Promotion: piece stays in place, mark the square
            r, c = move.from_rc
            planes[16, r, c] = 1.0
            planes[17, r, c] = 1.0
        elif isinstance(move, BombardAttack):
            fr, fc = move.from_rc
            tr, tc = move.target_rc
            planes[16, fr, fc] = 1.0
            planes[17, tr, tc] = 1.0

    return planes


def move_to_policy_index(move: Move) -> int:
    """Encode a move as an index into the flat policy vector."""
    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        dr = tr - fr
        dc = tc - fc
        # Determine direction and distance
        dist = max(abs(dr), abs(dc))
        if dist == 0:
            return 0  # Should not happen
        # Normalize direction
        d = (dr // dist, dc // dist)
        if d not in DIR_TO_IDX:
            return 0
        dir_idx = DIR_TO_IDX[d]
        slot = dir_idx * 7 + (dist - 1)  # 0-55
        sq_idx = fr * BOARD_SIZE + fc
        return sq_idx * MOVES_PER_SQUARE + slot

    elif isinstance(move, BombardAttack):
        fr, fc = move.from_rc
        tr, tc = move.target_rc
        dr = tr - fr
        dc = tc - fc
        dist = max(abs(dr), abs(dc))
        d = (dr // dist, dc // dist)
        dir_idx = DIR_TO_IDX.get(d, 0)
        slot = 56 + dir_idx  # 56-63
        sq_idx = fr * BOARD_SIZE + fc
        return sq_idx * MOVES_PER_SQUARE + slot

    elif isinstance(move, Promote):
        r, c = move.from_rc
        promote_map = {
            PieceType.RIDER: 0, PieceType.BOMBARD: 1,
            PieceType.LANCER: 2,
        }
        slot = 64 + promote_map[move.to_type]  # 64-66
        sq_idx = r * BOARD_SIZE + c
        return sq_idx * MOVES_PER_SQUARE + slot

    return 0


def policy_index_to_move(index: int, state: GameState) -> Move | None:
    """Decode a policy index back to a move (best effort, may not be legal)."""
    sq_idx = index // MOVES_PER_SQUARE
    slot = index % MOVES_PER_SQUARE
    row = sq_idx // BOARD_SIZE
    col = sq_idx % BOARD_SIZE

    if slot < 56:
        # Direction move
        dir_idx = slot // 7
        dist = (slot % 7) + 1
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * dist, col + dc * dist
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            target = state.board[tr][tc]
            is_capture = target is not None and target.player != state.current_player
            return MoveStep((row, col), (tr, tc), is_capture=is_capture)

    elif slot < 64:
        # Bombard ranged attack
        dir_idx = slot - 56
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * 2, col + dc * 2
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            return BombardAttack((row, col), (tr, tc))

    elif slot < 67:
        # Promotion
        promote_map = {
            0: PieceType.RIDER, 1: PieceType.BOMBARD,
            2: PieceType.LANCER,
        }
        to_type = promote_map[slot - 64]
        return Promote((row, col), to_type)

    return None


if HAS_TORCH:
    class ResBlock(nn.Module):
        """Residual block with two conv layers and batch norm."""

        def __init__(self, num_filters: int):
            super().__init__()
            self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_filters)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x = x + residual
            x = F.relu(x)
            return x

    class DaveChessNetwork(nn.Module):
        """ResNet policy+value network for DaveChess."""

        def __init__(self, num_res_blocks: int = 5, num_filters: int = 64,
                     input_planes: int = NUM_INPUT_PLANES):
            super().__init__()

            # Initial convolution
            self.conv_input = nn.Conv2d(input_planes, num_filters, 3, padding=1, bias=False)
            self.bn_input = nn.BatchNorm2d(num_filters)

            # Residual blocks
            self.res_blocks = nn.ModuleList([
                ResBlock(num_filters) for _ in range(num_res_blocks)
            ])

            # Policy head
            self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)

            # Value head
            self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
            self.value_fc2 = nn.Linear(64, 1)

        def forward(self, x):
            """Forward pass.

            Args:
                x: (batch, input_planes, 8, 8) tensor.

            Returns:
                (policy_logits, value) where policy is (batch, POLICY_SIZE)
                and value is (batch, 1).
            """
            # Common trunk
            x = F.relu(self.bn_input(self.conv_input(x)))
            for block in self.res_blocks:
                x = block(x)

            # Policy head
            p = F.relu(self.policy_bn(self.policy_conv(x)))
            p = p.view(p.size(0), -1)
            p = self.policy_fc(p)

            # Value head
            v = F.relu(self.value_bn(self.value_conv(x)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v))

            return p, v

        def predict(self, state: GameState) -> tuple[np.ndarray, float]:
            """Get policy and value for a single state.

            Returns:
                (policy_probs, value) - policy is softmax over legal moves.
            """
            self.eval()
            planes = state_to_planes(state)
            x = torch.from_numpy(planes).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()

            with torch.no_grad():
                logits, value = self(x)

            policy = F.softmax(logits[0], dim=0).cpu().numpy()
            return policy, value.item()

        @classmethod
        def from_checkpoint(cls, path: str, device: str = "cpu"):
            """Load network from checkpoint, inferring architecture from weights.

            Returns:
                (network, checkpoint_dict)
            """
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state_dict = ckpt["network_state"]

            # Infer architecture from state dict
            conv_input_weight = state_dict["conv_input.weight"]
            num_filters = conv_input_weight.shape[0]
            input_planes = conv_input_weight.shape[1]
            max_block = max(
                int(k.split(".")[1]) for k in state_dict if k.startswith("res_blocks.")
            )
            num_res_blocks = max_block + 1

            net = cls(num_res_blocks=num_res_blocks, num_filters=num_filters,
                      input_planes=input_planes)
            net.load_state_dict(state_dict)
            net.eval()
            if device != "cpu":
                net = net.to(device)
            return net, ckpt
