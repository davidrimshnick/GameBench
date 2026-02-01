"""ResNet policy+value network for DaveChess AlphaZero.

Input: 15 planes of 8x8
  Planes 0-4: Current player's C/W/R/B/L positions (binary)
  Planes 5-9: Opponent's C/W/R/B/L positions (binary)
  Plane 10: Gold node positions (binary)
  Plane 11: Power node positions (binary)
  Plane 12: Current player indicator (all 1s or 0s)
  Plane 13: Current player resources (scalar broadcast, normalized)
  Plane 14: Opponent resources (scalar broadcast, normalized)

Output:
  Policy: flat logit vector over all possible moves (2816 logits)
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

from davechess.game.board import BOARD_SIZE, GOLD_NODES, POWER_NODES
from davechess.game.state import GameState, Player, PieceType, Move, MoveStep, Deploy, BombardAttack

# Move encoding per source square:
#   Slots 0-31:  direction moves (8 dirs x 4 max distances)
#   Slots 32-39: bombard ranged attacks (8 dirs, always dist 2)
#   Slots 40-43: deploy piece types (W=0, R=1, B=2, L=3)
# Total: 44 slots per square, 64 squares = 2816 total
MOVES_PER_SQUARE = 44
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * MOVES_PER_SQUARE  # 2816

# Direction encoding
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(ALL_DIRS)}

# Resource normalization constant
RESOURCE_NORM = 50.0


def state_to_planes(state: GameState) -> np.ndarray:
    """Convert game state to 15x8x8 input planes.

    The planes are always from the perspective of the current player.
    """
    planes = np.zeros((15, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
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

    # Power nodes
    for r, c in POWER_NODES:
        planes[11, r, c] = 1.0

    # Current player indicator
    planes[12, :, :] = 1.0 if current == Player.WHITE else 0.0

    # Resources (normalized, broadcast)
    planes[13, :, :] = min(state.resources[current] / RESOURCE_NORM, 1.0)
    planes[14, :, :] = min(state.resources[opponent] / RESOURCE_NORM, 1.0)

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
        slot = dir_idx * 4 + (dist - 1)  # 0-31
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
        slot = 32 + dir_idx  # 32-39
        sq_idx = fr * BOARD_SIZE + fc
        return sq_idx * MOVES_PER_SQUARE + slot

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        deploy_map = {
            PieceType.WARRIOR: 0, PieceType.RIDER: 1,
            PieceType.BOMBARD: 2, PieceType.LANCER: 3,
        }
        slot = 40 + deploy_map[move.piece_type]  # 40-43
        sq_idx = tr * BOARD_SIZE + tc
        return sq_idx * MOVES_PER_SQUARE + slot

    return 0


def policy_index_to_move(index: int, state: GameState) -> Move | None:
    """Decode a policy index back to a move (best effort, may not be legal)."""
    sq_idx = index // MOVES_PER_SQUARE
    slot = index % MOVES_PER_SQUARE
    row = sq_idx // BOARD_SIZE
    col = sq_idx % BOARD_SIZE

    if slot < 32:
        # Direction move
        dir_idx = slot // 4
        dist = (slot % 4) + 1
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * dist, col + dc * dist
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            target = state.board[tr][tc]
            is_capture = target is not None and target.player != state.current_player
            return MoveStep((row, col), (tr, tc), is_capture=is_capture)

    elif slot < 40:
        # Bombard ranged attack
        dir_idx = slot - 32
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * 2, col + dc * 2
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            return BombardAttack((row, col), (tr, tc))

    elif slot < 44:
        # Deploy
        deploy_map = {
            0: PieceType.WARRIOR, 1: PieceType.RIDER,
            2: PieceType.BOMBARD, 3: PieceType.LANCER,
        }
        piece_type = deploy_map[slot - 40]
        return Deploy(piece_type, (row, col))

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
                     input_planes: int = 15):
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
                x: (batch, 15, 8, 8) tensor.

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
