"""ResNet policy+value network for DaveChess AlphaZero.

Input: 12 planes of 8x8
  Planes 0-3: Current player's C/W/R/B positions (binary)
  Planes 4-7: Opponent's C/W/R/B positions (binary)
  Plane 8: Resource node positions (binary)
  Plane 9: Current player indicator (all 1s or 0s)
  Plane 10: Current player resources (scalar broadcast, normalized)
  Plane 11: Opponent resources (scalar broadcast, normalized)

Output:
  Policy: flat logit vector over all possible moves (~4736 logits)
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

from davechess.game.board import BOARD_SIZE, RESOURCE_NODES
from davechess.game.state import GameState, Player, PieceType, Move, MoveStep, Deploy, BombardAttack

# Move encoding:
# For each source square (64), encode move type:
#   - Direction moves: 8 directions x 3 max distance = 24
#   - Deploy: 3 piece types (W, R, B) = 3 (source square = deploy target)
#   - Bombard: 8 directions x 1 (always distance 2) = 8
# Total per square: 24 + 3 + 8 = 35
# But we only need deploy on deploy squares, so we use a flat encoding.
# For simplicity: 64 * (24 + 8) + 64 * 3 = 64 * 35 = 2240
# Actually, let's use a cleaner encoding:

# Move types per source square:
#   Slots 0-23: direction moves (8 dirs x 3 distances)
#   Slots 24-31: bombard ranged attacks (8 dirs, always dist 2)
#   Slots 32-34: deploy piece types (W=0, R=1, B=2) — only meaningful if source=deploy target
# Total: 35 slots per square, 64 squares = 2240 total
MOVES_PER_SQUARE = 35
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * MOVES_PER_SQUARE  # 2240

# Direction encoding
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(ALL_DIRS)}

# Resource normalization constant
RESOURCE_NORM = 50.0


def state_to_planes(state: GameState) -> np.ndarray:
    """Convert game state to 12x8x8 input planes.

    The planes are always from the perspective of the current player.
    """
    planes = np.zeros((12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    current = state.current_player
    opponent = Player(1 - current)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None:
                continue
            pt = int(piece.piece_type)
            if piece.player == current:
                planes[pt, row, col] = 1.0
            else:
                planes[4 + pt, row, col] = 1.0

    # Resource nodes
    for r, c in RESOURCE_NODES:
        planes[8, r, c] = 1.0

    # Current player indicator
    planes[9, :, :] = 1.0 if current == Player.WHITE else 0.0

    # Resources (normalized, broadcast)
    planes[10, :, :] = min(state.resources[current] / RESOURCE_NORM, 1.0)
    planes[11, :, :] = min(state.resources[opponent] / RESOURCE_NORM, 1.0)

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
            # Handle non-unit diagonal (shouldn't happen with our rules)
            return 0
        dir_idx = DIR_TO_IDX[d]
        slot = dir_idx * 3 + (dist - 1)  # 0-23
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
        slot = 24 + dir_idx  # 24-31
        sq_idx = fr * BOARD_SIZE + fc
        return sq_idx * MOVES_PER_SQUARE + slot

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        # Map piece type to deploy slot
        deploy_map = {PieceType.WARRIOR: 0, PieceType.RIDER: 1, PieceType.BOMBARD: 2}
        slot = 32 + deploy_map[move.piece_type]
        sq_idx = tr * BOARD_SIZE + tc
        return sq_idx * MOVES_PER_SQUARE + slot

    return 0


def policy_index_to_move(index: int, state: GameState) -> Move | None:
    """Decode a policy index back to a move (best effort, may not be legal)."""
    sq_idx = index // MOVES_PER_SQUARE
    slot = index % MOVES_PER_SQUARE
    row = sq_idx // BOARD_SIZE
    col = sq_idx % BOARD_SIZE

    if slot < 24:
        # Direction move
        dir_idx = slot // 3
        dist = (slot % 3) + 1
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * dist, col + dc * dist
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            target = state.board[tr][tc]
            is_capture = target is not None and target.player != state.current_player
            return MoveStep((row, col), (tr, tc), is_capture=is_capture)

    elif slot < 32:
        # Bombard ranged attack
        dir_idx = slot - 24
        dr, dc = ALL_DIRS[dir_idx]
        tr, tc = row + dr * 2, col + dc * 2
        if 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
            return BombardAttack((row, col), (tr, tc))

    elif slot < 35:
        # Deploy
        deploy_map = {0: PieceType.WARRIOR, 1: PieceType.RIDER, 2: PieceType.BOMBARD}
        piece_type = deploy_map[slot - 32]
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
                     input_planes: int = 12):
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
                x: (batch, 12, 8, 8) tensor.

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
