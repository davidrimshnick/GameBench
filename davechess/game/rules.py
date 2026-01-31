"""Legal move generation, move execution, capture resolution, win conditions."""

from __future__ import annotations

from typing import Optional

from davechess.game.board import BOARD_SIZE, RESOURCE_NODES
from davechess.game.state import (
    BASE_STRENGTH, DEPLOY_COST, GameState, Move, MoveStep, Deploy,
    BombardAttack, Piece, PieceType, Player,
)

# Orthogonal directions
ORTHOGONAL = [(0, 1), (0, -1), (1, 0), (-1, 0)]
# All 8 directions
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# Straight-line directions (orthogonal + diagonal)
STRAIGHT_DIRS = ALL_DIRS

_RESOURCE_SET = frozenset(RESOURCE_NODES)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_resource_income(state: GameState, player: Player) -> int:
    """Calculate resource income for a player.

    +1 per resource node that the player has a piece on or orthogonally adjacent to.
    """
    income = 0
    for nr, nc in RESOURCE_NODES:
        # Check if player has a piece on or orthogonally adjacent to this node
        if _player_controls_node(state, player, nr, nc):
            income += 1
    return income


def _player_controls_node(state: GameState, player: Player, nr: int, nc: int) -> bool:
    """Check if player has a piece on or orthogonally adjacent to node at (nr, nc)."""
    # On the node
    piece = state.board[nr][nc]
    if piece is not None and piece.player == player:
        return True
    # Orthogonally adjacent
    for dr, dc in ORTHOGONAL:
        r2, c2 = nr + dr, nc + dc
        if _in_bounds(r2, c2):
            p2 = state.board[r2][c2]
            if p2 is not None and p2.player == player:
                return True
    return False


def _count_controlled_nodes(state: GameState, player: Player) -> int:
    """Count resource nodes exclusively controlled by player.

    A node is exclusively controlled if the player controls it and the opponent does not.
    """
    opponent = Player(1 - player)
    count = 0
    for nr, nc in RESOURCE_NODES:
        if _player_controls_node(state, player, nr, nc) and \
           not _player_controls_node(state, opponent, nr, nc):
            count += 1
    return count


def _count_pieces(state: GameState, player: Player) -> int:
    """Count total pieces for a player."""
    count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = state.board[row][col]
            if p is not None and p.player == player:
                count += 1
    return count


def _get_warrior_strength(state: GameState, row: int, col: int, player: Player) -> int:
    """Get warrior strength including adjacency bonus.

    +1 per adjacent friendly Warrior (orthogonal only).
    """
    strength = BASE_STRENGTH[PieceType.WARRIOR]
    for dr, dc in ORTHOGONAL:
        r2, c2 = row + dr, col + dc
        if _in_bounds(r2, c2):
            p2 = state.board[r2][c2]
            if p2 is not None and p2.player == player and p2.piece_type == PieceType.WARRIOR:
                strength += 1
    return strength


def _get_piece_strength(state: GameState, row: int, col: int) -> int:
    """Get total strength of piece at (row, col)."""
    piece = state.board[row][col]
    if piece is None:
        return 0
    if piece.piece_type == PieceType.WARRIOR:
        return _get_warrior_strength(state, row, col, piece.player)
    return BASE_STRENGTH[piece.piece_type]


def generate_legal_moves(state: GameState) -> list[Move]:
    """Generate all legal moves for the current player."""
    if state.done:
        return []

    player = state.current_player
    moves: list[Move] = []

    # Movement/capture moves for each piece
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None or piece.player != player:
                continue

            if piece.piece_type == PieceType.COMMANDER:
                _gen_commander_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.WARRIOR:
                _gen_warrior_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.RIDER:
                _gen_rider_moves(state, row, col, player, moves)
            elif piece.piece_type == PieceType.BOMBARD:
                _gen_bombard_moves(state, row, col, player, moves)

    # Deployment moves
    _gen_deploy_moves(state, player, moves)

    return moves


def _gen_commander_moves(state: GameState, row: int, col: int, player: Player,
                         moves: list[Move]):
    """Commander: 1-2 squares, any direction."""
    for dr, dc in ALL_DIRS:
        for dist in (1, 2):
            r2, c2 = row + dr * dist, col + dc * dist
            if not _in_bounds(r2, c2):
                break
            target = state.board[r2][c2]
            if target is None:
                moves.append(MoveStep((row, col), (r2, c2)))
            elif target.player != player:
                moves.append(MoveStep((row, col), (r2, c2), is_capture=True))
                break  # Can't jump over enemy
            else:
                break  # Blocked by friendly piece
            # For dist=1, if we could move there, try dist=2
            # But if dist=1 was a capture, we already broke
            # Check if path is clear for dist=2
            if dist == 1 and target is not None:
                break


def _gen_warrior_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Warrior: 1 square, orthogonal only."""
    for dr, dc in ORTHOGONAL:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))


def _gen_rider_moves(state: GameState, row: int, col: int, player: Player,
                     moves: list[Move]):
    """Rider: up to 3 squares, straight line, no jumping."""
    for dr, dc in STRAIGHT_DIRS:
        for dist in range(1, 4):
            r2, c2 = row + dr * dist, col + dc * dist
            if not _in_bounds(r2, c2):
                break
            target = state.board[r2][c2]
            if target is None:
                moves.append(MoveStep((row, col), (r2, c2)))
            elif target.player != player:
                moves.append(MoveStep((row, col), (r2, c2), is_capture=True))
                break
            else:
                break  # Blocked by friendly piece


def _gen_bombard_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Bombard: 1 square movement (any direction) + ranged capture at exactly 2 squares."""
    # Normal movement: 1 square, any direction
    for dr, dc in ALL_DIRS:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
            # Melee capture with strength 0 - still allowed, will likely lose
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))

    # Ranged attack: exactly 2 squares away, straight line, clear path
    for dr, dc in STRAIGHT_DIRS:
        # Check intermediate square is clear
        mid_r, mid_c = row + dr, col + dc
        if not _in_bounds(mid_r, mid_c):
            continue
        if state.board[mid_r][mid_c] is not None:
            continue  # Path blocked

        target_r, target_c = row + dr * 2, col + dc * 2
        if not _in_bounds(target_r, target_c):
            continue
        target = state.board[target_r][target_c]
        if target is not None and target.player != player:
            moves.append(BombardAttack((row, col), (target_r, target_c)))


def _gen_deploy_moves(state: GameState, player: Player, moves: list[Move]):
    """Generate deployment moves onto back 2 rows."""
    if player == Player.WHITE:
        deploy_rows = [0, 1]
    else:
        deploy_rows = [6, 7]

    for piece_type in (PieceType.WARRIOR, PieceType.RIDER, PieceType.BOMBARD):
        cost = DEPLOY_COST[piece_type]
        if state.resources[player] < cost:
            continue
        for row in deploy_rows:
            for col in range(BOARD_SIZE):
                if state.board[row][col] is None:
                    moves.append(Deploy(piece_type, (row, col)))


def apply_move(state: GameState, move: Move) -> GameState:
    """Apply a move and return the new state.

    This modifies the state in-place for performance, so clone first if needed.
    """
    player = state.current_player
    opponent = Player(1 - player)

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]

        if move.is_capture:
            defender = state.board[tr][tc]
            atk_str = _get_piece_strength(state, fr, fc)
            def_str = _get_piece_strength(state, tr, tc)

            if atk_str > def_str:
                # Attacker wins
                state.board[tr][tc] = attacker
                state.board[fr][fc] = None
                # Check if defender was Commander
                if defender.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = player
            elif atk_str < def_str:
                # Defender wins, attacker removed
                state.board[fr][fc] = None
                if attacker.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = opponent
            else:
                # Tie: both removed
                state.board[fr][fc] = None
                state.board[tr][tc] = None
                if attacker.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = opponent
                if defender.piece_type == PieceType.COMMANDER:
                    state.done = True
                    # If both commanders die, the attacker's side loses
                    # (their commander walked into a tie)
                    if attacker.piece_type == PieceType.COMMANDER:
                        state.winner = opponent
                    else:
                        state.winner = player
        else:
            # Simple move
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        cost = DEPLOY_COST[move.piece_type]
        state.resources[player] -= cost
        state.board[tr][tc] = Piece(move.piece_type, player)

    elif isinstance(move, BombardAttack):
        # Ranged capture: target is simply removed, bombard stays
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    state.move_history.append(move)

    # Check resource domination win (if not already won)
    if not state.done:
        exclusive = _count_controlled_nodes(state, player)
        if exclusive >= 6:
            state.done = True
            state.winner = player

    # Switch player and advance turn
    if not state.done:
        state.current_player = opponent
        if player == Player.BLACK:
            state.turn += 1

        # Gain resources at start of new player's turn
        income = get_resource_income(state, state.current_player)
        state.resources[state.current_player] += income

        # Check turn limit
        if state.turn > 200:
            state.done = True
            white_nodes = _count_controlled_nodes(state, Player.WHITE)
            black_nodes = _count_controlled_nodes(state, Player.BLACK)
            if white_nodes > black_nodes:
                state.winner = Player.WHITE
            elif black_nodes > white_nodes:
                state.winner = Player.BLACK
            else:
                # Tiebreak by piece count
                white_pieces = _count_pieces(state, Player.WHITE)
                black_pieces = _count_pieces(state, Player.BLACK)
                if white_pieces > black_pieces:
                    state.winner = Player.WHITE
                elif black_pieces > white_pieces:
                    state.winner = Player.BLACK
                else:
                    state.winner = None  # Draw

    return state


def check_winner(state: GameState) -> tuple[bool, Optional[Player]]:
    """Check if the game is over.

    Returns (is_done, winner) where winner is None for draw.
    """
    return state.done, state.winner


def get_controlled_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of resource nodes controlled by player (on or adjacent)."""
    return [(nr, nc) for nr, nc in RESOURCE_NODES
            if _player_controls_node(state, player, nr, nc)]


def get_exclusive_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of resource nodes exclusively controlled by player."""
    opponent = Player(1 - player)
    return [(nr, nc) for nr, nc in RESOURCE_NODES
            if _player_controls_node(state, player, nr, nc)
            and not _player_controls_node(state, opponent, nr, nc)]
