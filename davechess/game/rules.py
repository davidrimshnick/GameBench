"""Legal move generation, move execution, capture resolution, win conditions.

DaveChess v3: Chess-style capture (attacker always takes defender).
No deployment. Promotion: spend Gold resources to upgrade pieces in place.
Warriors capture diagonally forward (like pawns).
Bombard has ranged attack at exactly 2 squares (stays in place).
"""

from __future__ import annotations

from typing import Optional

from davechess.game.board import BOARD_SIZE, GOLD_NODES, ALL_NODES, RESOURCE_NODES
from davechess.game.state import (
    PROMOTION_COST, GameState, Move, MoveStep, Promote,
    BombardAttack, Piece, PieceType, Player,
)

# Orthogonal directions
ORTHOGONAL = [(0, 1), (0, -1), (1, 0), (-1, 0)]
# All 8 directions
ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# Straight-line directions (orthogonal + diagonal)
STRAIGHT_DIRS = ALL_DIRS
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

_GOLD_SET = frozenset(GOLD_NODES)
_ALL_NODES_SET = frozenset(ALL_NODES)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_resource_income(state: GameState, player: Player) -> int:
    """Calculate resource income for a player.

    +1 per Gold node that the player has a piece directly on.
    """
    income = 0
    for nr, nc in GOLD_NODES:
        piece = state.board[nr][nc]
        if piece is not None and piece.player == player:
            income += 1
    return income


def _player_controls_node(state: GameState, player: Player, nr: int, nc: int) -> bool:
    """Check if player has a piece on or orthogonally adjacent to node at (nr, nc).

    Note: resource income only requires being ON the node. This broader check
    is used for node-control queries (e.g. UI, analysis).
    """
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
    """Count resource nodes exclusively controlled by player."""
    opponent = Player(1 - player)
    count = 0
    for nr, nc in ALL_NODES:
        if _player_controls_node(state, player, nr, nc) and \
           not _player_controls_node(state, opponent, nr, nc):
            count += 1
    return count


def _find_commander(state: GameState, player: Player) -> tuple[int, int] | None:
    """Find the Commander's position for a player."""
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = state.board[row][col]
            if p is not None and p.player == player and p.piece_type == PieceType.COMMANDER:
                return (row, col)
    return None


def _is_square_attacked(state: GameState, tr: int, tc: int, by_player: Player) -> bool:
    """Check if any piece of by_player can attack the square (tr, tc).

    Checks outward from the target square along attack patterns.
    """
    board = state.board

    # Check all 8 adjacent squares and 2-square Rider reach
    for dr, dc in ALL_DIRS:
        # Distance 1: Commander (any dir), Bombard (any dir melee), Rider (any dir)
        r1, c1 = tr + dr, tc + dc
        if _in_bounds(r1, c1):
            p = board[r1][c1]
            if p is not None and p.player == by_player:
                pt = p.piece_type
                if pt == PieceType.COMMANDER or pt == PieceType.BOMBARD or pt == PieceType.RIDER:
                    return True
                # Warrior captures diagonally forward only
                if pt == PieceType.WARRIOR:
                    # Warrior at (r1,c1) captures by moving (-dr,-dc) to reach target
                    # For White: forward is +row, so capture dirs are (+1,+1) and (+1,-1)
                    # A White Warrior at r1 captures at r1+1 diagonally, so target is at r1+1
                    # From target perspective: dr = r1 - tr, so Warrior row = tr + dr
                    # The Warrior moves -dr to reach target. For White forward capture:
                    # move_dr must be +1 (forward), so -dr == +1, meaning dr == -1
                    # i.e. Warrior is one row BEHIND target (lower row for White)
                    move_dr = -dr
                    move_dc = -dc
                    # Must be diagonal
                    if move_dr != 0 and move_dc != 0:
                        forward = 1 if by_player == Player.WHITE else -1
                        if move_dr == forward:
                            return True

        # Distance 2: Rider only (straight line, clear path)
        r2, c2 = tr + dr * 2, tc + dc * 2
        if _in_bounds(r2, c2):
            p2 = board[r2][c2]
            if p2 is not None and p2.player == by_player and p2.piece_type == PieceType.RIDER:
                if _in_bounds(r1, c1) and board[r1][c1] is None:
                    return True

    # Lancer: diagonal directions, distance 1-4, can jump over one piece
    for dr, dc in DIAGONAL_DIRS:
        blocking = 0
        for dist in range(1, 5):
            r, c = tr + dr * dist, tc + dc * dist
            if not _in_bounds(r, c):
                break
            p = board[r][c]
            if p is not None:
                if p.player == by_player and p.piece_type == PieceType.LANCER and blocking <= 1:
                    return True
                blocking += 1
                if blocking > 1:
                    break

    # Bombard ranged: exactly 2 squares orthogonal, clear path, but NOT against Commander
    # We only need this for general "is square attacked" — but Bombard can't target Commander
    # So for check detection this doesn't apply (Commanders can't be bombarded)
    # For other purposes (like protecting squares), we check:
    target_piece = board[tr][tc]
    if target_piece is None or target_piece.piece_type != PieceType.COMMANDER:
        for dr, dc in STRAIGHT_DIRS:
            br, bc = tr + dr * 2, tc + dc * 2
            if _in_bounds(br, bc):
                bp = board[br][bc]
                if bp is not None and bp.player == by_player and bp.piece_type == PieceType.BOMBARD:
                    # Check clear path
                    mr, mc = tr + dr, tc + dc
                    if _in_bounds(mr, mc) and board[mr][mc] is None:
                        return True

    return False


def is_in_check(state: GameState, player: Player) -> bool:
    """Check if the given player's Commander is under attack."""
    cmd_pos = _find_commander(state, player)
    if cmd_pos is None:
        return False  # Commander already captured
    opponent = Player(1 - player)
    return _is_square_attacked(state, cmd_pos[0], cmd_pos[1], opponent)


def _generate_pseudo_legal_moves(state: GameState) -> list[Move]:
    """Generate all pseudo-legal moves (ignoring check)."""
    player = state.current_player
    moves: list[Move] = []

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
            elif piece.piece_type == PieceType.LANCER:
                _gen_lancer_moves(state, row, col, player, moves)

    _gen_promotion_moves(state, player, moves)
    return moves


def _apply_move_no_checks(state: GameState, move: Move) -> GameState:
    """Apply a move without win-condition or turn-switching logic.

    Used internally for check detection (just updates the board).
    Modifies state in place.
    """
    player = state.current_player

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]
        # Chess-style: attacker always takes the square
        state.board[tr][tc] = attacker
        state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        state.board[tr][tc] = None

    return state


def generate_legal_moves(state: GameState) -> list[Move]:
    """Generate all legal moves for the current player.

    Filters out moves that leave the player's own Commander in check.
    Also detects checkmate/stalemate: if no legal moves exist, sets state.done.
    """
    if state.done:
        return []

    player = state.current_player
    pseudo_moves = _generate_pseudo_legal_moves(state)

    legal_moves = []
    for move in pseudo_moves:
        if _is_move_legal(state, move, player):
            legal_moves.append(move)

    # Detect checkmate/stalemate
    if not legal_moves:
        state.done = True
        if is_in_check(state, player):
            # Checkmate: current player loses
            state.winner = Player(1 - player)
        else:
            # Stalemate: draw
            state.winner = None

    return legal_moves


def _is_move_legal(state: GameState, move: Move, player: Player) -> bool:
    """Check if a move is legal (doesn't leave own Commander in check).

    Uses make/unmake on the board to avoid cloning.
    """
    board = state.board

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        moving_piece = board[fr][fc]
        captured_piece = board[tr][tc]

        # Chess-style: attacker always takes the square
        board[fr][fc] = None
        board[tr][tc] = moving_piece

        # Check if our Commander is safe
        safe = not is_in_check(state, player)

        # Unmake
        board[fr][fc] = moving_piece
        board[tr][tc] = captured_piece
        return safe

    elif isinstance(move, Promote):
        r, c = move.from_rc
        old_piece = board[r][c]
        board[r][c] = Piece(move.to_type, player)
        safe = not is_in_check(state, player)
        board[r][c] = old_piece  # unmake
        return safe

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        captured_piece = board[tr][tc]
        board[tr][tc] = None  # target removed
        safe = not is_in_check(state, player)
        board[tr][tc] = captured_piece  # unmake
        return safe

    return True


def _gen_commander_moves(state: GameState, row: int, col: int, player: Player,
                         moves: list[Move]):
    """Commander: 1 square, any direction. Captures same as movement."""
    for dr, dc in ALL_DIRS:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))


def _gen_warrior_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Warrior: moves 1 square forward, captures 1 square diagonal-forward.

    Like a chess pawn. Forward is +row for White, -row for Black.
    """
    forward = 1 if player == Player.WHITE else -1

    # Forward move (non-capture only)
    r2 = row + forward
    if _in_bounds(r2, col) and state.board[r2][col] is None:
        moves.append(MoveStep((row, col), (r2, col)))

    # Diagonal-forward captures
    for dc in (-1, 1):
        c2 = col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is not None and target.player != player:
            moves.append(MoveStep((row, col), (r2, c2), is_capture=True))


def _gen_rider_moves(state: GameState, row: int, col: int, player: Player,
                     moves: list[Move]):
    """Rider: up to 2 squares, any straight line (orthogonal + diagonal), no jumping."""
    for dr, dc in STRAIGHT_DIRS:
        for dist in range(1, 3):
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
    """Bombard: 1 square movement (any direction) + ranged capture at exactly 2 squares.

    Melee: moves 1 square any direction, can capture adjacent enemies.
    Ranged: attacks at exactly 2 squares orthogonal/diagonal, clear path,
            Bombard stays in place. Cannot target Commanders.
    """
    # Normal movement/capture: 1 square, any direction
    for dr, dc in ALL_DIRS:
        r2, c2 = row + dr, col + dc
        if not _in_bounds(r2, c2):
            continue
        target = state.board[r2][c2]
        if target is None:
            moves.append(MoveStep((row, col), (r2, c2)))
        elif target.player != player:
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
        if target is not None and target.player != player \
                and target.piece_type != PieceType.COMMANDER:
            moves.append(BombardAttack((row, col), (target_r, target_c)))


def _gen_lancer_moves(state: GameState, row: int, col: int, player: Player,
                       moves: list[Move]):
    """Lancer: diagonal up to 4 squares, can jump over exactly one piece (any color)."""
    for dr, dc in DIAGONAL_DIRS:
        pieces_in_way = 0
        for dist in range(1, 5):
            r2, c2 = row + dr * dist, col + dc * dist
            if not _in_bounds(r2, c2):
                break
            occupant = state.board[r2][c2]
            if occupant is None:
                # Empty: can land here
                moves.append(MoveStep((row, col), (r2, c2)))
            elif occupant.player == player:
                # Friendly piece: jump over if first, blocked if second
                pieces_in_way += 1
                if pieces_in_way > 1:
                    break
            else:
                # Enemy piece: can capture, then this square blocks further
                if pieces_in_way <= 1:
                    moves.append(MoveStep((row, col), (r2, c2), is_capture=True))
                pieces_in_way += 1
                if pieces_in_way > 1:
                    break


def _gen_promotion_moves(state: GameState, player: Player, moves: list[Move]):
    """Generate promotion moves: upgrade a friendly piece in place.

    Any non-Commander piece can promote to a higher-cost type.
    Cost = full price of target type.
    """
    resources = state.resources[player]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = state.board[row][col]
            if piece is None or piece.player != player:
                continue
            if piece.piece_type == PieceType.COMMANDER:
                continue  # Commander can't promote

            for target_type, cost in PROMOTION_COST.items():
                if resources < cost:
                    continue
                # Can only promote to a different type
                if piece.piece_type == target_type:
                    continue
                moves.append(Promote((row, col), target_type))


def apply_move(state: GameState, move: Move) -> GameState:
    """Apply a move and return the new state.

    This modifies the state in-place for performance, so clone first if needed.
    Chess-style capture: attacker always takes the defender's square.
    """
    player = state.current_player
    opponent = Player(1 - player)

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]

        if move.is_capture:
            defender = state.board[tr][tc]
            # Chess-style: attacker always wins
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None
            # Check if defender was Commander
            if defender.piece_type == PieceType.COMMANDER:
                state.done = True
                state.winner = player
        else:
            # Simple move
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        cost = PROMOTION_COST[move.to_type]
        state.resources[player] -= cost
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        # Ranged capture: target is simply removed, bombard stays
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    state.move_history.append(move)

    # Update halfmove clock (50-move rule): reset on capture/promote/bombard, else increment
    if isinstance(move, MoveStep) and move.is_capture:
        state.halfmove_clock = 0
    elif isinstance(move, (Promote, BombardAttack)):
        state.halfmove_clock = 0
    else:
        state.halfmove_clock += 1

    # Switch player and advance turn
    if not state.done:
        state.current_player = opponent
        if player == Player.BLACK:
            state.turn += 1

        # Gain resources at start of new player's turn
        income = get_resource_income(state, state.current_player)
        state.resources[state.current_player] += income

        # Check turn limit — draw if no checkmate by turn 100
        if state.turn > 100:
            state.done = True
            state.winner = None  # Draw

        # Check 50-move rule — draw if 100 halfmoves with no capture or promotion
        if not state.done and state.halfmove_clock >= 100:
            state.done = True
            state.winner = None  # Draw by 50-move rule

        # Check threefold repetition — draw if same position occurs 3 times
        if not state.done:
            pos_key = state.get_position_key()
            state.position_counts[pos_key] = state.position_counts.get(pos_key, 0) + 1
            if state.position_counts[pos_key] >= 3:
                state.done = True
                state.winner = None  # Draw by repetition

    return state


def generate_pseudo_legal_moves(state: GameState) -> list[Move]:
    """Generate pseudo-legal moves (no check filtering). For fast rollouts."""
    if state.done:
        return []
    return _generate_pseudo_legal_moves(state)


def apply_move_fast(state: GameState, move: Move) -> GameState:
    """Apply a move without checkmate/stalemate detection. For fast rollouts.

    Still handles captures, win conditions, resource income, and turn switching,
    but skips the expensive check for whether the opponent has legal moves.
    """
    player = state.current_player
    opponent = Player(1 - player)

    if isinstance(move, MoveStep):
        fr, fc = move.from_rc
        tr, tc = move.to_rc
        attacker = state.board[fr][fc]

        if move.is_capture:
            defender = state.board[tr][tc]
            # Chess-style: attacker always wins
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None
            if defender.piece_type == PieceType.COMMANDER:
                state.done = True
                state.winner = player
        else:
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Promote):
        r, c = move.from_rc
        cost = PROMOTION_COST[move.to_type]
        state.resources[player] -= cost
        state.board[r][c] = Piece(move.to_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    # Update halfmove clock (50-move rule)
    if isinstance(move, MoveStep) and move.is_capture:
        state.halfmove_clock = 0
    elif isinstance(move, (Promote, BombardAttack)):
        state.halfmove_clock = 0
    else:
        state.halfmove_clock += 1

    if not state.done:
        state.current_player = opponent
        if player == Player.BLACK:
            state.turn += 1

        income = get_resource_income(state, state.current_player)
        state.resources[state.current_player] += income

        # Check turn limit — draw if no checkmate by turn 100
        if state.turn > 100:
            state.done = True
            state.winner = None  # Draw

        # Check 50-move rule — draw if 100 halfmoves with no capture or promotion
        if not state.done and state.halfmove_clock >= 100:
            state.done = True
            state.winner = None  # Draw by 50-move rule

        # Check threefold repetition (same logic as apply_move)
        if not state.done:
            pos_key = state.get_position_key()
            state.position_counts[pos_key] = state.position_counts.get(pos_key, 0) + 1
            if state.position_counts[pos_key] >= 3:
                state.done = True
                state.winner = None  # Draw by repetition

    return state


def check_winner(state: GameState) -> tuple[bool, Optional[Player]]:
    """Check if the game is over.

    Returns (is_done, winner) where winner is None for draw.
    """
    return state.done, state.winner


def get_controlled_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of all nodes controlled by player (on or adjacent)."""
    return [(nr, nc) for nr, nc in ALL_NODES
            if _player_controls_node(state, player, nr, nc)]


def get_exclusive_nodes(state: GameState, player: Player) -> list[tuple[int, int]]:
    """Return list of all nodes exclusively controlled by player."""
    opponent = Player(1 - player)
    return [(nr, nc) for nr, nc in ALL_NODES
            if _player_controls_node(state, player, nr, nc)
            and not _player_controls_node(state, opponent, nr, nc)]
