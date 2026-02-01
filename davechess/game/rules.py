"""Legal move generation, move execution, capture resolution, win conditions."""

from __future__ import annotations

from typing import Optional

from davechess.game.board import BOARD_SIZE, GOLD_NODES, POWER_NODES, ALL_NODES, RESOURCE_NODES
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
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

_GOLD_SET = frozenset(GOLD_NODES)
_POWER_SET = frozenset(POWER_NODES)
_ALL_NODES_SET = frozenset(ALL_NODES)
_ORTHOGONAL_SET = frozenset(ORTHOGONAL)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_resource_income(state: GameState, player: Player) -> int:
    """Calculate resource income for a player.

    +1 per Gold node that the player has a piece on or orthogonally adjacent to.
    """
    income = 0
    for nr, nc in GOLD_NODES:
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
    for nr, nc in ALL_NODES:
        if _player_controls_node(state, player, nr, nc) and \
           not _player_controls_node(state, opponent, nr, nc):
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


def _get_power_node_bonus(row: int, col: int) -> int:
    """Return +1 if (row, col) is on or adjacent (8-directional) to a Power node."""
    for pr, pc in POWER_NODES:
        if abs(row - pr) <= 1 and abs(col - pc) <= 1:
            return 1
    return 0


def _get_piece_strength(state: GameState, row: int, col: int) -> int:
    """Get total strength of piece at (row, col), including Power node bonus."""
    piece = state.board[row][col]
    if piece is None:
        return 0
    if piece.piece_type == PieceType.WARRIOR:
        base = _get_warrior_strength(state, row, col, piece.player)
    else:
        base = BASE_STRENGTH[piece.piece_type]
    return base + _get_power_node_bonus(row, col)


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

    # Check all 8 adjacent squares and 2-square Rider reach in one pass
    for dr, dc in ALL_DIRS:
        # Distance 1: Commander, Warrior (ortho only), Bombard melee, Rider
        r1, c1 = tr + dr, tc + dc
        if _in_bounds(r1, c1):
            p = board[r1][c1]
            if p is not None and p.player == by_player:
                pt = p.piece_type
                if pt == PieceType.COMMANDER or pt == PieceType.BOMBARD or pt == PieceType.RIDER:
                    return True
                if pt == PieceType.WARRIOR and (dr == 0 or dc == 0):
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

    _gen_deploy_moves(state, player, moves)
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

        if move.is_capture:
            defender = state.board[tr][tc]
            atk_str = _get_piece_strength(state, fr, fc)
            def_str = _get_piece_strength(state, tr, tc)

            if atk_str > def_str:
                state.board[tr][tc] = attacker
                state.board[fr][fc] = None
            elif atk_str < def_str:
                state.board[fr][fc] = None
            else:
                state.board[fr][fc] = None
                state.board[tr][tc] = None
        else:
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        state.board[tr][tc] = Piece(move.piece_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        state.board[tr][tc] = None

    return state


def generate_legal_moves(state: GameState) -> list[Move]:
    """Generate all legal moves for the current player.

    Filters out moves that leave the player's own Commander in check.
    Uses make/unmake optimization to avoid full state cloning.
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

        if move.is_capture:
            atk_str = _get_piece_strength(state, fr, fc)
            def_str = _get_piece_strength(state, tr, tc)
            if atk_str > def_str:
                # Attacker wins: piece moves to target
                board[fr][fc] = None
                board[tr][tc] = moving_piece
            elif atk_str < def_str:
                # Attacker loses: attacker removed
                board[fr][fc] = None
            else:
                # Tie: both removed
                board[fr][fc] = None
                board[tr][tc] = None
        else:
            board[fr][fc] = None
            board[tr][tc] = moving_piece

        # If the moving piece was our Commander and it was removed
        # (lost a capture or tied), the move is illegal.
        if moving_piece.piece_type == PieceType.COMMANDER and board[tr][tc] is not moving_piece:
            board[fr][fc] = moving_piece
            board[tr][tc] = captured_piece
            return False

        # Check if our Commander is safe
        safe = not is_in_check(state, player)

        # Unmake
        board[fr][fc] = moving_piece
        board[tr][tc] = captured_piece
        return safe

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        # Deploy adds a piece; doesn't move anything
        board[tr][tc] = Piece(move.piece_type, player)
        safe = not is_in_check(state, player)
        board[tr][tc] = None  # unmake
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
    """Commander: 1 square, any direction."""
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
    """Rider: up to 2 squares, straight line, no jumping."""
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


def _gen_deploy_moves(state: GameState, player: Player, moves: list[Move]):
    """Generate deployment moves onto back 2 rows."""
    if player == Player.WHITE:
        deploy_rows = [0, 1]
    else:
        deploy_rows = [6, 7]

    for piece_type in (PieceType.WARRIOR, PieceType.RIDER, PieceType.BOMBARD, PieceType.LANCER):
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

    # Checkmate/stalemate detection is handled lazily by generate_legal_moves()
    # when the next player tries to move and has no legal moves.

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
            atk_str = _get_piece_strength(state, fr, fc)
            def_str = _get_piece_strength(state, tr, tc)

            if atk_str > def_str:
                state.board[tr][tc] = attacker
                state.board[fr][fc] = None
                if defender.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = player
            elif atk_str < def_str:
                state.board[fr][fc] = None
                if attacker.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = opponent
            else:
                state.board[fr][fc] = None
                state.board[tr][tc] = None
                if attacker.piece_type == PieceType.COMMANDER:
                    state.done = True
                    state.winner = opponent
                if defender.piece_type == PieceType.COMMANDER:
                    state.done = True
                    if attacker.piece_type == PieceType.COMMANDER:
                        state.winner = opponent
                    else:
                        state.winner = player
        else:
            state.board[tr][tc] = attacker
            state.board[fr][fc] = None

    elif isinstance(move, Deploy):
        tr, tc = move.to_rc
        cost = DEPLOY_COST[move.piece_type]
        state.resources[player] -= cost
        state.board[tr][tc] = Piece(move.piece_type, player)

    elif isinstance(move, BombardAttack):
        tr, tc = move.target_rc
        defender = state.board[tr][tc]
        state.board[tr][tc] = None
        if defender is not None and defender.piece_type == PieceType.COMMANDER:
            state.done = True
            state.winner = player

    # Skip move_history for fast rollouts
    # state.move_history.append(move)

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
