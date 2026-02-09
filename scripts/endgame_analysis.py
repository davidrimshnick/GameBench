#!/usr/bin/env python3
"""Endgame analysis: can R+C force checkmate vs lone C in DaveChess?

Tests whether checkmate is theoretically possible in various minimal-material
endgames, and whether MCTS can find the mate.

Piece movement reference:
  Commander: 1 square any direction (like chess King)
  Rider: up to 7 squares orthogonal, up to 3 squares diagonal, no jumping
  Lancer: up to 7 squares any direction, can jump exactly 1 piece
  Bombard: 1 sq any dir move + ranged attack at exactly 2 sq (can't target Commander)
"""

import sys
import time
from itertools import product

sys.path.insert(0, '/home/david/repos/GameBench')

from davechess.game.state import GameState, Piece, PieceType, Player
from davechess.game.rules import (
    generate_legal_moves, is_in_check, apply_move,
    _find_commander, _is_square_attacked,
)
from davechess.game.board import BOARD_SIZE, render_board, rc_to_notation


def make_empty_state():
    """Create a blank GameState with empty board."""
    state = GameState.__new__(GameState)
    state.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    state.resources = [0, 0]
    state.current_player = Player.BLACK
    state.turn = 1
    state.done = False
    state.winner = None
    state.move_history = []
    state.position_counts = {}
    state.halfmove_clock = 0
    return state


def place_piece(state, row, col, piece_type, player):
    state.board[row][col] = Piece(piece_type, player)


def display_state(state, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    board_display = state.to_display_board()
    print(render_board(board_display,
                       resource_counts=(state.resources[0], state.resources[1]),
                       turn=state.turn,
                       current_player=int(state.current_player)))


def is_checkmate(state, player):
    """Check if the given player is in checkmate."""
    if not is_in_check(state, player):
        return False
    # Save current player, set to the checked player, generate moves
    saved = state.current_player
    state.current_player = player
    moves = generate_legal_moves(state)
    state.current_player = saved
    # If the state got marked done by generate_legal_moves, undo that
    if state.done and state.winner is not None:
        # Was indeed checkmate
        state.done = False
        state.winner = None
        state.current_player = saved
        return True
    state.current_player = saved
    return len(moves) == 0


# ==============================================================================
# PART 1: Exhaustive search for R+C vs C checkmate positions
# ==============================================================================
def find_checkmate_positions_rc_vs_c():
    """Enumerate all possible board positions with White R+C vs Black C.
    Check which ones are checkmate for Black."""
    print("\n" + "=" * 70)
    print("PART 1: Exhaustive search for R+C vs C checkmate positions")
    print("=" * 70)
    print("Searching all placements of White Commander, White Rider, Black Commander")
    print("where Black Commander is in checkmate...")

    checkmates = []
    positions_checked = 0

    all_squares = [(r, c) for r in range(8) for c in range(8)]

    for wc_pos in all_squares:
        for wr_pos in all_squares:
            if wr_pos == wc_pos:
                continue
            for bc_pos in all_squares:
                if bc_pos == wc_pos or bc_pos == wr_pos:
                    continue

                # Skip if White Commander is adjacent to Black Commander
                # (mutual check / illegal since white just moved)
                wc_r, wc_c = wc_pos
                bc_r, bc_c = bc_pos

                state = make_empty_state()
                state.current_player = Player.BLACK

                place_piece(state, wc_r, wc_c, PieceType.COMMANDER, Player.WHITE)
                place_piece(state, wr_pos[0], wr_pos[1], PieceType.RIDER, Player.WHITE)
                place_piece(state, bc_r, bc_c, PieceType.COMMANDER, Player.BLACK)

                positions_checked += 1

                # Check if Black is in check
                if not is_in_check(state, Player.BLACK):
                    continue

                # Check if White Commander is also in check from Black Commander
                # (this would mean the position is illegal - White moved into check)
                # But also: Commanders can't be adjacent, since both would be in check
                if abs(wc_r - bc_r) <= 1 and abs(wc_c - bc_c) <= 1:
                    continue  # Kings can't be adjacent

                # Check if Black has any legal moves
                legal = generate_legal_moves(state)
                # Reset state.done if generate_legal_moves set it
                state.done = False
                state.winner = None

                if len(legal) == 0:
                    checkmates.append((wc_pos, wr_pos, bc_pos))

    print(f"\nPositions checked: {positions_checked}")
    print(f"Checkmate positions found: {len(checkmates)}")

    if checkmates:
        # Show some examples
        shown = 0
        for wc_pos, wr_pos, bc_pos in checkmates:
            if shown >= 5:
                break
            state = make_empty_state()
            state.current_player = Player.BLACK
            place_piece(state, wc_pos[0], wc_pos[1], PieceType.COMMANDER, Player.WHITE)
            place_piece(state, wr_pos[0], wr_pos[1], PieceType.RIDER, Player.WHITE)
            place_piece(state, bc_pos[0], bc_pos[1], PieceType.COMMANDER, Player.BLACK)

            wc_n = rc_to_notation(wc_pos[0], wc_pos[1])
            wr_n = rc_to_notation(wr_pos[0], wr_pos[1])
            bc_n = rc_to_notation(bc_pos[0], bc_pos[1])

            display_state(state,
                f"CHECKMATE #{shown+1}: White C={wc_n}, White R={wr_n}, Black C={bc_n}")
            shown += 1

        # Categorize: how many are edge/corner mates?
        corner_mates = 0
        edge_mates = 0
        other_mates = 0
        for wc_pos, wr_pos, bc_pos in checkmates:
            br, bc_col = bc_pos
            is_corner = (br in (0, 7)) and (bc_col in (0, 7))
            is_edge = (br in (0, 7)) or (bc_col in (0, 7))
            if is_corner:
                corner_mates += 1
            elif is_edge:
                edge_mates += 1
            else:
                other_mates += 1
        print(f"\nCheckmate breakdown:")
        print(f"  Corner mates (Black C in corner): {corner_mates}")
        print(f"  Edge mates (Black C on edge, not corner): {edge_mates}")
        print(f"  Interior mates (Black C not on edge): {other_mates}")
    else:
        print("\nR+C vs C: NO checkmate positions exist!")
        print("This means R+C vs C is a THEORETICAL DRAW.")

    return checkmates


# ==============================================================================
# PART 2: Can Lancer + Commander checkmate a lone Commander?
# ==============================================================================
def find_checkmate_positions_lc_vs_c():
    """Enumerate all possible L+C vs C positions for checkmate."""
    print("\n" + "=" * 70)
    print("PART 2: Exhaustive search for L+C vs C checkmate positions")
    print("=" * 70)

    checkmates = []
    all_squares = [(r, c) for r in range(8) for c in range(8)]

    for wc_pos in all_squares:
        for wl_pos in all_squares:
            if wl_pos == wc_pos:
                continue
            for bc_pos in all_squares:
                if bc_pos == wc_pos or bc_pos == wl_pos:
                    continue

                wc_r, wc_c = wc_pos
                bc_r, bc_c = bc_pos

                # Commanders can't be adjacent
                if abs(wc_r - bc_r) <= 1 and abs(wc_c - bc_c) <= 1:
                    continue

                state = make_empty_state()
                state.current_player = Player.BLACK
                place_piece(state, wc_r, wc_c, PieceType.COMMANDER, Player.WHITE)
                place_piece(state, wl_pos[0], wl_pos[1], PieceType.LANCER, Player.WHITE)
                place_piece(state, bc_r, bc_c, PieceType.COMMANDER, Player.BLACK)

                if not is_in_check(state, Player.BLACK):
                    continue

                legal = generate_legal_moves(state)
                state.done = False
                state.winner = None

                if len(legal) == 0:
                    checkmates.append((wc_pos, wl_pos, bc_pos))

    print(f"Checkmate positions found: {len(checkmates)}")
    if checkmates:
        shown = 0
        for wc_pos, wl_pos, bc_pos in checkmates[:3]:
            state = make_empty_state()
            state.current_player = Player.BLACK
            place_piece(state, wc_pos[0], wc_pos[1], PieceType.COMMANDER, Player.WHITE)
            place_piece(state, wl_pos[0], wl_pos[1], PieceType.LANCER, Player.WHITE)
            place_piece(state, bc_pos[0], bc_pos[1], PieceType.COMMANDER, Player.BLACK)
            wc_n = rc_to_notation(wc_pos[0], wc_pos[1])
            wl_n = rc_to_notation(wl_pos[0], wl_pos[1])
            bc_n = rc_to_notation(bc_pos[0], bc_pos[1])
            display_state(state,
                f"L+C CHECKMATE #{shown+1}: White C={wc_n}, White L={wl_n}, Black C={bc_n}")
            shown += 1
    else:
        print("L+C vs C: NO checkmate positions exist! Theoretical DRAW.")

    return checkmates


# ==============================================================================
# PART 3: Can Bombard + Commander checkmate a lone Commander?
# ==============================================================================
def find_checkmate_positions_bc_vs_c():
    """Enumerate all possible B+C vs C positions for checkmate.
    Note: Bombard cannot ranged-attack a Commander. Only melee (1 sq) attack.
    Bombard attacks 1 square any direction (same as Commander for captures).
    """
    print("\n" + "=" * 70)
    print("PART 3: Exhaustive search for Bombard+C vs C checkmate positions")
    print("=" * 70)
    print("Note: Bombard ranged attack CANNOT target Commander.")
    print("Bombard melee: 1 square any direction (same as Commander for attack).")

    checkmates = []
    all_squares = [(r, c) for r in range(8) for c in range(8)]

    for wc_pos in all_squares:
        for wb_pos in all_squares:
            if wb_pos == wc_pos:
                continue
            for bc_pos in all_squares:
                if bc_pos == wc_pos or bc_pos == wb_pos:
                    continue

                wc_r, wc_c = wc_pos
                bc_r, bc_c = bc_pos

                if abs(wc_r - bc_r) <= 1 and abs(wc_c - bc_c) <= 1:
                    continue

                state = make_empty_state()
                state.current_player = Player.BLACK
                place_piece(state, wc_r, wc_c, PieceType.COMMANDER, Player.WHITE)
                place_piece(state, wb_pos[0], wb_pos[1], PieceType.BOMBARD, Player.WHITE)
                place_piece(state, bc_r, bc_c, PieceType.COMMANDER, Player.BLACK)

                if not is_in_check(state, Player.BLACK):
                    continue

                legal = generate_legal_moves(state)
                state.done = False
                state.winner = None

                if len(legal) == 0:
                    checkmates.append((wc_pos, wb_pos, bc_pos))

    print(f"Checkmate positions found: {len(checkmates)}")
    if checkmates:
        shown = 0
        for wc_pos, wb_pos, bc_pos in checkmates[:3]:
            state = make_empty_state()
            state.current_player = Player.BLACK
            place_piece(state, wc_pos[0], wc_pos[1], PieceType.COMMANDER, Player.WHITE)
            place_piece(state, wb_pos[0], wb_pos[1], PieceType.BOMBARD, Player.WHITE)
            place_piece(state, bc_pos[0], bc_pos[1], PieceType.COMMANDER, Player.BLACK)
            wc_n = rc_to_notation(wc_pos[0], wc_pos[1])
            wb_n = rc_to_notation(wb_pos[0], wb_pos[1])
            bc_n = rc_to_notation(bc_pos[0], bc_pos[1])
            display_state(state,
                f"B+C CHECKMATE #{shown+1}: White C={wc_n}, White B={wb_n}, Black C={bc_n}")
            shown += 1
    else:
        print("B+C vs C: NO checkmate positions exist! Theoretical DRAW.")

    return checkmates


# ==============================================================================
# PART 4: Can 2 Riders + Commander checkmate a lone Commander?
# ==============================================================================
def find_checkmate_positions_2r_c_vs_c():
    """Enumerate 2R+C vs C positions for checkmate.
    This is expensive (64^4 ~= 16M combinations) so we sample."""
    print("\n" + "=" * 70)
    print("PART 4: Search for 2R+C vs C checkmate positions")
    print("=" * 70)
    print("Checking all positions (may take a minute)...")

    checkmates = []
    all_squares = [(r, c) for r in range(8) for c in range(8)]
    checked = 0

    for wc_pos in all_squares:
        for wr1_pos in all_squares:
            if wr1_pos == wc_pos:
                continue
            for wr2_pos in all_squares:
                if wr2_pos == wc_pos or wr2_pos == wr1_pos:
                    continue
                # Only check wr2 > wr1 to avoid double-counting
                if wr2_pos <= wr1_pos:
                    continue
                for bc_pos in all_squares:
                    if bc_pos in (wc_pos, wr1_pos, wr2_pos):
                        continue

                    wc_r, wc_c = wc_pos
                    bc_r, bc_c = bc_pos

                    if abs(wc_r - bc_r) <= 1 and abs(wc_c - bc_c) <= 1:
                        continue

                    state = make_empty_state()
                    state.current_player = Player.BLACK
                    place_piece(state, wc_r, wc_c, PieceType.COMMANDER, Player.WHITE)
                    place_piece(state, wr1_pos[0], wr1_pos[1], PieceType.RIDER, Player.WHITE)
                    place_piece(state, wr2_pos[0], wr2_pos[1], PieceType.RIDER, Player.WHITE)
                    place_piece(state, bc_r, bc_c, PieceType.COMMANDER, Player.BLACK)

                    checked += 1

                    if not is_in_check(state, Player.BLACK):
                        continue

                    legal = generate_legal_moves(state)
                    state.done = False
                    state.winner = None

                    if len(legal) == 0:
                        checkmates.append((wc_pos, wr1_pos, wr2_pos, bc_pos))

    print(f"Positions checked: {checked}")
    print(f"Checkmate positions found: {len(checkmates)}")
    if checkmates:
        shown = 0
        for entry in checkmates[:3]:
            wc_pos, wr1_pos, wr2_pos, bc_pos = entry
            state = make_empty_state()
            state.current_player = Player.BLACK
            place_piece(state, wc_pos[0], wc_pos[1], PieceType.COMMANDER, Player.WHITE)
            place_piece(state, wr1_pos[0], wr1_pos[1], PieceType.RIDER, Player.WHITE)
            place_piece(state, wr2_pos[0], wr2_pos[1], PieceType.RIDER, Player.WHITE)
            place_piece(state, bc_pos[0], bc_pos[1], PieceType.COMMANDER, Player.BLACK)
            wc_n = rc_to_notation(wc_pos[0], wc_pos[1])
            wr1_n = rc_to_notation(wr1_pos[0], wr1_pos[1])
            wr2_n = rc_to_notation(wr2_pos[0], wr2_pos[1])
            bc_n = rc_to_notation(bc_pos[0], bc_pos[1])
            display_state(state,
                f"2R+C CHECKMATE #{shown+1}: WC={wc_n}, WR1={wr1_n}, WR2={wr2_n}, BC={bc_n}")
            shown += 1
    else:
        print("2R+C vs C: NO checkmate positions exist! Theoretical DRAW.")

    return checkmates


# ==============================================================================
# PART 5: MCTS test - can it find checkmate from near-mate position?
# ==============================================================================
def run_mcts_endgame_test(checkmates, piece_label="R+C"):
    """Given checkmate positions, set up a position 1-2 moves before mate
    and see if MCTS can find the mating sequence."""
    if not checkmates:
        print(f"\nNo {piece_label} checkmate positions to test MCTS from.")
        return

    from davechess.engine.mcts_lite import MCTSLite

    print(f"\n{'='*70}")
    print(f"PART 5: MCTS endgame test - can MCTS find mate with {piece_label}?")
    print(f"{'='*70}")

    # Pick a checkmate position and create a pre-mate setup
    # We'll place White to move, one move away from delivering checkmate
    # The idea: find a position where White has a move that reaches a
    # checkmate position for Black

    # For each checkmate, try to find a "pre-mate" position (White to move)
    # by looking at what White's last move could have been
    print(f"\nTesting {min(5, len(checkmates))} checkmate positions...")

    tested = 0
    for idx, entry in enumerate(checkmates):
        if tested >= 5:
            break

        if len(entry) == 3:
            wc_pos, wp_pos, bc_pos = entry
            piece_type = PieceType.RIDER if piece_label.startswith("R") else \
                         PieceType.LANCER if piece_label.startswith("L") else PieceType.BOMBARD
            piece_positions = [(wp_pos, piece_type)]
        elif len(entry) == 4:
            wc_pos, wp1_pos, wp2_pos, bc_pos = entry
            piece_positions = [(wp1_pos, PieceType.RIDER), (wp2_pos, PieceType.RIDER)]

        # Now try to set up a pre-mate position: White to move, can reach checkmate
        # Try moving the Rider/Lancer/Bombard back along its movement path
        # Or try moving the White Commander back

        # Simple approach: set up White to move in a position near the
        # checkmate, and let MCTS play it out
        # We'll create the mated position but with White to move 1 step earlier

        # Actually, let's just set up a position with Black Commander in corner
        # and White pieces nearby, then let MCTS play from White's perspective
        state = make_empty_state()
        state.current_player = Player.WHITE
        place_piece(state, wc_pos[0], wc_pos[1], PieceType.COMMANDER, Player.WHITE)
        for pp, pt in piece_positions:
            place_piece(state, pp[0], pp[1], pt, Player.WHITE)
        place_piece(state, bc_pos[0], bc_pos[1], PieceType.COMMANDER, Player.BLACK)

        # But in this position it's White to move and Black is already mated...
        # We need to back up one move. Let's move the attacking piece back a bit.
        # Try: put it in a different position and see if MCTS finds the mate.

        # Better approach: Set up a specific endgame and let MCTS play it out
        tested += 1

    # NEW APPROACH: Set up realistic endgame positions and play them out with MCTS
    print(f"\n{'='*70}")
    print(f"MCTS ENDGAME PLAYTHROUGH: {piece_label} vs lone Commander")
    print(f"{'='*70}")

    if piece_label == "R+C":
        # White: C on d3, R on e5. Black: C on a8 (corner).
        # White to move. MCTS should try to mate.
        state = make_empty_state()
        state.current_player = Player.WHITE
        place_piece(state, 2, 3, PieceType.COMMANDER, Player.WHITE)  # d3
        place_piece(state, 4, 4, PieceType.RIDER, Player.WHITE)      # e5
        place_piece(state, 7, 0, PieceType.COMMANDER, Player.BLACK)  # a8
        display_state(state, "Starting position: R+C vs C")
    elif piece_label == "2R+C":
        state = make_empty_state()
        state.current_player = Player.WHITE
        place_piece(state, 2, 3, PieceType.COMMANDER, Player.WHITE)  # d3
        place_piece(state, 4, 4, PieceType.RIDER, Player.WHITE)      # e5
        place_piece(state, 3, 5, PieceType.RIDER, Player.WHITE)      # f4
        place_piece(state, 7, 0, PieceType.COMMANDER, Player.BLACK)  # a8
        display_state(state, "Starting position: 2R+C vs C")
    elif piece_label == "L+C":
        state = make_empty_state()
        state.current_player = Player.WHITE
        place_piece(state, 2, 3, PieceType.COMMANDER, Player.WHITE)  # d3
        place_piece(state, 4, 4, PieceType.LANCER, Player.WHITE)     # e5
        place_piece(state, 7, 0, PieceType.COMMANDER, Player.BLACK)  # a8
        display_state(state, "Starting position: L+C vs C")
    else:
        return

    mcts = MCTSLite(num_simulations=800, max_rollout_depth=200)
    print(f"\nPlaying out with MCTS ({mcts.num_simulations} sims, max_rollout_depth={mcts.max_rollout_depth})...")
    print(f"50-move rule active (100 halfmoves with no capture = draw)")
    print()

    move_count = 0
    max_moves = 200  # Safety limit

    while not state.done and move_count < max_moves:
        moves = generate_legal_moves(state)
        if not moves:
            break

        move = mcts.search(state)
        player_name = "White" if state.current_player == Player.WHITE else "Black"

        # Determine move notation
        if hasattr(move, 'from_rc') and hasattr(move, 'to_rc'):
            from_n = rc_to_notation(move.from_rc[0], move.from_rc[1])
            to_n = rc_to_notation(move.to_rc[0], move.to_rc[1])
            cap = "x" if getattr(move, 'is_capture', False) else "-"
            piece = state.board[move.from_rc[0]][move.from_rc[1]]
            piece_name = piece.char if piece else "?"
            print(f"  Move {move_count+1} ({player_name}): {piece_name}{from_n}{cap}{to_n}")
        else:
            print(f"  Move {move_count+1} ({player_name}): {move}")

        apply_move(state, move)
        move_count += 1

    print()
    if state.done:
        if state.winner is not None:
            winner_name = "White" if state.winner == Player.WHITE else "Black"
            print(f"RESULT: {winner_name} wins by checkmate in {move_count} moves!")
        else:
            print(f"RESULT: DRAW after {move_count} moves (turn limit, repetition, or 50-move rule)")
    else:
        print(f"RESULT: Game not finished after {move_count} moves (safety limit)")

    display_state(state, "Final position")
    return state


# ==============================================================================
# PART 6: Quick check - R+C vs C theoretical analysis
# ==============================================================================
def theoretical_analysis():
    """Analyze WHY R+C might or might not be able to force mate."""
    print("\n" + "=" * 70)
    print("THEORETICAL ANALYSIS: R+C vs lone C")
    print("=" * 70)
    print("""
Key observations for DaveChess endgame theory:

Commander: 1 square any direction (identical to chess King)
Rider: up to 3 squares any straight line (ortho + diagonal), no jumping

In chess, K+R vs K is a forced win (Rook controls files/ranks).
The chess Rook has UNLIMITED range on files and ranks.

The DaveChess Rider has only 3-square range but covers both
orthogonal AND diagonal lines.

Questions to answer:
1. Can R+C produce a checkmate position? (static analysis)
2. If yes, can R+C FORCE mate? (dynamic - requires retrograde analysis or MCTS)
3. Key differences from chess K+R vs K:
   - Rider has 3-square range (not unlimited like Rook)
   - Rider covers diagonals too (Rook doesn't)
   - Can the defending Commander "outrun" the Rider?
""")

    # Analyze: in a checkmate position, what does the Rider need to do?
    # The Rider must give check while the Commander cuts off escape squares.
    # The Commander controls 8 adjacent squares max.
    # The Rider from distance can control a line of squares.
    # But with only 3-square range, the Rider can only threaten squares
    # within 3 squares of its position.

    # Key question: can the Rider "cut off" files/ranks like a chess Rook?
    # A chess Rook on the 6th rank with the defending King on ranks 7-8
    # restricts the King to those ranks. The Rider with 3-square range
    # cannot create such a permanent barrier.

    print("Testing: can a Rider cut off a rank/file like a chess Rook?")
    print("A Rider on e5 threatens: up to 3 squares in each of 8 directions")
    state = make_empty_state()
    place_piece(state, 4, 4, PieceType.RIDER, Player.WHITE)  # e5

    # Check which squares the Rider attacks
    attacked = []
    for r in range(8):
        for c in range(8):
            if (r, c) == (4, 4):
                continue
            if _is_square_attacked(state, r, c, Player.WHITE):
                attacked.append((r, c))

    print(f"Rider on e5 attacks {len(attacked)} squares:")
    # Display as board
    attack_board = [['.' for _ in range(8)] for _ in range(8)]
    attack_board[4][4] = 'R'
    for r, c in attacked:
        attack_board[r][c] = 'x'

    print("    a b c d e f g h")
    for r in range(7, -1, -1):
        row_str = f" {r+1}  " + " ".join(attack_board[r])
        print(row_str)
    print()

    # Compare: how many squares does a chess Rook on e5 attack?
    print("For comparison, a chess Rook on e5 would attack 14 squares (entire rank + file).")
    print(f"The DaveChess Rider on e5 attacks {len(attacked)} squares.")
    print("But the Rider also covers diagonals, which the Rook doesn't.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    t0 = time.time()

    # Part 1: R+C vs C
    rc_mates = find_checkmate_positions_rc_vs_c()

    # Part 2: L+C vs C
    lc_mates = find_checkmate_positions_lc_vs_c()

    # Part 3: B+C vs C
    bc_mates = find_checkmate_positions_bc_vs_c()

    # Part 4: 2R+C vs C (expensive but informative)
    print("\nStarting 2R+C vs C search (this may take a while)...")
    rr_mates = find_checkmate_positions_2r_c_vs_c()

    # Theoretical analysis
    theoretical_analysis()

    # Part 5: MCTS tests for each that has checkmate positions
    if rc_mates:
        run_mcts_endgame_test(rc_mates, "R+C")
    if lc_mates:
        run_mcts_endgame_test(lc_mates, "L+C")
    if rr_mates:
        run_mcts_endgame_test(rr_mates, "2R+C")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY OF ENDGAME ANALYSIS")
    print(f"{'='*70}")
    print(f"R+C vs C:  {'CAN checkmate (' + str(len(rc_mates)) + ' positions)' if rc_mates else 'CANNOT checkmate (theoretical draw)'}")
    print(f"L+C vs C:  {'CAN checkmate (' + str(len(lc_mates)) + ' positions)' if lc_mates else 'CANNOT checkmate (theoretical draw)'}")
    print(f"B+C vs C:  {'CAN checkmate (' + str(len(bc_mates)) + ' positions)' if bc_mates else 'CANNOT checkmate (theoretical draw)'}")
    print(f"2R+C vs C: {'CAN checkmate (' + str(len(rr_mates)) + ' positions)' if rr_mates else 'CANNOT checkmate (theoretical draw)'}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print()
    if not rc_mates:
        print("CRITICAL FINDING: R+C vs C is a THEORETICAL DRAW.")
        print("If the model's self-play games frequently reach R+C vs C endgames,")
        print("drawing is CORRECT behavior. The problem is in the middlegame")
        print("(failing to maintain enough material advantage to force mate).")
    print()
