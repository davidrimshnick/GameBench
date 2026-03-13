#!/usr/bin/env python3
"""Deep move-by-move analysis of NN losses vs random opponent.

Replays each game using the actual game engine to identify the CAUSAL
mechanism behind losses, not just symptoms like "shuttling."

Key complexity: games from iterations with selfplay_opening_plies > 0
start from a randomized position. The DCN files record moves from that
randomized position, NOT from the standard starting position. We must
reconstruct the starting position by analyzing the first few moves.
"""

import sys
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from davechess.game.state import GameState, Player, PieceType, Piece, PIECE_NAMES, PIECE_CHARS, Move, MoveStep, Promote, BombardAttack
from davechess.game.rules import apply_move, generate_legal_moves, is_in_check, _find_commander
from davechess.game.board import BOARD_SIZE, GOLD_NODES, STARTING_POSITIONS
from davechess.game.notation import move_to_dcn, dcn_to_game
from davechess.engine.probe_openings import build_probe_start_state


GOLD_SET = set(GOLD_NODES)

PIECE_VALUES = {
    PieceType.COMMANDER: 100,
    PieceType.LANCER: 7,
    PieceType.BOMBARD: 5,
    PieceType.RIDER: 3,
    PieceType.WARRIOR: 1,
}


def infer_start_state_from_moves(dcn_text):
    """Try to reconstruct the starting position by attempting replay
    from standard position. If it fails, try all possible opening seeds
    for 4-ply opening randomization.

    Returns (start_state, adjusted_moves) where start_state is the position
    before the first recorded move, or None if we can't figure it out.
    """
    headers, moves, result = dcn_to_game(dcn_text)

    # First, try standard starting position
    state = GameState()
    success = True
    for ply, move in enumerate(moves):
        try:
            # Validate: check if the piece exists at from_rc
            if isinstance(move, MoveStep):
                piece = state.board[move.from_rc[0]][move.from_rc[1]]
                if piece is None:
                    success = False
                    break
                if move.is_capture:
                    target = state.board[move.to_rc[0]][move.to_rc[1]]
                    if target is None:
                        success = False
                        break
            elif isinstance(move, BombardAttack):
                piece = state.board[move.from_rc[0]][move.from_rc[1]]
                if piece is None:
                    success = False
                    break
                target = state.board[move.target_rc[0]][move.target_rc[1]]
                if target is None:
                    success = False
                    break
            apply_move(state, move)
        except Exception:
            success = False
            break

    if success:
        return GameState(), moves, result, headers

    # Opening randomization used — brute-force find the seed
    # Try seeds 0..10000 with 4-ply opening
    for seed in range(10001):
        try:
            start = build_probe_start_state(seed, 4)
            state = start.clone()
            ok = True
            for ply, move in enumerate(moves):
                if isinstance(move, MoveStep):
                    piece = state.board[move.from_rc[0]][move.from_rc[1]]
                    if piece is None:
                        ok = False
                        break
                    if move.is_capture:
                        target = state.board[move.to_rc[0]][move.to_rc[1]]
                        if target is None:
                            ok = False
                            break
                elif isinstance(move, BombardAttack):
                    piece = state.board[move.from_rc[0]][move.from_rc[1]]
                    if piece is None:
                        ok = False
                        break
                    target = state.board[move.target_rc[0]][move.target_rc[1]]
                    if target is None:
                        ok = False
                        break
                apply_move(state, move)
                if state.done:
                    break
            if ok:
                return start, moves, result, headers
        except Exception:
            continue

    return None, moves, result, headers


def try_replay_game(moves, start_state=None):
    """Try to replay a game from the given start state. Returns (states, final_state, ok).
    If replay fails, returns as far as we got."""
    state = start_state.clone() if start_state else GameState()
    states_before = []
    for ply, move in enumerate(moves):
        states_before.append(state.clone())
        try:
            if isinstance(move, MoveStep):
                piece = state.board[move.from_rc[0]][move.from_rc[1]]
                if piece is None:
                    return states_before, state, False
                if move.is_capture:
                    target = state.board[move.to_rc[0]][move.to_rc[1]]
                    if target is None:
                        return states_before, state, False
            elif isinstance(move, BombardAttack):
                piece = state.board[move.from_rc[0]][move.from_rc[1]]
                if piece is None:
                    return states_before, state, False
                target = state.board[move.target_rc[0]][move.target_rc[1]]
                if target is None:
                    return states_before, state, False
            apply_move(state, move)
        except Exception:
            return states_before, state, False
    return states_before, state, True


def count_pieces(state, player):
    """Count pieces by type for a player."""
    counts = Counter()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = state.board[row][col]
            if p is not None and p.player == player:
                counts[p.piece_type] += 1
    return counts


def material_value(state, player):
    """Calculate material value for a player (excluding commander from value)."""
    total = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = state.board[row][col]
            if p is not None and p.player == player:
                total += PIECE_VALUES[p.piece_type]
    return total


def pieces_on_gold(state, player):
    """Return list of piece types on gold nodes for a player."""
    result = []
    for r, c in GOLD_NODES:
        p = state.board[r][c]
        if p is not None and p.player == player:
            result.append((PIECE_NAMES[p.piece_type], (r, c)))
    return result


def load_iteration_dcn_sections(iter_num):
    """Load raw DCN text sections for each game in an iteration."""
    filepath = f"/home/david/repos/GameBench/logs/on_policy_fix_20260308/games/iter_{iter_num:04d}.dcn"
    if not os.path.exists(filepath):
        return []

    with open(filepath) as f:
        content = f.read()

    sections = []
    current_section = []

    for line in content.split("\n"):
        if line.startswith("[Iteration") and current_section:
            sections.append("\n".join(current_section))
            current_section = []
        current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))

    return [s.strip() for s in sections if s.strip()]


@dataclass
class GameAnalysis:
    """Complete analysis of a single game."""
    iteration: int
    game_num: int
    nn_color: str
    result: str
    nn_result: str
    total_moves: int
    replay_ok: bool = False
    end_reason: str = ""
    final_commander_alive: dict = field(default_factory=dict)

    # Captures
    nn_captures: int = 0
    opp_captures: int = 0
    nn_pieces_lost: dict = field(default_factory=dict)
    opp_pieces_lost: dict = field(default_factory=dict)
    nn_pieces_lost_detail: list = field(default_factory=list)  # (ply, piece, captor)
    opp_pieces_lost_detail: list = field(default_factory=list)

    # Material
    nn_ever_ahead: bool = False
    nn_max_advantage: int = 0
    nn_max_advantage_move: int = 0
    nn_fell_behind_move: int = 0
    final_material_balance: int = 0

    # Critical moment
    critical_move: int = 0
    critical_move_dcn: str = ""
    critical_captured_piece: str = ""
    nn_move_before_critical: str = ""

    # Activity
    nn_commander_moves: int = 0
    opp_commander_moves: int = 0
    nn_in_check_count: int = 0

    # Opening
    opening_moves: list = field(default_factory=list)

    # Move type breakdown
    nn_non_capture_moves: int = 0
    nn_capture_moves: int = 0
    nn_promotion_moves: int = 0

    # Last 10 moves
    final_moves: list = field(default_factory=list)

    # Capture timeline (ply, who, attacker_type, victim_type, dcn, material_balance)
    capture_timeline: list = field(default_factory=list)


def analyze_game_from_replay(iteration, game_num, moves, result_str, start_state, states_before, final_state, replay_ok):
    """Analyze a game from its replay data."""
    game_idx = game_num - 1
    nn_plays_white = (game_idx % 2 == 0)
    nn_player = Player.WHITE if nn_plays_white else Player.BLACK
    opp_player = Player(1 - nn_player)
    nn_color = "White" if nn_plays_white else "Black"

    if result_str == "1-0":
        winner = Player.WHITE
    elif result_str == "0-1":
        winner = Player.BLACK
    else:
        winner = None

    if winner is None:
        nn_result = "draw"
    elif winner == nn_player:
        nn_result = "win"
    else:
        nn_result = "loss"

    a = GameAnalysis(
        iteration=iteration,
        game_num=game_num,
        nn_color=nn_color,
        result=result_str,
        nn_result=nn_result,
        total_moves=len(moves),
        replay_ok=replay_ok,
    )

    # Track material balance
    nn_fell_behind = False
    prev_nn_dcn = ""

    num_plies = min(len(moves), len(states_before))
    for ply in range(num_plies):
        state = states_before[ply]
        move = moves[ply]
        is_nn_turn = (state.current_player == nn_player)

        # Opening moves
        if ply < 4:
            try:
                a.opening_moves.append(move_to_dcn(state, move))
            except:
                a.opening_moves.append("?")

        # DCN
        try:
            dcn = move_to_dcn(state, move)
        except:
            dcn = "?"

        # Check if NN is in check
        if is_nn_turn and is_in_check(state, nn_player):
            a.nn_in_check_count += 1

        # Commander moves
        if isinstance(move, MoveStep):
            piece = state.board[move.from_rc[0]][move.from_rc[1]]
            if piece and piece.piece_type == PieceType.COMMANDER:
                if is_nn_turn:
                    a.nn_commander_moves += 1
                else:
                    a.opp_commander_moves += 1

        # Move type classification for NN
        if is_nn_turn:
            if isinstance(move, MoveStep):
                if move.is_capture:
                    a.nn_capture_moves += 1
                else:
                    a.nn_non_capture_moves += 1
            elif isinstance(move, Promote):
                a.nn_promotion_moves += 1

        # Capture tracking
        captured_piece = None
        capturing_piece = None
        if isinstance(move, MoveStep) and move.is_capture:
            captured_piece = state.board[move.to_rc[0]][move.to_rc[1]]
            capturing_piece = state.board[move.from_rc[0]][move.from_rc[1]]
        elif isinstance(move, BombardAttack):
            captured_piece = state.board[move.target_rc[0]][move.target_rc[1]]
            capturing_piece = state.board[move.from_rc[0]][move.from_rc[1]]

        if captured_piece:
            cap_name = PIECE_NAMES[captured_piece.piece_type]
            atk_name = PIECE_NAMES[capturing_piece.piece_type] if capturing_piece else "?"

            if is_nn_turn:
                a.nn_captures += 1
                a.opp_pieces_lost[cap_name] = a.opp_pieces_lost.get(cap_name, 0) + 1
                a.opp_pieces_lost_detail.append((ply + 1, cap_name, atk_name))
            else:
                a.opp_captures += 1
                a.nn_pieces_lost[cap_name] = a.nn_pieces_lost.get(cap_name, 0) + 1
                a.nn_pieces_lost_detail.append((ply + 1, cap_name, atk_name))

        # Material balance after this move (compute from next state if available)
        if ply + 1 < len(states_before):
            next_state = states_before[ply + 1]
        elif ply == num_plies - 1:
            next_state = final_state
        else:
            next_state = state  # fallback

        nn_mat = material_value(next_state, nn_player)
        opp_mat = material_value(next_state, opp_player)
        nn_balance = nn_mat - opp_mat

        if captured_piece:
            a.capture_timeline.append({
                "ply": ply + 1,
                "who": "NN" if is_nn_turn else "RND",
                "attacker": atk_name,
                "victim": cap_name,
                "dcn": dcn,
                "balance": nn_balance,
                "is_ranged": isinstance(move, BombardAttack),
            })

        if nn_balance > a.nn_max_advantage:
            a.nn_max_advantage = nn_balance
            a.nn_max_advantage_move = ply + 1
            a.nn_ever_ahead = True

        if nn_balance < 0 and not nn_fell_behind:
            nn_fell_behind = True
            a.nn_fell_behind_move = ply + 1
            a.critical_move = ply + 1
            a.critical_move_dcn = dcn
            if captured_piece and not is_nn_turn:
                a.critical_captured_piece = PIECE_NAMES[captured_piece.piece_type]
            a.nn_move_before_critical = prev_nn_dcn

        if is_nn_turn:
            prev_nn_dcn = dcn

        # Final moves
        if ply >= len(moves) - 10:
            is_check_nn = is_in_check(state, nn_player)
            is_check_opp = is_in_check(state, opp_player)
            nn_pc = sum(count_pieces(state, nn_player).values())
            opp_pc = sum(count_pieces(state, opp_player).values())
            cap_info = ""
            if captured_piece:
                cap_info = f" (takes {PIECE_NAMES[captured_piece.piece_type]})"
            check_info = ""
            if is_check_nn:
                check_info = " [NN IN CHECK]"
            elif is_check_opp:
                check_info = " [RND IN CHECK]"
            a.final_moves.append(
                f"Ply {ply+1:3d} {'NN' if is_nn_turn else 'RND':3s}: {dcn:12s}{cap_info}{check_info}  "
                f"[NN:{nn_pc}pc RND:{opp_pc}pc Gold:W{state.resources[0]}/B{state.resources[1]}]"
            )

    a.final_material_balance = nn_balance if 'nn_balance' in dir() else 0

    # End reason
    if final_state.done:
        if final_state.winner is not None:
            w_cmd = _find_commander(final_state, Player.WHITE)
            b_cmd = _find_commander(final_state, Player.BLACK)
            a.final_commander_alive = {"White": w_cmd is not None, "Black": b_cmd is not None}
            if w_cmd is None or b_cmd is None:
                a.end_reason = "commander_captured"
            else:
                a.end_reason = "checkmate"
        else:
            if final_state.turn > 100:
                a.end_reason = "turn_limit_draw"
            elif final_state.halfmove_clock >= 100:
                a.end_reason = "50_move_draw"
            else:
                a.end_reason = "repetition_draw"
    elif num_plies < len(moves):
        a.end_reason = "replay_incomplete"

    return a


def main():
    iterations = range(450, 459)

    all_vs_random = []
    all_losses = []
    all_wins = []
    all_draws = []

    failed_replays = 0
    total_games = 0

    for iter_num in iterations:
        print(f"Loading iteration {iter_num}...")
        sections = load_iteration_dcn_sections(iter_num)
        print(f"  Found {len(sections)} game sections")

        # Games 1-10 are vs_random
        for game_num in range(1, min(11, len(sections) + 1)):
            total_games += 1
            dcn_text = sections[game_num - 1]
            headers, moves, result_str = dcn_to_game(dcn_text)

            if not moves:
                print(f"  Skipping iter {iter_num} game {game_num}: no moves parsed")
                continue

            # Try replay from standard position
            states_before, final_state, ok = try_replay_game(moves)

            if not ok:
                # Try opening randomization seeds
                found = False
                for seed in range(5000):
                    try:
                        start = build_probe_start_state(seed, 4)
                        states_before2, final_state2, ok2 = try_replay_game(moves, start)
                        if ok2:
                            states_before = states_before2
                            final_state = final_state2
                            ok = True
                            found = True
                            break
                    except:
                        continue

                if not found:
                    failed_replays += 1

            analysis = analyze_game_from_replay(
                iter_num, game_num, moves, result_str,
                None, states_before, final_state, ok
            )

            all_vs_random.append(analysis)
            if analysis.nn_result == "loss":
                all_losses.append(analysis)
            elif analysis.nn_result == "win":
                all_wins.append(analysis)
            else:
                all_draws.append(analysis)

    print(f"\n{'#'*80}")
    print(f"REPLAY STATISTICS")
    print(f"  Total games: {total_games}")
    print(f"  Failed replays: {failed_replays}")
    print(f"  Successful: {total_games - failed_replays}")
    print(f"{'#'*80}")

    print(f"\n{'#'*80}")
    print(f"VS-RANDOM SUMMARY: {len(all_vs_random)} games (iters 450-458)")
    print(f"  Wins: {len(all_wins)}, Losses: {len(all_losses)}, Draws: {len(all_draws)}")
    if all_vs_random:
        print(f"  Win rate: {len(all_wins)/len(all_vs_random)*100:.1f}%")
    print(f"{'#'*80}")

    # =====================================================
    # SECTION 1: DETAILED LOSS ANALYSIS
    # =====================================================
    if all_losses:
        print(f"\n{'#'*80}")
        print(f"SECTION 1: DETAILED LOSS ANALYSIS ({len(all_losses)} losses)")
        print(f"{'#'*80}")

        for a in all_losses:
            print(f"\n{'='*80}")
            print(f"  Iter {a.iteration}, Game {a.game_num}: NN={a.nn_color}, "
                  f"Result={a.result} => NN {a.nn_result.upper()}")
            print(f"  Replay: {'OK' if a.replay_ok else 'PARTIAL'}, "
                  f"Total moves: {a.total_moves}, End: {a.end_reason}")
            print(f"  Commander alive: {a.final_commander_alive}")
            print(f"  Captures: NN made {a.nn_captures}, Random made {a.opp_captures}")
            print(f"  NN pieces lost: {dict(a.nn_pieces_lost)}")
            print(f"  Random pieces lost: {dict(a.opp_pieces_lost)}")
            print(f"  Commander moves: NN={a.nn_commander_moves}, Random={a.opp_commander_moves}")
            print(f"  NN in check: {a.nn_in_check_count} times")
            print(f"  Opening: {' '.join(a.opening_moves)}")
            print(f"  NN move breakdown: {a.nn_non_capture_moves} non-capture, "
                  f"{a.nn_capture_moves} capture, {a.nn_promotion_moves} promotion")

            if a.nn_ever_ahead:
                print(f"  NN was ahead by {a.nn_max_advantage} at ply {a.nn_max_advantage_move}")
            else:
                print(f"  NN was NEVER ahead in material")

            print(f"  Final material balance (NN perspective): {a.final_material_balance:+d}")

            if a.critical_move > 0:
                print(f"\n  CRITICAL MOMENT at ply {a.critical_move}:")
                print(f"    Move: {a.critical_move_dcn}")
                print(f"    Piece captured: {a.critical_captured_piece or 'N/A'}")
                print(f"    NN's previous move: {a.nn_move_before_critical or 'N/A'}")

            if a.capture_timeline:
                print(f"\n  Capture timeline:")
                for cap in a.capture_timeline:
                    ranged = " (ranged)" if cap["is_ranged"] else ""
                    print(f"    Ply {cap['ply']:3d} [{cap['who']:3s}]: {cap['dcn']:12s} "
                          f"({cap['attacker']} takes {cap['victim']}{ranged}) bal={cap['balance']:+d}")

            if a.final_moves:
                print(f"\n  Last moves:")
                for m in a.final_moves:
                    print(f"    {m}")

    # =====================================================
    # SECTION 2: CROSS-LOSS PATTERN ANALYSIS
    # =====================================================
    if all_losses:
        print(f"\n{'#'*80}")
        print(f"SECTION 2: CROSS-LOSS PATTERN ANALYSIS")
        print(f"{'#'*80}")

        # How do losses end?
        end_reasons = Counter(a.end_reason for a in all_losses)
        print(f"\n  How losses END:")
        for reason, count in end_reasons.most_common():
            print(f"    {reason}: {count} ({count/len(all_losses)*100:.0f}%)")

        # Which commander gets captured?
        cmd_captured = Counter()
        for a in all_losses:
            for side, alive in a.final_commander_alive.items():
                if not alive:
                    is_nn = (side == a.nn_color)
                    cmd_captured[f"{'NN' if is_nn else 'Random'} commander"] += 1
        print(f"\n  Commander captured in losses:")
        for desc, count in cmd_captured.most_common():
            print(f"    {desc}: {count}")

        # Pieces lost by NN
        nn_piece_losses = Counter()
        for a in all_losses:
            for piece, count in a.nn_pieces_lost.items():
                nn_piece_losses[piece] += count
        print(f"\n  NN pieces lost across all losses:")
        for piece, count in nn_piece_losses.most_common():
            print(f"    {piece}: {count}")

        # Pieces lost by Random
        opp_piece_losses = Counter()
        for a in all_losses:
            for piece, count in a.opp_pieces_lost.items():
                opp_piece_losses[piece] += count
        print(f"\n  Random pieces lost across all losses:")
        for piece, count in opp_piece_losses.most_common():
            print(f"    {piece}: {count}")

        # NN ever ahead
        ever_ahead = sum(1 for a in all_losses if a.nn_ever_ahead)
        print(f"\n  NN ever ahead in losses: {ever_ahead}/{len(all_losses)}")

        nn_cap_total = sum(a.nn_captures for a in all_losses)
        opp_cap_total = sum(a.opp_captures for a in all_losses)
        print(f"  Total captures in losses: NN={nn_cap_total}, Random={opp_cap_total}")

        # Critical moment details
        print(f"\n  Critical moments (when NN first fell behind):")
        for a in all_losses:
            if a.critical_move > 0:
                print(f"    Iter {a.iteration} G{a.game_num}: ply {a.critical_move} "
                      f"({a.critical_move_dcn}), lost={a.critical_captured_piece or '?'}, "
                      f"NN prev={a.nn_move_before_critical or 'start'}")

        # Color bias
        nn_color_losses = Counter(a.nn_color for a in all_losses)
        white_total = sum(1 for a in all_vs_random if a.nn_color == "White")
        black_total = sum(1 for a in all_vs_random if a.nn_color == "Black")
        print(f"\n  NN color in losses: {dict(nn_color_losses)}")
        print(f"    NN as White: {nn_color_losses.get('White', 0)}/{white_total} games "
              f"({nn_color_losses.get('White', 0)/max(1,white_total)*100:.0f}% loss rate)")
        print(f"    NN as Black: {nn_color_losses.get('Black', 0)}/{black_total} games "
              f"({nn_color_losses.get('Black', 0)/max(1,black_total)*100:.0f}% loss rate)")

        avg_loss_len = sum(a.total_moves for a in all_losses) / len(all_losses)
        print(f"\n  Avg loss game length: {avg_loss_len:.1f}")

        nn_cmd_avg = sum(a.nn_commander_moves for a in all_losses) / len(all_losses)
        opp_cmd_avg = sum(a.opp_commander_moves for a in all_losses) / len(all_losses)
        print(f"  Avg commander moves: NN={nn_cmd_avg:.1f}, Random={opp_cmd_avg:.1f}")

        avg_checks = sum(a.nn_in_check_count for a in all_losses) / len(all_losses)
        print(f"  Avg times NN in check: {avg_checks:.1f}")

        # What piece delivers the killing blow?
        print(f"\n  Final capture in losses (the kill shot):")
        final_captor = Counter()
        final_victim = Counter()
        for a in all_losses:
            if a.capture_timeline:
                last = a.capture_timeline[-1]
                if last["who"] == "RND":
                    final_captor[last["attacker"]] += 1
                    final_victim[last["victim"]] += 1
                    print(f"    Iter {a.iteration} G{a.game_num}: {last['attacker']} takes "
                          f"{last['victim']} at ply {last['ply']} ({last['dcn']})")
                else:
                    # NN made last capture but still lost
                    print(f"    Iter {a.iteration} G{a.game_num}: NN made last capture "
                          f"({last['dcn']}) but still lost!")
            else:
                print(f"    Iter {a.iteration} G{a.game_num}: NO captures recorded "
                      f"(replay may be incomplete)")
        if final_captor:
            print(f"  Kill shot piece: {dict(final_captor)}")
            print(f"  Kill shot victim: {dict(final_victim)}")

        # NN move types in losses
        total_nn_moves_in_losses = sum(a.nn_non_capture_moves + a.nn_capture_moves + a.nn_promotion_moves for a in all_losses)
        total_nn_noncap = sum(a.nn_non_capture_moves for a in all_losses)
        total_nn_cap = sum(a.nn_capture_moves for a in all_losses)
        total_nn_promo = sum(a.nn_promotion_moves for a in all_losses)
        if total_nn_moves_in_losses > 0:
            print(f"\n  NN move type distribution in losses:")
            print(f"    Non-capture: {total_nn_noncap} ({total_nn_noncap/total_nn_moves_in_losses*100:.1f}%)")
            print(f"    Capture: {total_nn_cap} ({total_nn_cap/total_nn_moves_in_losses*100:.1f}%)")
            print(f"    Promotion: {total_nn_promo} ({total_nn_promo/total_nn_moves_in_losses*100:.1f}%)")

    # =====================================================
    # SECTION 3: WIN COMPARISON
    # =====================================================
    if all_wins:
        print(f"\n{'#'*80}")
        print(f"SECTION 3: WIN COMPARISON ({len(all_wins)} wins)")
        print(f"{'#'*80}")

        win_sample = all_wins[:10]
        for a in win_sample:
            print(f"\n  Iter {a.iteration} G{a.game_num}: NN={a.nn_color}, "
                  f"{a.total_moves} moves, End={a.end_reason}")
            print(f"    Captures: NN={a.nn_captures}, RND={a.opp_captures}")
            print(f"    NN lost: {dict(a.nn_pieces_lost)}")
            print(f"    RND lost: {dict(a.opp_pieces_lost)}")
            print(f"    Cmdr moves: NN={a.nn_commander_moves}, RND={a.opp_commander_moves}")
            print(f"    Opening: {' '.join(a.opening_moves)}")

        if all_losses:
            print(f"\n  --- Statistical Comparison ---")
            avg_win_len = sum(a.total_moves for a in all_wins) / len(all_wins)
            avg_win_nn_cap = sum(a.nn_captures for a in all_wins) / len(all_wins)
            avg_win_opp_cap = sum(a.opp_captures for a in all_wins) / len(all_wins)
            avg_win_cmd = sum(a.nn_commander_moves for a in all_wins) / len(all_wins)
            avg_win_checks = sum(a.nn_in_check_count for a in all_wins) / len(all_wins)
            avg_win_noncap = sum(a.nn_non_capture_moves for a in all_wins) / len(all_wins)

            print(f"                          Wins ({len(all_wins)})    Losses ({len(all_losses)})")
            print(f"  Avg game length:        {avg_win_len:6.1f}         {avg_loss_len:6.1f}")
            print(f"  Avg NN captures:        {avg_win_nn_cap:6.1f}         {nn_cap_total/len(all_losses):6.1f}")
            print(f"  Avg RND captures:       {avg_win_opp_cap:6.1f}         {opp_cap_total/len(all_losses):6.1f}")
            print(f"  Avg NN cmdr moves:      {avg_win_cmd:6.1f}         {nn_cmd_avg:6.1f}")
            print(f"  Avg NN in check:        {avg_win_checks:6.1f}         {avg_checks:6.1f}")
            print(f"  Avg NN non-cap moves:   {avg_win_noncap:6.1f}         {total_nn_noncap/len(all_losses):6.1f}")

            # Win pieces lost by NN
            win_pieces_lost = Counter()
            for a in all_wins:
                for p, c in a.nn_pieces_lost.items():
                    win_pieces_lost[p] += c

            print(f"\n  NN pieces lost per game:")
            all_pieces = sorted(set(list(win_pieces_lost.keys()) + list(nn_piece_losses.keys())))
            for p in all_pieces:
                w = win_pieces_lost.get(p, 0) / len(all_wins)
                l = nn_piece_losses.get(p, 0) / len(all_losses)
                print(f"    {p}: wins={w:.2f}/game  losses={l:.2f}/game  diff={l-w:+.2f}")

            # NN move types in wins
            total_win_nn_moves = sum(a.nn_non_capture_moves + a.nn_capture_moves + a.nn_promotion_moves for a in all_wins)
            win_nn_noncap = sum(a.nn_non_capture_moves for a in all_wins)
            win_nn_cap = sum(a.nn_capture_moves for a in all_wins)
            win_nn_promo = sum(a.nn_promotion_moves for a in all_wins)
            if total_win_nn_moves > 0:
                print(f"\n  NN move type distribution in WINS:")
                print(f"    Non-capture: {win_nn_noncap} ({win_nn_noncap/total_win_nn_moves*100:.1f}%)")
                print(f"    Capture: {win_nn_cap} ({win_nn_cap/total_win_nn_moves*100:.1f}%)")
                print(f"    Promotion: {win_nn_promo} ({win_nn_promo/total_win_nn_moves*100:.1f}%)")

            # Win end reasons
            win_end = Counter(a.end_reason for a in all_wins)
            print(f"\n  Win end reasons:")
            for reason, count in win_end.most_common():
                print(f"    {reason}: {count}")

    # =====================================================
    # SECTION 4: DRAW ANALYSIS
    # =====================================================
    if all_draws:
        print(f"\n{'#'*80}")
        print(f"SECTION 4: DRAW ANALYSIS ({len(all_draws)} draws)")
        print(f"{'#'*80}")

        draw_reasons = Counter(a.end_reason for a in all_draws)
        for reason, count in draw_reasons.most_common():
            print(f"  {reason}: {count}")
        for a in all_draws:
            print(f"  Iter {a.iteration} G{a.game_num}: NN={a.nn_color}, {a.total_moves} moves, "
                  f"NN cap={a.nn_captures}, RND cap={a.opp_captures}, end={a.end_reason}")

    # =====================================================
    # SECTION 5: OPENING CORRELATION
    # =====================================================
    print(f"\n{'#'*80}")
    print(f"SECTION 5: OPENING CORRELATION")
    print(f"{'#'*80}")

    loss_first_moves = Counter()
    win_first_moves = Counter()
    for a in all_losses:
        if a.opening_moves:
            loss_first_moves[a.opening_moves[0]] += 1
    for a in all_wins:
        if a.opening_moves:
            win_first_moves[a.opening_moves[0]] += 1

    print(f"\n  First move in losses:")
    for m, c in loss_first_moves.most_common(10):
        print(f"    {m}: {c}")
    print(f"  First move in wins:")
    for m, c in win_first_moves.most_common(10):
        print(f"    {m}: {c}")

    # =====================================================
    # SECTION 6: CAUSAL HYPOTHESIS SYNTHESIS
    # =====================================================
    if all_losses:
        print(f"\n{'#'*80}")
        print(f"SECTION 6: CAUSAL HYPOTHESIS SYNTHESIS")
        print(f"{'#'*80}")

        cmd_cap = sum(1 for a in all_losses if a.end_reason == "commander_captured")
        checkmate = sum(1 for a in all_losses if a.end_reason == "checkmate")
        draw_end = sum(1 for a in all_losses if "draw" in a.end_reason)
        incomplete = sum(1 for a in all_losses if a.end_reason == "replay_incomplete")
        n = len(all_losses)

        print(f"\n  H1: Random wins by commander capture: {cmd_cap}/{n} ({cmd_cap/n*100:.0f}%)")
        print(f"  H2: Random wins by checkmate: {checkmate}/{n} ({checkmate/n*100:.0f}%)")
        print(f"  H3: Losses that are actually draws: {draw_end}/{n} ({draw_end/n*100:.0f}%)")
        print(f"  H4: Incomplete replays (can't determine): {incomplete}/{n}")

        nn_fewer = sum(1 for a in all_losses if a.nn_captures < a.opp_captures)
        print(f"  H5: NN made fewer captures: {nn_fewer}/{n} ({nn_fewer/n*100:.0f}%)")

        nn_passive_cmd = sum(1 for a in all_losses if a.nn_commander_moves < a.opp_commander_moves)
        print(f"  H6: NN commander less active: {nn_passive_cmd}/{n}")

        long_games = sum(1 for a in all_losses if a.total_moves > 100)
        print(f"  H7: Long games (>100 moves): {long_games}/{n}")

        # Unique analysis: what's the NN's capture-per-move rate?
        if all_wins:
            win_cap_rate = sum(a.nn_captures for a in all_wins) / sum(
                a.nn_non_capture_moves + a.nn_capture_moves + a.nn_promotion_moves for a in all_wins) if sum(
                a.nn_non_capture_moves + a.nn_capture_moves + a.nn_promotion_moves for a in all_wins) > 0 else 0
            loss_cap_rate = nn_cap_total / total_nn_moves_in_losses if total_nn_moves_in_losses > 0 else 0
            print(f"\n  NN capture rate (captures/NN_moves):")
            print(f"    In wins:   {win_cap_rate:.3f}")
            print(f"    In losses: {loss_cap_rate:.3f}")

        # Specific piece loss analysis per game
        print(f"\n  Per-game piece loss detail in losses:")
        for a in all_losses:
            print(f"    Iter {a.iteration} G{a.game_num} (NN={a.nn_color}):")
            if a.nn_pieces_lost_detail:
                for ply, piece, captor in a.nn_pieces_lost_detail:
                    print(f"      Ply {ply}: NN loses {piece} to {captor}")
            else:
                print(f"      No NN pieces lost (replay may be incomplete)")

        # Check: how many losses have replay_ok?
        ok_losses = sum(1 for a in all_losses if a.replay_ok)
        print(f"\n  Losses with full replay: {ok_losses}/{n}")
        if ok_losses < n:
            print(f"  WARNING: {n - ok_losses} losses had incomplete replay — "
                  f"opening randomization seed not found in first 5000 seeds.")
            print(f"  The incomplete replays may have missing capture data.")


if __name__ == "__main__":
    main()
