"""Smart seed game generation using heuristic strategies and endgame patterns."""

import logging
import time
import random
import numpy as np
from typing import List, Tuple, Optional
from davechess.game.state import GameState, Piece, Player, Move, PieceType
from davechess.game.board import BOARD_SIZE
from davechess.game.rules import apply_move, generate_legal_moves, is_in_check
from davechess.engine.heuristic_player import HeuristicPlayer, SmartMCTS
from davechess.engine.network import state_to_planes, move_to_policy_index, POLICY_SIZE
from davechess.engine.selfplay import ReplayBuffer

logger = logging.getLogger("davechess.smart_seeds")


class CommanderHunter:
    """Ultra-aggressive player that hunts enemy commander."""

    def get_move(self, state: GameState):
        """Get move that most threatens enemy commander."""
        moves = generate_legal_moves(state)
        if not moves:
            return None

        # Find enemy commander
        enemy_commander_pos = None
        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece and piece.piece_type == PieceType.COMMANDER:
                    if piece.player != state.current_player:
                        enemy_commander_pos = (r, c)
                        break

        if not enemy_commander_pos:
            return random.choice(moves)

        # Score moves by distance to enemy commander
        best_moves = []
        best_score = -999

        for move in moves:
            score = 0

            # Check if this directly captures commander
            if hasattr(move, 'to_rc') and move.to_rc == enemy_commander_pos:
                return move  # Instant win!

            # Score by manhattan distance
            if hasattr(move, 'to_rc'):
                to_r, to_c = move.to_rc
                cmd_r, cmd_c = enemy_commander_pos
                dist = abs(to_r - cmd_r) + abs(to_c - cmd_c)
                score = -dist

                # Bonus for threatening
                if dist == 1:
                    score += 10

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(moves)


def play_heuristic_game(white_player, black_player, max_moves: int = 100) -> Tuple[List[Move], Player, List[GameState]]:
    """Play a game between two heuristic players."""
    state = GameState()
    moves = []
    states = []

    for turn in range(max_moves):
        states.append(state.clone())

        # Get move from current player
        if state.current_player == Player.WHITE:
            move = white_player.get_move(state)
        else:
            move = black_player.get_move(state)

        # If no legal moves, game ends
        if move is None:
            break

        moves.append(move)
        apply_move(state, move)

        if state.done:
            break

    return moves, state.winner, states


def generate_smart_seeds(num_games: int = 50, verbose: bool = False) -> ReplayBuffer:
    """Generate seed games using smart heuristic strategies."""
    buffer = ReplayBuffer(max_size=100000)

    # Mostly CommanderHunter matchups since those actually finish
    players = [
        ("Commander Hunter", CommanderHunter()),
        ("Commander Hunter", CommanderHunter()),
        ("Commander Hunter", CommanderHunter()),
        ("Aggressive", HeuristicPlayer(exploration=0.05, aggression=0.95)),
    ]

    game_count = 0
    total_positions = 0
    skipped = 0

    if verbose:
        print(f"Generating {num_games} smart seed games...")

    while game_count < num_games:
        white_name, white_player = random.choice(players)
        black_name, black_player = random.choice(players)

        # Play game
        moves, winner, states = play_heuristic_game(white_player, black_player)

        # Skip games that hit max length
        if len(moves) >= 100:
            skipped += 1
            if verbose:
                print(f"  [skip {skipped}] {white_name} vs {black_name}: {len(moves)} moves (max length) | {game_count}/{num_games} done", flush=True)
            continue

        # Convert to training examples
        for i, (state, move) in enumerate(zip(states, moves)):
            # Create planes
            planes = state_to_planes(state)

            # Create policy (one-hot for actual move)
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            policy[move_to_policy_index(move)] = 1.0

            # Create value from game outcome: +1 win, -1 loss
            if winner == state.current_player:
                value = 1.0
            else:
                value = -1.0

            buffer.push(planes, policy, value)
            total_positions += 1

        game_count += 1

        if verbose:
            winner_str = "White" if winner == Player.WHITE else "Black" if winner == Player.BLACK else "Draw"
            print(f"  [game {game_count}/{num_games}] {white_name} vs {black_name}: {len(moves)} moves, {winner_str} | {total_positions} positions, {skipped} skipped", flush=True)

    if verbose:
        print(f"Complete: {game_count} games, {total_positions} positions, {skipped} skipped")

    return buffer


def _make_endgame_state(
    white_pieces: list[tuple[int, int, PieceType]],
    black_pieces: list[tuple[int, int, PieceType]],
    current_player: Player = Player.WHITE,
) -> GameState:
    """Create a GameState with only the specified pieces on the board."""
    state = GameState.__new__(GameState)
    state.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    state.resources = [0, 0]
    state.current_player = current_player
    state.turn = 1
    state.done = False
    state.winner = None
    state.move_history = []
    state.position_counts = {}
    state.halfmove_clock = 0
    for r, c, pt in white_pieces:
        state.board[r][c] = Piece(pt, Player.WHITE)
    for r, c, pt in black_pieces:
        state.board[r][c] = Piece(pt, Player.BLACK)
    state.position_counts[state.get_position_key()] = 1
    return state


def _random_endgame_position(
    attacker_piece: PieceType,
    two_attackers: bool = False,
) -> Optional[GameState]:
    """Generate a structured endgame position biased toward winnability.

    Black Commander is placed on edge/corner. White Commander is placed
    nearby (2-3 squares away, helping cut off escape). Attacker piece(s)
    are placed at medium distance (3-5 squares) to create realistic
    mating scenarios that MCTS can solve in reasonable time.
    """
    # Black Commander: always on edge, 50% chance corner
    if random.random() < 0.5:
        # Corner
        bc_r, bc_c = random.choice([(0, 0), (0, 7), (7, 0), (7, 7)])
    else:
        # Edge (not corner)
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            bc_r, bc_c = 7, random.randint(1, 6)
        elif edge == 'bottom':
            bc_r, bc_c = 0, random.randint(1, 6)
        elif edge == 'left':
            bc_r, bc_c = random.randint(1, 6), 0
        else:
            bc_r, bc_c = random.randint(1, 6), 7

    # White Commander: 2-3 squares away from Black Commander (opposition)
    wc_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 2 <= dist <= 4 and (r, c) != (bc_r, bc_c):
                # Not adjacent (would be mutual check)
                if abs(r - bc_r) > 1 or abs(c - bc_c) > 1:
                    wc_candidates.append((r, c))
    if not wc_candidates:
        return None
    wc_r, wc_c = random.choice(wc_candidates)

    # Attacker piece: 2-5 squares from Black Commander, not adjacent
    occupied = {(bc_r, bc_c), (wc_r, wc_c)}
    wp_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in occupied:
                continue
            # Not adjacent to Black Commander
            if abs(r - bc_r) <= 1 and abs(c - bc_c) <= 1:
                continue
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 2 <= dist <= 6:
                wp_candidates.append((r, c))
    if not wp_candidates:
        return None
    wp_r, wp_c = random.choice(wp_candidates)
    occupied.add((wp_r, wp_c))

    white_pieces = [
        (wc_r, wc_c, PieceType.COMMANDER),
        (wp_r, wp_c, attacker_piece),
    ]

    if two_attackers:
        wp2_candidates = [(r, c) for r, c in wp_candidates if (r, c) not in occupied]
        if not wp2_candidates:
            return None
        wp2_r, wp2_c = random.choice(wp2_candidates)
        white_pieces.append((wp2_r, wp2_c, attacker_piece))

    black_pieces = [(bc_r, bc_c, PieceType.COMMANDER)]

    state = _make_endgame_state(white_pieces, black_pieces, Player.WHITE)

    if is_in_check(state, Player.WHITE):
        return None

    return state


def _random_mixed_endgame_position() -> Optional[GameState]:
    """Generate R+L+C vs lone C endgame position."""
    # Black Commander on edge/corner
    if random.random() < 0.5:
        bc_r, bc_c = random.choice([(0, 0), (0, 7), (7, 0), (7, 7)])
    else:
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            bc_r, bc_c = 7, random.randint(1, 6)
        elif edge == 'bottom':
            bc_r, bc_c = 0, random.randint(1, 6)
        elif edge == 'left':
            bc_r, bc_c = random.randint(1, 6), 0
        else:
            bc_r, bc_c = random.randint(1, 6), 7

    # White Commander nearby
    wc_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 2 <= dist <= 4 and (abs(r - bc_r) > 1 or abs(c - bc_c) > 1):
                wc_candidates.append((r, c))
    if not wc_candidates:
        return None
    wc_r, wc_c = random.choice(wc_candidates)

    occupied = {(bc_r, bc_c), (wc_r, wc_c)}
    piece_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in occupied:
                continue
            if abs(r - bc_r) <= 1 and abs(c - bc_c) <= 1:
                continue
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 2 <= dist <= 6:
                piece_candidates.append((r, c))
    if len(piece_candidates) < 2:
        return None
    chosen = random.sample(piece_candidates, 2)

    white_pieces = [
        (wc_r, wc_c, PieceType.COMMANDER),
        (chosen[0][0], chosen[0][1], PieceType.RIDER),
        (chosen[1][0], chosen[1][1], PieceType.LANCER),
    ]
    black_pieces = [(bc_r, bc_c, PieceType.COMMANDER)]

    state = _make_endgame_state(white_pieces, black_pieces, Player.WHITE)
    if is_in_check(state, Player.WHITE):
        return None
    return state


def _random_endgame_position_generic(
    attacker_pieces: list,
) -> Optional[GameState]:
    """Generate an endgame position with any combination of attacker pieces vs lone Commander.

    attacker_pieces: list of PieceType for white's attacking pieces (Commander added automatically).
    Black Commander is placed on edge/corner. White Commander nearby. Attackers at medium distance.
    """
    # Black Commander: always on edge, 50% chance corner
    if random.random() < 0.5:
        bc_r, bc_c = random.choice([(0, 0), (0, 7), (7, 0), (7, 7)])
    else:
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            bc_r, bc_c = 7, random.randint(1, 6)
        elif edge == 'bottom':
            bc_r, bc_c = 0, random.randint(1, 6)
        elif edge == 'left':
            bc_r, bc_c = random.randint(1, 6), 0
        else:
            bc_r, bc_c = random.randint(1, 6), 7

    # White Commander: 2-4 Manhattan distance, not adjacent
    wc_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 2 <= dist <= 4 and (abs(r - bc_r) > 1 or abs(c - bc_c) > 1):
                wc_candidates.append((r, c))
    if not wc_candidates:
        return None
    wc_r, wc_c = random.choice(wc_candidates)

    occupied = {(bc_r, bc_c), (wc_r, wc_c)}

    # Place each attacker piece: 2-6 Manhattan from Black Commander, not adjacent
    white_pieces = [(wc_r, wc_c, PieceType.COMMANDER)]
    for pt in attacker_pieces:
        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in occupied:
                    continue
                if abs(r - bc_r) <= 1 and abs(c - bc_c) <= 1:
                    continue
                dist = abs(r - bc_r) + abs(c - bc_c)
                if 2 <= dist <= 6:
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        white_pieces.append((r, c, pt))
        occupied.add((r, c))

    black_pieces = [(bc_r, bc_c, PieceType.COMMANDER)]
    state = _make_endgame_state(white_pieces, black_pieces, Player.WHITE)
    if is_in_check(state, Player.WHITE):
        return None
    return state


def generate_endgame_seeds(
    num_positions: int = 500,
    mcts_sims: int = 400,
    max_game_moves: int = 80,
    verbose: bool = False,
) -> ReplayBuffer:
    """Generate training data from endgame positions where MCTS finds checkmate.

    Creates random R+C vs C and L+C vs C positions, plays them out with
    MCTSLite, and records positions from games that end in checkmate.
    This teaches the network what winning endgames look like and how to
    deliver checkmate.
    """
    from davechess.engine.mcts_lite import MCTSLite

    buffer = ReplayBuffer(max_size=200000)
    mcts = MCTSLite(num_simulations=mcts_sims, max_rollout_depth=200)

    # Endgame types to generate, weighted by importance.
    # Single-piece endgames (R+C, L+C) are too hard for MCTS to win from
    # random positions, so we focus on multi-piece endgames that teach
    # cornering and mating technique. The model generalizes to single-piece.
    # Format: (piece_types_list, label, weight)
    # piece_types_list is a list of PieceTypes for the attackers (Commander added automatically)
    endgame_configs = [
        # 2-piece endgames (original)
        ([PieceType.RIDER, PieceType.RIDER], "2R+C vs C", 3),
        ([PieceType.LANCER, PieceType.LANCER], "2L+C vs C", 2),
        ([PieceType.RIDER, PieceType.LANCER], "R+L+C vs C", 3),
        # Bombard combos — Bombard can't ranged-attack Commander, needs partner
        ([PieceType.RIDER, PieceType.BOMBARD], "R+B+C vs C", 3),
        ([PieceType.LANCER, PieceType.BOMBARD], "L+B+C vs C", 2),
        # 3-piece endgames — model often has 3+ pieces left
        ([PieceType.RIDER, PieceType.RIDER, PieceType.LANCER], "2R+L+C vs C", 2),
        ([PieceType.RIDER, PieceType.LANCER, PieceType.BOMBARD], "R+L+B+C vs C", 2),
        ([PieceType.RIDER, PieceType.RIDER, PieceType.BOMBARD], "2R+B+C vs C", 2),
    ]
    total_weight = sum(w for _, _, w in endgame_configs)

    wins = 0
    draws = 0
    total_positions = 0
    attempts = 0
    max_attempts = num_positions * 20  # Give up after enough tries

    if verbose:
        print(f"Generating endgame seeds ({mcts_sims} MCTS sims per move)...")

    while wins < num_positions and attempts < max_attempts:
        # Pick endgame type by weight
        rand_val = random.random() * total_weight
        cumulative = 0
        for piece_types, label, weight in endgame_configs:
            cumulative += weight
            if rand_val <= cumulative:
                break

        state = _random_endgame_position_generic(piece_types)
        if state is None:
            attempts += 1
            continue

        attempts += 1

        # Play out with MCTS, recording positions
        game_states = []
        game_moves = []
        game_policies = []
        move_count = 0

        while not state.done and move_count < max_game_moves:
            moves = generate_legal_moves(state)
            if not moves:
                break

            # Get MCTS move with visit-count policy
            best_move, visit_policy = mcts.search_with_policy(state)

            # Record this position
            game_states.append(state.clone())
            game_moves.append(best_move)

            # Convert visit policy to policy vector
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            for m, prob in visit_policy.items():
                try:
                    idx = move_to_policy_index(m)
                    policy[idx] = prob
                except (ValueError, IndexError):
                    pass
            game_policies.append(policy)

            apply_move(state, best_move)
            move_count += 1

        # Only keep games that end in checkmate with at least 4 moves
        # (short games are trivial captures, not real mating technique)
        if state.done and state.winner is not None and move_count >= 4:
            wins += 1
            winner = state.winner

            for gs, policy in zip(game_states, game_policies):
                planes = state_to_planes(gs)
                # +1 if current player wins, -1 if they lose
                value = 1.0 if gs.current_player == winner else -1.0
                buffer.push(planes, policy, value)
                total_positions += 1

            if verbose:
                print(
                    f"  [win {wins}/{num_positions}] {label}: "
                    f"{move_count} moves, {'White' if winner == Player.WHITE else 'Black'} wins | "
                    f"{total_positions} positions ({attempts} attempts)",
                    flush=True,
                )
        else:
            draws += 1

    if verbose:
        win_rate = wins / max(attempts, 1) * 100
        print(
            f"Endgame seeds complete: {wins} wins, {draws} draws "
            f"({win_rate:.0f}% win rate), {total_positions} positions"
        )

    return buffer


def _random_middlegame_position(
    white_attackers: list[PieceType],
    black_defenders: list[PieceType],
) -> Optional[GameState]:
    """Generate a middlegame position with material imbalance.

    White has Commander + attackers, Black has Commander + defenders.
    Black Commander on edge/corner, Black defenders nearby.
    White Commander farther away (center-ish), White attackers spread out.
    """
    # Black Commander: edge/corner (realistic mating target)
    if random.random() < 0.5:
        bc_r, bc_c = random.choice([(0, 0), (0, 7), (7, 0), (7, 7)])
    else:
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            bc_r, bc_c = 7, random.randint(1, 6)
        elif edge == 'bottom':
            bc_r, bc_c = 0, random.randint(1, 6)
        elif edge == 'left':
            bc_r, bc_c = random.randint(1, 6), 0
        else:
            bc_r, bc_c = random.randint(1, 6), 7

    occupied = {(bc_r, bc_c)}
    black_pieces = [(bc_r, bc_c, PieceType.COMMANDER)]

    # Black defenders: 2-4 Manhattan from Black Commander
    for pt in black_defenders:
        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in occupied:
                    continue
                dist = abs(r - bc_r) + abs(c - bc_c)
                if 1 <= dist <= 4:
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        black_pieces.append((r, c, pt))
        occupied.add((r, c))

    # White Commander: 4-6 Manhattan from Black Commander
    wc_candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in occupied:
                continue
            dist = abs(r - bc_r) + abs(c - bc_c)
            if 4 <= dist <= 7:
                # Not adjacent to Black Commander
                if abs(r - bc_r) > 1 or abs(c - bc_c) > 1:
                    wc_candidates.append((r, c))
    if not wc_candidates:
        return None
    wc_r, wc_c = random.choice(wc_candidates)
    occupied.add((wc_r, wc_c))
    white_pieces = [(wc_r, wc_c, PieceType.COMMANDER)]

    # White attackers: 3-7 Manhattan from Black Commander, not adjacent
    for pt in white_attackers:
        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in occupied:
                    continue
                if abs(r - bc_r) <= 1 and abs(c - bc_c) <= 1:
                    continue
                dist = abs(r - bc_r) + abs(c - bc_c)
                if 2 <= dist <= 7:
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        white_pieces.append((r, c, pt))
        occupied.add((r, c))

    state = _make_endgame_state(white_pieces, black_pieces, Player.WHITE)

    # Validate: neither side in check at start
    if is_in_check(state, Player.WHITE):
        return None
    if is_in_check(state, Player.BLACK):
        return None

    return state


def generate_middlegame_checkmate_seeds(
    num_positions: int = 300,
    mcts_sims: int = 100,
    max_game_moves: int = 60,
    verbose: bool = False,
) -> ReplayBuffer:
    """Generate middlegame positions with material imbalance played to checkmate.

    Bridges the gap between heuristic full games (natural but often drawn) and
    endgame seeds (always checkmate but bare-minimum pieces). These positions
    have White with extra pieces and Black with some defenders, teaching the
    network how to break through defenses and convert advantages into checkmate.
    """
    from davechess.engine.mcts_lite import MCTSLite

    buffer = ReplayBuffer(max_size=200000)
    mcts = MCTSLite(num_simulations=mcts_sims, max_rollout_depth=200)

    # (white_attackers, black_defenders, label, weight)
    middlegame_configs = [
        # Rider-heavy attacks vs light defense
        ([PieceType.RIDER, PieceType.RIDER, PieceType.LANCER],
         [PieceType.WARRIOR, PieceType.BOMBARD],
         "2R+L+C vs W+B+C", 3),
        # Lancer-heavy attack vs moderate defense
        ([PieceType.RIDER, PieceType.LANCER, PieceType.LANCER],
         [PieceType.RIDER, PieceType.WARRIOR],
         "R+2L+C vs R+W+C", 3),
        # Riders + Bombard vs Warrior wall
        ([PieceType.RIDER, PieceType.RIDER, PieceType.BOMBARD],
         [PieceType.WARRIOR, PieceType.WARRIOR],
         "2R+B+C vs 2W+C", 2),
        # Mixed attack vs mixed defense
        ([PieceType.RIDER, PieceType.LANCER, PieceType.BOMBARD],
         [PieceType.WARRIOR, PieceType.BOMBARD],
         "R+L+B+C vs W+B+C", 2),
        # Three Riders vs heavier defense
        ([PieceType.RIDER, PieceType.RIDER, PieceType.RIDER],
         [PieceType.WARRIOR, PieceType.WARRIOR, PieceType.BOMBARD],
         "3R+C vs 2W+B+C", 2),
        # Dominant Lancers vs minimal defense
        ([PieceType.LANCER, PieceType.LANCER],
         [PieceType.WARRIOR],
         "2L+C vs W+C", 2),
    ]
    total_weight = sum(w for _, _, _, w in middlegame_configs)

    wins = 0
    draws = 0
    total_positions = 0
    attempts = 0
    max_attempts = num_positions * 30

    if verbose:
        print(f"Generating middlegame checkmate seeds ({mcts_sims} MCTS sims per move)...")

    while wins < num_positions and attempts < max_attempts:
        # Pick config by weight
        rand_val = random.random() * total_weight
        cumulative = 0
        for white_attackers, black_defenders, label, weight in middlegame_configs:
            cumulative += weight
            if rand_val <= cumulative:
                break

        state = _random_middlegame_position(white_attackers, black_defenders)
        if state is None:
            attempts += 1
            continue

        attempts += 1

        # Play out with MCTS, recording positions
        game_states = []
        game_moves = []
        game_policies = []
        move_count = 0

        while not state.done and move_count < max_game_moves:
            moves = generate_legal_moves(state)
            if not moves:
                break

            best_move, visit_policy = mcts.search_with_policy(state)

            game_states.append(state.clone())
            game_moves.append(best_move)

            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            for m, prob in visit_policy.items():
                try:
                    idx = move_to_policy_index(m)
                    policy[idx] = prob
                except (ValueError, IndexError):
                    pass
            game_policies.append(policy)

            apply_move(state, best_move)
            move_count += 1

        # Keep games with checkmate and ≥8 moves (enough to show conversion)
        if state.done and state.winner is not None and move_count >= 8:
            wins += 1
            winner = state.winner

            for gs, policy in zip(game_states, game_policies):
                planes = state_to_planes(gs)
                value = 1.0 if gs.current_player == winner else -1.0
                buffer.push(planes, policy, value)
                total_positions += 1

            if verbose:
                print(
                    f"  [win {wins}/{num_positions}] {label}: "
                    f"{move_count} moves, {'White' if winner == Player.WHITE else 'Black'} wins | "
                    f"{total_positions} positions ({attempts} attempts)",
                    flush=True,
                )
        else:
            draws += 1

    if verbose:
        win_rate = wins / max(attempts, 1) * 100
        print(
            f"Middlegame seeds complete: {wins} wins, {draws} draws "
            f"({win_rate:.0f}% win rate), {total_positions} positions"
        )

    return buffer