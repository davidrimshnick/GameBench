"""
Estimate ELO of 50-sim random-rollout MCTS in chess.

Key insight: pure MCTS with random rollouts can't deliver checkmate
even with overwhelming material. We use material adjudication:
if one side is up 15+ points of material, it's a win.
This matches how human games would be resigned.
"""
import chess
import math
import random
import time


class MCTSNode:
    __slots__ = ['parent', 'move', 'children', 'visits', 'value', 'untried_moves']

    def __init__(self, legal_moves, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(legal_moves)
        random.shuffle(self.untried_moves)

    def ucb1(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def best_move_child(self):
        return max(self.children, key=lambda c: c.visits)


PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def material_balance(board):
    """Material score from white's perspective (centipawns-ish)."""
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_VALUES[p.piece_type]
            if p.color:
                score += v
            else:
                score -= v
    return score


def material_eval_sigmoid(board):
    score = material_balance(board)
    return 1.0 / (1.0 + math.exp(-score * 0.3))


def fast_rollout(board, max_depth=40):
    b = board.copy(stack=False)
    side = board.turn
    for _ in range(max_depth):
        moves = list(b.legal_moves)
        if not moves:
            if b.is_check():
                winner = not b.turn
                return 1.0 if (winner == chess.WHITE) == (side == chess.WHITE) else 0.0
            return 0.5
        b.push(random.choice(moves))
    wv = material_eval_sigmoid(b)
    return wv if side == chess.WHITE else 1.0 - wv


def mcts_select_move(board, num_simulations=50):
    root = MCTSNode(board.legal_moves)
    for _ in range(num_simulations):
        node = root
        b = board.copy()
        while not node.untried_moves and node.children:
            node = node.best_child()
            b.push(node.move)
        if node.untried_moves:
            move = node.untried_moves.pop()
            b.push(move)
            child = MCTSNode(b.legal_moves, parent=node, move=move)
            node.children.append(child)
            node = child
        result = fast_rollout(b)
        # Rollout returns value for side-to-move at leaf.
        # Leaf node stores value for the player who CHOSE the move (parent's side).
        # So flip once before starting backprop.
        result = 1.0 - result
        while node is not None:
            node.visits += 1
            node.value += result
            result = 1.0 - result
            node = node.parent
    return root.best_move_child().move


def play_random_move(board):
    moves = list(board.legal_moves)
    return random.choice(moves) if moves else None


def play_game(white_fn, black_fn, max_moves=500, adjudicate_material=15):
    """
    Play game with proper termination + material adjudication.
    If one side is up by adjudicate_material points, that side wins.
    Returns (result_for_white, move_count, reason).
    """
    board = chess.Board()
    move_count = 0

    for _ in range(max_moves):
        if board.is_checkmate():
            return (1.0 if board.turn == chess.BLACK else 0.0), move_count, "checkmate"
        if board.is_stalemate():
            return 0.5, move_count, "stalemate"
        if board.is_insufficient_material():
            return 0.5, move_count, "insufficient"
        if board.can_claim_fifty_moves():
            return 0.5, move_count, "fifty_move"
        if board.can_claim_threefold_repetition():
            return 0.5, move_count, "threefold"

        # Material adjudication after move 20
        if move_count > 40 and adjudicate_material > 0:
            mb = material_balance(board)
            if mb >= adjudicate_material:
                return 1.0, move_count, "adjudicate_white"
            elif mb <= -adjudicate_material:
                return 0.0, move_count, "adjudicate_black"

        if board.turn == chess.WHITE:
            move = white_fn(board)
        else:
            move = black_fn(board)
        if move is None:
            break
        board.push(move)
        move_count += 1

    return 0.5, move_count, "max_moves"


def wr_to_elo(wr):
    if wr <= 0.001: return -800
    if wr >= 0.999: return 800
    return 400 * math.log10(wr / (1.0 - wr))


def run_match(name, p1_fn, p2_fn, num_games=30, adjudicate=15):
    wins = draws = losses = 0
    reasons = {}
    t0 = time.time()
    for i in range(num_games):
        if i % 2 == 0:
            r, moves, reason = play_game(p1_fn, p2_fn, adjudicate_material=adjudicate)
        else:
            r, moves, reason = play_game(p2_fn, p1_fn, adjudicate_material=adjudicate)
            r = 1.0 - r

        if r == 1.0: wins += 1
        elif r == 0.0: losses += 1
        else: draws += 1
        reasons[reason] = reasons.get(reason, 0) + 1

        elapsed = time.time() - t0
        wr = (wins + 0.5 * draws) / (i + 1)
        print(f"  {name} {i+1}/{num_games}: "
              f"{'W' if r==1 else 'L' if r==0 else 'D'} "
              f"({reason},{moves}mv) "
              f"[{wins}W {draws}D {losses}L WR={wr:.2f} Elo={wr_to_elo(wr):+.0f}] "
              f"{elapsed:.0f}s", flush=True)

    total = wins + draws + losses
    wr = (wins + 0.5 * draws) / total
    print(f"\n  === {name} FINAL: {total}g {wins}W {draws}D {losses}L "
          f"WR={wr:.3f} Elo={wr_to_elo(wr):+.0f} ({time.time()-t0:.0f}s) ===")
    print(f"  Reasons: {reasons}\n", flush=True)
    return wr, wr_to_elo(wr)


if __name__ == "__main__":
    random.seed(42)

    # Quick timing
    print("Timing: 1 game MCTS-50 vs Random (with adjudication)...", flush=True)
    t = time.time()
    r, mv, reason = play_game(
        lambda b: mcts_select_move(b, 50), play_random_move, adjudicate_material=15)
    print(f"  {time.time()-t:.1f}s, {mv} moves, {reason}, "
          f"{'W' if r==1 else 'L' if r==0 else 'D'}\n", flush=True)

    mcts50 = lambda b: mcts_select_move(b, 50)
    mcts100 = lambda b: mcts_select_move(b, 100)
    mcts200 = lambda b: mcts_select_move(b, 200)

    # MCTSLite-50 vs Random (anchor match)
    print("=" * 60)
    print("Match 1: MCTSLite-50 vs Random (30 games)")
    print("=" * 60, flush=True)
    wr50, elo50 = run_match("M50vR", mcts50, play_random_move, 30, adjudicate=15)

    # MCTSLite-100 vs MCTSLite-50
    print("=" * 60)
    print("Match 2: MCTSLite-100 vs MCTSLite-50 (20 games)")
    print("=" * 60, flush=True)
    wr100v50, elo100v50 = run_match("M100vM50", mcts100, mcts50, 20, adjudicate=15)

    # MCTSLite-200 vs MCTSLite-100
    print("=" * 60)
    print("Match 3: MCTSLite-200 vs MCTSLite-100 (20 games)")
    print("=" * 60, flush=True)
    wr200v100, elo200v100 = run_match("M200vM100", mcts200, mcts100, 20, adjudicate=15)

    # Bootstrap ELO ladder from MCTS-50 = 650
    base_elo = 650
    elo_100 = base_elo + elo100v50
    elo_200 = elo_100 + elo200v100

    print("=" * 60)
    print("RESULTS — Bootstrapped ELO Ladder")
    print("=" * 60)
    print(f"\nAnchor: MCTSLite-50 = {base_elo} FIDE (from previous experiment)")
    print(f"\nMCTSLite-50 vs Random:    WR={wr50:.3f} gap={elo50:+.0f}")
    print(f"MCTSLite-100 vs MCTS-50:  WR={wr100v50:.3f} gap={elo100v50:+.0f}")
    print(f"MCTSLite-200 vs MCTS-100: WR={wr200v100:.3f} gap={elo200v100:+.0f}")
    print()
    print(f"  MCTSLite-50:  {base_elo}")
    print(f"  MCTSLite-100: {elo_100:.0f}")
    print(f"  MCTSLite-200: {elo_200:.0f}")
    print(flush=True)
