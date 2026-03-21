# %% [markdown]
# # DaveChess: Measuring Learning Through Novel Game Mastery
#
# A benchmark for the **Learning** track of "Measuring Progress Toward AGI."
#
# DaveChess is a custom board game absent from all LLM training data. This
# benchmark measures whether an AI system can **learn** a novel strategic game
# by studying expert examples and improving through experience.
#
# **Key design**: Agents get general-purpose tools (file I/O, Python execution)
# and 200 expert games. The benchmark does NOT prescribe how to learn — agents
# that build effective external memory structures (notes, databases, analysis
# scripts) will outperform those limited to in-context learning.

# %%
import os
import sys
import glob
import json
import random
import re
import tempfile
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

# Force unbuffered stdout so logs appear immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"

_kbench_available = False
try:
    import kaggle_benchmarks as kbench
    _kbench_available = True
    print("[INFO] kaggle_benchmarks SDK loaded")
except ImportError:
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "--upgrade", "protobuf>=5.29.6"])
        subprocess.check_call([
            "pip", "install", "-q",
            "kaggle_benchmarks @ git+https://github.com/Kaggle/kaggle-benchmarks.git"
        ])
        import kaggle_benchmarks as kbench
        _kbench_available = True
        print("[INFO] kaggle_benchmarks installed from GitHub and loaded")
    except Exception as exc:
        print(f"[WARN] Failed to install kaggle_benchmarks: {exc}")
        pass

if not _kbench_available:
    e = "not installed"
    print(f"[INFO] kaggle_benchmarks not available ({e}), using local mock")
    # Minimal mock for local testing (no Kaggle SDK available)
    import types
    import contextlib

    class _MockTask:
        """Mock task that wraps a function with .run()."""
        def __init__(self, fn, **kw):
            self.fn = fn
            self.__name__ = fn.__name__
        def run(self, *args, **kwargs):
            return types.SimpleNamespace(result=self.fn(*args, **kwargs))
        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    kbench = types.ModuleType("kaggle_benchmarks")
    kbench.task = lambda **kw: lambda fn: _MockTask(fn, **kw)
    kbench.chats = types.SimpleNamespace(
        new=lambda name: contextlib.nullcontext()
    )
    kbench.assertions = types.SimpleNamespace(
        assert_true=lambda cond, expectation="": None,
    )
    kbench.tools = types.SimpleNamespace(
        python=types.SimpleNamespace(
            script_runner=types.SimpleNamespace(run_code=lambda code: None)
        )
    )
    kbench.llm = None
    _kbench_available = False

# Find dataset - try multiple possible locations
DATASET_DIR = None
_candidates = [
    "/kaggle/input/davechess-benchmark-data",
    "/kaggle/input/datasets/davidrimshnick/davechess-benchmark-data",
]
# Also search /kaggle/input recursively for davechess_engine.py
for candidate in _candidates:
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "davechess_engine.py")):
        DATASET_DIR = candidate
        break

if DATASET_DIR is None:
    # Search for it
    for root, dirs, files in os.walk("/kaggle/input"):
        if "davechess_engine.py" in files:
            DATASET_DIR = root
            break

if DATASET_DIR is None:
    # Local development fallback
    DATASET_DIR = os.getcwd()

sys.path.insert(0, DATASET_DIR)
print(f"[INFO] Dataset dir: {DATASET_DIR}")
if os.path.isdir("/kaggle/input"):
    # Debug: show what's in /kaggle/input
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        if level < 3:
            indent = " " * 2 * level
            print(f"[DEBUG] {indent}{os.path.basename(root)}/")
            if level < 2:
                for f in files[:5]:
                    print(f"[DEBUG] {indent}  {f}")

from davechess_engine import (
    GameState, Player, PieceType, Piece, Move, MoveStep, Promote, BombardAttack,
    generate_legal_moves, apply_move, MCTSLite, play_random_game,
    move_to_dcn, dcn_to_move, render_board, rc_to_notation, GOLD_NODES,
    BOARD_SIZE,
)
from rules_text import get_rules_prompt, build_game_state_message

# Try loading NN opponent
_NN_SIMS = 10
_nn_opponent = None
if True:
    try:
        from nn_opponent import load_nn_opponent
        weights_path = os.path.join(DATASET_DIR, "model_weights.npz")
        if os.path.isfile(weights_path):
            _nn_opponent = load_nn_opponent(weights_path, num_simulations=_NN_SIMS)
            print(f"[INFO] Neural network opponent loaded ({_NN_SIMS} sims)")
        else:
            print(f"[INFO] model_weights.npz not found, using MCTSLite")
    except Exception as e:
        print(f"[INFO] NN opponent failed to load: {e}, using MCTSLite")

# %%
# === Configuration ===
MCTS_SIMS = 100         # MCTSLite opponent strength (fallback)
NN_SIMS = 10            # Neural network MCTS sims (primary opponent)
USE_NN_OPPONENT = True  # Use trained NN as opponent (falls back to MCTSLite if unavailable)
MAX_GAME_TURNS = 40     # Cap game length (draw if no checkmate)
MAX_RETRIES = 3         # Illegal move retries before forfeit
STUDY_BUDGET = 5        # Max study batches (10 GM games each)
PHASE_A_GAMES = 7       # Baseline games (no study)
PHASE_B_GAMES = 7       # Post-study games
PHASE_C_GAMES = 7       # Experience learning games
TOKEN_BUDGET = 10_000_000  # 10M token limit for the entire benchmark
# Budget allocation: ~10% baseline, ~30% study, ~30% post-study, ~30% experience
PHASE_A_TOKEN_BUDGET = 1_000_000    # 1M for baseline (no study needed)
STUDY_TOKEN_BUDGET = 3_000_000      # 3M for studying GM games
PHASE_B_TOKEN_BUDGET = 3_000_000    # 3M for post-study play
PHASE_C_TOKEN_BUDGET = 3_000_000    # 3M for experience learning

# Agent memory workspace
AGENT_WORKSPACE = "/kaggle/working/agent_memory"
if not os.path.isdir("/kaggle/working"):
    AGENT_WORKSPACE = tempfile.mkdtemp(prefix="davechess_agent_")
os.makedirs(AGENT_WORKSPACE, exist_ok=True)


# === Token Tracking ===
class TokenTracker:
    """Track total token usage across all LLM calls.

    Estimates tokens from text length since Model Proxy doesn't return usage.
    Rough estimate: 1 token ≈ 4 chars for English text.
    """
    CHARS_PER_TOKEN = 4

    def __init__(self, budget: int = TOKEN_BUDGET):
        self.budget = budget
        self.total_tokens = 0
        self.total_calls = 0

    def add(self, prompt_text: str, response_text: str):
        """Estimate and add tokens from a prompt/response pair."""
        prompt_tokens = len(prompt_text) // self.CHARS_PER_TOKEN
        response_tokens = len(response_text) // self.CHARS_PER_TOKEN
        self.total_tokens += prompt_tokens + response_tokens
        self.total_calls += 1

    @property
    def remaining(self):
        return max(0, self.budget - self.total_tokens)

    @property
    def exceeded(self):
        return self.total_tokens >= self.budget

    def log(self, phase: str):
        pct = (self.total_tokens / self.budget * 100) if self.budget > 0 else 0
        print(f"[TOKENS] {phase}: {self.total_tokens:,} / {self.budget:,} ({pct:.1f}%) | calls={self.total_calls}", flush=True)

_tracker = TokenTracker()


# %%
# === Tool Definitions ===
# These tools are provided to the LLM agent during study and play phases.
# The agent can use them however it wants — the benchmark is mechanism-agnostic.

def write_file(path: str, content: str) -> str:
    """Write content to a file in the agent's workspace.

    Use this to create notes, strategy documents, analysis results,
    databases, or any other knowledge structure you find useful.
    Path is relative to your workspace directory.

    Args:
        path: Relative file path (e.g., "strategy.md", "patterns/openings.json")
        content: Content to write

    Returns:
        Confirmation message with the full path written.
    """
    full_path = os.path.join(AGENT_WORKSPACE, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def read_file(path: str) -> str:
    """Read content from a file in the agent's workspace.

    Use this to consult your notes, strategy documents, or any other
    files you created during study.

    Args:
        path: Relative file path (e.g., "strategy.md")

    Returns:
        File content as string, or error message if not found.
    """
    full_path = os.path.join(AGENT_WORKSPACE, path)
    if not os.path.isfile(full_path):
        return f"File not found: {path}"
    with open(full_path, "r") as f:
        return f.read()


def list_files(directory: str = ".") -> str:
    """List files in the agent's workspace directory.

    Args:
        directory: Relative directory path (default: workspace root)

    Returns:
        Newline-separated list of files and directories.
    """
    full_path = os.path.join(AGENT_WORKSPACE, directory)
    if not os.path.isdir(full_path):
        return f"Directory not found: {directory}"
    entries = []
    for name in sorted(os.listdir(full_path)):
        entry_path = os.path.join(full_path, name)
        if os.path.isdir(entry_path):
            entries.append(f"  {name}/")
        else:
            size = os.path.getsize(entry_path)
            entries.append(f"  {name} ({size} bytes)")
    return "\n".join(entries) if entries else "(empty directory)"


def get_gm_game(index: int) -> str:
    """Get a specific expert-level DaveChess game in DCN notation.

    These are games played by a strong AlphaZero-trained neural network.
    Study them to learn strategic patterns, opening principles, and
    endgame techniques.

    Args:
        index: Game index (1-200)

    Returns:
        Full game in DCN notation including headers and move list.
    """
    gm_dir = os.path.join(DATASET_DIR, "gm_games")
    path = os.path.join(gm_dir, f"gm_{index:04d}.dcn")
    if not os.path.isfile(path):
        return f"Game {index} not found. Available: 1-200."
    with open(path) as f:
        return f.read()


def get_gm_games_batch(start: int, count: int) -> str:
    """Get a batch of expert games.

    Args:
        start: Starting game index (1-200)
        count: Number of games to retrieve (max 20 per call)

    Returns:
        Multiple games separated by blank lines.
    """
    count = min(count, 20)  # Cap to prevent context overflow
    games = []
    for i in range(start, min(start + count, 201)):
        game = get_gm_game(i)
        if not game.startswith("Game"):
            games.append(game)
    return "\n\n".join(games)


def count_gm_games() -> int:
    """Return the number of expert games available for study.

    Returns:
        Number of available GM games (200).
    """
    gm_dir = os.path.join(DATASET_DIR, "gm_games")
    return len(glob.glob(os.path.join(gm_dir, "gm_*.dcn")))


def run_python_script(code: str) -> str:
    """Execute a Python script and return its output.

    Use this to analyze games programmatically, compute statistics,
    or build data structures. The davechess_engine module is available
    for import.

    Args:
        code: Python code to execute

    Returns:
        Script stdout and stderr output.
    """
    if kbench is not None:
        result = kbench.tools.python.script_runner.run_code(code)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        return output.strip() or "(no output)"
    else:
        # Local fallback: use subprocess
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
            cwd=AGENT_WORKSPACE,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        return output.strip() or "(no output)"


AGENT_TOOLS = [write_file, read_file, list_files, get_gm_game,
               get_gm_games_batch, count_gm_games, run_python_script]

MEMORY_TOOLS = [write_file, read_file, list_files]  # Tools available during play


# %%
# === Game Wrapper ===

@dataclass
class DaveChessMove:
    """Structured output for LLM move selection."""
    move: str       # DCN notation, e.g. "Wd2-d3"
    reasoning: str = ""


class DaveChessGame:
    """Manages a single DaveChess game: LLM vs opponent (NN or MCTSLite)."""

    def __init__(self, mcts_sims: int = MCTS_SIMS, seed: int = 0):
        self.state = GameState()
        self.opponent = _nn_opponent if _nn_opponent is not None else MCTSLite(num_simulations=mcts_sims)
        self.opponent_type = "NN" if _nn_opponent is not None else "MCTSLite"
        self.move_history_dcn: list[str] = []
        self.illegal_attempts = 0
        self.legal_first_attempts = 0
        self.total_llm_turns = 0
        self._seed = seed
        random.seed(seed)

    def get_legal_moves_dcn(self) -> list[str]:
        """Get legal moves as DCN strings."""
        moves = generate_legal_moves(self.state)
        return [move_to_dcn(self.state, m) for m in moves]

    def try_llm_move(self, dcn_str: str) -> tuple[bool, str]:
        """Parse and apply an LLM's move. Returns (success, error_msg)."""
        dcn_str = dcn_str.strip()
        try:
            move = dcn_to_move(dcn_str)
        except (ValueError, IndexError) as e:
            return False, f"Cannot parse '{dcn_str}': {e}"

        # Validate against legal moves
        legal_moves = generate_legal_moves(self.state)
        matched = None
        for lm in legal_moves:
            if move_to_dcn(self.state, lm) == dcn_str:
                matched = lm
                break

        if matched is None:
            # Try case-insensitive match
            for lm in legal_moves:
                if move_to_dcn(self.state, lm).lower() == dcn_str.lower():
                    matched = lm
                    break

        if matched is None:
            legal_dcn = [move_to_dcn(self.state, m) for m in legal_moves[:15]]
            return False, f"'{dcn_str}' is not a legal move. Legal moves include: {legal_dcn}"

        apply_move(self.state, matched)
        self.move_history_dcn.append(dcn_str)
        return True, ""

    def play_opponent_move(self) -> str:
        """Have opponent (NN or MCTSLite) play a move. Returns DCN string."""
        move = self.opponent.search(self.state)
        dcn = move_to_dcn(self.state, move)
        apply_move(self.state, move)
        self.move_history_dcn.append(dcn)
        return dcn

    def get_board_display(self) -> str:
        """Render board for display."""
        display = self.state.to_display_board()
        return render_board(
            display,
            resource_counts=tuple(self.state.resources),
            turn=self.state.turn,
            current_player=int(self.state.current_player),
        )

    @property
    def is_over(self) -> bool:
        return self.state.done or self.state.turn > MAX_GAME_TURNS

    @property
    def result_str(self) -> str:
        if not self.state.done:
            return "draw"  # Hit turn limit
        if self.state.winner is None:
            return "draw"
        return "white" if self.state.winner == Player.WHITE else "black"


# %%
# === Single Game Task ===

def _extract_move_from_response(response_text: str, legal_dcn: list[str]) -> str | None:
    """Extract a DCN move from LLM response text using multiple strategies.

    Returns the matched legal DCN string, or None if no match found.
    """
    text = str(response_text).strip()

    # Strategy 1: exact match against legal moves
    for dcn in legal_dcn:
        if dcn in text:
            return dcn

    # Strategy 2: case-insensitive match
    text_lower = text.lower()
    for dcn in legal_dcn:
        if dcn.lower() in text_lower:
            return dcn

    # Strategy 3: regex for DCN patterns
    # Move: Wd2-d3, Rb1xd3
    # Promote: Wa1>R
    # Bombard: Bc3~e3
    patterns = [
        r'[CWRBL][a-h][1-8][-x][a-h][1-8]',  # Move/capture
        r'[CWRBL][a-h][1-8]>[RBL]',            # Promote
        r'B[a-h][1-8]~[a-h][1-8]',             # Bombard
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group()
            # Uppercase the piece char
            candidate = candidate[0].upper() + candidate[1:]
            # Check against legal moves (case-insensitive)
            for dcn in legal_dcn:
                if dcn.lower() == candidate.lower():
                    return dcn

    return None


@kbench.task(name="DaveChess single game", store_task=False, store_run=False)
def play_single_game(llm, mcts_sims: int, system_prompt: str,
                     llm_is_white: bool, game_seed: int,
                     can_use_memory: bool = False) -> dict:
    """Play one game of DaveChess: LLM vs MCTSLite opponent.

    Returns dict with game result and statistics.
    """
    game = DaveChessGame(mcts_sims=mcts_sims, seed=game_seed)
    llm_color = Player.WHITE if llm_is_white else Player.BLACK
    color_name = "White" if llm_is_white else "Black"
    forfeited = False

    while not game.is_over:
        if _tracker.exceeded:
            print(f"[TOKENS] Budget exceeded during game, forfeiting.")
            forfeited = True
            break

        legal_moves = generate_legal_moves(game.state)
        if not legal_moves:
            break

        if game.state.current_player == llm_color:
            # LLM's turn
            game.total_llm_turns += 1
            print(f"  [MOVE] Turn {game.state.turn}, LLM thinking...", end="", flush=True)
            _turn_start = time.time()
            legal_dcn = game.get_legal_moves_dcn()

            state_msg = build_game_state_message(
                game.state, game.move_history_dcn, legal_moves
            )

            # Build prompt — keep it focused, put move instruction first
            move_prompt = (
                f"You are playing DaveChess as {color_name}.\n"
                f"Token budget: {_tracker.remaining:,} remaining of {_tracker.budget:,} total.\n\n"
                f"{state_msg}\n\n"
                f"Reply with ONLY your chosen move in DCN notation "
                f"(e.g., {legal_dcn[0]}). Nothing else."
            )

            with kbench.chats.new(f"Turn {game.state.turn}"):
                success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        # Don't mix schema + tools — just get raw text
                        _call_start = time.time()
                        raw_response = llm.prompt(move_prompt)
                        _call_time = time.time() - _call_start
                        response_text = str(raw_response)
                        _tracker.add(move_prompt, response_text)
                    except Exception as e:
                        print(f"[ERROR] LLM prompt failed (turn {game.state.turn}, attempt {attempt}): {type(e).__name__}: {str(e)[:200]}")
                        _tracker.add(move_prompt, "")
                        game.illegal_attempts += 1
                        continue

                    # Extract move from response
                    matched_dcn = _extract_move_from_response(
                        response_text, legal_dcn
                    )

                    if matched_dcn is not None:
                        ok, err = game.try_llm_move(matched_dcn)
                        if ok:
                            if attempt == 0:
                                game.legal_first_attempts += 1
                            _turn_time = time.time() - _turn_start
                            print(f" {matched_dcn} ({_call_time:.1f}s call, {_turn_time:.1f}s total)", flush=True)
                            success = True
                            break
                        else:
                            game.illegal_attempts += 1
                    else:
                        game.illegal_attempts += 1

                    # Retry with clearer prompt
                    move_prompt = (
                        f"That was not a valid move. "
                        f"Choose ONE move from this list and reply with ONLY that move:\n"
                        f"{', '.join(legal_dcn[:20])}\n"
                        f"Reply with just the move, nothing else."
                    )

                if not success:
                    forfeited = True
                    break
        else:
            # MCTSLite opponent's turn
            try:
                game.play_opponent_move()
            except ValueError:
                break  # No legal moves for opponent

    # Determine result
    if forfeited:
        result = "forfeit"
        llm_won = False
    elif game.result_str == "draw":
        result = "draw"
        llm_won = False
    elif (game.result_str == "white" and llm_is_white) or \
         (game.result_str == "black" and not llm_is_white):
        result = "win"
        llm_won = True
    else:
        result = "loss"
        llm_won = False

    legal_rate = (game.legal_first_attempts / game.total_llm_turns
                  if game.total_llm_turns > 0 else 0.0)

    return {
        "result": result,
        "llm_won": llm_won,
        "llm_color": color_name,
        "game_length": game.state.turn,
        "total_moves": len(game.move_history_dcn),
        "illegal_attempts": game.illegal_attempts,
        "legal_move_rate": legal_rate,
        "forfeited": forfeited,
        "move_history": game.move_history_dcn,
    }


# %%
# === Phase A: Baseline (rules only, no study, no tools) ===

def play_phase_a(llm, n_games: int = PHASE_A_GAMES) -> dict:
    """Play baseline games with only rules knowledge. No examples, no tools."""
    print(f"\n{'='*60}")
    print(f"PHASE A: Baseline ({n_games} games, rules only)")
    print(f"{'='*60}")
    _tracker.log("Phase A start")

    rules_prompt = get_rules_prompt()
    system_prompt = (rules_prompt +
                     "\n\nPick the best move from the legal moves list. "
                     "Respond with the move in DCN notation.")

    results = []
    for i in range(n_games):
        if _tracker.exceeded:
            print(f"[TOKENS] Budget exceeded, skipping remaining Phase A games")
            break
        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=1000 + i,
            can_use_memory=False,
        )
        r = run.result
        results.append(r)
        print(f"[GAME] Phase A game {i+1}/{n_games}: {r['result']} ({r['llm_color']}, {r['game_length']} turns, {r['illegal_attempts']} illegal)")
        _tracker.log(f"Phase A game {i+1}")

    agg = _aggregate_results(results, "Phase A (Baseline)")
    print(f"[PHASE A] Win rate: {agg['win_rate']:.0%} | ELO: {agg['elo_estimate']} | Legal move rate: {agg['legal_move_rate']:.0%}", flush=True)
    return agg


# %%
# === Study Phase: Agent studies GM games with tools ===

def study_phase(llm, budget: int = STUDY_BUDGET):
    """Give agent time to study GM games and build knowledge structures.

    The agent receives batches of GM games and is asked to analyze them
    and produce strategy notes. The notes are saved to the workspace
    so the agent can consult them during play.

    This approach works with any LLM API (no tool calling required).
    The agent's analysis quality determines how well it plays later.
    """
    print(f"\n{'='*60}")
    print(f"STUDY PHASE: Analyzing GM games ({budget} batches)")
    print(f"{'='*60}")
    _tracker.log("Study start")

    rules = get_rules_prompt()

    # Feed GM games in batches and ask for analysis
    games_per_batch = 10
    total_games = count_gm_games()
    n_batches = min(budget, total_games // games_per_batch)

    all_notes = []

    for batch_idx in range(n_batches):
        if _tracker.exceeded:
            print(f"[TOKENS] Budget exceeded, stopping study at batch {batch_idx}")
            break
        start = batch_idx * games_per_batch + 1
        games_text = get_gm_games_batch(start, games_per_batch)

        if batch_idx == 0:
            # First batch: include rules and full instructions
            study_prompt = f"""{rules}

TOKEN BUDGET: You have {_tracker.remaining:,} tokens remaining out of {_tracker.budget:,} total.
This budget covers studying, playing baseline games, post-study games, and experience games.
Be efficient — focus on extracting the most useful strategic insights per token spent.

Here are expert-level DaveChess games (batch {batch_idx + 1}/{n_batches}).
Study them carefully and write detailed strategic notes.

{games_text}

Analyze these games. Write notes covering:
1. Common opening moves and strategies
2. How players use Gold nodes for resource accumulation
3. Promotion timing and piece choices (Rider vs Bombard vs Lancer)
4. Tactical patterns (captures, forks, checkmate threats)
5. Winning strategies you observe

Write your analysis as concise, actionable strategy notes that will help you play well."""
        else:
            # Subsequent batches: build on previous notes
            prev_notes = "\n".join(all_notes[-2:])  # Last 2 analyses
            study_prompt = f"""Here are more expert DaveChess games (batch {batch_idx + 1}/{n_batches}).

Your previous notes:
{prev_notes}

New games to study:
{games_text}

Update and refine your strategic notes based on these new games.
Focus on patterns you see across multiple games. Be concise and actionable."""

        with kbench.chats.new(f"Study turn {batch_idx + 1}/{n_batches}"):
            try:
                response = llm.prompt(study_prompt)
                notes = str(response)
                all_notes.append(notes)
                write_file(f"study_notes_batch_{batch_idx + 1}.md", notes)
                _tracker.add(study_prompt, notes)
                print(f"[STUDY] Batch {batch_idx + 1}/{n_batches}: {len(notes)} chars of notes")
                _tracker.log(f"Study batch {batch_idx + 1}")
            except Exception as e:
                print(f"[ERROR] Study batch {batch_idx + 1} failed: {type(e).__name__}: {str(e)[:200]}")
                continue

    # Create a consolidated strategy document
    if all_notes:
        summary_prompt = (
            "You've studied multiple batches of expert DaveChess games. "
            "Here are your batch notes:\n\n" +
            "\n---\n".join(all_notes) +
            "\n\nWrite a FINAL consolidated strategy guide. "
            "Include: opening principles, mid-game tactics, resource/promotion strategy, "
            "and endgame patterns. Be concise — this will be your reference during play."
        )
        with kbench.chats.new("Study consolidation"):
            try:
                final_notes = str(llm.prompt(summary_prompt))
                write_file("strategy.md", final_notes)
            except Exception:
                # Save whatever we have
                write_file("strategy.md", "\n---\n".join(all_notes))


# %%
# === Phase B: Post-study (can consult memory) ===

def _load_strategy_notes() -> str:
    """Load the agent's strategy notes from workspace."""
    notes = read_file("strategy.md")
    if "not found" in notes.lower():
        # Try batch notes
        all_notes = []
        for i in range(1, 30):
            n = read_file(f"study_notes_batch_{i}.md")
            if "not found" not in n.lower():
                all_notes.append(n)
            else:
                break
        notes = "\n---\n".join(all_notes) if all_notes else ""
    return notes


def play_phase_b(llm, n_games: int = PHASE_B_GAMES) -> dict:
    """Play games after studying. Strategy notes included in prompt."""
    print(f"\n{'='*60}")
    print(f"PHASE B: Post-study play ({n_games} games)")
    print(f"{'='*60}")
    _tracker.log("Phase B start")

    rules_prompt = get_rules_prompt()
    strategy_notes = _load_strategy_notes()
    print(f"[PHASE B] Strategy notes: {len(strategy_notes)} chars")

    system_prompt = rules_prompt
    if strategy_notes:
        # Truncate notes to avoid context overflow
        max_notes = 3000
        if len(strategy_notes) > max_notes:
            strategy_notes = strategy_notes[:max_notes] + "\n...(truncated)"
        system_prompt += f"\n\n# Your Strategy Notes\n{strategy_notes}"

    results = []
    for i in range(n_games):
        if _tracker.exceeded:
            print(f"[TOKENS] Budget exceeded, skipping remaining Phase B games")
            break
        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=2000 + i,
            can_use_memory=False,
        )
        r = run.result
        results.append(r)
        print(f"[GAME] Phase B game {i+1}/{n_games}: {r['result']} ({r['llm_color']}, {r['game_length']} turns, {r['illegal_attempts']} illegal)")
        _tracker.log(f"Phase B game {i+1}")

    agg = _aggregate_results(results, "Phase B (Post-study)")
    print(f"[PHASE B] Win rate: {agg['win_rate']:.0%} | ELO: {agg['elo_estimate']} | Legal move rate: {agg['legal_move_rate']:.0%}", flush=True)
    return agg


# %%
# === Phase C: Experience learning (play + reflect loop) ===

def play_phase_c(llm, n_games: int = PHASE_C_GAMES) -> dict:
    """Play games with reflection between each game.

    After each game, the agent reflects and updates its strategy notes.
    Updated notes are included in the prompt for subsequent games.
    Measures whether the agent improves through experience.
    """
    print(f"\n{'='*60}")
    print(f"PHASE C: Experience learning ({n_games} games + reflection)")
    print(f"{'='*60}")
    _tracker.log("Phase C start")

    rules_prompt = get_rules_prompt()
    results = []
    experience_notes = _load_strategy_notes()

    for i in range(n_games):
        if _tracker.exceeded:
            print(f"[TOKENS] Budget exceeded, skipping remaining Phase C games")
            break
        # Build system prompt with current strategy + experience
        system_prompt = rules_prompt
        if experience_notes:
            notes_truncated = experience_notes[:3000]
            if len(experience_notes) > 3000:
                notes_truncated += "\n...(truncated)"
            system_prompt += f"\n\n# Your Strategy Notes\n{notes_truncated}"

        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=3000 + i,
            can_use_memory=False,
        )
        game_result = run.result
        results.append(game_result)
        print(f"[GAME] Phase C game {i+1}/{n_games}: {game_result['result']} ({game_result['llm_color']}, {game_result['game_length']} turns, {game_result['illegal_attempts']} illegal)")
        _tracker.log(f"Phase C game {i+1}")

        # Reflection phase: ask LLM to update strategy based on game
        if i < n_games - 1 and not _tracker.exceeded:
            result_word = game_result["result"]
            moves_str = ", ".join(game_result["move_history"][:20])
            reflection_prompt = (
                f"You just played DaveChess game {i + 1}.\n"
                f"Result: {result_word}. Played as: {game_result['llm_color']}. "
                f"Length: {game_result['game_length']} turns. "
                f"Illegal attempts: {game_result['illegal_attempts']}.\n"
                f"Moves: {moves_str}{'...' if len(game_result['move_history']) > 20 else ''}\n\n"
                f"Your current strategy notes:\n{experience_notes[:2000]}\n\n"
                f"Based on this game, write UPDATED strategy notes. "
                f"What worked? What failed? What will you do differently? "
                f"Be concise and actionable."
            )
            with kbench.chats.new(f"Reflection after game {i + 1}"):
                try:
                    updated = str(llm.prompt(reflection_prompt))
                    experience_notes = updated
                    write_file("strategy.md", updated)
                    _tracker.add(reflection_prompt, updated)
                    print(f"[REFLECT] After game {i+1}: updated strategy ({len(updated)} chars)")
                except Exception:
                    pass  # Reflection is optional

    agg = _aggregate_results(results, "Phase C (Experience)")
    print(f"[PHASE C] Win rate: {agg['win_rate']:.0%} | ELO: {agg['elo_estimate']}", flush=True)
    _tracker.log("Phase C done")
    return agg


# %%
# === Scoring Helpers ===

def _aggregate_results(results: list[dict], phase_name: str) -> dict:
    """Aggregate game results into phase statistics."""
    wins = sum(1 for r in results if r["result"] == "win")
    losses = sum(1 for r in results if r["result"] == "loss")
    draws = sum(1 for r in results if r["result"] == "draw")
    forfeits = sum(1 for r in results if r["result"] == "forfeit")
    total = len(results)

    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    legal_rates = [r["legal_move_rate"] for r in results if r["legal_move_rate"] > 0]
    avg_legal_rate = sum(legal_rates) / len(legal_rates) if legal_rates else 0.0

    # Estimate ELO relative to opponent (NN-10 = 1000)
    import math
    OPPONENT_ELO = 1000
    if win_rate >= 1.0:
        elo_estimate = OPPONENT_ELO + 400
    elif win_rate <= 0.0:
        elo_estimate = OPPONENT_ELO - 400
    else:
        elo_estimate = OPPONENT_ELO - 400 * math.log10(1.0 / win_rate - 1.0)

    return {
        "phase": phase_name,
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "forfeits": forfeits,
        "win_rate": win_rate,
        "elo_estimate": round(elo_estimate),
        "legal_move_rate": avg_legal_rate,
        "results": results,
    }


def compute_learning_score(phase_a: dict, phase_b: dict, phase_c: dict) -> float:
    """Compute composite learning score (0.0 - 1.0).

    Weights:
    - 35%: Study learning (Phase B - Phase A win rate delta)
    - 30%: Experience learning (Phase C 2nd half - 1st half improvement)
    - 25%: Final performance (Phase C last half win rate)
    - 10%: Rule comprehension (Phase A legal move rate)
    """
    # Study delta: did studying examples help?
    study_delta = max(0.0, phase_b["win_rate"] - phase_a["win_rate"])

    # Experience delta: did the agent improve across Phase C games?
    c_results = phase_c["results"]
    mid = len(c_results) // 2
    first_half = c_results[:mid]
    second_half = c_results[mid:]

    first_wr = sum(1 for r in first_half if r["result"] == "win") / len(first_half) if first_half else 0
    second_wr = sum(1 for r in second_half if r["result"] == "win") / len(second_half) if second_half else 0
    exp_delta = max(0.0, second_wr - first_wr)

    # Final performance
    final_perf = second_wr

    # Rule comprehension
    legal_rate = phase_a["legal_move_rate"]

    score = (
        0.35 * study_delta +
        0.30 * exp_delta +
        0.25 * final_perf +
        0.10 * legal_rate
    )

    return min(1.0, score)


# %%
# === Main Benchmark Task ===

@kbench.task(name="DaveChess Learning Benchmark")
def davechess_learning_benchmark(llm) -> float:
    """Measure an AI system's ability to learn a novel strategic game.

    DaveChess is a custom board game absent from all training data.
    This benchmark tests three dimensions of learning:

    Phase A - Baseline: Play with only rule knowledge (no examples)
    Study:   Analyze 200 expert games using tools (files, Python, etc.)
    Phase B - Post-study: Play after studying (can consult notes)
    Phase C - Experience: Play + reflect loop (iterative improvement)

    The agent has access to general-purpose tools and can build any
    memory structures it finds useful. Better learners = higher scores.

    Returns a learning score from 0.0 to 1.0.
    """
    # Clean workspace for this run
    if os.path.isdir(AGENT_WORKSPACE):
        shutil.rmtree(AGENT_WORKSPACE)
    os.makedirs(AGENT_WORKSPACE, exist_ok=True)

    # Phase A: Baseline (rules only)
    phase_a = play_phase_a(llm)

    # Study period: agent uses tools to learn from GM games
    study_phase(llm, budget=STUDY_BUDGET)

    # Phase B: Post-study play (can consult memory)
    phase_b = play_phase_b(llm)

    # Phase C: Experience learning (play + reflect + improve)
    phase_c = play_phase_c(llm)

    # Compute learning score
    learning_score = compute_learning_score(phase_a, phase_b, phase_c)

    # === Assertions for leaderboard ===

    kbench.assertions.assert_true(
        phase_a["legal_move_rate"] > 0.3,
        expectation=(
            f"Rule comprehension: {phase_a['legal_move_rate']:.0%} legal moves on first attempt "
            f"(Phase A). Expect >30% to show basic rule understanding."
        ),
    )

    kbench.assertions.assert_true(
        phase_b["win_rate"] > phase_a["win_rate"],
        expectation=(
            f"Learning from study: Phase B win rate ({phase_b['win_rate']:.0%}) should exceed "
            f"Phase A baseline ({phase_a['win_rate']:.0%}). "
            f"Delta: {phase_b['win_rate'] - phase_a['win_rate']:+.0%}."
        ),
    )

    # Phase C improvement
    c_results = phase_c["results"]
    mid = len(c_results) // 2
    first_wr = sum(1 for r in c_results[:mid] if r["result"] == "win") / mid if mid > 0 else 0
    second_wr = sum(1 for r in c_results[mid:] if r["result"] == "win") / (len(c_results) - mid) if len(c_results) > mid else 0

    kbench.assertions.assert_true(
        second_wr >= first_wr,
        expectation=(
            f"Learning from experience: Phase C 2nd half win rate ({second_wr:.0%}) "
            f"should improve over 1st half ({first_wr:.0%}). "
            f"Delta: {second_wr - first_wr:+.0%}."
        ),
    )

    kbench.assertions.assert_true(
        learning_score > 0.15,
        expectation=(
            f"Overall learning score: {learning_score:.3f}. "
            f"Expect >0.15 for meaningful learning. "
            f"Breakdown: study_delta={phase_b['win_rate'] - phase_a['win_rate']:.2f}, "
            f"exp_delta={second_wr - first_wr:.2f}, "
            f"final_perf={second_wr:.2f}, "
            f"legal_rate={phase_a['legal_move_rate']:.2f}."
        ),
    )

    return learning_score


# %%
# Run the benchmark
def _load_llm():
    """Try multiple approaches to get an LLM for the benchmark."""
    if not _kbench_available:
        print("[INFO] SDK not available, skipping.")
        return None

    # Approach 1: kbench.llm already configured (Benchmarks environment)
    llm = getattr(kbench, 'llm', None)
    if llm is not None:
        print("[INFO] Using kbench.llm (Benchmarks environment)")
        return llm

    # Approach 2: Set MODEL_PROXY env vars from KAGGLE_DATA_PROXY_TOKEN
    proxy_key = os.environ.get("KAGGLE_DATA_PROXY_TOKEN")
    if proxy_key and "MODEL_PROXY_API_KEY" not in os.environ:
        os.environ["MODEL_PROXY_API_KEY"] = proxy_key
        os.environ.setdefault("MODEL_PROXY_URL", "https://mp.kaggle.net/models/openapi")
        os.environ.setdefault("LLM_DEFAULT", "google/gemini-2.5-flash")
        print("[INFO] Set MODEL_PROXY from KAGGLE_DATA_PROXY_TOKEN")

    # Approach 3: Kaggle user secrets
    if "MODEL_PROXY_API_KEY" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("MODEL_PROXY_API_KEY")
            if key:
                os.environ["MODEL_PROXY_API_KEY"] = key
                os.environ.setdefault("MODEL_PROXY_URL", "https://mp.kaggle.net/models/openapi")
                os.environ.setdefault("LLM_DEFAULT", "google/gemini-2.5-flash")
                print("[INFO] Got MODEL_PROXY_API_KEY from Kaggle secrets")
        except Exception as e:
            print(f"[INFO] Kaggle secrets: {e}")

    # Try loading a model
    try:
        from kaggle_benchmarks.kaggle import models as kmodels
        llm = kmodels.load_model("google/gemini-2.5-flash")
        print("[INFO] Loaded google/gemini-2.5-flash")
        return llm
    except Exception as e:
        print(f"[INFO] Model loading failed: {e}")

    return None

_llm = _load_llm()
if _llm is not None:
    davechess_learning_benchmark.run(_llm)
else:
    print("[INFO] No LLM available. To run this benchmark:")
    print("  1. Create notebook at kaggle.com/benchmarks/tasks/new")
    print("  2. Or add MODEL_PROXY_API_KEY to Kaggle secrets")

# %%
# %choose davechess_learning_benchmark
