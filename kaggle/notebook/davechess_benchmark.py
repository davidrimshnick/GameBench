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
from dataclasses import dataclass, field
from typing import Optional

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
    DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

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

# %%
# === Configuration ===
MCTS_SIMS = 10          # MCTSLite opponent strength (weak, beatable by learning agents)
MAX_GAME_TURNS = 80     # Cap game length (native draw at turn 100)
MAX_RETRIES = 3         # Illegal move retries before forfeit
STUDY_BUDGET = 20       # Max tool-use turns during study phase
PHASE_A_GAMES = 3       # Baseline games (no study)
PHASE_B_GAMES = 5       # Post-study games
PHASE_C_GAMES = 6       # Experience learning games

# Agent memory workspace
AGENT_WORKSPACE = "/kaggle/working/agent_memory"
if not os.path.isdir("/kaggle/working"):
    AGENT_WORKSPACE = tempfile.mkdtemp(prefix="davechess_agent_")
os.makedirs(AGENT_WORKSPACE, exist_ok=True)


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
    """Manages a single DaveChess game: LLM vs MCTSLite."""

    def __init__(self, mcts_sims: int = MCTS_SIMS, seed: int = 0):
        self.state = GameState()
        self.mcts = MCTSLite(num_simulations=mcts_sims)
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

    def play_mcts_move(self) -> str:
        """Have MCTSLite play a move. Returns DCN string."""
        move = self.mcts.search(self.state)
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
        legal_moves = generate_legal_moves(game.state)
        if not legal_moves:
            break

        if game.state.current_player == llm_color:
            # LLM's turn
            game.total_llm_turns += 1
            state_msg = build_game_state_message(
                game.state, game.move_history_dcn, legal_moves
            )

            turn_prompt = f"{system_prompt}\n\nYou are playing as {color_name}.\n\n{state_msg}"

            tools = MEMORY_TOOLS if can_use_memory else []

            with kbench.chats.new(f"Turn {game.state.turn}"):
                success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        if tools:
                            response = llm.prompt(turn_prompt, schema=DaveChessMove,
                                                  tools=tools)
                        else:
                            response = llm.prompt(turn_prompt, schema=DaveChessMove)
                    except Exception:
                        # Fallback: try to extract move from raw text
                        try:
                            raw = llm.prompt(turn_prompt)
                            dcn_match = re.search(r'[CWRBL][a-h][1-8][-x>~][a-hRBL1-8]+', str(raw))
                            if dcn_match:
                                response = DaveChessMove(move=dcn_match.group())
                            else:
                                game.illegal_attempts += 1
                                continue
                        except Exception:
                            game.illegal_attempts += 1
                            continue

                    ok, err = game.try_llm_move(response.move)
                    if ok:
                        if attempt == 0:
                            game.legal_first_attempts += 1
                        success = True
                        break
                    else:
                        game.illegal_attempts += 1
                        turn_prompt = (
                            f"Your move '{response.move}' was illegal: {err}\n"
                            f"Pick a legal move from the list. Respond with just the move."
                        )

                if not success:
                    forfeited = True
                    break
        else:
            # MCTSLite opponent's turn
            try:
                game.play_mcts_move()
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
    rules_prompt = get_rules_prompt()
    system_prompt = (rules_prompt +
                     "\n\nPick the best move from the legal moves list. "
                     "Respond with the move in DCN notation.")

    results = []
    for i in range(n_games):
        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=1000 + i,
            can_use_memory=False,
        )
        results.append(run.result)

    return _aggregate_results(results, "Phase A (Baseline)")


# %%
# === Study Phase: Agent studies GM games with tools ===

def study_phase(llm, budget: int = STUDY_BUDGET):
    """Give agent time to study GM games and build knowledge structures.

    The agent has access to:
    - 200 expert games (get_gm_game, get_gm_games_batch)
    - File I/O (write_file, read_file, list_files)
    - Python execution (run_python_script)

    The agent decides how to use these tools. It might:
    - Write strategy notes
    - Analyze games statistically
    - Build opening/pattern databases
    - Create Python analysis scripts
    - Or anything else
    """
    rules = get_rules_prompt()
    study_prompt = f"""{rules}

You have access to 200 expert-level DaveChess games and general-purpose tools.
Your goal: study these games and prepare to play well.

Available tools:
- get_gm_game(index): Get game 1-200 in DCN notation
- get_gm_games_batch(start, count): Get up to 20 games at once
- count_gm_games(): Returns 200
- write_file(path, content): Save notes/analysis to your workspace
- read_file(path): Read files from your workspace
- list_files(): See what's in your workspace
- run_python_script(code): Run Python analysis (davechess_engine is importable)

You have {budget} turns to study. Build whatever knowledge structures
you think will help you play better. When done, say "DONE STUDYING".

Study strategically: look for opening patterns, common tactics,
resource/promotion strategies, and winning endgame patterns."""

    for turn in range(budget):
        with kbench.chats.new(f"Study turn {turn + 1}/{budget}"):
            try:
                response = llm.prompt(study_prompt, tools=AGENT_TOOLS)
                response_text = str(response).lower()
                if "done studying" in response_text:
                    break
            except Exception:
                continue

            # Update prompt to acknowledge progress
            study_prompt = (
                f"Continue studying. You have {budget - turn - 1} turns remaining. "
                f"Use tools to analyze more games or refine your notes. "
                f"Say 'DONE STUDYING' when ready to play."
            )


# %%
# === Phase B: Post-study (can consult memory) ===

def play_phase_b(llm, n_games: int = PHASE_B_GAMES) -> dict:
    """Play games after studying. Agent can consult its notes."""
    rules_prompt = get_rules_prompt()
    system_prompt = (
        rules_prompt +
        "\n\nYou have studied expert games and may have notes in your workspace. "
        "You can use read_file() and list_files() to consult your notes. "
        "Pick the best move from the legal moves list."
    )

    results = []
    for i in range(n_games):
        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=2000 + i,
            can_use_memory=True,
        )
        results.append(run.result)

    return _aggregate_results(results, "Phase B (Post-study)")


# %%
# === Phase C: Experience learning (play + reflect loop) ===

def play_phase_c(llm, n_games: int = PHASE_C_GAMES) -> dict:
    """Play games with reflection between each game.

    After each game, the agent can update its notes/strategy.
    Measures whether the agent improves through experience.
    """
    rules_prompt = get_rules_prompt()
    results = []

    for i in range(n_games):
        # Play a game
        system_prompt = (
            rules_prompt +
            "\n\nYou have studied expert games and played previous games. "
            "Consult your notes (read_file, list_files) for strategy. "
            "Pick the best move from the legal moves list."
        )

        llm_is_white = (i % 2 == 0)
        run = play_single_game.run(
            llm=llm, mcts_sims=MCTS_SIMS, system_prompt=system_prompt,
            llm_is_white=llm_is_white, game_seed=3000 + i,
            can_use_memory=True,
        )
        game_result = run.result
        results.append(game_result)

        # Reflection phase: agent can update its strategy
        if i < n_games - 1:  # No need to reflect after the last game
            result_word = game_result["result"]
            moves_str = ", ".join(game_result["move_history"][:20])
            reflection_prompt = (
                f"Game {i + 1} result: {result_word}. "
                f"You played as {game_result['llm_color']}. "
                f"Game lasted {game_result['game_length']} turns. "
                f"Illegal move attempts: {game_result['illegal_attempts']}. "
                f"Moves: {moves_str}{'...' if len(game_result['move_history']) > 20 else ''}\n\n"
                f"Reflect on this game. Update your notes and strategy. "
                f"Use write_file() to revise your approach. "
                f"What patterns did you notice? What would you do differently?"
            )
            with kbench.chats.new(f"Reflection after game {i + 1}"):
                try:
                    llm.prompt(reflection_prompt, tools=AGENT_TOOLS)
                except Exception:
                    pass  # Reflection is optional

    return _aggregate_results(results, "Phase C (Experience)")


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

    return {
        "phase": phase_name,
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "forfeits": forfeits,
        "win_rate": win_rate,
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
