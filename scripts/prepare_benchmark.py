#!/usr/bin/env python3
"""Prepare all benchmark data: download model, extract GM games, calibrate.

This is the single script to run on the Jetson (or any machine) to go from
a trained W&B model to a fully-prepared benchmark environment.

Pipeline:
    1. Find the highest-ELO model across all W&B runs
    2. Download best.pt to checkpoints/
    3. Download game logs from high-ELO runs, split into individual DCN files
    4. Run ELO calibration (round-robin tournament with NN-MCTS at varying sims)
    5. Print summary and readiness check

Usage:
    # Full pipeline (on Jetson with CUDA)
    python scripts/prepare_benchmark.py --all

    # Full pipeline on CPU (slower calibration)
    python scripts/prepare_benchmark.py --all --device cpu

    # Individual steps
    python scripts/prepare_benchmark.py --download-model
    python scripts/prepare_benchmark.py --download-games
    python scripts/prepare_benchmark.py --calibrate
    python scripts/prepare_benchmark.py --calibrate --games-per-pair 30
    python scripts/prepare_benchmark.py --check  # Just check readiness

    # Force re-download even if files exist
    python scripts/prepare_benchmark.py --all --force

    # Use a specific W&B run instead of auto-detecting best
    python scripts/prepare_benchmark.py --download-model --run-id xrjpggwn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_benchmark")

WANDB_ENTITY = "david-rimshnick-david-rimshnick"
WANDB_PROJECT = "davechess"
WANDB_FULL = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

CHECKPOINT_DIR = "checkpoints"
BEST_PT = os.path.join(CHECKPOINT_DIR, "best.pt")
CALIBRATION_JSON = os.path.join(CHECKPOINT_DIR, "calibration.json")
GAMES_DIR = "data/gm_games"


# ── Step 1: Find and download the best model ─────────────────────────


def find_latest_run(api, run_id: str | None = None):
    """Find the latest W&B run to use for the benchmark.

    Uses the most recently created run. Older runs used different rule sets
    and are NOT comparable.

    Args:
        api: W&B API object.
        run_id: Specific run ID to use (overrides auto-detection).

    Returns:
        W&B run object, or None on failure.
    """
    if run_id:
        run = api.run(f"{WANDB_FULL}/{run_id}")
        elo = run.summary.get("best_elo", 0)
        iteration = run.summary.get("iteration", 0)
        log.info(f"Using specified run: {run.name} (ELO {elo:.0f}, iter {iteration})")
        return run

    log.info(f"Finding latest run in {WANDB_FULL}...")
    runs = api.runs(WANDB_FULL, order="-created_at", per_page=5)

    for run in runs:
        elo = run.summary.get("best_elo", 0)
        iteration = run.summary.get("iteration", 0)
        log.info(f"  {run.name}: ELO {elo:.0f}, iter {iteration}, state={run.state}")

    # Most recently created run (first in the list)
    runs = api.runs(WANDB_FULL, order="-created_at", per_page=1)
    for run in runs:
        elo = run.summary.get("best_elo", 0)
        iteration = run.summary.get("iteration", 0)
        log.info(f"Selected (latest): {run.name} (ELO {elo:.0f}, iter {iteration}, state={run.state})")
        return run

    log.error("No runs found!")
    return None


def download_model(api, run, force: bool = False) -> bool:
    """Download the best (highest-iteration) model artifact from a W&B run.

    Args:
        api: W&B API object.
        run: The W&B run object.
        force: Re-download even if best.pt exists.

    Returns:
        True on success.
    """
    if os.path.exists(BEST_PT) and not force:
        size_mb = os.path.getsize(BEST_PT) / (1024 * 1024)
        log.info(f"Model already exists: {BEST_PT} ({size_mb:.1f} MB) — use --force to re-download")
        return True

    # Find all model artifacts and pick the highest iteration
    all_models = [a for a in run.logged_artifacts() if a.type == "model"]
    if not all_models:
        log.error(f"No model artifacts found in run {run.name}")
        return False

    # Sort by iteration number extracted from name (e.g., best-model-iter13)
    all_models.sort(key=lambda a: _parse_iter_from_name(a.name))
    best = all_models[-1]
    best_iter = _parse_iter_from_name(best.name)
    log.info(f"Best model in {run.name}: {best.name} (iter {best_iter}, "
             f"{best.size / 1024 / 1024:.1f} MB)")

    log.info(f"Downloading {best.name}...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best.download(root=CHECKPOINT_DIR)

    if os.path.exists(BEST_PT):
        size_mb = os.path.getsize(BEST_PT) / (1024 * 1024)
        log.info(f"Saved to {BEST_PT} ({size_mb:.1f} MB)")
        return True
    else:
        downloaded = os.listdir(CHECKPOINT_DIR)
        log.error(f"best.pt not found after download. Files in {CHECKPOINT_DIR}: {downloaded}")
        return False


# ── Step 2: Download and split GM game logs ───────────────────────────


def download_games(api, run, max_games: int = 200,
                   force: bool = False) -> bool:
    """Download game logs from a specific W&B run and split into individual DCN files.

    Only downloads from the specified run to avoid mixing games from different
    rule sets. Games from later iterations are prioritized (higher quality).

    Args:
        api: W&B API object.
        run: The W&B run to download games from.
        max_games: Maximum number of individual games to extract.
        force: Re-download even if games exist.

    Returns:
        True on success.
    """
    existing = []
    if os.path.isdir(GAMES_DIR):
        existing = [f for f in os.listdir(GAMES_DIR) if f.endswith(".dcn")]

    if existing and not force:
        log.info(f"GM games already exist: {len(existing)} files in {GAMES_DIR}/ — use --force to re-download")
        return True

    run_elo = run.summary.get("best_elo", 0)
    log.info(f"Downloading game logs from {run.name} (ELO {run_elo:.0f})...")

    game_logs = [a for a in run.logged_artifacts() if a.type == "game-log"]
    if not game_logs:
        log.error(f"No game log artifacts found in run {run.name}")
        return False

    # Collect artifacts with iteration numbers
    artifact_candidates = []
    for art in game_logs:
        iter_num = _parse_iter_from_name(art.name)
        artifact_candidates.append({
            "artifact": art,
            "run_name": run.name,
            "run_elo": run_elo,
            "iter_num": iter_num,
        })

    # Sort by iteration (desc) — later iterations have higher quality play
    artifact_candidates.sort(key=lambda x: x["iter_num"], reverse=True)

    log.info(f"Found {len(artifact_candidates)} game log artifacts, extracting up to {max_games} games...")

    os.makedirs(GAMES_DIR, exist_ok=True)

    # Clear existing games if forcing
    if force:
        for f in existing:
            os.remove(os.path.join(GAMES_DIR, f))

    game_count = 0
    decisive_count = 0
    artifacts_used = 0

    for candidate in artifact_candidates:
        if game_count >= max_games:
            break

        art = candidate["artifact"]
        try:
            # Download artifact to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                art.download(root=tmpdir)

                # Find the .dcn file in the download
                dcn_files = [f for f in os.listdir(tmpdir) if f.endswith(".dcn")]
                if not dcn_files:
                    continue

                for dcn_file in dcn_files:
                    dcn_path = os.path.join(tmpdir, dcn_file)
                    with open(dcn_path, "r") as f:
                        content = f.read()

                    # Split multi-game file into individual games
                    games = _split_dcn_games(content)
                    artifacts_used += 1

                    for game_text in games:
                        if game_count >= max_games:
                            break

                        game_text = game_text.strip()
                        if not game_text:
                            continue

                        # Check if decisive (not a draw)
                        is_decisive = game_text.rstrip().endswith(("1-0", "0-1"))

                        # Add source metadata
                        source_header = f'[Source "{candidate["run_name"]} iter {candidate["iter_num"]}"]'
                        elo_header = f'[ELO "{candidate["run_elo"]:.0f}"]'
                        if not game_text.startswith("["):
                            game_text = f"{source_header}\n{elo_header}\n\n{game_text}"
                        else:
                            # Insert after existing headers
                            lines = game_text.split("\n")
                            header_end = 0
                            for idx, line in enumerate(lines):
                                if line.startswith("["):
                                    header_end = idx + 1
                                elif line.strip() == "":
                                    break
                            lines.insert(header_end, elo_header)
                            lines.insert(header_end, source_header)
                            game_text = "\n".join(lines)

                        game_count += 1
                        if is_decisive:
                            decisive_count += 1

                        filepath = os.path.join(GAMES_DIR, f"gm_{game_count:04d}.dcn")
                        with open(filepath, "w") as f:
                            f.write(game_text)

        except Exception as e:
            log.warning(f"Failed to process {art.name}: {e}")
            continue

    log.info(f"Extracted {game_count} games ({decisive_count} decisive) from "
             f"{artifacts_used} artifacts into {GAMES_DIR}/")
    return game_count > 0


def _parse_iter_from_name(name: str) -> int:
    """Extract iteration number from artifact name like 'game-log-iter0013:v2'."""
    import re
    match = re.search(r"iter(\d+)", name)
    return int(match.group(1)) if match else 0


def _split_dcn_games(content: str) -> list[str]:
    """Split a multi-game DCN file into individual game texts.

    Games are separated by double newlines. A new game starts with a '[' header.
    """
    # First try splitting on triple newlines (storage.py convention)
    sections = content.split("\n\n\n")
    if len(sections) > 1:
        return [s for s in sections if s.strip()]

    # Fall back: split on double newline followed by '[' bracket
    parts = []
    current = []
    for line in content.split("\n"):
        if line.startswith("[") and current and not current[-1].strip():
            parts.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        parts.append("\n".join(current))

    return [p for p in parts if p.strip()]


# ── Step 3: Calibrate opponents ───────────────────────────────────────


def run_calibration(device: str = "cuda", games_per_pair: int = 20,
                    sim_counts: str | None = None,
                    force: bool = False) -> bool:
    """Run ELO calibration with the NN model.

    Args:
        device: 'cuda' or 'cpu'.
        games_per_pair: Games per pair in round-robin.
        sim_counts: Comma-separated sim counts, or None for defaults.
        force: Re-calibrate even if calibration.json exists.

    Returns:
        True on success.
    """
    if os.path.exists(CALIBRATION_JSON) and not force:
        with open(CALIBRATION_JSON) as f:
            data = json.load(f)
        n_levels = len(data.get("levels", []))
        log.info(f"Calibration already exists: {CALIBRATION_JSON} ({n_levels} levels) — use --force to re-calibrate")
        return True

    if not os.path.exists(BEST_PT):
        log.error(f"Model not found at {BEST_PT} — run --download-model first")
        return False

    # Build calibration command
    cmd = [
        sys.executable, "scripts/calibrate_opponents.py",
        "--checkpoint", BEST_PT,
        "--device", device,
        "--games-per-pair", str(games_per_pair),
        "--output", CALIBRATION_JSON,
        "--resume",  # Always enable resume for crash recovery
        "-v",
    ]
    if sim_counts:
        cmd.extend(["--sim-counts", sim_counts])

    log.info(f"Running calibration: {' '.join(cmd)}")
    log.info(f"This may take a while ({games_per_pair} games/pair, "
             f"{'GPU' if device == 'cuda' else 'CPU'} inference)...")

    import subprocess
    result = subprocess.run(cmd)

    if result.returncode != 0:
        log.error(f"Calibration failed with exit code {result.returncode}")
        return False

    if os.path.exists(CALIBRATION_JSON):
        with open(CALIBRATION_JSON) as f:
            data = json.load(f)
        log.info(f"Calibration saved: {len(data['levels'])} levels")
        for lvl in data["levels"]:
            log.info(f"  {lvl['sim_count']:>4d} sims -> ELO {lvl['elo']}")
        return True

    log.error("Calibration completed but no output file found")
    return False


# ── Step 4: Readiness check ───────────────────────────────────────────


def check_readiness():
    """Check if all benchmark prerequisites are met."""
    print("\n" + "=" * 60)
    print("  BENCHMARK READINESS CHECK")
    print("=" * 60)

    all_ready = True

    # Model
    if os.path.exists(BEST_PT):
        size_mb = os.path.getsize(BEST_PT) / (1024 * 1024)
        try:
            import torch
            ckpt = torch.load(BEST_PT, map_location="cpu", weights_only=False)
            elo = ckpt.get("elo_estimate", "?")
            iteration = ckpt.get("iteration", "?")
            print(f"  [OK] Model: {BEST_PT} ({size_mb:.1f} MB, ELO {elo}, iter {iteration})")
        except Exception:
            print(f"  [OK] Model: {BEST_PT} ({size_mb:.1f} MB)")
    else:
        print(f"  [!!] Model: {BEST_PT} NOT FOUND")
        all_ready = False

    # GM Games
    if os.path.isdir(GAMES_DIR):
        dcn_files = [f for f in os.listdir(GAMES_DIR) if f.endswith(".dcn")]
        if dcn_files:
            print(f"  [OK] GM Games: {len(dcn_files)} files in {GAMES_DIR}/")
        else:
            print(f"  [!!] GM Games: {GAMES_DIR}/ exists but is EMPTY")
            all_ready = False
    else:
        print(f"  [!!] GM Games: {GAMES_DIR}/ NOT FOUND")
        all_ready = False

    # Calibration
    if os.path.exists(CALIBRATION_JSON):
        with open(CALIBRATION_JSON) as f:
            data = json.load(f)
        levels = data.get("levels", [])
        elo_range = f"{levels[0]['elo']}-{levels[-1]['elo']}" if levels else "?"
        print(f"  [OK] Calibration: {len(levels)} levels, ELO range {elo_range}")
    else:
        print(f"  [!!] Calibration: {CALIBRATION_JSON} NOT FOUND")
        all_ready = False

    print()
    if all_ready:
        print("  ALL READY! You can run the benchmark:")
        print()
        print("  # Set up sandbox")
        print('  SANDBOX="/tmp/benchmark-sandbox"')
        print('  mkdir -p "$SANDBOX/scripts" "$SANDBOX/data" "$SANDBOX/checkpoints"')
        print(f'  cp scripts/agent_cli.py "$SANDBOX/scripts/"')
        print(f'  cp -r {GAMES_DIR} "$SANDBOX/data/"')
        print(f'  cp {BEST_PT} "$SANDBOX/checkpoints/"')
        print(f'  cp {CALIBRATION_JSON} "$SANDBOX/checkpoints/"')
        print()
        print("  # Launch agent")
        print('  cd "$SANDBOX" && claude -p "Read scripts/agent_cli.py..." \\')
        print(f"    --checkpoint checkpoints/best.pt \\")
        print(f"    --calibration checkpoints/calibration.json")
    else:
        print("  NOT READY — run missing steps above.")

    print("=" * 60 + "\n")
    return all_ready


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare benchmark: download model, games, calibrate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on Jetson
  python scripts/prepare_benchmark.py --all

  # Full pipeline on CPU
  python scripts/prepare_benchmark.py --all --device cpu

  # Just download, skip calibration
  python scripts/prepare_benchmark.py --download-model --download-games

  # Just check readiness
  python scripts/prepare_benchmark.py --check
""",
    )

    # Steps
    parser.add_argument("--all", action="store_true",
                        help="Run all steps: download model + games + calibrate")
    parser.add_argument("--download-model", action="store_true",
                        help="Download best model from W&B")
    parser.add_argument("--download-games", action="store_true",
                        help="Download GM game logs from W&B")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run ELO calibration")
    parser.add_argument("--check", action="store_true",
                        help="Check benchmark readiness (no downloads)")

    # Options
    parser.add_argument("--device", default="cuda",
                        help="Device for calibration: cuda or cpu (default: cuda)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download/re-calibrate even if files exist")
    parser.add_argument("--run-id", default=None,
                        help="Use a specific W&B run ID instead of auto-detecting")
    parser.add_argument("--max-games", type=int, default=200,
                        help="Max GM games to extract (default: 200)")
    parser.add_argument("--games-per-pair", type=int, default=20,
                        help="Games per pair for calibration (default: 20)")
    parser.add_argument("--sim-counts", default=None,
                        help="Comma-separated sim counts for calibration")

    args = parser.parse_args()

    if args.all:
        args.download_model = True
        args.download_games = True
        args.calibrate = True

    if not any([args.download_model, args.download_games, args.calibrate, args.check]):
        parser.print_help()
        return

    if args.check:
        check_readiness()
        return

    # Steps that need W&B
    if args.download_model or args.download_games:
        try:
            import wandb
        except ImportError:
            log.error("wandb not installed. Run: pip install wandb")
            sys.exit(1)

        api = wandb.Api()

        # Find the run once — model and games come from the same run
        run = find_latest_run(api, run_id=args.run_id)
        if run is None:
            sys.exit(1)

        if args.download_model:
            if not download_model(api, run, force=args.force):
                sys.exit(1)

        if args.download_games:
            if not download_games(api, run,
                                  max_games=args.max_games, force=args.force):
                sys.exit(1)

    if args.calibrate:
        if not run_calibration(device=args.device,
                               games_per_pair=args.games_per_pair,
                               sim_counts=args.sim_counts,
                               force=args.force):
            sys.exit(1)

    # Always show readiness at the end
    check_readiness()


if __name__ == "__main__":
    main()
