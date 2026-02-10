#!/usr/bin/env python3
"""Project ELO growth from W&B training data.

Pulls iteration history from the active W&B run, fits growth curves,
and projects when the model will reach target ELO milestones.

Usage:
    python scripts/project_elo.py              # Plot + project
    python scripts/project_elo.py --no-plot    # Text-only projection
    python scripts/project_elo.py --target 1500  # Custom target ELO
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

try:
    import wandb
except ImportError:
    print("wandb not installed: pip install wandb")
    sys.exit(1)

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Cache file for offline use
CACHE_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'checkpoints', 'elo_history.json'
)


def fetch_history(run_name=None):
    """Fetch ELO history from W&B."""
    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f'{entity}/davechess', per_page=10)

    target_run = None
    for r in runs:
        if run_name and r.name != run_name:
            continue
        if r.state == 'running' or run_name:
            target_run = r
            break

    if target_run is None:
        # Fall back to most recent run
        runs = api.runs(f'{entity}/davechess', per_page=1)
        target_run = next(iter(runs))

    print(f"Run: {target_run.name} (state={target_run.state})")
    print(f"Created: {target_run.created_at}")

    # Fetch history
    history = target_run.history(samples=5000, pandas=False)

    iterations = []
    elos = []
    policy_losses = []
    value_losses = []
    timestamps = []
    game_lengths = []
    draw_rates = []

    for row in history:
        it = row.get('iteration')
        elo = row.get('elo_estimate') or row.get('elo/estimate')
        pl = row.get('train/policy_loss') or row.get('iteration/avg_policy_loss')
        vl = row.get('train/value_loss') or row.get('iteration/avg_value_loss')
        ts = row.get('_timestamp')
        gl = row.get('selfplay/avg_game_length')
        dr = row.get('selfplay/draw_rate')

        if it is not None and elo is not None:
            iterations.append(int(it))
            elos.append(float(elo))
            if pl is not None:
                policy_losses.append(float(pl))
            if vl is not None:
                value_losses.append(float(vl))
            if ts is not None:
                timestamps.append(float(ts))
            if gl is not None:
                game_lengths.append(float(gl))
            if dr is not None:
                draw_rates.append(float(dr))

    data = {
        'run_name': target_run.name,
        'run_state': target_run.state,
        'created_at': str(target_run.created_at),
        'fetched_at': datetime.now().isoformat(),
        'iterations': iterations,
        'elos': elos,
        'policy_losses': policy_losses[:len(iterations)],
        'value_losses': value_losses[:len(iterations)],
        'timestamps': timestamps[:len(iterations)],
        'game_lengths': game_lengths[:len(iterations)],
        'draw_rates': draw_rates[:len(iterations)],
    }

    # Cache locally
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Cached {len(iterations)} data points to {CACHE_FILE}")

    return data


def load_cached():
    """Load cached history."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


# --- Curve models ---

def log_model(x, a, b):
    """ELO = a * ln(x + 1) + b"""
    return a * np.log(x + 1) + b


def power_model(x, a, b, c):
    """ELO = a * x^b + c"""
    return a * np.power(x + 1, b) + c


def sqrt_model(x, a, b):
    """ELO = a * sqrt(x) + b"""
    return a * np.sqrt(x) + b


def fit_and_project(iterations, elos, targets):
    """Fit curves and project when targets will be reached."""
    if not HAS_SCIPY:
        print("scipy not installed - can't fit curves. pip install scipy")
        return {}

    iters = np.array(iterations, dtype=float)
    elo_arr = np.array(elos, dtype=float)

    if len(iters) < 3:
        print(f"Only {len(iters)} data points - need at least 3 for curve fitting")
        return {}

    results = {}

    # Try each model
    models = [
        ('logarithmic', log_model, [500, 0], {}),
        ('sqrt', sqrt_model, [100, 0], {}),
        ('power', power_model, [10, 0.5, 0], {'maxfev': 5000}),
    ]

    for name, func, p0, kwargs in models:
        try:
            popt, pcov = curve_fit(func, iters, elo_arr, p0=p0, **kwargs)
            residuals = elo_arr - func(iters, *popt)
            rmse = np.sqrt(np.mean(residuals**2))
            r_squared = 1 - np.sum(residuals**2) / np.sum((elo_arr - np.mean(elo_arr))**2)

            projections = {}
            for target in targets:
                # Binary search for iteration where ELO reaches target
                lo, hi = 0, 100000
                for _ in range(50):
                    mid = (lo + hi) / 2
                    if func(mid, *popt) < target:
                        lo = mid
                    else:
                        hi = mid
                pred_iter = int((lo + hi) / 2)
                if pred_iter < 100000:
                    projections[target] = pred_iter
                else:
                    projections[target] = None

            results[name] = {
                'params': popt.tolist(),
                'rmse': rmse,
                'r_squared': r_squared,
                'projections': projections,
                'func': func,
            }
        except (RuntimeError, ValueError) as e:
            print(f"  {name}: fit failed ({e})")

    return results


def estimate_time(data, target_iter):
    """Estimate wall time to reach target iteration."""
    if not data['timestamps'] or len(data['timestamps']) < 2:
        return None

    ts = np.array(data['timestamps'])
    iters = np.array(data['iterations'][:len(ts)])

    # Use recent iterations for time estimate (pace may change)
    n_recent = min(5, len(ts))
    if n_recent < 2:
        return None

    recent_ts = ts[-n_recent:]
    recent_iters = iters[-n_recent:]

    time_per_iter = (recent_ts[-1] - recent_ts[0]) / (recent_iters[-1] - recent_iters[0])
    current_iter = iters[-1]
    remaining_iters = target_iter - current_iter

    if remaining_iters <= 0:
        return timedelta(0)

    remaining_sec = remaining_iters * time_per_iter
    return timedelta(seconds=remaining_sec)


def print_report(data, results, targets):
    """Print projection report."""
    iters = data['iterations']
    elos = data['elos']

    print(f"\n{'='*60}")
    print(f"ELO Projection Report - {data['run_name']}")
    print(f"{'='*60}")
    print(f"Data points: {len(iters)}")
    print(f"Iterations: {min(iters)} - {max(iters)}")
    print(f"ELO range: {min(elos):.0f} - {max(elos):.0f}")
    print(f"Current ELO: {elos[-1]:.0f} (iter {iters[-1]})")

    if data['policy_losses']:
        print(f"Policy loss: {data['policy_losses'][-1]:.3f}")
    if data['value_losses']:
        print(f"Value loss: {data['value_losses'][-1]:.4f}")
    if data['game_lengths']:
        print(f"Avg game length: {data['game_lengths'][-1]:.0f}")

    if not results:
        return

    # Find best model by R²
    best_name = max(results, key=lambda k: results[k]['r_squared'])
    best = results[best_name]

    print(f"\n--- Model Fits ---")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['r_squared']):
        marker = " <-- best" if name == best_name else ""
        print(f"  {name:15s}  R²={r['r_squared']:.4f}  RMSE={r['rmse']:.1f}{marker}")

    print(f"\n--- Projections (using {best_name}, R²={best['r_squared']:.4f}) ---")
    current_iter = iters[-1]
    current_elo = elos[-1]

    for target in targets:
        if target <= current_elo:
            print(f"  ELO {target:5d}: already reached!")
            continue

        pred_iter = best['projections'].get(target)
        if pred_iter is None:
            print(f"  ELO {target:5d}: beyond projection range")
            continue

        iters_remaining = pred_iter - current_iter
        eta = estimate_time(data, pred_iter)
        eta_str = ""
        if eta:
            hours = eta.total_seconds() / 3600
            if hours < 1:
                eta_str = f" (~{eta.total_seconds()/60:.0f} min)"
            elif hours < 48:
                eta_str = f" (~{hours:.1f} hrs)"
            else:
                eta_str = f" (~{hours/24:.1f} days)"

        print(f"  ELO {target:5d}: iter ~{pred_iter:,d} ({iters_remaining:,d} remaining){eta_str}")

    # Also show all models' projections for comparison
    if len(results) > 1:
        print(f"\n--- All Model Projections (iteration to reach target) ---")
        header = f"  {'Target':>7s}"
        for name in sorted(results.keys()):
            header += f"  {name:>15s}"
        print(header)
        for target in targets:
            if target <= current_elo:
                continue
            row = f"  {target:>7d}"
            for name in sorted(results.keys()):
                pred = results[name]['projections'].get(target)
                if pred:
                    row += f"  {pred:>15,d}"
                else:
                    row += f"  {'---':>15s}"
            print(row)


def plot_projection(data, results, targets, output_path=None):
    """Plot ELO trajectory and projections."""
    if not HAS_MPL:
        print("matplotlib not installed - skipping plot. pip install matplotlib")
        return

    iters = np.array(data['iterations'], dtype=float)
    elos = np.array(data['elos'], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Projection - {data['run_name']}", fontsize=14, fontweight='bold')

    # --- Plot 1: ELO vs Iteration with projections ---
    ax = axes[0][0]
    ax.scatter(iters, elos, c='#22d3ee', s=20, zorder=5, label='Actual')

    if results:
        best_name = max(results, key=lambda k: results[k]['r_squared'])
        max_proj_iter = max(iters[-1] * 2, 200)
        for target in targets:
            pred = results[best_name]['projections'].get(target)
            if pred and pred > iters[-1]:
                max_proj_iter = max(max_proj_iter, pred * 1.1)

        x_proj = np.linspace(0, min(max_proj_iter, 5000), 500)
        colors = {'logarithmic': '#f87171', 'sqrt': '#34d399', 'power': '#fbbf24'}
        for name, r in results.items():
            y_proj = r['func'](x_proj, *r['params'])
            style = '-' if name == best_name else '--'
            alpha = 1.0 if name == best_name else 0.5
            ax.plot(x_proj, y_proj, style, color=colors.get(name, '#888'),
                    alpha=alpha, label=f'{name} (R²={r["r_squared"]:.3f})')

        # Target lines
        for target in targets:
            ax.axhline(y=target, color='#50546e', linestyle=':', alpha=0.5)
            ax.text(ax.get_xlim()[1] * 0.98, target + 10, f'{target}',
                    color='#7c819a', fontsize=8, ha='right')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELO')
    ax.set_title('ELO vs Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- Plot 2: Policy Loss vs Iteration ---
    ax = axes[0][1]
    if data['policy_losses']:
        pl_iters = iters[:len(data['policy_losses'])]
        ax.plot(pl_iters, data['policy_losses'], color='#a78bfa', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss vs Iteration')
        ax.grid(True, alpha=0.2)
        # Add random baseline
        ax.axhline(y=3.9, color='#f87171', linestyle=':', alpha=0.5, label='Random (~ln50)')
        ax.legend(fontsize=8)

    # --- Plot 3: ELO vs Policy Loss ---
    ax = axes[1][0]
    if data['policy_losses']:
        n = min(len(elos), len(data['policy_losses']))
        ax.scatter(data['policy_losses'][:n], elos[:n], c='#22d3ee', s=20)
        ax.set_xlabel('Policy Loss')
        ax.set_ylabel('ELO')
        ax.set_title('ELO vs Policy Loss')
        ax.grid(True, alpha=0.2)
        ax.invert_xaxis()

    # --- Plot 4: Game Length & Draw Rate ---
    ax = axes[1][1]
    if data['game_lengths']:
        gl_iters = iters[:len(data['game_lengths'])]
        ax.plot(gl_iters, data['game_lengths'], color='#34d399', linewidth=1.5, label='Avg Game Length')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Game Length', color='#34d399')
        ax.set_title('Game Length & Draw Rate')
        ax.grid(True, alpha=0.2)
        ax.axhline(y=200, color='#f87171', linestyle=':', alpha=0.3, label='Max (turn 100)')

        if data['draw_rates']:
            ax2 = ax.twinx()
            dr_iters = iters[:len(data['draw_rates'])]
            ax2.plot(dr_iters, [d * 100 for d in data['draw_rates']],
                     color='#fbbf24', linewidth=1.5, alpha=0.7, label='Draw %')
            ax2.set_ylabel('Draw %', color='#fbbf24')
            ax2.set_ylim(0, 100)

        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', 'checkpoints', 'elo_projection.png'
        )
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#08090d', edgecolor='none')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Project ELO growth from W&B training')
    parser.add_argument('--run', type=str, help='W&B run name (default: latest running)')
    parser.add_argument('--target', type=int, nargs='+', default=[500, 1000, 1500, 2000],
                        help='Target ELO milestones')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--offline', action='store_true', help='Use cached data only')
    parser.add_argument('--output', type=str, help='Plot output path')
    args = parser.parse_args()

    if args.offline:
        data = load_cached()
        if data is None:
            print("No cached data found. Run without --offline first.")
            sys.exit(1)
        print(f"Using cached data from {data.get('fetched_at', '?')}")
    else:
        data = fetch_history(run_name=args.run)

    if not data['iterations']:
        print("No iteration data found with ELO estimates.")
        sys.exit(1)

    results = fit_and_project(data['iterations'], data['elos'], args.target)
    print_report(data, results, args.target)

    if not args.no_plot:
        plot_projection(data, results, args.target, output_path=args.output)


if __name__ == '__main__':
    main()
