#!/usr/bin/env python3
"""Post a training progress update to the current W&B run notes."""
import sys
import json
import time

import wandb

ENTITY = "david-rimshnick-david-rimshnick"
PROJECT = "davechess"


def get_latest_run_id():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at")
    for run in runs:
        if run.state == "running":
            return run.id
    # Fall back to most recent
    return runs[0].id if runs else None


def post_update(message: str, run_id: str = None):
    api = wandb.Api()
    if run_id is None:
        run_id = get_latest_run_id()
    if run_id is None:
        print("No runs found")
        return

    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    existing = run.notes or ""
    run.notes = existing + f"\n\n---\n**[{timestamp}]** {message}"
    run.update()
    print(f"Posted update to run {run.name} ({run_id})")


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Status check"
    post_update(msg)
