"""
Incremental checkpoint system for benchmark results.

Saves after each individual measurement so that crashed runs can resume
from exactly where they left off.
"""

import json
from pathlib import Path


CHECKPOINT_FILE = "checkpoint.json"


def _path(out_dir: Path) -> Path:
    return out_dir / CHECKPOINT_FILE


def load_checkpoint(out_dir: Path) -> dict:
    """Load existing checkpoint or return empty structure."""
    p = _path(out_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_checkpoint(out_dir: Path, data: dict):
    """Atomically save checkpoint data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _path(out_dir)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(p)


def db_is_complete(ckpt: dict, db_key: str, num_runs: int, pareto_count: int) -> bool:
    """Check if a database benchmark is fully complete."""
    db = ckpt.get(db_key, {})
    if not db.get("ingest"):
        return False
    if len(db.get("runs", [])) < num_runs:
        return False
    if len(db.get("pareto", [])) < pareto_count:
        return False
    return True
