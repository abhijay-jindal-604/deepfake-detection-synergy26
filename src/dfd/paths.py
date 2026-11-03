"""Dataset-root resolution and output directories. No hardcoded absolute paths."""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DF_DATA_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"          # small generated metadata (committed)
OUTPUTS_DIR = REPO_ROOT / "outputs"    # runs, audit images (git-ignored)
AUDIT_DIR = OUTPUTS_DIR / "audit"
RUNS_DIR = OUTPUTS_DIR / "runs"


def resolve_data_root(cli_value: str | None = None) -> Path:
    """CLI value wins, then $DF_DATA_ROOT. Must contain train/ and train_metadata.csv."""
    value = cli_value or os.environ.get(ENV_VAR)
    if not value:
        raise SystemExit(f"Dataset root not given: pass --data_root or set ${ENV_VAR}.")
    root = Path(value).expanduser().resolve()
    if not (root / "train_metadata.csv").is_file() or not (root / "train").is_dir():
        raise SystemExit(f"{root} does not look like the dataset root (needs train/ and train_metadata.csv).")
    return root


def train_image_path(root: Path, image_id: str, label: int) -> Path:
    return root / "train" / ("fake" if int(label) == 1 else "real") / f"{image_id}.jpg"


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIT_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
