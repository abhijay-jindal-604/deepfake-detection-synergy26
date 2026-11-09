#!/usr/bin/env python
"""Assemble the Kaggle T4 kit (local, CPU, no GPU): one folder + one zip to upload as a PRIVATE Kaggle dataset.

  python scripts/kaggle/make_kit.py --bundle outputs/bundle/model_weights.pt \
      --local_pred outputs/predictions_local.csv            # test images default: $DF_DATA_ROOT/test/images
  -> outputs/kaggle_kit/  and  outputs/kaggle_kit.zip

Kit layout (t4_run.py finds it anywhere under /kaggle/input via kit_manifest.json):
  kit_manifest.json        sha256 + size of every kit file, n_images, git commit, creation time
  model_weights.pt         the bundle (scripts/bundle.py)
  inference_script.py      repo root copy (what the organisers run)
  requirements-infer.txt
  validate_predictions.py  scripts/validate_predictions.py
  t4_run.py                the runner (this folder)
  predictions_local.csv    optional: the local-GPU CSV to diff against
  test_images/*.jpg        the test images (inference only; never used for training)

--limit_images N copies only the first N images (smoke tests only; never for the real timing).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = (".jpg", ".jpeg", ".png")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="model_weights.pt from scripts/bundle.py")
    ap.add_argument("--test_dir", help="default: $DF_DATA_ROOT/test/images")
    ap.add_argument("--local_pred", help="local predictions.csv from the same bundle (for the cross-GPU diff)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "kaggle_kit"))
    ap.add_argument("--limit_images", type=int, help="SMOKE TEST ONLY: copy the first N images")
    ap.add_argument("--no_zip", action="store_true")
    a = ap.parse_args()

    test_dir = Path(a.test_dir) if a.test_dir else Path(os.environ.get("DF_DATA_ROOT", "")) / "test" / "images"
    if not test_dir.is_dir():
        raise SystemExit(f"test images not found at {test_dir} (pass --test_dir or set DF_DATA_ROOT)")
    imgs = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    if a.limit_images:
        imgs = imgs[: a.limit_images]
        print(f"WARNING: --limit_images {a.limit_images}: smoke kit, NOT for the real T4 timing")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "test_images").mkdir(parents=True)
    files = {
        "model_weights.pt": Path(a.bundle),
        "inference_script.py": ROOT / "inference_script.py",
        "requirements-infer.txt": ROOT / "requirements-infer.txt",
        "validate_predictions.py": ROOT / "scripts" / "validate_predictions.py",
        "t4_run.py": ROOT / "scripts" / "kaggle" / "t4_run.py",
    }
    if a.local_pred:
        files["predictions_local.csv"] = Path(a.local_pred)
    for dst, src in files.items():
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / dst)
    for p in imgs:
        shutil.copy2(p, out / "test_images" / p.name)

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    manifest = {
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "n_images": len(imgs), "smoke": bool(a.limit_images),
        "files": {k: {"sha256": sha256(out / k), "bytes": (out / k).stat().st_size} for k in files},
    }
    (out / "kit_manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"kit -> {out}  ({len(imgs)} images, {size / 1e6:.1f} MB)")
    for k, v in manifest["files"].items():
        print(f"  {k:<24} {v['bytes'] / 1e6:8.1f} MB  sha256 {v['sha256'][:16]}...")

    if not a.no_zip:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # JPEGs and weights do not compress
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(out.name) / p.relative_to(out))
        print(f"zip -> {z}  ({z.stat().st_size / 1e6:.1f} MB): upload this as a PRIVATE Kaggle dataset")


if __name__ == "__main__":
    main()
