#!/usr/bin/env python
"""Kaggle T4 run: sm_75 check, timed end-to-end inference (x2), validator, determinism and cross-GPU diff.

Run it with the Python 3.10 venv that has requirements-infer.txt installed (see scripts/kaggle/README.md):
  /kaggle/working/venv/bin/python /kaggle/input/<dataset>/.../t4_run.py
Options: --kit_dir (default /kaggle/input, searched recursively for kit_manifest.json), --out (default
/kaggle/working/t4_out), --runs 2, --batch_size 64, --allow_cpu (local smoke test without a GPU).

Writes into --out: predictions_t4_run{1,2}.csv, predictions.csv (= run 1, the candidate submission), inference
logs, validate_*.txt, t4_report.json and t4_report.md. Exit code 1 if a hard check fails.

Timing = wall clock of `python inference_script.py ...` as a fresh process (interpreter start, imports, bundle
load, decode, inference, CSV write), i.e. what the organisers measure. Run 1 is the cold-cache number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

BANDS = [(5, 20), (10, 15), (20, 10), (40, 5)]  # (minutes, efficiency points); > 40 min or OOM = 0
TARGET_MIN = 2.5  # PHASE5 §5 budget
CROSS_TOL = 1e-3  # PHASE6 §7: local GPU vs T4 confidences


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def points(minutes: float) -> int:
    for lim, pts in BANDS:
        if minutes < lim:
            return pts
    return 0


def find_kit(kit_dir: Path) -> Path:
    hits = sorted(kit_dir.rglob("kit_manifest.json"))
    if not hits:
        raise SystemExit(f"no kit_manifest.json under {kit_dir}: is the Kaggle dataset attached (Add Input)?")
    if len(hits) > 1:
        print(f"[WARN] {len(hits)} kits found, using {hits[0].parent}")
    return hits[0].parent


def env_report(allow_cpu: bool) -> dict:
    import torch

    r = {
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "compiled_archs": torch._C._cuda_getArchFlags() if torch.version.cuda else "",
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        r["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        r["nvidia_smi"] = ""
    if not torch.cuda.is_available():
        r["gpu_ok"] = False
        if not allow_cpu:
            raise SystemExit("CUDA not available: set the Kaggle accelerator to 'GPU T4 x2' (or pass --allow_cpu)")
        return r
    maj, mnr = torch.cuda.get_device_capability(0)
    r["device"] = torch.cuda.get_device_name(0)
    r["capability"] = f"sm_{maj}{mnr}"
    r["is_t4"] = "T4" in r["device"]
    r["arch_compiled"] = r["capability"] in torch.cuda.get_arch_list()
    try:  # prove kernels for this GPU exist: "no kernel image is available" would raise here
        x = torch.randn(8, 3, 64, 64, device="cuda", dtype=torch.float16)
        w = torch.randn(16, 3, 3, 3, device="cuda", dtype=torch.float16)
        y = torch.nn.functional.conv2d(x, w).float().sum() + (x.flatten(1) @ x.flatten(1).T).float().sum()
        torch.cuda.synchronize()
        r["kernel_smoke"] = bool(torch.isfinite(y))
    except RuntimeError as e:
        r["kernel_smoke"] = False
        r["kernel_error"] = str(e).splitlines()[0]
    r["gpu_ok"] = bool(r["arch_compiled"] and r["kernel_smoke"])
    return r


def run_inference(kit: Path, out: Path, i: int, batch_size: int) -> dict:
    pred = out / f"predictions_t4_run{i}.csv"
    cmd = [sys.executable, str(kit / "inference_script.py"), "--test_dir", str(kit / "test_images"),
           "--weights", str(kit / "model_weights.pt"), "--output", str(pred), "--batch_size", str(batch_size)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}  # one T4, as graded
    print(f"\n[run {i}] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    log = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
    (out / f"inference_run{i}.log").write_text(log)
    print(log, flush=True)
    m = re.search(r"threshold ([0-9.]+)", p.stdout)
    t = re.search(r"Time: load ([0-9.]+)s \| inference ([0-9.]+)s \(([0-9.]+) img/s\)", p.stdout)
    return {"run": i, "returncode": p.returncode, "wall_s": round(wall, 2), "wall_min": round(wall / 60, 3),
            "points_band": points(wall / 60), "threshold": float(m.group(1)) if m else None,
            "load_s": float(t.group(1)) if t else None, "inference_s": float(t.group(2)) if t else None,
            "img_per_s": float(t.group(3)) if t else None, "csv": str(pred)}


def validate(kit: Path, out: Path, pred: Path, name: str, extra: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, str(kit / "validate_predictions.py"), str(pred), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / f"validate_{name}.txt").write_text(p.stdout + p.stderr)
    print(f"\n[validate {name}] {' '.join(cmd[1:])}\n{p.stdout}{p.stderr}", flush=True)
    return p.returncode == 0, p.stdout


def cross_diff(a: Path, b: Path) -> dict:
    import numpy as np
    import pandas as pd

    x = pd.read_csv(a, dtype={"image_id": str})
    y = pd.read_csv(b, dtype={"image_id": str})
    m = x.merge(y, on="image_id", suffixes=("_t4", "_local"), how="outer", indicator=True)
    both = m[m._merge == "both"]
    d = (both.confidence_t4 - both.confidence_local).abs()
    flips = both[both.label_t4 != both.label_local]
    rank = float(both.confidence_t4.rank().corr(both.confidence_local.rank())) if len(both) > 1 else None
    return {"n_common": int(len(both)), "n_only_one_side": int((m._merge != "both").sum()),
            "max_abs_dconf": float(d.max()) if len(d) else None, "mean_abs_dconf": float(d.mean()) if len(d) else None,
            "n_over_tol": int((d > CROSS_TOL).sum()), "label_flips": int(len(flips)),
            "flip_ids": flips.image_id.head(20).tolist(), "spearman": rank,
            "flip_confs": [round(float(v), 6) for v in flips.confidence_t4.head(20)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit_dir", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/t4_out")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--allow_cpu", action="store_true", help="local smoke test without a GPU")
    a = ap.parse_args()

    kit = find_kit(Path(a.kit_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((kit / "kit_manifest.json").read_text())
    fails = []

    # 1. kit integrity: the files on Kaggle are byte-identical to the ones packed locally
    integrity = {}
    for k, v in man["files"].items():
        integrity[k] = (kit / k).is_file() and sha256(kit / k) == v["sha256"]
    n_img = sum(1 for p in (kit / "test_images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"kit {kit} | git {man.get('git_commit', '')[:10]} | images {n_img} (manifest {man['n_images']}) | "
          f"sha256 {'OK' if all(integrity.values()) else 'MISMATCH ' + str([k for k, ok in integrity.items() if not ok])}")
    if not all(integrity.values()) or n_img != man["n_images"]:
        fails.append("kit integrity")
    if man.get("smoke"):
        print("[WARN] smoke kit (--limit_images): timing is NOT representative")

    # 2. environment + sm_75
    env = env_report(a.allow_cpu)
    print("\n[env] " + json.dumps(env, indent=2))
    if env.get("cuda_available") and not env["gpu_ok"]:
        fails.append(f"GPU arch {env.get('capability')} not supported by this torch build")
    if env.get("cuda_available") and not env.get("is_t4"):
        print(f"[WARN] GPU is {env.get('device')}, not a T4: timing is not the graded number")

    # 3. timed runs
    runs = [run_inference(kit, out, i, a.batch_size) for i in range(1, a.runs + 1)]
    if any(r["returncode"] != 0 for r in runs):
        fails.append("inference_script.py failed")
        ok_runs = []
    else:
        ok_runs = runs
    # 4. validator (format + ID set + label/threshold consistency) on run 1
    val_ok = det = None
    cross = None
    if ok_runs:
        r1 = Path(runs[0]["csv"])
        extra = ["--test_dir", str(kit / "test_images")]
        if runs[0]["threshold"] is not None:
            extra += ["--threshold", str(runs[0]["threshold"])]
        val_ok, _ = validate(kit, out, r1, "format", extra)
        if not val_ok:
            fails.append("validator")
        # 5. determinism: same machine, same command -> byte-identical CSV
        if len(runs) > 1:
            det = all(Path(r["csv"]).read_bytes() == r1.read_bytes() for r in runs[1:])
            print(f"\n[determinism] run1 vs run2..{len(runs)} byte-identical: {det}")
            if not det:
                fails.append("re-run not byte-identical")
        (out / "predictions.csv").write_bytes(r1.read_bytes())
        # 6. cross-GPU diff against the local CSV (informational; FP16 differs slightly between GPU generations)
        if (kit / "predictions_local.csv").is_file():
            validate(kit, out, r1, "vs_local", ["--compare", str(kit / "predictions_local.csv"), "--tol", str(CROSS_TOL)])
            cross = cross_diff(r1, kit / "predictions_local.csv")
            print("\n[cross-GPU vs local] " + json.dumps(cross, indent=2))

    rep = {"kit_commit": man.get("git_commit"), "n_images": n_img, "env": env, "runs": runs,
           "validator_pass": val_ok, "deterministic": det, "cross_vs_local": cross, "fails": fails,
           "target_min": TARGET_MIN}
    (out / "t4_report.json").write_text(json.dumps(rep, indent=2))
    lines = [f"# T4 run report ({time.strftime('%Y-%m-%d %H:%M %Z')})", "",
             f"- GPU: {env.get('device', 'none')} ({env.get('capability', '-')}), driver/nvidia-smi: {env.get('nvidia_smi', '')}",
             f"- torch {env['torch']} (CUDA {env['torch_cuda']}, cuDNN {env['cudnn']}), archs: {env['compiled_archs']}; "
             f"sm check: {'PASS' if env.get('gpu_ok') else 'FAIL/none'}; Python {env['python']}",
             f"- images: {n_img}; kit commit {str(man.get('git_commit'))[:10]}", ""]
    lines += ["| run | wall (s) | wall (min) | load (s) | inference (s) | img/s | efficiency band |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['run']} | {r['wall_s']} | {r['wall_min']} | {r['load_s']} | {r['inference_s']} | {r['img_per_s']} | "
                     f"{r['points_band']} pts{' (rc ' + str(r['returncode']) + ')' if r['returncode'] else ''} |")
    lines += ["", f"- validator: {val_ok}; re-run byte-identical: {det}; target <= {TARGET_MIN} min: "
              f"{'met' if runs and runs[0]['wall_min'] <= TARGET_MIN else 'NOT met'} (run 1, cold)"]
    if cross:
        lines.append(f"- vs local CSV: max |dconf| {cross['max_abs_dconf']:.2e}, mean {cross['mean_abs_dconf']:.2e}, "
                     f"> {CROSS_TOL:g}: {cross['n_over_tol']}, label flips {cross['label_flips']}, Spearman {cross['spearman']:.6f}")
    lines.append(f"- hard failures: {fails or 'none'}")
    (out / "t4_report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
