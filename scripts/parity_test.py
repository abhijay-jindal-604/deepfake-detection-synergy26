"""Parity + robustness test for inference_script.py (docs/PHASE5.md §4.2). CPU is fine. Uses TRAIN images only.

  DF_DATA_ROOT=<root> python scripts/parity_test.py --run p2_ref --fold 0 [--runs_dir <runs>] [--n 0] [--keep]

1. Logit parity (FP32): inference_script.Member (FP32 bundle) vs dfd.models.Classifier with the same checkpoint
   on 64 of the fold's val images, each through its own decoder (inference_script.read_rgb vs dfd.data.read_rgb).
   Pass: max |dlogit| < 1e-4.
2. End-to-end parity: the fold model is bundled like the shipped bundle (--dtype, default fp32; T = w = 1) and
   inference_script.py is run as a subprocess (like the organisers) on a temp folder of symlinks to the fold's
   val images. Its confidences are compared
   (a) to dfd.models FP32 in-process on the same images (the exact reference; pass on CPU + FP32 bundle:
       max |dp| <= 2e-6, i.e. only the 6-decimal rounding), and
   (b) to the run's oof_fold{k}.csv, which was produced on the GPU under BF16 autocast. BF16 alone moves B0
       logits by up to ~0.75 (measured 4 Nov), so per-image |dp| can reach ~0.2; the gate here is the AUC:
       |AUC_script - AUC_oof| < 2e-3. Also reports mean |dp| and label flips at 0.5.
   --device cpu (default) hides the GPU from the subprocess (CUDA_VISIBLE_DEVICES=""); --device cuda tests FP16.
3. Robustness folder: PNG (lossless) copies, .JPG / .jpeg names, greyscale, RGBA, 300x300 and 160x160 resizes.
   Checks every file gets a row, lossless PNG copies and renamed JPEGs score exactly like the original, and a
   second run writes a byte-identical CSV.
All temporary files go to a fresh temp directory (--workdir to choose one), never into the repo.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bundle import load_inference_module, make_bundle, member_entry  # noqa: E402
from dfd.data import load_folds, read_rgb, to_tensor  # noqa: E402
from dfd.models import build_model  # noqa: E402
from dfd.paths import RUNS_DIR, resolve_data_root, train_image_path  # noqa: E402


DEVICE = "cpu"


def run_script(test_dir: Path, weights: Path, out: Path, batch_size: int = 64) -> float:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if DEVICE == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "inference_script.py"), "--test_dir", str(test_dir),
                        "--weights", str(weights), "--output", str(out), "--batch_size", str(batch_size)],
                       env=env, capture_output=True, text=True, cwd=str(test_dir.parent))
    dt_ = time.time() - t0
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("inference_script.py failed")
    for line in r.stdout.splitlines():
        if line.startswith(("[INFO] device", "  Time")):
            print("   script:", line.strip())
    return dt_


def reference_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ref = build_model(ck["config"], pretrained=False)
    ref.load_state_dict(ck.get("state_dict_ema") or ck["state_dict"])
    return ref.eval()


@torch.inference_mode()
def reference_probs(ckpt_path: Path, df: pd.DataFrame, root: Path, bs: int = 64) -> np.ndarray:
    """dfd.models FP32 on CPU, dfd.data decoding: the exact reference for the script."""
    ref = reference_model(ckpt_path)
    paths = [train_image_path(root, i, l) for i, l in zip(df.image_id, df.label)]
    out = []
    for s in range(0, len(paths), bs):
        x = torch.stack([to_tensor(read_rgb(p)) for p in paths[s:s + bs]])
        out.append(torch.sigmoid(ref(x).float()).numpy())
    return np.concatenate(out).astype(np.float64)


def logit_parity(ckpt_path: Path, df: pd.DataFrame, root: Path, work: Path, n: int = 64) -> float:
    inf = load_inference_module()
    ref = reference_model(ckpt_path)
    b = make_bundle([member_entry(ckpt_path, 1.0, 1.0, "fp32")], 0.5, 1.0, False)
    bp = work / "bundle_fp32.pt"
    torch.save(b, bp)
    (model, _, _), = inf.build_members(inf.load_bundle(bp), "cpu")
    sub = df.head(n)
    paths = [train_image_path(root, i, l) for i, l in zip(sub.image_id, sub.label)]
    x_ref = torch.stack([to_tensor(read_rgb(p)) for p in paths])
    x_inf = torch.stack([torch.from_numpy(inf.read_rgb(p).transpose(2, 0, 1).copy()) for p in paths])
    assert torch.equal(x_ref, x_inf), "decoded pixels differ between dfd.data.read_rgb and inference_script.read_rgb"
    with torch.inference_mode():
        z_ref = ref(x_ref).float()
        z_inf = model(x_inf).float()
    return float((z_ref - z_inf).abs().max())


def make_variants(src_ids: list[tuple[str, int]], root: Path, d: Path) -> dict[str, str]:
    """Write robustness variants; returns {variant_id: original_id} for the ones that must score identically."""
    d.mkdir(parents=True, exist_ok=True)
    same = {}
    for k, (iid, lab) in enumerate(src_ids):
        p = train_image_path(root, iid, lab)
        shutil.copy(p, d / f"{iid}.jpg")
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if k == 0:
            cv2.imwrite(str(d / f"{iid}_png.png"), bgr)  # lossless: identical pixels
            same[f"{iid}_png"] = iid
            shutil.copy(p, d / f"{iid}_upper.JPG")
            same[f"{iid}_upper"] = iid
            shutil.copy(p, d / f"{iid}_jpeg.jpeg")
            same[f"{iid}_jpeg"] = iid
            cv2.imwrite(str(d / f"{iid}_UPPERPNG.PNG"), bgr)
            same[f"{iid}_UPPERPNG"] = iid
        elif k == 1:
            cv2.imwrite(str(d / f"{iid}_grey.jpg"), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[..., 3] = 128
            cv2.imwrite(str(d / f"{iid}_rgba.png"), rgba)
        elif k == 2:
            cv2.imwrite(str(d / f"{iid}_300.png"), cv2.resize(bgr, (300, 300), interpolation=cv2.INTER_CUBIC))
            cv2.imwrite(str(d / f"{iid}_160.jpg"), cv2.resize(bgr, (160, 160), interpolation=cv2.INTER_AREA))
    (d / "notes.txt").write_text("not an image: must be ignored\n")
    return same


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default="last", choices=["last", "best"])
    ap.add_argument("--runs_dir", default=None, help="default: $DFD_RUNS_DIR or outputs/runs")
    ap.add_argument("--data_root")
    ap.add_argument("--n", type=int, default=0, help="limit val images for the end-to-end check (0 = all)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workdir", help="temp directory (default: a fresh tempfile.mkdtemp())")
    ap.add_argument("--keep", action="store_true", help="keep the temp directory")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"], help="bundle weight storage")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device for inference_script.py")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device

    root = resolve_data_root(args.data_root)
    runs = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else RUNS_DIR
    fdir = runs / args.run / f"fold{args.fold}"
    ckpt = fdir / f"{args.ckpt}.pt"
    oof = pd.read_csv(fdir / f"oof_fold{args.fold}.csv", dtype={"image_id": str})
    folds = load_folds()
    df = folds[folds.fold == args.fold].reset_index(drop=True)
    assert set(df.image_id) == set(oof.image_id), "oof file does not match the fold's val split"
    work = Path(args.workdir or tempfile.mkdtemp(prefix="dfd_parity_"))
    work.mkdir(parents=True, exist_ok=True)
    assert ROOT not in work.resolve().parents, "keep temp files out of the repo"
    print(f"workdir {work}")
    ok = True
    try:
        # 1. logit parity, FP32
        d = logit_parity(ckpt, df, root, work)
        print(f"1. logit parity FP32 (64 imgs, script vs dfd.models): max |dlogit| = {d:.2e} -> {'PASS' if d < 1e-4 else 'FAIL'}")
        ok &= d < 1e-4

        # 2. end-to-end vs OOF
        sub = df if args.n <= 0 else df.sample(n=min(args.n, len(df)), random_state=42)
        tdir = work / "val_images"
        tdir.mkdir(exist_ok=True)
        for i, l in zip(sub.image_id, sub.label):
            (tdir / f"{i}.jpg").symlink_to(train_image_path(root, i, l))
        bp = work / "model_weights.pt"
        torch.save(make_bundle([member_entry(ckpt, 1.0, 1.0, args.dtype)], 0.5, 1.0, False), bp)
        out = work / "pred_val.csv"
        secs = run_script(tdir, bp, out, args.batch_size)
        pred = pd.read_csv(out, dtype={"image_id": str})
        m = pred.merge(oof, on="image_id", how="inner")
        assert len(m) == len(sub) == len(pred), "row mismatch"
        print(f"2. end-to-end: {len(m)} val images, {secs:.1f}s wall incl. start-up ({len(m) / secs:.1f} img/s), "
              f"bundle {args.dtype}, device {DEVICE}")
        # (a) exact reference
        sub_ref = sub.set_index("image_id").loc[m.image_id].reset_index()
        p_ref = reference_probs(ckpt, sub_ref, root)
        da = np.abs(m.confidence.to_numpy() - p_ref)
        tol_a = 2e-6 if (args.dtype == "fp32" and DEVICE == "cpu") else 2e-2
        print(f"   (a) vs dfd.models FP32: max |dp| {da.max():.2e} | mean {da.mean():.2e} -> "
              f"{'PASS' if da.max() <= tol_a else 'FAIL'} (tol {tol_a:g})")
        ok &= bool(da.max() <= tol_a)
        # (b) vs the BF16 OOF file
        dp = (m.confidence - m.prob).abs()
        auc_s, auc_o = roc_auc_score(m.label_y, m.confidence), roc_auc_score(m.label_y, m.prob)
        flips = int(((m.confidence >= 0.5) != (m.prob >= 0.5)).sum())
        print(f"   (b) vs oof_fold{args.fold}.csv (BF16): max |dp| {dp.max():.2e} | mean {dp.mean():.2e} | "
              f"p99 {dp.quantile(0.99):.2e} | AUC script {auc_s:.6f} vs OOF {auc_o:.6f} (d {auc_s - auc_o:+.2e}) | "
              f"label flips @0.5: {flips}/{len(m)} -> {'PASS' if abs(auc_s - auc_o) < 2e-3 else 'FAIL'} (|dAUC| < 2e-3)")
        ok &= bool(abs(auc_s - auc_o) < 2e-3)

        # 3. robustness + determinism
        rdir = work / "robust_images"
        same = make_variants(list(zip(df.image_id[:3], df.label[:3])), root, rdir)
        n_img = sum(1 for p in rdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        o1, o2 = work / "pred_robust_1.csv", work / "pred_robust_2.csv"
        run_script(rdir, bp, o1, 4)
        run_script(rdir, bp, o2, 3)  # different batch size too
        r = pd.read_csv(o1, dtype={"image_id": str}).set_index("image_id")
        rows_ok = len(r) == n_img
        same_ok = all(r.confidence[v] == r.confidence[o] for v, o in same.items())
        ident = filecmp.cmp(o1, o2, shallow=False)
        hdr = o1.read_text().splitlines()[0] == "image_id,label,confidence"
        print(f"3. robustness: {len(r)}/{n_img} rows | header ok {hdr} | lossless/renamed copies identical {same_ok} | "
              f"re-run (batch 4 vs 3) byte-identical {ident}")
        print(r.to_string())
        ok &= rows_ok and same_ok and ident and hdr
    finally:
        if not args.keep and not args.workdir:
            shutil.rmtree(work, ignore_errors=True)
    print("PARITY TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
