"""
inference_script.py - Deepfake ML Hackathon (Synergy '26), self-contained inference.

    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
        [--batch_size 64] [--threshold T]   (threshold default: the value stored in the weights bundle)

Writes predictions.csv with columns image_id,label,confidence (confidence = P(fake), label = confidence >= threshold).

The weights file is ONE bundle (made by scripts/bundle.py) holding every ensemble member:
    {"format_version": 1, "members": [{"timm_name", "input_size", "state_dict", "temperature", "weight", ...}],
     "temperature_ens", "threshold", "tta_flip", "mean", "std", "image_size", ...}
A raw training checkpoint ({"state_dict"/"state_dict_ema", "config"}) is also accepted (single model, T=1).

Pipeline (identical to training, src/dfd/data.py + src/dfd/models.py):
  decode with OpenCV (IMREAD_COLOR: greyscale -> 3 channels, alpha dropped) -> RGB -> resize to 224 with INTER_AREA
  if needed -> uint8 CHW; inside the model: (x - mean*255) / (std*255), then bilinear upsampling to the member's
  input_size (e.g. 320) if it differs from 224 -> timm backbone -> 1 logit.
Ensemble: s = sum_m w_m * logit_m / T_m (logit_m averaged over the horizontal flip if tta_flip), confidence =
sigmoid(s / T_ens).

Speed / reproducibility: all models loaded once; batched DataLoader with workers; CUDA -> FP16 autocast +
channels_last + inference_mode (a batch with a non-finite FP16 logit is recomputed in FP32); CPU -> FP32.
cudnn.benchmark is OFF and cudnn.deterministic ON so a re-run gives the same CSV; files are processed in sorted
order; nothing is random. Models are built with pretrained=False: no network access is needed or attempted.

Dependencies: torch, timm, numpy, pandas, opencv-python-headless, pillow (fallback decoder). Does not import the
training package. Python 3.10.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download anything (weights come from --weights)

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # matched case-insensitively (.JPG, .PNG, ...)
DEFAULT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.5  # only used if the bundle carries no threshold


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def list_images(test_dir: str | Path) -> list[Path]:
    d = Path(test_dir)
    if not d.is_dir():
        raise SystemExit(f"--test_dir {d} is not a directory")
    paths = sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name)
    if not paths:
        raise SystemExit(f"no .jpg/.jpeg/.png images found in {d}")
    return paths


def read_rgb(path: Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    """uint8 HxWx3 RGB at size x size; same decoder and resize as training (dfd.data.read_rgb)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR; greyscale -> 3 channels; alpha dropped; 16-bit -> 8-bit
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:  # fallback decoder for files OpenCV cannot read (fail loudly if neither can)
        try:
            from PIL import Image

            with Image.open(path) as im:
                img = np.asarray(im.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot decode image {path}: {exc}") from exc
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


class TestDataset(Dataset):
    def __init__(self, paths: list[Path], size: int = DEFAULT_SIZE):
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = read_rgb(self.paths[i], self.size)
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), i


def _worker_init(_: int) -> None:
    cv2.setNumThreads(1)  # avoid thread oversubscription inside DataLoader workers


# ---------------------------------------------------------------------------------------------------------------
# Model (copy of dfd.models.Classifier's eval path: normalisation + optional upsampling inside forward)
# ---------------------------------------------------------------------------------------------------------------
class Member(nn.Module):
    def __init__(self, timm_name: str, input_size: int, mean, std, drop_rate: float = 0.0):
        super().__init__()
        self.input_size = int(input_size)
        self.net = timm.create_model(timm_name, pretrained=False, num_classes=1, drop_rate=drop_rate)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: uint8 (B,3,H,W) -> (B,) logits
        x = (x.float() - self.mean) / self.std
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net(x).squeeze(1)


def _members_from_checkpoint(ck: dict) -> dict:
    """A raw training checkpoint ({state_dict[_ema], config}) as a 1-member bundle (T = w = 1, threshold 0.5)."""
    m = ck["config"]["model"]
    sd = ck.get("state_dict_ema") or ck["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("mask_head.")}
    return {
        "format_version": 1,
        "members": [{"timm_name": m["name"], "input_size": int(m.get("input_size", DEFAULT_SIZE)),
                     "drop_rate": 0.0, "state_dict": sd, "temperature": 1.0, "weight": 1.0}],
        "temperature_ens": 1.0, "threshold": DEFAULT_THRESHOLD, "tta_flip": False,
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "image_size": DEFAULT_SIZE,
    }


def load_bundle(weights_path: str | Path) -> dict:
    try:
        b = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (older/raw checkpoints with non-tensor objects)
        b = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(b, dict):
        raise SystemExit(f"{weights_path}: expected a weights bundle dict, got {type(b).__name__}")
    if "members" not in b:
        if "config" in b and ("state_dict" in b or "state_dict_ema" in b):
            b = _members_from_checkpoint(b)
        else:
            raise SystemExit(f"{weights_path}: not a weights bundle (no 'members') nor a training checkpoint")
    if int(b.get("format_version", 1)) != 1:
        raise SystemExit(f"unsupported bundle format_version {b.get('format_version')}")
    return b


def build_members(bundle: dict, device: str) -> list[tuple[Member, float, float]]:
    """[(model, weight, temperature)] with weights loaded strictly, eval mode, on device."""
    mean = bundle.get("mean", IMAGENET_MEAN)
    std = bundle.get("std", IMAGENET_STD)
    out = []
    for i, m in enumerate(bundle["members"]):
        model = Member(m["timm_name"], m.get("input_size", DEFAULT_SIZE), mean, std, float(m.get("drop_rate", 0.0)))
        sd = {k: v.float() if v.is_floating_point() else v for k, v in m["state_dict"].items()}
        model.load_state_dict(sd, strict=True)
        model.eval().to(device)
        if device.startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
        w, t = float(m.get("weight", 1.0)), float(m.get("temperature", 1.0))
        if not (math.isfinite(w) and math.isfinite(t) and t > 0):
            raise SystemExit(f"member {i}: bad weight {w} / temperature {t}")
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[INFO] member {i}: {m['timm_name']} @ {model.input_size} | {n_par:.1f} M params | w {w:.3f} | T {t:.3f}")
        out.append((model, w, t))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------------------
def _forward(model: Member, x: torch.Tensor, fp16: bool) -> torch.Tensor:
    if not fp16:
        return model(x).float()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        z = model(x).float()
    if not torch.isfinite(z).all():  # FP16 overflow guard: redo this batch in FP32
        z = model(x).float()
    return z


def member_logits(model: Member, x: torch.Tensor, tta_flip: bool, fp16: bool) -> torch.Tensor:
    z = _forward(model, x, fp16)
    if tta_flip:
        z = 0.5 * (z + _forward(model, torch.flip(x, dims=[3]), fp16))
    return z


@torch.inference_mode()
def run_inference(members, paths: list[Path], device: str, batch_size: int, tta_flip: bool,
                  temperature_ens: float, image_size: int = DEFAULT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Returns (confidence float64 (N,), per-member logits float32 (N, M)) in `paths` order."""
    cuda = device.startswith("cuda")
    workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(TestDataset(paths, image_size), batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=cuda, worker_init_fn=_worker_init if workers > 0 else None)
    logits = np.zeros((len(paths), len(members)), dtype=np.float32)
    seen = np.zeros(len(paths), dtype=bool)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        if cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        idx = idx.numpy()
        for j, (model, _, _) in enumerate(members):
            logits[idx, j] = member_logits(model, x, tta_flip, cuda).cpu().numpy()
        seen[idx] = True
    assert seen.all(), "some images were not processed"
    w = np.array([m[1] for m in members], dtype=np.float64)
    t = np.array([m[2] for m in members], dtype=np.float64)
    s = (logits.astype(np.float64) / t) @ w
    conf = 1.0 / (1.0 + np.exp(-s / float(temperature_ens)))
    return conf, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon - Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images directory")
    parser.add_argument("--weights", required=True, help="Path to model weights file (bundle)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=None, help="default: the threshold stored in the bundle")
    args = parser.parse_args()

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False  # deterministic algorithm choice -> identical CSV on re-run
    torch.backends.cudnn.deterministic = True
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    paths = list_images(args.test_dir)
    ids = [p.stem for p in paths]
    if len(set(ids)) != len(ids):
        dup = pd.Series(ids)[pd.Series(ids).duplicated()].tolist()[:5]
        raise SystemExit(f"duplicate image ids (same stem, different extension?): {dup}")

    bundle = load_bundle(args.weights)
    members = build_members(bundle, device)
    threshold = float(args.threshold) if args.threshold is not None else float(bundle.get("threshold", DEFAULT_THRESHOLD))
    tta_flip = bool(bundle.get("tta_flip", False))
    t_ens = float(bundle.get("temperature_ens", 1.0))
    image_size = int(bundle.get("image_size", DEFAULT_SIZE))
    if device == "cuda":
        torch.cuda.synchronize()
    t_loaded = time.time()
    print(f"[INFO] device {device}{' (' + torch.cuda.get_device_name(0) + ', FP16)' if device == 'cuda' else ' (FP32)'} | "
          f"{len(members)} member(s) | tta_flip {tta_flip} | T_ens {t_ens:.3f} | threshold {threshold:.4f} | "
          f"{len(paths)} images | batch {args.batch_size} | load {t_loaded - t_start:.1f}s")

    conf, _ = run_inference(members, paths, device, args.batch_size, tta_flip, t_ens, image_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time() - t_loaded

    conf = np.round(conf, 6)  # what is written; the label is derived from the written value
    predictions = pd.DataFrame({"image_id": ids, "label": (conf >= threshold).astype(int), "confidence": conf})

    # Validate output
    assert list(predictions.columns) == ["image_id", "label", "confidence"], "bad columns"
    assert len(predictions) == len(paths), "row count != number of images"
    assert predictions["image_id"].is_unique, "duplicate image_id"
    assert predictions["confidence"].notna().all() and np.isfinite(predictions["confidence"]).all(), "NaN confidence"
    assert predictions["confidence"].between(0, 1).all(), "Confidence values must be in [0, 1]"
    assert predictions["label"].isin([0, 1]).all(), "Labels must be 0 or 1"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False, float_format="%.6f")
    total = time.time() - t_start
    print(f"[DONE] Predictions saved to {out}")
    print(f"  Total images: {len(predictions)}")
    print(f"  Predicted FAKE: {predictions['label'].sum()} ({predictions['label'].mean() * 100:.1f}%)")
    print(f"  Predicted REAL: {(predictions['label'] == 0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")
    print(f"  Time: load {t_loaded - t_start:.1f}s | inference {t_inf:.1f}s ({len(paths) / max(t_inf, 1e-9):.1f} img/s) | "
          f"total {total:.1f}s")


if __name__ == "__main__":
    main()
