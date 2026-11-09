"""Grad-CAM for the timm classifiers used here (PHASE4 §6.5, PHASE6 §1). VALIDATION images only, never test.

Where the CAM is taken (no extra package; autograd on an intermediate tensor):
  CNNs (EfficientNet B0/B4, ConvNeXt, ...): the last feature map, net.forward_features(x) (7x7 at 224, 10x10 for
      B4@320). NHWC outputs (Swin) are permuted to NCHW.
  ViT: the input of the last block's norm1 (= the residual stream entering the last block), patch tokens reshaped
      to the patch grid (14x14). The final normed tokens cannot be used: a cls-token head gives zero gradient to
      the patch tokens there. This is the standard "reshape transform" Grad-CAM for ViTs.
Target score: the fake logit for a fake prediction and -logit for a real prediction (`--target pred`, default), i.e.
"evidence for the predicted class"; `--target fake` always explains the fake logit.

  # TP / TN / FP / FN grid from a fold run (selection uses the run's oof_fold<k>.csv; CAMs recomputed in FP32)
  python -m dfd.gradcam grid --ckpt outputs/runs/p2_final/fold0/last.pt --per_class 4
  # corrupted val fakes (attention should stay on the face): rows = clean + each corruption
  python -m dfd.gradcam corrupt --ckpt outputs/runs/p2_final/fold0/last.pt --n 6 --corruptions snow:2,block_patches:2,pixelate:2
  # before/after (e.g. p2_final vs best Phase 4 model) on the same val images
  python -m dfd.gradcam compare --ckpt A/fold0/last.pt B/fold0/last.pt --labels p2_final p4_sbi --n 8
  # CPU self-test of every backbone family (random init, no download)
  python -m dfd.gradcam selftest
Figures go to outputs/figures/gradcam/ by default. CPU works (set --threads); ~0.1-1 s per image per model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .models import Classifier, build_model
from .paths import OUTPUTS_DIR, resolve_data_root, train_image_path

FIG_DIR = OUTPUTS_DIR / "figures" / "gradcam"
CAT_ORDER = ("TP", "TN", "FP", "FN")
CAT_TEXT = {"TP": "fake, called fake", "TN": "real, called real", "FP": "real, called FAKE", "FN": "fake, called REAL"}
CAT_COLOUR = {"TP": (200, 40, 40), "TN": (40, 160, 60), "FP": (220, 120, 0), "FN": (120, 40, 160)}


# ---------------------------------------------------------------- core
def is_vit(net) -> bool:
    from timm.models.vision_transformer import VisionTransformer

    return isinstance(net, VisionTransformer)


class GradCAM:
    """cams, probs = GradCAM(model)(x_uint8) -> (B, H, W) float in [0,1] at the input resolution, (B,) P(fake)."""

    def __init__(self, model: Classifier, target: str = "pred"):
        assert target in ("pred", "fake")
        self.model = model.eval()
        self.net = model.net
        self.vit = is_vit(self.net)
        self.target = target
        self._feat = None
        if self.vit:
            self.net.blocks[-1].norm1.register_forward_hook(self._grab_input)

    def _grab_input(self, module, inp, out):
        self._feat = inp[0]
        self._feat.retain_grad()

    def _forward(self, x: torch.Tensor):
        xp = self.model._prep(x)
        if self.vit:
            logit = self.net(xp).squeeze(1)
            tok = self._feat  # (B, N, D) incl. prefix tokens
            n_pre = getattr(self.net, "num_prefix_tokens", 1)
            gh, gw = self.net.patch_embed.grid_size
            return logit, tok, lambda t: t[:, n_pre:].reshape(t.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
        f = self.net.forward_features(xp)
        f.retain_grad()
        logit = self.net.forward_head(f).squeeze(1)
        nf = getattr(self.net, "num_features", f.shape[1])
        nhwc = f.ndim == 4 and f.shape[1] != nf and f.shape[-1] == nf
        return logit, f, (lambda t: t.permute(0, 3, 1, 2)) if nhwc else (lambda t: t)

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit, feat, to_nchw = self._forward(x)
            sign = torch.ones_like(logit) if self.target == "fake" else torch.where(logit >= 0, 1.0, -1.0)
            (logit * sign).sum().backward()
            a, g = to_nchw(feat.detach()), to_nchw(feat.grad)
            w = g.mean(dim=(2, 3), keepdim=True)  # Grad-CAM channel weights = spatially averaged gradients
            cam = torch.relu((w * a).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.cpu().numpy()
        mx = cam.reshape(len(cam), -1).max(1).reshape(-1, 1, 1)
        cam = np.where(mx > 0, cam / np.maximum(mx, 1e-12), 0.0)  # per-image [0,1]; all-zero stays zero
        return cam.astype(np.float32), torch.sigmoid(logit.detach().float()).cpu().numpy()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8((1 - alpha) * img + alpha * heat)


def load_ckpt(path: str | Path, device: str = "cpu") -> tuple[Classifier, dict]:
    """Fold checkpoint (train.py last.pt / best.pt): EMA weights if present, built with pretrained=False."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"], pretrained=False)
    model.load_state_dict(ck.get("state_dict_ema") or ck["state_dict"])
    for p in model.parameters():
        p.requires_grad_(True)
    return model.to(device).eval(), ck


def cams_for(model: Classifier, imgs: list[np.ndarray], target: str, device: str, bs: int = 16):
    gc = GradCAM(model, target)
    cams, probs = [], []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im in imgs[i : i + bs]])).to(device)
        c, p = gc(x)
        cams.append(c)
        probs.append(p)
    return np.concatenate(cams), np.concatenate(probs)


# ---------------------------------------------------------------- selection
def categorise(oof: pd.DataFrame, thr: float) -> pd.Series:
    pred = (oof.prob >= thr).astype(int)
    y = oof.label.astype(int)
    return pd.Series(np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1)],
                               ["TP", "TN", "FP"], "FN"), index=oof.index)


def pick(oof: pd.DataFrame, n: int, how: str, seed: int) -> pd.DataFrame:
    """n rows per category. random = representative; confident = most confident (furthest from threshold)."""
    out = []
    for cat in CAT_ORDER:
        d = oof[oof.cat == cat]
        if how == "confident":
            d = d.reindex((d.prob - 0.5).abs().sort_values(ascending=False).index).head(n)
        else:
            d = d.sample(n=min(n, len(d)), random_state=seed)
        out.append(d)
    return pd.concat(out)


def read_rgb224(root: Path, image_id: str, label: int) -> np.ndarray:
    from .data import read_rgb

    return read_rgb(train_image_path(root, image_id, label))


def fold_oof(ckpt_path: Path, ck: dict) -> pd.DataFrame:
    k = ck["fold"]
    p = Path(ckpt_path).parent / f"oof_fold{k}.csv"
    if not p.is_file():
        raise SystemExit(f"{p} missing (fold run not finished?); Grad-CAM grids need a fold checkpoint with OOF")
    return pd.read_csv(p, dtype={"image_id": str})


# ---------------------------------------------------------------- figures
def _cell(img: np.ndarray, cam: np.ndarray, caption: str, colour, size: int = 160) -> Image.Image:
    from .viz import _tile

    a = _tile(Image.fromarray(img), caption, colour, size=size)
    b = _tile(Image.fromarray(overlay(img, cam)), "Grad-CAM", (60, 60, 60), size=size)
    c = Image.new("RGB", (a.width * 2 + 2, a.height), "white")
    c.paste(a, (0, 0))
    c.paste(b, (a.width + 2, 0))
    return c


def _name(ck: dict) -> str:
    return f"{ck['config']['model']['name'].split('.')[0]} fold{ck['fold']}"


def cmd_grid(a) -> None:
    from .viz import grid_from_cells

    root = resolve_data_root(a.data_root)
    model, ck = load_ckpt(a.ckpt, a.device)
    oof = fold_oof(a.ckpt, ck)
    oof["cat"] = categorise(oof, a.threshold)
    sel = pick(oof, a.per_class, a.pick, a.seed)
    imgs = [read_rgb224(root, r.image_id, r.label) for r in sel.itertuples()]
    cams, probs = cams_for(model, imgs, a.target, a.device)
    cells = [_cell(im, cm, f"{r.cat} p={p:.2f}", CAT_COLOUR[r.cat]) for im, cm, p, r in zip(imgs, cams, probs, sel.itertuples())]
    counts = oof.cat.value_counts().to_dict()
    title = (f"Grad-CAM {_name(ck)} on fold-{ck['fold']} VALIDATION images (thr {a.threshold}); rows TP / TN / FP / FN "
             f"({', '.join(f'{c} {counts.get(c, 0)}' for c in CAT_ORDER)}); target={a.target}")
    out = Path(a.out) if a.out else FIG_DIR / f"gradcam_{Path(a.ckpt).parent.parent.name}_fold{ck['fold']}.jpg"
    grid_from_cells(cells, out, title, cols=a.per_class, gap=6)
    sel.assign(prob_fp32=probs).to_csv(out.with_suffix(".csv"), index=False)
    print(f"-> {out}  (+ .csv with the selected ids; max |p_fp32 - p_oof| = {np.abs(probs - sel.prob.to_numpy()).max():.4f})")


def cmd_corrupt(a) -> None:
    from .corruptions import corrupt, setting_seed
    from .viz import grid_from_cells

    root = resolve_data_root(a.data_root)
    model, ck = load_ckpt(a.ckpt, a.device)
    oof = fold_oof(a.ckpt, ck)
    oof["cat"] = categorise(oof, a.threshold)
    sel = oof[oof.cat == "TP"].sample(n=min(a.n, int((oof.cat == "TP").sum())), random_state=a.seed)
    settings = [("clean", 0)] + [(s.split(":")[0], int(s.split(":")[1])) for s in a.corruptions.split(",")]
    cells = []
    for name, sev in settings:
        imgs = []
        for r in sel.itertuples():
            im = read_rgb224(root, r.image_id, r.label)
            imgs.append(im if name == "clean" else corrupt(im, name, sev, setting_seed(r.image_id, name, sev)))
        cams, probs = cams_for(model, imgs, a.target, a.device)
        tag = name if name == "clean" else f"{name}@{sev}"
        cells += [_cell(im, cm, f"p={p:.2f} {tag}", (30, 80, 160)) for im, cm, p in zip(imgs, cams, probs)]
    title = f"Grad-CAM {_name(ck)}: fold-{ck['fold']} val fakes (TP when clean), rows = clean + corruptions"
    out = Path(a.out) if a.out else FIG_DIR / f"gradcam_corrupt_{Path(a.ckpt).parent.parent.name}_fold{ck['fold']}.jpg"
    grid_from_cells(cells, out, title, cols=len(sel), gap=4)
    print(f"-> {out}")


def cmd_compare(a) -> None:
    from .viz import _tile, grid_from_cells

    root = resolve_data_root(a.data_root)
    labels = a.labels or [Path(c).parent.parent.name for c in a.ckpt]
    if len(labels) != len(a.ckpt):
        raise SystemExit("--labels must match --ckpt")
    models = [load_ckpt(c, a.device) for c in a.ckpt]
    if len({ck["fold"] for _, ck in models}) != 1:
        raise SystemExit("all checkpoints must come from the same fold (same validation images)")
    oof = fold_oof(a.ckpt[0], models[0][1])
    fakes = oof[oof.label == 1].sample(n=a.n, random_state=a.seed)
    imgs = [read_rgb224(root, r.image_id, 1) for r in fakes.itertuples()]
    cells = [_tile(Image.fromarray(im), "val fake", (200, 40, 40), size=160) for im in imgs]
    for (m, _), lab in zip(models, labels):
        cams, probs = cams_for(m, imgs, a.target, a.device)
        cells += [_tile(Image.fromarray(overlay(im, cm)), f"{lab} p={p:.2f}", (60, 60, 60), size=160)
                  for im, cm, p in zip(imgs, cams, probs)]
    out = Path(a.out) if a.out else FIG_DIR / f"gradcam_compare_{'_vs_'.join(labels)}.jpg"
    grid_from_cells(cells, out, f"Grad-CAM before/after on {a.n} fold-{models[0][1]['fold']} val fakes: rows = image, "
                    + ", ".join(labels), cols=a.n, gap=4)
    print(f"-> {out}")


SELFTEST_ARCHS = (
    ("efficientnet_b0", 224), ("efficientnet_b4", 320), ("convnext_small", 224),
    ("vit_base_patch16_224", 224), ("swin_small_patch4_window7_224", 224),
)


def cmd_selftest(a) -> None:
    """Every backbone family used here: random-init (pretrained=False, no download), random non-zero head, 2 random
    images -> CAM shape (2,224,224), finite, in [0,1], not all zero; CAM resolution printed."""
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    for name, size in SELFTEST_ARCHS:
        m = Classifier(name, pretrained=False, input_size=size).eval()
        head = m.net.get_classifier()
        torch.nn.init.normal_(head.weight, std=0.05)  # Classifier zero-inits the head -> zero gradients otherwise
        gc = GradCAM(m, a.target)
        cam, p = gc(x)
        _, feat, to_nchw = gc._forward(x)
        grid = tuple(to_nchw(feat.detach()).shape[-2:])
        ok = cam.shape == (2, 224, 224) and np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1 and cam.max() > 0
        print(f"{'OK ' if ok else 'BAD'} {name:<32} input {size}  CAM grid {grid}  p={[round(float(v), 3) for v in p]}")
        if not ok:
            raise SystemExit(f"selftest failed for {name}")
    print("selftest passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_root")
    common.add_argument("--device", default=None, help="default: $DFD_DEVICE, else cuda if available")
    common.add_argument("--threads", type=int, default=4, help="CPU threads (keep low while the GPU trains)")
    common.add_argument("--target", choices=["pred", "fake"], default="pred")
    common.add_argument("--threshold", type=float, default=0.5)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--out")
    g = sub.add_parser("grid", parents=[common])
    g.add_argument("--ckpt", required=True)
    g.add_argument("--per_class", type=int, default=4)
    g.add_argument("--pick", choices=["random", "confident"], default="random")
    c = sub.add_parser("corrupt", parents=[common])
    c.add_argument("--ckpt", required=True)
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--corruptions", default="snow:2,block_patches:2,pixelate:2")
    m = sub.add_parser("compare", parents=[common])
    m.add_argument("--ckpt", nargs="+", required=True)
    m.add_argument("--labels", nargs="*")
    m.add_argument("--n", type=int, default=8)
    sub.add_parser("selftest", parents=[common])
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.device is None:
        from .evaluate import default_device

        a.device = default_device()
    {"grid": cmd_grid, "corrupt": cmd_corrupt, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
