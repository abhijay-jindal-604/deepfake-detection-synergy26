"""Build the single weights bundle `model_weights.pt` for inference_script.py (docs/PHASE5.md §4.1).

  # from the ensemble decision (members, weights, temperatures, threshold, TTA flag):
  python scripts/bundle.py --ensemble outputs/ensemble/ensemble.json --source full --out outputs/bundle/model_weights.pt
  # fallback (b-lite, PHASE5 §3): average a member's fold-0 and fold-1 models (2 entries, weight w/2 each)
  python scripts/bundle.py --ensemble outputs/ensemble/ensemble.json --source fold0,fold1 \
      --member_source p3_vitb=fold0,fold1
  # ad hoc (equal weights, T = 1, threshold 0.5 unless given):
  python scripts/bundle.py --member p2_ref:fold0 --member p2_corr_p09:fold0 --out /tmp/x/model_weights.pt

Each member checkpoint is <runs_dir>/<run>/<source>/<ckpt>.pt (source = full | foldK). Stored per entry: timm name
(models are rebuilt with pretrained=False), input_size, drop_rate, state_dict (EMA weights when present,
`mask_head.*` stripped, FP32 by default; --dtype fp16 halves the size but FP16-rounded weights alone shift
B0 logits by up to ~0.09), temperature, weight. Top level: T_ens,
threshold, tta_flip, mean/std, image_size, versions, git commit, creation date. Only tensors and plain Python
types, so it loads with torch.load(weights_only=True).

After saving, the bundle is re-loaded and every member is rebuilt twice with load_state_dict(strict=True):
as dfd.models.Classifier and through inference_script.build_members (what the organisers run).
$DFD_RUNS_DIR (or --runs_dir) selects the runs directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfd.models import IMAGENET_MEAN, IMAGENET_STD, Classifier, strip_mask_head  # noqa: E402
from dfd.paths import RUNS_DIR  # noqa: E402

DATA_SIZE = 224


def load_inference_module():
    """Import the repo-root inference_script.py as a module (it is not part of the dfd package)."""
    spec = importlib.util.spec_from_file_location("inference_script", ROOT / "inference_script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def member_entry(ckpt_path: Path, weight: float, temperature: float, dtype: str = "fp16", tag: str = "") -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    use_ema = "state_dict_ema" in ck
    sd = strip_mask_head(ck["state_dict_ema"] if use_ema else ck["state_dict"])
    if dtype == "fp16":
        sd = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}
    else:
        sd = {k: v.contiguous() for k, v in sd.items()}
    return {
        "timm_name": str(m["name"]),
        "input_size": int(m.get("input_size", DATA_SIZE)),
        "drop_rate": 0.0,  # inference only: dropout is inactive in eval mode anyway
        "state_dict": sd,
        "temperature": float(temperature),
        "weight": float(weight),
        "source": tag or str(ckpt_path),
        "weights": "ema" if use_ema else "raw",
        "epoch": int(ck.get("epoch", -1)),
    }


def make_bundle(entries: list[dict], threshold: float, temperature_ens: float, tta_flip: bool,
                extra: dict | None = None) -> dict:
    return {
        "format_version": 1,
        "members": entries,
        "temperature_ens": float(temperature_ens),
        "threshold": float(threshold),
        "tta_flip": bool(tta_flip),
        "mean": [float(v) for v in IMAGENET_MEAN],
        "std": [float(v) for v in IMAGENET_STD],
        "image_size": DATA_SIZE,
        "versions": {"torch": str(torch.__version__), "timm": str(timm.__version__)},
        "train_git_commit": git_commit(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }


def verify_bundle(path: Path) -> dict:
    """Strict reload through dfd.models.Classifier and through inference_script; returns the loaded bundle."""
    b = torch.load(path, map_location="cpu", weights_only=True)
    for i, m in enumerate(b["members"]):
        model = Classifier(m["timm_name"], pretrained=False, input_size=m["input_size"])
        sd = {k: v.float() if v.is_floating_point() else v for k, v in m["state_dict"].items()}
        model.load_state_dict(sd, strict=True)
    inf = load_inference_module()
    inf.build_members(inf.load_bundle(path), "cpu")
    return b


def size_report(path: Path, b: dict) -> str:
    lines = [f"{path}: {path.stat().st_size / 2**20:.1f} MiB, {len(b['members'])} member entr{'y' if len(b['members']) == 1 else 'ies'}"]
    for m in b["members"]:
        n = sum(v.numel() for v in m["state_dict"].values() if v.is_floating_point())
        nb = sum(v.numel() * v.element_size() for v in m["state_dict"].values())
        lines.append(f"  {m['timm_name']:<40} @{m['input_size']} {n / 1e6:6.2f} M params {nb / 2**20:7.1f} MiB "
                     f"w {m['weight']:.3f} T {m['temperature']:.3f} [{m['weights']}, ep {m['epoch']}] {m['source']}")
    lines.append(f"  T_ens {b['temperature_ens']:.3f} | threshold {b['threshold']:.4f} | tta_flip {b['tta_flip']}")
    return "\n".join(lines)


def _sources(spec: str) -> list[str]:
    out = [s.strip() for s in spec.split(",") if s.strip()]
    for s in out:
        if s != "full" and not (s.startswith("fold") and s[4:].isdigit()):
            raise SystemExit(f"bad source {s!r} (full | foldK)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", help="ensemble.json from dfd.ensemble (members, weights, temperatures, threshold)")
    ap.add_argument("--member", action="append", default=[],
                    help="RUN[:SOURCE] (repeatable, without --ensemble): equal weights, T = 1")
    ap.add_argument("--source", default="full", help="checkpoint dir per member: full | foldK | fold0,fold1 (averaged)")
    ap.add_argument("--member_source", action="append", default=[], help="RUN=SOURCE override, e.g. p3_vitb=fold0,fold1")
    ap.add_argument("--ckpt", default="last", choices=["last", "best"])
    ap.add_argument("--runs_dir", default=None, help="default: $DFD_RUNS_DIR or outputs/runs")
    ap.add_argument("--threshold", type=float, help="override the ensemble threshold")
    tta = ap.add_mutually_exclusive_group()
    tta.add_argument("--tta_flip", dest="tta_flip", action="store_true", default=None)
    tta.add_argument("--no_tta_flip", dest="tta_flip", action="store_false")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"],
                    help="weight storage; fp32 default: FP16-rounded weights alone move EfficientNet logits by ~0.09")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "bundle" / "model_weights.pt"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else RUNS_DIR
    overrides = {}
    for s in args.member_source:
        run, _, src = s.partition("=")
        overrides[run] = _sources(src)

    plan = []  # (run, weight, temperature, [sources])
    extra = {}
    if args.ensemble:
        if args.member:
            raise SystemExit("--member and --ensemble are exclusive")
        ens = json.loads(Path(args.ensemble).read_text())
        for m in ens["members"]:
            plan.append((m["run"], float(m["weight"]), float(m["temperature"]),
                         overrides.get(m["run"], _sources(args.source))))
        t_ens = float(ens.get("temperature_ens", 1.0))
        threshold = float(ens.get("threshold", 0.5))
        tta_flip = ens.get("variant", "plain") == "tta_flip"
        extra["ensemble_json"] = {k: ens[k] for k in ("objective", "variant", "folds") if k in ens}
    else:
        if not args.member:
            raise SystemExit("give --ensemble or at least one --member")
        for spec in args.member:
            run, _, src = spec.partition(":")
            plan.append((run, 1.0 / len(args.member), 1.0, _sources(src) if src else overrides.get(run, _sources(args.source))))
        t_ens, threshold, tta_flip = 1.0, 0.5, False
    if args.threshold is not None:
        threshold = args.threshold
    if args.tta_flip is not None:
        tta_flip = args.tta_flip

    entries = []
    for run, w, t, srcs in plan:
        for src in srcs:
            p = runs_dir / run / src / f"{args.ckpt}.pt"
            if not p.is_file():
                raise SystemExit(f"missing checkpoint {p}")
            if src == "full" and not (p.parent / "done.json").is_file():
                print(f"WARNING: {p.parent / 'done.json'} missing: full-data run may be unfinished")
            entries.append(member_entry(p, w / len(srcs), t, args.dtype, tag=f"{run}/{src}/{args.ckpt}.pt"))

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle(entries, threshold, t_ens, tta_flip, extra)
    tmp = out.with_suffix(".tmp")
    torch.save(bundle, tmp)
    tmp.replace(out)
    b = verify_bundle(out)
    print("strict reload OK (dfd.models.Classifier + inference_script)")
    print(size_report(out, b))


if __name__ == "__main__":
    main()
