"""Audit visualisations: pair grids, image grids with captions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_NAME = {0: "REAL", 1: "FAKE"}
LABEL_COLOUR = {0: (40, 160, 60), 1: (200, 40, 40)}


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _tile(img: Image.Image, caption: str, colour=(0, 0, 0), size: int = 224, bar: int = 22) -> Image.Image:
    t = Image.new("RGB", (size, size + bar), "white")
    t.paste(img.convert("RGB").resize((size, size)), (0, bar))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, size, bar], fill=colour)
    d.text((4, 3), caption, fill="white", font=_font(13))
    return t


def pair_grid(pairs, id2path: dict, out: Path, title: str = "", cols: int = 4) -> None:
    """Each cell = left/right images of one candidate pair, captioned with label and scores."""
    cells = []
    for r in pairs.itertuples():
        a = _tile(Image.open(id2path[r.id_a]), f"{LABEL_NAME[r.label_a]}  cos {r.cos_sim:.3f}", LABEL_COLOUR[r.label_a])
        b = _tile(Image.open(id2path[r.id_b]), f"{LABEL_NAME[r.label_b]}  pH {r.phash_dist}", LABEL_COLOUR[r.label_b])
        cell = Image.new("RGB", (a.width * 2 + 4, a.height), "white")
        cell.paste(a, (0, 0))
        cell.paste(b, (a.width + 4, 0))
        cells.append(cell)
    grid_from_cells(cells, out, title, cols=cols, gap=14)


def grid_from_cells(cells: list[Image.Image], out: Path, title: str = "", cols: int = 4, gap: int = 6) -> None:
    if not cells:
        return
    w, h = cells[0].size
    rows = int(np.ceil(len(cells) / cols))
    top = 30 if title else 0
    canvas = Image.new("RGB", (cols * (w + gap) + gap, top + rows * (h + gap) + gap), (235, 235, 235))
    if title:
        ImageDraw.Draw(canvas).text((gap, 6), title, fill="black", font=_font(18))
    for k, c in enumerate(cells):
        r, cc = divmod(k, cols)
        canvas.paste(c, (gap + cc * (w + gap), top + gap + r * (h + gap)))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)


def valc_examples(img_rgb: np.ndarray, image_id: str, out: Path, tile: int = 160) -> None:
    """Rows = corruptions (held-out marked), cols = clean + severities 1..3."""
    from .corruptions import ALL, HELD_OUT, SEVERITIES, corrupt, setting_seed

    cells = []
    for name in ALL:
        tag = " [held-out]" if name in HELD_OUT else ""
        row = [_tile(Image.fromarray(img_rgb), f"{name}{tag}"[:24], (90, 90, 90) if tag else (30, 80, 160), size=tile)]
        for s in SEVERITIES:
            im = corrupt(img_rgb, name, s, setting_seed(image_id, name, s))
            row.append(_tile(Image.fromarray(im), f"severity {s}", (60, 60, 60), size=tile))
        cells.extend(row)
    grid_from_cells(cells, out, "val-C: clean | severity 1 | 2 | 3  (grey header = held out from training)", cols=1 + len(SEVERITIES), gap=4)


if __name__ == "__main__":
    import argparse

    import cv2
    import pandas as pd

    from .paths import AUDIT_DIR, DATA_DIR, resolve_data_root, train_image_path

    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["valc"])
    p.add_argument("--data_root")
    p.add_argument("--index", type=int, default=0, help="which fold-0 validation image")
    a = p.parse_args()
    root = resolve_data_root(a.data_root)
    folds = pd.read_csv(DATA_DIR / "folds.csv", dtype={"image_id": str})
    r = folds[folds.fold == 0].iloc[a.index]
    img = cv2.cvtColor(cv2.imread(str(train_image_path(root, r.image_id, r.label))), cv2.COLOR_BGR2RGB)
    valc_examples(img, r.image_id, AUDIT_DIR / "valC_examples.jpg")
    print("wrote", AUDIT_DIR / "valC_examples.jpg")
