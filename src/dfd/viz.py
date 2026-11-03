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
