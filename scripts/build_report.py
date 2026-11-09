#!/usr/bin/env python
"""Render the technical report (Markdown) to a print-ready PDF (docs/PHASE6.md §2).

  python scripts/build_report.py                                   # docs/report/report.md -> outputs/report/report.pdf
  python scripts/build_report.py --src docs/report/report.md --out outputs/report/report.pdf [--show_todos]
  cat report.md | python scripts/build_report.py --src - --out /tmp/report.pdf   # from stdin (relative images
                                                                                 # then resolve against --base)

Pipeline (no new installs): python-markdown (already in requirements.txt via tensorboard) -> standalone HTML with
print CSS (A4, page numbers via @page margin boxes) -> PDF with a headless Chromium-family browser
(google-chrome / chromium / chromium-browser / msedge, `--print-to-pdf`). If no browser is found, WeasyPrint is used
when importable; otherwise only the HTML is written and the script says so (open it and "Print to PDF").

`<!-- TODO(...) -->` placeholders are HTML comments: invisible in the PDF by default. --show_todos renders them as
yellow boxes (draft review). The script always prints how many TODOs remain and the PDF page count (the README
asks for >= 4 pages).
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "docs" / "report" / "report.md"
DEFAULT_OUT = ROOT / "outputs" / "report" / "report.pdf"
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge")
TODO_RE = re.compile(r"<!--\s*(TODO\([^)]*\):?.*?)-->", re.S)

CSS = """
@page { size: A4; margin: 16mm 15mm 17mm 15mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8.5pt 'DejaVu Sans', Arial, sans-serif; color: #666; } }
html { font-size: 10pt; }
body { font-family: 'DejaVu Serif', 'Liberation Serif', Georgia, serif; line-height: 1.38; color: #111; margin: 0; }
h1 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 17pt; margin: 0 0 4pt; }
h2 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 12.5pt; margin: 14pt 0 5pt; border-bottom: 1px solid #bbb;
     padding-bottom: 2pt; break-after: avoid; }
h3 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; margin: 10pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; text-align: justify; hyphens: auto; }
ul, ol { padding-left: 16pt; margin: 3pt 0; }
hr { border: 0; border-top: 1px solid #ccc; margin: 8pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt; font-size: 8.3pt; line-height: 1.25; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #eef1f5; font-family: 'DejaVu Sans', Arial, sans-serif; }
code { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; font-size: 8.3pt; background: #f4f4f4; padding: 0 1.5pt; }
pre { background: #f6f7f9; border: 1px solid #e1e4e8; padding: 5pt 6pt; font-size: 7.6pt; line-height: 1.3;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
img { max-width: 100%; display: block; margin: 4pt auto; break-inside: avoid; }
em { color: #333; }
.todo { border: 1px dashed #c9a400; background: #fff8d6; color: #5c4b00; font: 8pt 'DejaVu Sans Mono', monospace;
        padding: 3pt 5pt; margin: 4pt 0; white-space: pre-wrap; }
"""


def find_browser() -> str | None:
    for b in BROWSERS:
        p = shutil.which(b)
        if p:
            return p
    return None


def md_to_html(text: str, base: Path, title: str, show_todos: bool) -> str:
    try:
        import markdown
    except ImportError:
        sys.exit("python-markdown is not installed (pip install markdown; it is pinned in requirements.txt)")
    if show_todos:
        text = TODO_RE.sub(lambda m: f'\n<div class="todo">{html.escape(m.group(1).strip())}</div>\n', text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"], output_format="html")
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<base href='{base.resolve().as_uri()}/'><title>{html.escape(title)}</title>"
        f"<style>{CSS}</style></head><body>\n{body}\n</body></html>\n"
    )


def pdf_pages(pdf: Path) -> int:
    data = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?!s)", data))


def render_pdf(html_path: Path, out: Path) -> str:
    browser = find_browser()
    if browser:
        with tempfile.TemporaryDirectory(prefix="dfd_report_") as prof:
            cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                   f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                   html_path.resolve().as_uri()]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            return Path(browser).name
        print(f"[warn] {browser} failed (exit {r.returncode}): {r.stderr.strip()[-500:]}", file=sys.stderr)
    try:
        from weasyprint import HTML  # optional; not in requirements.txt
    except ImportError:
        return ""
    HTML(filename=str(html_path)).write_pdf(str(out))
    return "weasyprint"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="Markdown source, or - for stdin")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output PDF (the .html is written next to it)")
    ap.add_argument("--base", help="directory that relative image paths resolve against (default: the source's dir)")
    ap.add_argument("--title", default="Deepfake detection - technical report")
    ap.add_argument("--show_todos", action="store_true", help="render TODO placeholders as visible boxes")
    ap.add_argument("--min_pages", type=int, default=4, help="warn if the PDF is shorter (README rule: >= 4 pages)")
    a = ap.parse_args()

    if a.src == "-":
        text, base = sys.stdin.read(), Path(a.base or DEFAULT_SRC.parent)
    else:
        src = Path(a.src)
        if not src.is_file():
            sys.exit(f"source not found: {src}")
        text, base = src.read_text(encoding="utf-8"), Path(a.base or src.parent)
    todos = [t.split("\n")[0][:90] for t in TODO_RE.findall(text)]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_path = out.with_suffix(".html")
    html_path.write_text(md_to_html(text, base, a.title, a.show_todos), encoding="utf-8")
    print(f"wrote {html_path}")

    engine = render_pdf(html_path, out)
    if not engine:
        print("no PDF engine found (google-chrome/chromium or weasyprint); open the HTML above and print it to PDF")
        return 1
    n = pdf_pages(out)
    print(f"wrote {out} via {engine}: {n} pages, {out.stat().st_size / 1024:.0f} KiB")
    if n < a.min_pages:
        print(f"[warn] {n} pages < required {a.min_pages}")
    print(f"TODO placeholders remaining: {len(todos)}{' (hidden in the PDF; --show_todos to see them)' if todos and not a.show_todos else ''}")
    for t in todos:
        print(f"  - {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
