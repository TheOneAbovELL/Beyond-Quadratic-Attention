"""Assemble docs/book/NN_*.md into one Markdown file, a self-contained HTML file and (if Edge is present) a PDF.

    python scripts/build_book.py [--no-pdf]

Outputs (all local):
    docs/book/Beyond_Quadratic_Attention_Book.md
    docs/book/Beyond_Quadratic_Attention_Book.html   (figures embedded as base64; opens offline)
    docs/book/Beyond_Quadratic_Attention_Book.pdf    (Microsoft Edge headless print, if available)
"""
import argparse
import base64
import glob
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

import mistune

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "docs", "book")
NAME = "Beyond_Quadratic_Attention_Book"
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1b1b1a; --muted:#5a5955; --rule:#dcdad4; --accent:#2a78d6; --box:#f4f3ef; --code:#f6f6f4; }
html { background:#ffffff; }
body { font-family: Georgia, 'Times New Roman', serif; color: var(--ink); background:#ffffff;
       font-size: 11pt; line-height: 1.5; max-width: 820px; margin: 0 auto; padding: 0 16px; }
h1 { font-family: 'Segoe UI', Arial, sans-serif; font-size: 22pt; margin: 0 0 10pt; padding-top: 6pt;
     border-bottom: 2px solid var(--accent); page-break-before: always; break-before: page; }
h1.first { page-break-before: avoid; break-before: avoid; }
h2 { font-family: 'Segoe UI', Arial, sans-serif; font-size: 15pt; margin-top: 18pt; color: #10233f; }
h3 { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12.5pt; margin-top: 14pt; }
p, li { text-align: left; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt; background: var(--code);
       padding: 0 2px; border-radius: 2px; }
pre { background: var(--code); border: 1px solid var(--rule); padding: 8px 10px; overflow-x: auto;
      font-size: 9pt; line-height: 1.35; page-break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
blockquote { background: var(--box); border-left: 4px solid var(--accent); margin: 10pt 0; padding: 6pt 10pt;
             page-break-inside: avoid; }
blockquote p { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.2pt;
        page-break-inside: auto; }
th, td { border: 1px solid var(--rule); padding: 3px 6px; vertical-align: top; }
th { background: #eef2f8; font-family: 'Segoe UI', Arial, sans-serif; }
tr { page-break-inside: avoid; }
img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
.figcap { text-align: center; color: var(--muted); font-size: 9pt; margin-top: -4pt; }
hr { border: none; border-top: 1px solid var(--rule); }
nav.toc { page-break-after: always; break-after: page; }
nav.toc ol { list-style: none; padding-left: 0; }
nav.toc li.h1 { margin-top: 5pt; font-weight: bold; }
nav.toc li.h2 { padding-left: 18px; font-size: 9.5pt; }
nav.toc a { color: var(--ink); text-decoration: none; }
.titlepage { text-align: center; padding-top: 120pt; page-break-after: always; break-after: page; }
.titlepage h1 { border: none; font-size: 30pt; page-break-before: avoid; }
.titlepage .sub { font-size: 14pt; color: var(--muted); margin-top: 12pt; }
.titlepage .meta { margin-top: 60pt; font-size: 10.5pt; color: var(--muted); }
"""


def slug(text: str, used: dict) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "section"
    n = used.get(s, 0)
    used[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def embed_images(html_text: str) -> str:
    def repl(m):
        src, alt = m.group(1), m.group(2)
        path = os.path.normpath(os.path.join(BOOK, src))
        if not os.path.isfile(path):
            return f'<p class="figcap">[missing figure: {html.escape(src)}]</p>'
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{alt}" /><p class="figcap">{alt}</p>'
    return re.sub(r'<img src="([^"]+)" alt="([^"]*)" />', repl, html_text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    chapters = sorted(glob.glob(os.path.join(BOOK, "[0-9][0-9]_*.md")))
    if not chapters:
        print("no chapters found", file=sys.stderr)
        return 1
    parts = [open(p, encoding="utf-8").read().strip() for p in chapters]
    md_all = "\n\n---\n\n".join(parts) + "\n"
    with open(os.path.join(BOOK, NAME + ".md"), "w", encoding="utf-8") as f:
        f.write(md_all)

    render = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
    body_parts = []
    for i, text in enumerate(parts):
        if i == 0:
            continue  # front matter becomes the title page + "how to read" section below
        body_parts.append(render(text))
    front = render(parts[0])
    body = "\n".join(body_parts)

    used: dict = {}
    toc = []

    def add_anchor(m):
        level, inner = m.group(1), m.group(2)
        anchor = slug(inner, used)
        if level in ("1", "2"):
            toc.append((level, anchor, re.sub(r"<[^>]+>", "", inner)))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    body = re.sub(r"<h([123])>(.*?)</h\1>", add_anchor, body)
    body = re.sub(r"<hr />", "", body)
    toc_html = ['<nav class="toc"><h2>Contents</h2><ol>']
    for level, anchor, text in toc:
        toc_html.append(f'<li class="h{level}"><a href="#{anchor}">{html.escape(text)}</a></li>')
    toc_html.append("</ol></nav>")

    front_html = front.replace("<h1>", '<h1 class="first">', 1)
    title = ('<div class="titlepage"><h1>Beyond Quadratic Attention</h1>'
             '<div class="sub">The complete book of the project: what we built, why, how,<br/>'
             'what we found, and everything that went wrong along the way</div>'
             '<div class="meta">A local research laboratory for attention algorithms<br/>'
             'AMD Ryzen 5 7530U · CPU only · PyTorch 2.10 · October 2026</div></div>')
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            f'<title>Beyond Quadratic Attention — The Book</title><style>{CSS}</style></head><body>'
            f'{title}{"".join(toc_html)}<section>{front_html}</section>{body}</body></html>')
    page = embed_images(page)
    html_path = os.path.join(BOOK, NAME + ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", html_path, f"({len(page) / 1e6:.1f} MB)")

    if a.no_pdf:
        return 0
    edge = next((p for p in EDGE_CANDIDATES if os.path.isfile(p)), shutil.which("msedge"))
    if not edge:
        print("Microsoft Edge not found; skipping PDF (open the HTML and print to PDF instead).")
        return 0
    pdf_path = os.path.join(BOOK, NAME + ".pdf")
    url = "file:///" + html_path.replace("\\", "/")
    profile = tempfile.mkdtemp(prefix="book_edge_profile_")   # throwaway profile: never touches the user's Edge
    cmd = [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
           "--disable-sync", "--disable-background-networking", "--disable-component-update",
           "--disable-extensions", f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", url]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if os.path.isfile(pdf_path):
        with open(pdf_path, "rb") as f:
            pages = len(re.findall(rb"/Type\s*/Page[^s]", f.read()))
        print("wrote", pdf_path, f"({pages} pages)")
    else:
        print("PDF was not produced; open the HTML in a browser and print to PDF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
