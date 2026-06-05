"""
scripts/download_texts.py
--------------------------
Downloads freely available source texts into texts/plato/ and texts/feynman/.

Sources:
  Plato:             Project Gutenberg (public domain, Benjamin Jowett translations)
  Feynman essays:    Caltech Archives + Nobelprize.org
  Feynman Lectures:  feynmanlectures.caltech.edu (free to read; personal use)

Usage:
  python scripts/download_texts.py                    # everything
  python scripts/download_texts.py --plato-only
  python scripts/download_texts.py --feynman-only     # essays + lectures
  python scripts/download_texts.py --lectures-only    # just FLP (all 3 volumes)
  python scripts/download_texts.py --lectures-only --vol 1   # just Vol I

After this, run:
  python scripts/ingest.py --dry-run    # preview chunk counts
  python scripts/ingest.py              # build the DB (requires API key)
"""

from __future__ import annotations

import html
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_URL     = "https://feynmanlectures.caltech.edu"
HEADERS      = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Plato -- Project Gutenberg plain text (public domain)
# ---------------------------------------------------------------------------
PLATO_SOURCES = [
    ("republic",   "https://www.gutenberg.org/cache/epub/1497/pg1497.txt"),
    ("apology",    "https://www.gutenberg.org/cache/epub/1656/pg1656.txt"),
    ("meno",       "https://www.gutenberg.org/cache/epub/1643/pg1643.txt"),
    ("phaedo",     "https://www.gutenberg.org/cache/epub/1658/pg1658.txt"),
    ("symposium",  "https://www.gutenberg.org/cache/epub/1600/pg1600.txt"),
]

# ---------------------------------------------------------------------------
# Feynman essays -- freely available from official sources
# ---------------------------------------------------------------------------
FEYNMAN_ESSAYS = [
    (
        "cargo_cult_science",
        "https://calteches.library.caltech.edu/51/2/CargoCult.htm",
    ),
    (
        "nobel_lecture_1965",
        "https://www.nobelprize.org/prizes/physics/1965/feynman/lecture/",
    ),
]

# ---------------------------------------------------------------------------
# Feynman Lectures on Physics -- chapter counts per volume
# ---------------------------------------------------------------------------
FLP_VOLUMES = {
    1: ("I",   52),   # Mechanics, Radiation, Heat
    2: ("II",  42),   # Electromagnetism and Matter
    3: ("III", 21),   # Quantum Mechanics
}


# ---------------------------------------------------------------------------
# HTML stripper (stdlib only)
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "br", "li", "tr"):
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._buf.append(data)

    def get_text(self) -> str:
        text = html.unescape("".join(self._buf))
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _strip_html(raw: str) -> str:
    p = _HTMLStripper()
    p.feed(raw)
    return p.get_text()


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF"
    end_marker   = "*** END OF"
    start = text.find(start_marker)
    if start != -1:
        start = text.index("\n", start) + 1
    else:
        start = 0
    end = text.find(end_marker)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# FLP-specific extractor
# ---------------------------------------------------------------------------

def _extract_flp_chapter(html_text: str) -> tuple[str, str]:
    """
    Extracts clean prose text from a Feynman Lectures HTML page.

    Returns (title, body_text).

    Strategy:
      - Title:  from <title> tag
      - Body:   all <p class="p"> paragraphs + <h3 class="section-title"> headings
      - Equations: $...$ and $$...$$ replaced with [eq] -- prose is what matters
      - Everything else (nav, figures, captions, audio): skipped
    """
    # Title
    title_m = re.search(r"<title>([^<]+)</title>", html_text)
    title   = html.unescape(title_m.group(1).strip()) if title_m else "Unknown"

    # Section headings
    headings = re.findall(
        r'<h[23][^>]*class="section-title"[^>]*>(.*?)</h[23]>',
        html_text, re.DOTALL
    )

    # Paragraphs -- <p class="p"> is the main prose container
    paragraphs = re.findall(
        r'<p[^>]*class="p"[^>]*>(.*?)</p>',
        html_text, re.DOTALL
    )

    def clean(fragment: str) -> str:
        # Strip all tags
        fragment = re.sub(r"<[^>]+>", " ", fragment)
        # Decode HTML entities
        fragment = html.unescape(fragment)
        # Strip LaTeX math: $$...$$  then $...$
        fragment = re.sub(r"\$\$[^$]+\$\$", "[eq]", fragment)
        fragment = re.sub(r"\$[^$\n]+\$",   "[eq]", fragment)
        # Collapse whitespace
        fragment = re.sub(r"[ \t]+", " ", fragment).strip()
        return fragment

    heading_texts   = [clean(h) for h in headings]
    paragraph_texts = [clean(p) for p in paragraphs if clean(p)]

    # Interleave headings with paragraphs is too complex without a full parser;
    # just put headings at the top as a section list, body text after.
    body_parts = []
    if heading_texts:
        body_parts.append("Sections: " + " | ".join(heading_texts))
        body_parts.append("")
    body_parts.extend(paragraph_texts)

    return title, "\n\n".join(body_parts)


# ---------------------------------------------------------------------------
# Fetch helper
# ---------------------------------------------------------------------------

def _fetch(url: str, label: str, retries: int = 3) -> str | None:
    """Fetch with automatic retry on 403/429 rate-limiting (backoff: 5s, 15s, 30s)."""
    import urllib.error
    delays = [5, 15, 30]
    print(f"  Fetching {label} ...", end=" ", flush=True)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                enc = resp.headers.get_content_charset() or "utf-8"
                print("OK", flush=True)
                return raw.decode(enc, errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < retries:
                wait = delays[attempt]
                print(f"throttled, waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"FAIL ({e})", flush=True)
                return None
        except Exception as e:
            print(f"FAIL ({e})", flush=True)
            return None


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------

def download_plato(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nPlato -- Project Gutenberg (Jowett translations)")
    downloaded = 0
    for name, url in PLATO_SOURCES:
        if (out_dir / f"{name}.txt").exists():
            print(f"  Skip {name}.txt (already exists)")
            downloaded += 1
            continue
        time.sleep(1)
        raw = _fetch(url, name)
        if raw is None:
            continue
        text = _strip_gutenberg_boilerplate(raw)
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"  Saved {name}.txt ({len(text):,} chars)", flush=True)
        downloaded += 1
    return downloaded


def download_feynman_essays(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nFeynman essays -- Caltech Archives + Nobelprize.org")
    downloaded = 0
    for name, url in FEYNMAN_ESSAYS:
        if (out_dir / f"{name}.txt").exists():
            print(f"  Skip {name}.txt (already exists)")
            downloaded += 1
            continue
        time.sleep(1)
        raw = _fetch(url, name)
        if raw is None:
            continue
        is_html = any(url.endswith(x) for x in (".htm", ".html", "/"))
        text    = _strip_html(raw) if is_html else raw
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"  Saved {name}.txt ({len(text):,} chars)", flush=True)
        downloaded += 1
    return downloaded


def download_feynman_lectures(out_dir: Path, volumes: list[int] | None = None) -> int:
    """
    Downloads the Feynman Lectures on Physics from feynmanlectures.caltech.edu.
    One .txt file per chapter. Equations replaced with [eq].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vols = volumes or list(FLP_VOLUMES.keys())

    total = 0
    for vol_num in vols:
        roman, chapter_count = FLP_VOLUMES[vol_num]
        print(f"\nFLP Volume {vol_num} ({roman}) -- {chapter_count} chapters")

        for ch in range(1, chapter_count + 1):
            filename = f"vol{vol_num}_ch{ch:02d}.txt"
            out_path = out_dir / filename
            if out_path.exists():
                print(f"  Skip {filename} (already exists)")
                total += 1
                continue

            url   = f"{BASE_URL}/{roman}_{ch:02d}.html"
            label = f"Vol {vol_num} Ch {ch:02d}"
            time.sleep(1.5)

            raw = _fetch(url, label)
            if raw is None:
                continue

            title, body = _extract_flp_chapter(raw)
            text = f"{title}\n{'=' * len(title)}\n\n{body}"
            out_path.write_text(text, encoding="utf-8")
            print(f"  Saved {filename}  {len(body):,} chars", flush=True)
            total += 1

    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args          = sys.argv[1:]
    plato_only    = "--plato-only"    in args
    feynman_only  = "--feynman-only"  in args
    lectures_only = "--lectures-only" in args

    # --vol 1  /  --vol 2  /  --vol 3  (only meaningful with --lectures-only)
    vol_arg  = None
    if "--vol" in args:
        idx     = args.index("--vol")
        vol_arg = [int(args[idx + 1])] if idx + 1 < len(args) else None

    feynman_dir  = PROJECT_ROOT / "texts" / "feynman"
    plato_dir    = PROJECT_ROOT / "texts" / "plato"

    total = 0

    if not feynman_only and not lectures_only:
        total += download_plato(plato_dir)

    if not plato_only:
        if not lectures_only:
            total += download_feynman_essays(feynman_dir)
        total += download_feynman_lectures(feynman_dir, volumes=vol_arg)

    # Summary
    print(f"\n{'-'*52}")
    print(f"Done. {total} file(s) written/skipped.")
    print("\nFiles on disk:")
    for d in (plato_dir, feynman_dir):
        if d.exists():
            files = sorted(d.glob("*.txt"))
            if files:
                size_mb = sum(f.stat().st_size for f in files) / 1_048_576
                print(f"  {d.relative_to(PROJECT_ROOT)}/  "
                      f"({len(files)} files, {size_mb:.1f} MB)")

    print(
        "\nNext steps:\n"
        "  python scripts/ingest.py --dry-run   # preview\n"
        "  python scripts/ingest.py             # build DB (~$3-5)\n"
        "  python app.py                         # start agent\n"
    )


if __name__ == "__main__":
    main()
