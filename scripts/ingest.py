"""
scripts/ingest.py
-----------------
ONE-TIME script to build the semantic layer from source texts.
Run manually: python scripts/ingest.py

DO NOT import this from the agent at runtime.
DO NOT run this more than once per corpus unless adding new texts.

What it does:
  1. Reads all .txt files from texts/ subdirectories
  2. Chunks them into 500-token passages with 50-token overlap
  3. Embeds each passage locally (sentence-transformers, free)
  4. Extracts concept nodes via Haiku (one-time cost ~$3-5)
  5. Writes everything to philosopher.db

Usage:
  python scripts/ingest.py                          # ingest all texts/
  python scripts/ingest.py --source texts/plato/    # ingest one directory
  python scripts/ingest.py --role context           # set role for all files
  python scripts/ingest.py --thinker "Aristotle"   # tag all files with thinker

After running:
  Check node count: python scripts/ingest.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# Add project root to path so imports work when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import numpy as np
import nltk

from graph.graph_db import get_connection, init_schema, upsert_node, upsert_edge
from memory.retriever import embed_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 500     # target tokens per chunk (rough: 1 token ≈ 4 chars → 2000 chars)
CHUNK_OVERLAP = 50      # overlap tokens between adjacent chunks
CHUNK_CHARS   = CHUNK_SIZE * 4
OVERLAP_CHARS = CHUNK_OVERLAP * 4

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Thinker → source text directory mapping
THINKER_DIRS = {
    "plato":    "texts/plato",
    "feynman":  "texts/feynman",
    "context":  "texts/context",
    "neighbor": "texts/neighbors",
}


# ---------------------------------------------------------------------------
# Text cleaning — strips boilerplate before chunking
# ---------------------------------------------------------------------------

# Project Gutenberg header markers — everything before these is boilerplate
_GUTENBERG_START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "*END*THE SMALL PRINT",
    "NOTICE: THIS WORK MAY BE PROTECTED",
]

# Common Gutenberg/scan footer markers — everything after is boilerplate
_GUTENBERG_END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
]

# Lines that are almost certainly not philosophy
_SKIP_LINE_PREFIXES = [
    "Produced by",
    "Transcribed by",
    "Scanned by",
    "HTML version",
    "This eBook is for the use of",
    "Copyright (C)",
    "Copyright ©",
]


def clean_text(text: str) -> str:
    """
    Strips Project Gutenberg headers, footers, and scan boilerplate
    before chunking. Works on any plain-text philosophical source.

    Strategy:
      1. Strip Gutenberg START marker — keep only what follows.
      2. Strip Gutenberg END marker — keep only what precedes.
      3. Remove lines that are clearly metadata (Produced by, etc.).
      4. Collapse runs of blank lines to a single blank line.
      5. Strip leading/trailing whitespace.
    """
    # ── Step 1: find content start ────────────────────────────────────────
    for marker in _GUTENBERG_START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            # Skip past the marker line itself
            newline_after = text.find("\n", idx)
            if newline_after != -1:
                text = text[newline_after + 1:]
            break

    # ── Step 2: find content end ──────────────────────────────────────────
    for marker in _GUTENBERG_END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # ── Step 3: remove boilerplate lines ─────────────────────────────────
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in _SKIP_LINE_PREFIXES):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # ── Step 4: collapse excessive blank lines ────────────────────────────
    import re
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def download_nltk_data() -> None:
    """Download required NLTK data if not present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab", quiet=True)


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Splits text into overlapping chunks using sentence boundaries.
    Returns list of {text, source, chunk_index, char_start}.

    Sentence-aware chunking: never splits mid-sentence.
    Falls back to hard splits if sentences are very long.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on periods
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

    chunks = []
    current_chars = 0
    current_sentences: list[str] = []
    chunk_index = 0
    char_start = 0

    for sentence in sentences:
        sentence_chars = len(sentence)

        # If adding this sentence would exceed chunk size, save current chunk
        if current_chars + sentence_chars > CHUNK_CHARS and current_sentences:
            chunk_text_str = " ".join(current_sentences).strip()
            if chunk_text_str:
                chunks.append({
                    "text":        chunk_text_str,
                    "source":      source,
                    "chunk_index": chunk_index,
                    "char_start":  char_start,
                })
                chunk_index += 1

            # Overlap: keep last N chars worth of sentences
            overlap_sentences = []
            overlap_chars = 0
            for s in reversed(current_sentences):
                if overlap_chars + len(s) > OVERLAP_CHARS:
                    break
                overlap_sentences.insert(0, s)
                overlap_chars += len(s)

            current_sentences = overlap_sentences
            current_chars     = overlap_chars
            char_start        = char_start + current_chars

        current_sentences.append(sentence)
        current_chars += sentence_chars

    # Final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences).strip()
        if chunk_text_str:
            chunks.append({
                "text":        chunk_text_str,
                "source":      source,
                "chunk_index": chunk_index,
                "char_start":  char_start,
            })

    return chunks


# ---------------------------------------------------------------------------
# Concept extraction via Haiku
# ---------------------------------------------------------------------------

_CONCEPT_EXTRACTION_PROMPT = """\
Read this passage from a philosophical or scientific text and extract key concepts.
Return ONLY valid JSON with these keys:
{
  "concepts": ["list of key philosophical or scientific concepts as strings"],
  "definitions": [{"concept": "name", "definition": "brief one-sentence definition"}],
  "claims": ["list of key claims or arguments made in this passage"]
}

Keep lists short — 3-5 items maximum per key.
No preamble. No markdown fences. JSON only."""


def extract_concepts_batch(
    passages: list[str],
    client: anthropic.Anthropic,
    max_tokens: int = 400,
) -> list[dict]:
    """
    Extracts concepts from a batch of passages using Haiku.
    Returns list of extraction dicts (empty dict on failure).
    Uses single call per passage — batching to a single call
    risks context window overflow for large corpora.
    """
    results = []
    for i, passage in enumerate(passages):
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                system=_CONCEPT_EXTRACTION_PROMPT,
                messages=[{
                    "role": "user",
                    "content": passage[:2000]  # cap to avoid large context costs
                }],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            results.append(json.loads(raw))
        except Exception as e:
            log.warning("concept_extraction_failed passage=%d error=%s", i, str(e))
            results.append({})

    return results


# ---------------------------------------------------------------------------
# Main ingest pipeline
# ---------------------------------------------------------------------------

def ingest_directory(
    source_dir: Path,
    thinker: str,
    role: str,
    conn,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> dict:
    """
    Ingests all .txt files from source_dir.
    Returns stats dict.
    """
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        log.warning("no_txt_files dir=%s", source_dir)
        return {"files": 0, "chunks": 0, "nodes": 0}

    stats = {"files": 0, "chunks": 0, "nodes": 0, "cost_usd": 0.0}

    for txt_file in txt_files:
        log.info("ingesting file=%s thinker=%s role=%s", txt_file.name, thinker, role)
        raw_text = txt_file.read_text(encoding="utf-8", errors="replace")

        # Clean boilerplate before chunking
        text = clean_text(raw_text)
        if len(text) < 200:
            log.warning("  skipping %s — too little content after cleaning (%d chars)",
                        txt_file.name, len(text))
            continue

        chars_removed = len(raw_text) - len(text)
        if chars_removed > 500:
            log.info("  cleaned %d boilerplate chars from %s", chars_removed, txt_file.name)

        # Derive source name from filename
        source = txt_file.stem.replace("_", " ").title()

        # Chunk the text
        chunks = chunk_text(text, source)
        log.info("  chunked into %d passages", len(chunks))

        if dry_run:
            stats["files"]  += 1
            stats["chunks"] += len(chunks)
            continue

        # Embed all chunks in one batch call
        texts_to_embed = [c["text"] for c in chunks]
        embeddings     = embed_batch(texts_to_embed)
        log.info("  embedded %d passages", len(embeddings))

        # Extract concepts from a sample (every 10th chunk to control cost)
        sample_indices = list(range(0, len(chunks), 10))
        sample_texts   = [chunks[i]["text"] for i in sample_indices]
        concept_data   = extract_concepts_batch(sample_texts, client)

        # Write passage nodes
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node_id = f"passage:{thinker}:{txt_file.stem}:{chunk['chunk_index']}"
            upsert_node(
                conn,
                id        = node_id,
                type      = "passage",
                label     = chunk["text"],
                source    = f"{source}, passage {chunk['chunk_index']+1}",
                thinker   = thinker,
                role      = role,
                metadata  = {
                    "file":        txt_file.name,
                    "chunk_index": chunk["chunk_index"],
                    "char_start":  chunk["char_start"],
                },
                embedding = embeddings[i],
            )
            stats["nodes"] += 1

        # Write concept nodes from sampled extraction
        concept_count = 0
        for sample_idx, extraction in zip(sample_indices, concept_data):
            for concept_label in extraction.get("concepts", []):
                if not concept_label.strip():
                    continue
                concept_id = f"concept:{thinker}:{concept_label.lower().replace(' ', '_')}"

                # Only write if not already present (avoid duplicates across passages)
                existing = conn.execute(
                    "SELECT id FROM nodes WHERE id = ?", (concept_id,)
                ).fetchone()

                if not existing:
                    upsert_node(
                        conn,
                        id      = concept_id,
                        type    = "concept",
                        label   = concept_label,
                        source  = source,
                        thinker = thinker,
                        role    = role,
                    )
                    concept_count += 1

                # Link concept to passage
                passage_id = f"passage:{thinker}:{txt_file.stem}:{chunks[sample_idx]['chunk_index']}"
                edge_id    = f"edge:{concept_id}:{passage_id}"
                upsert_edge(
                    conn,
                    id       = edge_id,
                    from_id  = concept_id,
                    to_id    = passage_id,
                    relation = "source_text_for",
                )

        log.info("  wrote %d concept nodes", concept_count)
        stats["files"]  += 1
        stats["chunks"] += len(chunks)

    return stats


def run_check(conn) -> None:
    """Prints a summary of what is currently in the database."""
    print("\n─── Database contents ───────────────────────────────")
    for node_type in ["passage", "concept", "method", "session"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE type = ?", (node_type,)
        ).fetchone()[0]
        print(f"  {node_type:12s} nodes: {count}")

    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    snap_count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    print(f"  {'edges':12s}      : {edge_count}")
    print(f"  {'snapshots':12s}  : {snap_count}")
    print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest source texts into philosopher.db")
    parser.add_argument("--source",  type=str, default=None,
                        help="Path to source directory (default: all texts/ subdirs)")
    parser.add_argument("--thinker", type=str, default=None,
                        help="Thinker name to tag all files with")
    parser.add_argument("--role",    type=str, default="primary",
                        choices=["primary", "context", "neighbor"],
                        help="Role for these texts (default: primary)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files and chunks without writing to DB")
    parser.add_argument("--check",   action="store_true",
                        help="Print DB contents and exit")
    args = parser.parse_args()

    # Setup
    project_root = Path(__file__).parent.parent
    db_path      = project_root / "philosopher.db"
    conn         = get_connection(db_path)
    init_schema(conn)

    if args.check:
        run_check(conn)
        conn.close()
        return

    # Download NLTK data
    download_nltk_data()

    # Anthropic client for concept extraction
    client = anthropic.Anthropic()

    total_stats = {"files": 0, "chunks": 0, "nodes": 0}

    if args.source:
        # Ingest a specific directory
        source_dir = Path(args.source)
        if not source_dir.exists():
            log.error("Source directory not found: %s", source_dir)
            sys.exit(1)
        thinker = args.thinker or source_dir.name
        stats   = ingest_directory(source_dir, thinker, args.role, conn, client, args.dry_run)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    else:
        # Ingest all standard directories
        for thinker, dir_name in THINKER_DIRS.items():
            source_dir = project_root / dir_name
            if not source_dir.exists():
                log.info("skipping_missing_dir dir=%s", dir_name)
                continue
            role   = "context" if thinker in ("context",) else "neighbor" if thinker == "neighbor" else "primary"
            stats  = ingest_directory(source_dir, thinker, role, conn, client, args.dry_run)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

    conn.close()

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Ingest complete:")
    print(f"  Files processed:   {total_stats['files']}")
    print(f"  Passages chunked:  {total_stats['chunks']}")
    if not args.dry_run:
        print(f"  Nodes written:     {total_stats['nodes']}")
    print()

    if args.dry_run:
        print("Run without --dry-run to write to the database.")


if __name__ == "__main__":
    main()
