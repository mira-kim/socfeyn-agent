"""
graph/graph_db.py
-----------------
SQLite database layer for the philosopher agent.
Single connection interface for all graph and relational operations.

Does:    Schema init, node/edge CRUD, basic queries, embedding storage.
Does NOT: Application-level logic, API calls, prompt assembly.
Depends on: Nothing (no internal imports — pure stdlib + sqlite3).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path — override via DB_PATH env var or pass explicitly
# ---------------------------------------------------------------------------
_DEFAULT_DB = Path(__file__).parent.parent / "philosopher.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """
    Returns a configured SQLite connection.

    - Row factory set for column-name access: row["label"]
    - WAL mode for better concurrent reads
    - Foreign keys enforced
    - 64MB page cache

    Caller is responsible for closing. Use as context manager where possible.
    """
    path = db_path or _DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB
    conn.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Creates all tables and indexes. Idempotent — safe to call on every startup.
    Call once at application start, never per request.
    """
    conn.executescript("""
        -- ── Semantic + Procedural ──────────────────────────────────────
        CREATE TABLE IF NOT EXISTS nodes (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            label       TEXT NOT NULL,
            source      TEXT,
            thinker     TEXT,
            role        TEXT DEFAULT 'primary',
            metadata    TEXT,
            embedding   BLOB,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS edges (
            id          TEXT PRIMARY KEY,
            from_id     TEXT NOT NULL REFERENCES nodes(id),
            to_id       TEXT NOT NULL REFERENCES nodes(id),
            relation    TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            session_id  TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- ── Episodic ────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            summary     TEXT,
            topics      TEXT,
            open_claims TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS turns (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            topics      TEXT,
            embedding   BLOB,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS threads (
            id          TEXT PRIMARY KEY,
            claim       TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'open',
            from_node   TEXT REFERENCES nodes(id),
            session_id  TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME
        );

        -- ── Experiments ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS snapshots (
            run_id      TEXT PRIMARY KEY,
            session_id  TEXT,
            scenario    TEXT NOT NULL,
            config      TEXT NOT NULL,
            response    TEXT NOT NULL,
            auto_scores TEXT,
            human_scores TEXT,
            combined_score REAL,
            tags        TEXT,
            notes       TEXT,
            promoted    INTEGER DEFAULT 0,
            flagged     INTEGER DEFAULT 0,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- ── Indexes ─────────────────────────────────────────────────────
        CREATE INDEX IF NOT EXISTS idx_nodes_type     ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_label    ON nodes(label);
        CREATE INDEX IF NOT EXISTS idx_nodes_thinker  ON nodes(thinker);
        CREATE INDEX IF NOT EXISTS idx_edges_from     ON edges(from_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to       ON edges(to_id);
        CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
        CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
        CREATE INDEX IF NOT EXISTS idx_turns_session  ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_score ON snapshots(combined_score);
    """)
    log.debug("Schema initialized")


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

def upsert_node(
    conn: sqlite3.Connection,
    *,
    id: str,
    type: str,
    label: str,
    source: str | None = None,
    thinker: str | None = None,
    role: str = "primary",
    metadata: dict | None = None,
    embedding: np.ndarray | None = None,
) -> None:
    """
    Insert or update a node. Uses INSERT OR REPLACE semantics.
    Embedding stored as float32 BLOB — efficient, not JSON.
    """
    emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else None
    meta_str  = json.dumps(metadata) if metadata else None

    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO nodes
                (id, type, label, source, thinker, role, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, type, label, source, thinker, role, meta_str, emb_bytes))


def get_node(conn: sqlite3.Connection, node_id: str) -> sqlite3.Row | None:
    """Returns a single node by ID, or None if not found."""
    return conn.execute(
        "SELECT * FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()


def get_nodes_by_type(
    conn: sqlite3.Connection, node_type: str
) -> list[sqlite3.Row]:
    """Returns all nodes of a given type."""
    return conn.execute(
        "SELECT * FROM nodes WHERE type = ?", (node_type,)
    ).fetchall()


def get_nodes_by_label(
    conn: sqlite3.Connection, label: str, type: str | None = None
) -> list[sqlite3.Row]:
    """Fuzzy label search. Optionally filter by type."""
    if type:
        return conn.execute(
            "SELECT * FROM nodes WHERE label LIKE ? AND type = ?",
            (f"%{label}%", type)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM nodes WHERE label LIKE ?", (f"%{label}%",)
    ).fetchall()


def get_node_embedding(conn: sqlite3.Connection, node_id: str) -> np.ndarray | None:
    """Returns the embedding for a node as a numpy array, or None."""
    row = conn.execute(
        "SELECT embedding FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row is None or row["embedding"] is None:
        return None
    return np.frombuffer(row["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------

def upsert_edge(
    conn: sqlite3.Connection,
    *,
    id: str,
    from_id: str,
    to_id: str,
    relation: str,
    weight: float = 1.0,
    session_id: str | None = None,
) -> None:
    """Insert or update an edge. On conflict, increments weight."""
    with conn:
        conn.execute("""
            INSERT INTO edges (id, from_id, to_id, relation, weight, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight = weight + 1.0,
                session_id = excluded.session_id
        """, (id, from_id, to_id, relation, weight, session_id))


def get_edges_from(
    conn: sqlite3.Connection,
    node_id: str,
    relation: str | None = None,
) -> list[sqlite3.Row]:
    """Returns all edges from a node. Optionally filter by relation type."""
    if relation:
        return conn.execute(
            "SELECT * FROM edges WHERE from_id = ? AND relation = ? ORDER BY weight DESC",
            (node_id, relation)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM edges WHERE from_id = ? ORDER BY weight DESC", (node_id,)
    ).fetchall()


def get_related_nodes(
    conn: sqlite3.Connection,
    node_id: str,
    relation: str | None = None,
    limit: int = 10,
) -> list[sqlite3.Row]:
    """
    Returns nodes connected to node_id via edges.
    Joins edges + nodes for direct use.
    """
    if relation:
        return conn.execute("""
            SELECT n.*, e.relation, e.weight
            FROM edges e JOIN nodes n ON e.to_id = n.id
            WHERE e.from_id = ? AND e.relation = ?
            ORDER BY e.weight DESC LIMIT ?
        """, (node_id, relation, limit)).fetchall()
    return conn.execute("""
        SELECT n.*, e.relation, e.weight
        FROM edges e JOIN nodes n ON e.to_id = n.id
        WHERE e.from_id = ?
        ORDER BY e.weight DESC LIMIT ?
    """, (node_id, limit)).fetchall()


# ---------------------------------------------------------------------------
# Session and turn operations
# ---------------------------------------------------------------------------

def create_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Creates a new session record. Called at session start."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,)
        )
    log.info("session_created session_id=%s", session_id)


def update_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    summary: str,
    topics: list[str],
    open_claims: list[str],
) -> None:
    """Updates session summary after a drift-triggered rebuild."""
    with conn:
        conn.execute("""
            UPDATE sessions
            SET summary = ?, topics = ?, open_claims = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (summary, json.dumps(topics), json.dumps(open_claims), session_id))


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    """Returns session metadata, or None if not found."""
    return conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()


def add_turn(
    conn: sqlite3.Connection,
    *,
    turn_id: str,
    session_id: str,
    role: str,
    content: str,
    topics: list[str] | None = None,
    embedding: np.ndarray | None = None,
) -> None:
    """Appends a single turn to the session history."""
    emb_bytes  = embedding.astype(np.float32).tobytes() if embedding is not None else None
    topics_str = json.dumps(topics) if topics else None
    with conn:
        conn.execute("""
            INSERT INTO turns (id, session_id, role, content, topics, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (turn_id, session_id, role, content, topics_str, emb_bytes))


def get_rolling_window(
    conn: sqlite3.Connection,
    session_id: str,
    n: int = 4,
) -> list[dict]:
    """
    Returns the last N turns as a list of dicts ready for the messages array.
    Ordered chronologically (oldest first).
    """
    rows = conn.execute("""
        SELECT role, content FROM turns
        WHERE session_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (session_id, n)).fetchall()

    # Reverse to chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Thread operations
# ---------------------------------------------------------------------------

def add_thread(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    claim: str,
    session_id: str,
    from_node: str | None = None,
) -> None:
    """Adds an unresolved claim thread."""
    with conn:
        conn.execute("""
            INSERT OR IGNORE INTO threads (id, claim, session_id, from_node)
            VALUES (?, ?, ?, ?)
        """, (thread_id, claim, session_id, from_node))


def get_open_threads(
    conn: sqlite3.Connection,
    session_id: str | None = None,
    limit: int = 5,
) -> list[sqlite3.Row]:
    """Returns open threads, optionally filtered by session."""
    if session_id:
        return conn.execute("""
            SELECT * FROM threads WHERE status = 'open' AND session_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
    return conn.execute("""
        SELECT * FROM threads WHERE status = 'open'
        ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()


def resolve_thread(conn: sqlite3.Connection, thread_id: str) -> None:
    """Marks a thread as resolved."""
    with conn:
        conn.execute("""
            UPDATE threads
            SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (thread_id,))


# ---------------------------------------------------------------------------
# Snapshot operations
# ---------------------------------------------------------------------------

def save_snapshot(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    session_id: str,
    scenario: str,
    config: dict,
    response: str,
    auto_scores: dict | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> None:
    """Saves an experiment snapshot after every turn."""
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO snapshots
                (run_id, session_id, scenario, config, response,
                 auto_scores, tags, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            session_id,
            scenario,
            json.dumps(config),
            response,
            json.dumps(auto_scores) if auto_scores else None,
            json.dumps(tags) if tags else None,
            notes,
        ))
    log.debug("snapshot_saved run_id=%s", run_id)


def update_human_scores(
    conn: sqlite3.Connection,
    run_id: str,
    human_scores: dict,
    combined_score: float,
    tags: list[str],
    notes: str,
    promoted: bool = False,
    flagged: bool = False,
) -> None:
    """Updates a snapshot with human feedback after scoring."""
    with conn:
        conn.execute("""
            UPDATE snapshots
            SET human_scores = ?, combined_score = ?, tags = ?,
                notes = ?, promoted = ?, flagged = ?
            WHERE run_id = ?
        """, (
            json.dumps(human_scores),
            combined_score,
            json.dumps(tags),
            notes,
            int(promoted),
            int(flagged),
            run_id,
        ))


def get_recent_snapshots(
    conn: sqlite3.Connection, limit: int = 10
) -> list[sqlite3.Row]:
    """Returns the most recent snapshots ordered by creation time."""
    return conn.execute("""
        SELECT * FROM snapshots ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()


def vector_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    node_type: str = "passage",
    top_k: int = 20,
) -> list[tuple[sqlite3.Row, float]]:
    """
    Cosine similarity search over embedded nodes.
    Returns list of (node_row, similarity_score) tuples.

    NOTE: This is a pure-Python implementation for Phase 1.
    sqlite-vec extension provides native vector ops — upgrade in Phase 3
    once the corpus is large enough to need it. For hundreds of passages,
    Python cosine similarity is fast enough.
    """
    query_vec = query_embedding.astype(np.float32)

    rows = conn.execute(
        "SELECT * FROM nodes WHERE type = ? AND embedding IS NOT NULL",
        (node_type,)
    ).fetchall()

    if not rows:
        log.warning("vector_search: no embedded nodes of type=%s found", node_type)
        return []

    results = []
    for row in rows:
        node_vec = np.frombuffer(row["embedding"], dtype=np.float32)
        # Cosine similarity — vectors are normalized at ingest time
        score = float(np.dot(query_vec, node_vec))
        results.append((row, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
