"""
tests/unit/test_database.py
----------------------------
Tests for graph/graph_db.py database operations.

What we test:
  - Schema initializes correctly and creates all expected tables
  - Schema init is idempotent (safe to call multiple times)
  - Basic node insert and read round-trip
  - Basic edge insert with valid foreign keys
  - Foreign key violation raises IntegrityError
  - WAL mode is enabled
  - Row factory enables column name access
  - Transactions roll back on failure

What we do NOT test:
  - Vector search (requires sqlite-vec extension)
  - Any API calls
  - Application-level graph logic

Each test uses an isolated tmp_path database.
Never share database state between tests.
"""

import sqlite3
import pytest

# ---------------------------------------------------------------------------
# Uncomment when graph/graph_db.py exists (Phase 1).
# ---------------------------------------------------------------------------
from graph.graph_db import get_connection, init_schema


EXPECTED_TABLES = {"nodes", "edges", "threads", "sessions", "turns"}
EXPECTED_INDEXES = {
    "idx_nodes_type", "idx_nodes_label",
    "idx_edges_from", "idx_edges_to", "idx_edges_relation",
    "idx_threads_status", "idx_turns_session",
}


class TestDatabaseSchema:

    @pytest.fixture
    def db(self, tmp_path):
        """Fresh in-memory-style database per test using tmp_path."""
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    
    def test_all_tables_created(self, db):
        init_schema(db)
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert EXPECTED_TABLES.issubset(tables)

    
    def test_all_indexes_created(self, db):
        init_schema(db)
        indexes = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
        assert EXPECTED_INDEXES.issubset(indexes)

    
    def test_schema_is_idempotent(self, db):
        """Calling init_schema twice must not raise."""
        init_schema(db)
        init_schema(db)  # should be a no-op

    
    def test_wal_mode_enabled(self, tmp_path):
        conn = get_connection(tmp_path / "wal_test.db")
        init_schema(conn)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestNodeOperations:

    @pytest.fixture
    def db(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        init_schema(conn)
        return conn

    
    def test_node_insert_and_read(self, db):
        with db:
            db.execute(
                "INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                ("n001", "concept", "justice")
            )
        row = db.execute("SELECT * FROM nodes WHERE id = ?", ("n001",)).fetchone()
        assert row["label"] == "justice"
        assert row["type"] == "concept"

    
    def test_node_id_is_primary_key(self, db):
        with db:
            db.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                       ("n001", "concept", "justice"))
        with pytest.raises(sqlite3.IntegrityError):
            with db:
                db.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                           ("n001", "concept", "duplicate"))

    
    def test_row_factory_enables_column_access(self, db):
        with db:
            db.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                       ("n001", "concept", "virtue"))
        row = db.execute("SELECT * FROM nodes WHERE id = 'n001'").fetchone()
        # Should work with column names, not just indexes
        assert row["label"] == "virtue"
        assert row["type"] == "concept"


class TestEdgeOperations:

    @pytest.fixture
    def db_with_nodes(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        init_schema(conn)
        # Insert two nodes to use as edge endpoints
        with conn:
            conn.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                         ("n001", "concept", "justice"))
            conn.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                         ("n002", "concept", "equality"))
        return conn

    
    def test_edge_insert_with_valid_nodes(self, db_with_nodes):
        with db_with_nodes:
            db_with_nodes.execute(
                "INSERT INTO edges (id, from_id, to_id, relation) VALUES (?, ?, ?, ?)",
                ("e001", "n001", "n002", "related_to")
            )
        row = db_with_nodes.execute(
            "SELECT * FROM edges WHERE id = ?", ("e001",)
        ).fetchone()
        assert row["relation"] == "related_to"
        assert row["weight"] == 1.0  # default weight

    
    def test_edge_rejects_nonexistent_node(self, db_with_nodes):
        with pytest.raises(sqlite3.IntegrityError):
            with db_with_nodes:
                db_with_nodes.execute(
                    "INSERT INTO edges (id, from_id, to_id, relation) VALUES (?, ?, ?, ?)",
                    ("e001", "n001", "ghost-node", "related_to")
                )

    
    def test_edge_weight_can_be_updated(self, db_with_nodes):
        with db_with_nodes:
            db_with_nodes.execute(
                "INSERT INTO edges (id, from_id, to_id, relation, weight) VALUES (?, ?, ?, ?, ?)",
                ("e001", "n001", "n002", "related_to", 1.0)
            )
        with db_with_nodes:
            db_with_nodes.execute(
                "UPDATE edges SET weight = weight + 1.0 WHERE id = ?", ("e001",)
            )
        row = db_with_nodes.execute(
            "SELECT weight FROM edges WHERE id = ?", ("e001",)
        ).fetchone()
        assert row["weight"] == 2.0


class TestTransactionBehavior:

    @pytest.fixture
    def db(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        init_schema(conn)
        return conn

    
    def test_failed_transaction_rolls_back(self, db):
        """
        Multi-step write that fails halfway should leave DB unchanged.
        This is critical for graph writes — partial writes corrupt the graph.
        """
        try:
            with db:
                db.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                           ("n001", "concept", "justice"))
                # Intentionally fail the second insert
                db.execute("INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                           ("n001", "concept", "duplicate"))  # duplicate PK
        except sqlite3.IntegrityError:
            pass  # expected

        # First insert should also have been rolled back
        row = db.execute("SELECT * FROM nodes WHERE id = 'n001'").fetchone()
        assert row is None, "Partial write was not rolled back"
