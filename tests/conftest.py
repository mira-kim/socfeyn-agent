"""
tests/conftest.py
------------------
Shared pytest fixtures available to all test files.

Add fixtures here when they are needed by multiple test files.
Keep test-specific fixtures in their own test files.
"""

import json
import sqlite3
import pytest
from pathlib import Path


@pytest.fixture
def tmp_config(tmp_path) -> Path:
    """
    A valid voices.json config file in a temp directory.
    Use in any test that needs a config without caring about specific values.
    """
    config_file = tmp_path / "voices.json"
    config_file.write_text(json.dumps({
        "socrates_weight": 0.6,
        "feynman_weight": 0.4,
        "temperature": 0.5,
        "prompt_version": "v1.0",
        "active_thinkers": ["socrates", "feynman"],
    }))
    return config_file


@pytest.fixture
def tmp_db(tmp_path) -> sqlite3.Connection:
    """
    A fresh SQLite connection with schema initialized.
    Each test gets its own isolated database — no shared state.

    Usage:
        def test_something(tmp_db):
            tmp_db.execute("INSERT INTO nodes ...")
    """
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    # Uncomment when graph/graph_db.py exists:
    # from graph.graph_db import init_schema
    # init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def tmp_prompt_dir(tmp_path) -> Path:
    """
    A temporary config/prompts directory with placeholder prompt files.
    Tests that need real prompt content should write their own.
    """
    prompt_dir = tmp_path / "config" / "prompts"
    prompt_dir.mkdir(parents=True)

    # Minimal placeholder prompts — enough for structural tests
    (prompt_dir / "system.txt").write_text(
        "You are a philosophical thinking partner. "
        "Socrates questions. Feynman grounds in first principles."
    )
    (prompt_dir / "socrates.txt").write_text(
        "You are Socrates. Always ask one probing question. Never lecture."
    )
    (prompt_dir / "feynman.txt").write_text(
        "You are Feynman. Use first principles. Plain language only. Give one analogy."
    )

    return prompt_dir


@pytest.fixture
def mock_api_response():
    """
    A mock Anthropic API response that looks like a valid Socratic exchange.
    Use in tests that need an API response without calling the real API.
    """
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.content = [MagicMock()]
    mock.content[0].text = (
        "SOCRATES: Tell me — when you say justice requires equality, "
        "do you mean equality of treatment or equality of outcome?\n\n"
        "FEYNMAN: Let us be precise about what we mean. Equal treatment "
        "and equal outcomes are two different systems entirely."
    )
    mock.stop_reason = "end_turn"
    mock.usage = MagicMock()
    mock.usage.input_tokens = 850
    mock.usage.output_tokens = 175
    return mock
