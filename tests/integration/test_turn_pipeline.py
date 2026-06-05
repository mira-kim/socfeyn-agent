"""
tests/integration/test_turn_pipeline.py
-----------------------------------------
Integration tests for a complete dialogue turn.

These tests wire multiple modules together but mock the API call.
The goal is to verify that data flows correctly through the pipeline
without spending money on API calls.

What we test:
  - A turn produces a DialogueResponse with expected structure
  - Retrieved passages appear in the assembled prompt
  - Session context flows into the prompt correctly
  - Memory write is called after the response
  - Snapshot is saved after each turn
  - Monitoring traceability data is populated

What we do NOT test:
  - Response quality or philosophical content
  - Real API calls (mocked throughout)
  - Vector search accuracy

Run with:  pytest tests/integration/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Uncomment when Phase 1 modules exist.
# ---------------------------------------------------------------------------
# from agent.runner import run_dialogue_turn
# from agent.config_loader import AgentConfig
# from graph.graph_db import get_connection, init_schema
run_dialogue_turn = None  # placeholder until Phase 1


class TestTurnPipeline:

    @pytest.fixture
    def config(self, tmp_path):
        """Minimal valid config for testing."""
        config_file = tmp_path / "voices.json"
        config_file.write_text(json.dumps({
            "socrates_weight": 0.6,
            "feynman_weight": 0.4,
            "temperature": 0.5,
            "prompt_version": "v1.0",
            "active_thinkers": ["socrates", "feynman"],
        }))
        # return AgentConfig.load(config_file)  ← uncomment Phase 1
        return MagicMock()

    @pytest.fixture
    def db(self, tmp_path):
        """Fresh test database with schema initialized."""
        import sqlite3
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        # init_schema(conn)  ← uncomment Phase 1
        return conn

    @pytest.fixture
    def mock_api_response(self):
        """A mock Anthropic API response that looks like Socrates."""
        mock = MagicMock()
        mock.content = [MagicMock()]
        mock.content[0].text = (
            "SOCRATES: Tell me — when you say justice requires equality, "
            "do you mean equality of treatment or equality of outcome?\n\n"
            "FEYNMAN: Let us be precise. These are two entirely different "
            "physical systems with different equilibria."
        )
        mock.stop_reason = "end_turn"
        mock.usage = MagicMock()
        mock.usage.input_tokens = 850
        mock.usage.output_tokens = 180
        return mock

    @pytest.mark.skip(reason="Waiting for Phase 1 modules")
    def test_turn_returns_dialogue_response(self, config, db, mock_api_response):
        with patch("agent.runner.call_api_with_retry", return_value=mock_api_response):
            response, trace = run_dialogue_turn(
                user_query="Justice means treating everyone equally.",
                session_id="test-session-001",
                history=[],
                config=config,
                db=db,
            )
        assert response.text is not None
        assert len(response.text) > 0
        assert response.stop_reason == "end_turn"
        assert response.was_truncated is False

    @pytest.mark.skip(reason="Waiting for Phase 1 modules")
    def test_turn_populates_traceability(self, config, db, mock_api_response):
        """
        Traceability trace should be populated after every turn.
        This is what the monitoring panel reads.
        """
        with patch("agent.runner.call_api_with_retry", return_value=mock_api_response):
            response, trace = run_dialogue_turn(
                user_query="What is justice?",
                session_id="test-session-001",
                history=[],
                config=config,
                db=db,
            )
        # Trace should have at minimum these fields
        assert hasattr(trace, "retrieved_passages")
        assert hasattr(trace, "voice_rule_fired")
        assert hasattr(trace, "confidence_scores")

    @pytest.mark.skip(reason="Waiting for Phase 1 modules")
    def test_turn_does_not_call_api_when_query_empty(self, config, db):
        """Empty query should return early without an API call."""
        with patch("agent.runner.call_api_with_retry") as mock_call:
            with pytest.raises(ValueError, match="empty"):
                run_dialogue_turn(
                    user_query="",
                    session_id="test-session-001",
                    history=[],
                    config=config,
                    db=db,
                )
            mock_call.assert_not_called()

    @pytest.mark.skip(reason="Waiting for Phase 1 modules")
    def test_turn_uses_haiku_for_scoring_not_sonnet(self, config, db, mock_api_response):
        """
        Verify the correct model is used for each task.
        Scoring must use Haiku. Dialogue must use Sonnet.
        """
        calls = []

        def track_calls(*, model, **kwargs):
            calls.append(model)
            return mock_api_response

        with patch("agent.runner.call_api_with_retry", side_effect=track_calls):
            run_dialogue_turn(
                user_query="What is justice?",
                session_id="test-session-001",
                history=[],
                config=config,
                db=db,
            )

        # There should be at least one Sonnet call (dialogue)
        assert any("sonnet" in c for c in calls), f"No Sonnet call found in: {calls}"
        # Scoring calls should use Haiku
        scoring_calls = [c for c in calls if "haiku" in c]
        assert len(scoring_calls) > 0, f"No Haiku calls found in: {calls}"
