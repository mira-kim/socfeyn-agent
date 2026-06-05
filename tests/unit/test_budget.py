"""
tests/unit/test_budget.py
--------------------------
Tests for context budget enforcement in memory/memory_manager.py.

What we test:
  - All components fit when under budget
  - Least critical component dropped first when over budget
  - Multiple components dropped in correct priority order
  - Empty input handled gracefully
  - Budget of zero drops everything
  - Components with exact budget boundary fit correctly

What we do NOT test:
  - Any API calls
  - Token counting accuracy (we use a rough estimate)
  - Database operations
"""

import pytest

# ---------------------------------------------------------------------------
# Uncomment when memory/memory_manager.py exists (Phase 2).
# ---------------------------------------------------------------------------
# from memory.memory_manager import fits_budget, CONTEXT_BUDGET_TOKENS
fits_budget = None  # placeholder until Phase 2


class TestContextBudget:
    """Budget enforcement drops the right components in the right order."""

    # Priority order from best_practices.py:
    # rolling_window > rag_context > session_summary > open_threads

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_all_components_fit_under_budget(self):
        components = {
            "rolling_window":  "a" * 400,   # ~100 tokens
            "rag_context":     "b" * 400,   # ~100 tokens
            "session_summary": "c" * 400,   # ~100 tokens
            "open_threads":    "d" * 400,   # ~100 tokens
        }
        result = fits_budget(components, budget=500)
        assert len(result) == 4

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_open_threads_dropped_first(self):
        """open_threads is the lowest priority and should go first."""
        components = {
            "rolling_window":  "a" * 2000,
            "rag_context":     "b" * 2000,
            "session_summary": "c" * 2000,
            "open_threads":    "d" * 2000,   # should be dropped
        }
        result = fits_budget(components, budget=1500)
        assert "open_threads" not in result
        assert "rolling_window" in result

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_session_summary_dropped_before_rag(self):
        components = {
            "rolling_window":  "a" * 2000,
            "rag_context":     "b" * 2000,
            "session_summary": "c" * 2000,   # should drop before rag
        }
        result = fits_budget(components, budget=1000)
        assert "rolling_window" in result
        assert "rag_context" in result
        assert "session_summary" not in result

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_rolling_window_kept_last(self):
        """rolling_window is highest priority — last to be dropped."""
        components = {
            "rolling_window":  "a" * 2000,
            "rag_context":     "b" * 2000,
            "session_summary": "c" * 2000,
            "open_threads":    "d" * 2000,
        }
        result = fits_budget(components, budget=600)
        assert "rolling_window" in result

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_empty_components_returns_empty(self):
        result = fits_budget({}, budget=2000)
        assert result == {}

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_unknown_component_keys_handled(self):
        """Unknown keys should not crash — just be treated as low priority."""
        components = {
            "rolling_window": "a" * 400,
            "some_future_key": "b" * 400,
        }
        # Should not raise
        result = fits_budget(components, budget=300)
        assert isinstance(result, dict)

    @pytest.mark.skip(reason="Waiting for memory/memory_manager.py — Phase 2")
    def test_zero_budget_drops_everything(self):
        components = {"rolling_window": "a" * 100}
        result = fits_budget(components, budget=0)
        assert result == {}
