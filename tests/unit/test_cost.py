"""
tests/unit/test_cost.py
------------------------
Tests for cost tracking in agent/runner.py.

What we test:
  - Cost estimates are mathematically correct for known prices
  - Cumulative session tracking accumulates correctly
  - Cost warning fires at the right threshold
  - Zero token calls do not crash
  - DialogueResponse correctly identifies truncated responses

What we do NOT test:
  - Any actual API calls
  - Whether prices are current (update COST constants when pricing changes)
"""



# ---------------------------------------------------------------------------
# Uncomment when agent/runner.py exists (Phase 1).
# ---------------------------------------------------------------------------
from agent.runner import (
    estimate_cost,
    SessionCostTracker,
    DialogueResponse,
    )


class TestCostEstimates:

    
    def test_sonnet_cost_estimate(self):
        # Sonnet: $0.003/1k input, $0.015/1k output
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.003 / 1000) + (500 * 0.015 / 1000)  # = 0.003 + 0.0075
        assert abs(cost - expected) < 0.0001

    
    def test_haiku_cost_estimate(self):
        # Haiku: $0.001/1k input, $0.005/1k output
        cost = estimate_cost("claude-haiku-4-5-20251001", input_tokens=500, output_tokens=200)
        expected = (500 * 0.001 / 1000) + (200 * 0.005 / 1000)
        assert abs(cost - expected) < 0.0001

    
    def test_zero_tokens_returns_zero(self):
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    
    def test_unknown_model_uses_default_pricing(self):
        """Unknown model should not crash — use a safe default."""
        cost = estimate_cost("some-future-model", input_tokens=1000, output_tokens=500)
        assert cost > 0  # some cost, not zero, not crash


class TestSessionCostTracker:

    
    def test_cost_accumulates_across_calls(self):
        tracker = SessionCostTracker("test-session-001")
        tracker.record("claude-sonnet-4-6", 1000, 500)
        tracker.record("claude-haiku-4-5-20251001", 500, 200)
        assert tracker.total_usd > 0
        assert tracker.call_count == 2

    
    def test_initial_state_is_zero(self):
        tracker = SessionCostTracker("test-session-002")
        assert tracker.total_usd == 0.0
        assert tracker.call_count == 0

    
    def test_warning_fires_above_threshold(self, caplog):
        import logging
        tracker = SessionCostTracker("expensive-session")
        # Simulate 20 expensive turns
        for _ in range(20):
            tracker.record("claude-sonnet-4-6", 2000, 800)
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_expensive(threshold_usd=0.10)
        assert any("spent" in record.message.lower() for record in caplog.records)

    
    def test_warning_does_not_fire_below_threshold(self, caplog):
        import logging
        tracker = SessionCostTracker("cheap-session")
        tracker.record("claude-haiku-4-5-20251001", 100, 50)
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_expensive(threshold_usd=1.00)
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0


class TestDialogueResponse:

    
    def test_truncated_response_detected(self):
        """
        max_tokens stop reason means the response was cut off.
        This should be flagged so we can increase the limit.
        """
        mock_api_response = _make_mock_response(
            text="Socrates began to speak but was cut",
            stop_reason="max_tokens",
            input_tokens=800,
            output_tokens=600,
        )
        response = DialogueResponse.from_api_response(mock_api_response, run_id="test-001")
        assert response.was_truncated is True

    
    def test_normal_response_not_truncated(self):
        mock_api_response = _make_mock_response(
            text="Tell me, what do you mean by justice?",
            stop_reason="end_turn",
            input_tokens=800,
            output_tokens=120,
        )
        response = DialogueResponse.from_api_response(mock_api_response, run_id="test-001")
        assert response.was_truncated is False

    
    def test_cost_estimate_property(self):
        mock_api_response = _make_mock_response(
            text="Tell me, what do you mean by justice?",
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=500,
        )
        response = DialogueResponse.from_api_response(mock_api_response, run_id="test-001")
        assert response.cost_estimate_usd > 0


# ---------------------------------------------------------------------------
# Test helper — builds a mock API response without calling the API.
# Replace with the real structure once you know what anthropic returns.
# ---------------------------------------------------------------------------

def _make_mock_response(*, text, stop_reason, input_tokens, output_tokens):
    """Creates a mock Anthropic API response for testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.content = [MagicMock()]
    mock.content[0].text = text
    mock.stop_reason = stop_reason
    mock.usage = MagicMock()
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    return mock
