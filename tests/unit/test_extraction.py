"""
tests/unit/test_extraction.py
------------------------------
Tests for ExtractionResult in agent/runner.py (or memory/extractor.py).

What we test:
  - Valid JSON parses into correct fields
  - Markdown fences are stripped before parsing
  - Malformed JSON returns empty result — never crashes
  - Missing keys use defaults — never KeyError
  - Empty model output is handled gracefully
  - Partial JSON (some keys present, some missing) works correctly

What we do NOT test:
  - The actual Haiku API call
  - Whether the extraction is philosophically correct
  - Database writes from extraction results
"""

import json
import pytest

# ---------------------------------------------------------------------------
# Uncomment when memory/extractor.py exists (Phase 2).
# ---------------------------------------------------------------------------
# from memory.extractor import ExtractionResult
ExtractionResult = None  # placeholder until Phase 2


class TestExtractionResult:
    """LLM output parsing handles all the ways the model can misbehave."""

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_valid_json_parses_correctly(self):
        raw = json.dumps({
            "concepts": ["justice", "equality"],
            "claims": [{"speaker": "user", "claim": "justice = equality", "status": "challenged"}],
            "relationships": [{"from": "justice", "edge": "requires", "to": "equality", "disputed": True}],
            "unresolved": ["whether equality is necessary for justice"],
        })
        result = ExtractionResult.from_llm_output(raw)
        assert "justice" in result.concepts
        assert "equality" in result.concepts
        assert len(result.claims) == 1
        assert result.claims[0]["speaker"] == "user"
        assert len(result.unresolved) == 1

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_markdown_fences_stripped(self):
        raw = '```json\n{"concepts": ["justice"]}\n```'
        result = ExtractionResult.from_llm_output(raw)
        assert "justice" in result.concepts

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_markdown_fences_with_language_tag(self):
        raw = '```\n{"concepts": ["virtue"]}\n```'
        result = ExtractionResult.from_llm_output(raw)
        assert "virtue" in result.concepts

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_malformed_json_returns_empty_not_crash(self):
        """
        The agent must never crash because the model returned bad JSON.
        This happens more often than you expect — especially on long inputs.
        """
        raw = "I could not extract structured data from this exchange."
        result = ExtractionResult.from_llm_output(raw)
        assert result.concepts == []
        assert result.claims == []
        assert result.relationships == []
        assert result.unresolved == []

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_empty_string_returns_empty_not_crash(self):
        result = ExtractionResult.from_llm_output("")
        assert result.concepts == []

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_missing_keys_use_defaults(self):
        """Partial JSON — model returned some keys but not all."""
        raw = json.dumps({"concepts": ["virtue"]})
        result = ExtractionResult.from_llm_output(raw)
        assert result.concepts == ["virtue"]
        assert result.claims == []
        assert result.relationships == []
        assert result.unresolved == []

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_empty_lists_are_valid(self):
        """Model explicitly returned empty lists — not an error."""
        raw = json.dumps({
            "concepts": [],
            "claims": [],
            "relationships": [],
            "unresolved": [],
        })
        result = ExtractionResult.from_llm_output(raw)
        assert result.concepts == []
        assert result.claims == []

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_preamble_before_json_stripped(self):
        """
        Model sometimes adds preamble: "Here is the extraction:\n{...}"
        Must be stripped before JSON parse.
        """
        raw = 'Here is the extracted data:\n{"concepts": ["courage"]}'
        result = ExtractionResult.from_llm_output(raw)
        assert "courage" in result.concepts
