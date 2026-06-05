"""
tests/smoke/test_api.py
------------------------
Smoke tests that make real API calls.

Run manually before deploying any prompt changes:
  pytest tests/smoke/ -v -m smoke

Never run in automated CI — these cost money and require
a valid ANTHROPIC_API_KEY in the environment.

What we test:
  - Haiku extraction prompt returns parseable JSON
  - Sonnet dialogue prompt produces a response containing a question
  - The API client connects and responds within a reasonable time
  - Truncation does not occur at our standard max_tokens limits

If any smoke test fails:
  - The prompt it tests needs updating before you proceed
  - Document the failure in experiments/changelog.md
  - Do not promote runs until the smoke test passes
"""

import os
import pytest


# ---------------------------------------------------------------------------
# Uncomment when agent modules exist.
# ---------------------------------------------------------------------------
# from agent.runner import get_client, call_api_with_retry
# from memory.extractor import ExtractionResult
get_client = None  # placeholder until Phase 1
call_api_with_retry = None  # placeholder until Phase 1
ExtractionResult = None  # placeholder until Phase 2


pytestmark = pytest.mark.smoke  # all tests in this file require -m smoke


@pytest.fixture(scope="module")
def require_api_key():
    """Skip all smoke tests if API key is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping smoke tests")


class TestHaikuExtraction:
    """Verify the Haiku extraction prompt works as designed."""

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_haiku_returns_valid_json(self, require_api_key):
        """
        The extraction prompt must return clean JSON.
        If this fails, update the extraction prompt in config/prompts/.
        """
        client = get_client()
        response = call_api_with_retry(
            model="claude-haiku-4-5-20251001",
            system=(
                "Extract from this conversation turn. "
                "Return ONLY valid JSON with keys: "
                "concepts (list of strings), "
                "claims (list of dicts with speaker/claim/status), "
                "relationships (list of dicts with from/edge/to), "
                "unresolved (list of strings). "
                "No preamble. No markdown fences. JSON only."
            ),
            messages=[{
                "role": "user",
                "content": (
                    "User: I think justice means treating everyone equally.\n"
                    "Socrates: But tell me — if a surgeon and a child both need "
                    "medicine and you have only one dose, does equal treatment "
                    "serve justice?\n"
                    "User: I suppose not in that case."
                ),
            }],
            max_tokens=400,
        )

        result = ExtractionResult.from_llm_output(response.content[0].text)
        assert len(result.concepts) > 0 or len(result.claims) > 0, (
            f"Extraction returned nothing useful.\n"
            f"Raw output: {response.content[0].text}\n"
            f"Action: Update config/prompts/extraction.txt"
        )

    @pytest.mark.skip(reason="Waiting for memory/extractor.py — Phase 2")
    def test_haiku_does_not_truncate_at_400_tokens(self, require_api_key):
        """Extraction should complete within 400 tokens."""
        client = get_client()
        response = call_api_with_retry(
            model="claude-haiku-4-5-20251001",
            system="Return a valid JSON object with key 'concepts' containing 3 strings.",
            messages=[{"role": "user", "content": "justice equality virtue"}],
            max_tokens=400,
        )
        assert response.stop_reason != "max_tokens", (
            "Extraction hit max_tokens limit. Either reduce extraction scope "
            "or increase max_tokens for the extraction call."
        )


class TestSonnetDialogue:
    """Verify the Sonnet dialogue prompt behaves as designed."""

    @pytest.mark.skip(reason="Waiting for agent/config_loader.py — Phase 1")
    def test_socratic_response_contains_question(self, require_api_key):
        """
        The single most important contract: Socrates must ask a question.
        If this fails, strengthen the 'never lecture' rule in socrates.txt.
        """
        from agent.config_loader import load_prompt
        system = load_prompt("system") + "\n\n" + load_prompt("socrates")

        client = get_client()
        response = call_api_with_retry(
            model="claude-sonnet-4-6",
            system=system,
            messages=[{
                "role": "user",
                "content": "I believe that a strong economy is more important than equality."
            }],
            max_tokens=600,
        )

        text = response.content[0].text
        assert "?" in text, (
            f"Socratic response contains no question mark.\n"
            f"Response: {text}\n"
            f"Action: Strengthen 'never lecture' instruction in config/prompts/socrates.txt"
        )

    @pytest.mark.skip(reason="Waiting for agent/config_loader.py — Phase 1")
    def test_response_does_not_truncate_at_600_tokens(self, require_api_key):
        """Dialogue should complete within 600 tokens."""
        from agent.config_loader import load_prompt
        client = get_client()
        response = call_api_with_retry(
            model="claude-sonnet-4-6",
            system=load_prompt("system"),
            messages=[{"role": "user", "content": "What is justice?"}],
            max_tokens=600,
        )
        assert response.stop_reason != "max_tokens", (
            "Dialogue response hit max_tokens=600. "
            "Socrates should be more concise. Add brevity instruction to socrates.txt."
        )

    @pytest.mark.skip(reason="Waiting for agent/config_loader.py — Phase 1")
    def test_feynman_uses_plain_language(self, require_api_key):
        """
        Feynman should never use jargon without an analogy.
        This is a heuristic check — no jargon in the first two sentences.
        """
        from agent.config_loader import load_prompt
        technical_jargon = [
            "quantum", "electromagnetic", "thermodynamic",
            "epistemological", "ontological",
        ]

        client = get_client()
        response = call_api_with_retry(
            model="claude-sonnet-4-6",
            system=load_prompt("system") + "\n\n" + load_prompt("feynman"),
            messages=[{
                "role": "user",
                "content": "Can you explain how knowledge works?"
            }],
            max_tokens=600,
        )

        first_two_sentences = ". ".join(response.content[0].text.split(".")[:2])
        jargon_found = [j for j in technical_jargon if j in first_two_sentences.lower()]

        assert not jargon_found, (
            f"Feynman used jargon in opening sentences: {jargon_found}\n"
            f"Text: {first_two_sentences}\n"
            f"Action: Strengthen 'plain language only' in config/prompts/feynman.txt"
        )
