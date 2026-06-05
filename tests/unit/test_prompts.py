"""
tests/unit/test_prompts.py
---------------------------
Tests for prompt loading in agent/config_loader.py.

What we test:
  - Valid prompt files load and return their content
  - Leading and trailing whitespace is stripped
  - Missing prompt file raises FileNotFoundError with a clear message
  - Empty prompt file raises or returns empty string (define the contract)
  - Prompt assembly produces correct positional order
  - System prompt is always the first element
  - User query is always the last element

What we do NOT test:
  - Prompt quality or philosophical content
  - Any API calls
"""

import pytest


# ---------------------------------------------------------------------------
# Uncomment when agent/config_loader.py exists (Phase 1).
# ---------------------------------------------------------------------------
from agent.config_loader import load_prompt, clear_prompt_cache
from agent.config_loader import assemble_prompt


class TestPromptLoading:

    @pytest.fixture
    def prompt_dir(self, tmp_path):
        """Creates a temporary config/prompts directory."""
        d = tmp_path / "config" / "prompts"
        d.mkdir(parents=True)
        return d

    
    def test_valid_prompt_loads(self, tmp_path):
        """Prompt loads from a given directory path."""
        clear_prompt_cache()
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "test_p.txt").write_text("You are Socrates.")
        result = load_prompt("test_p", prompts_dir=prompt_dir)
        assert result == "You are Socrates."
        clear_prompt_cache()

    
    def test_whitespace_stripped(self, tmp_path):
        """Leading and trailing whitespace is stripped."""
        clear_prompt_cache()
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "test_ws.txt").write_text("   \n  You are Socrates.  \n  ")
        result = load_prompt("test_ws", prompts_dir=prompt_dir)
        assert result == "You are Socrates."
        clear_prompt_cache()

    
    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt("nonexistent_prompt")
        # Error message should tell you what file is missing
        assert "nonexistent_prompt" in str(exc_info.value)

    
    def test_all_required_prompts_exist(self):
        """
        Verify all prompts the agent depends on actually exist.
        This test fails if someone deletes a prompt file.
        """
        required = ["system", "socrates", "feynman"]
        for name in required:
            result = load_prompt(name)
            assert len(result) > 0, f"Prompt '{name}' is empty"


class TestPromptAssembly:
    """
    Prompt assembly puts components in the right order.
    The 'Lost in the Middle' paper shows this matters for attention.
    """

    
    def test_user_query_is_last_message(self):
        system, messages = assemble_prompt(
            system="You are Socrates.",
            rag_context="Relevant passage from Republic.",
            session_summary="We discussed justice.",
            open_threads="Justice = equality unresolved.",
            rolling_window=[],
            user_query="What is courage?",
        )
        last_message = messages[-1]
        assert last_message["role"] == "user"
        assert "What is courage?" in last_message["content"]

    
    def test_system_prompt_returned_separately(self):
        system, messages = assemble_prompt(
            system="You are Socrates.",
            rag_context="",
            session_summary="",
            open_threads="",
            rolling_window=[],
            user_query="What is courage?",
        )
        # System is returned as a string, not in messages
        assert system == "You are Socrates."
        # System should not also appear in messages
        for msg in messages:
            assert "You are Socrates." not in msg.get("content", "")

    
    def test_empty_components_excluded_from_messages(self):
        system, messages = assemble_prompt(
            system="You are Socrates.",
            rag_context="",          # empty — should not appear
            session_summary="",      # empty — should not appear
            open_threads="",         # empty — should not appear
            rolling_window=[],
            user_query="What is courage?",
        )
        # Only the user query should be in messages
        assert len(messages) == 1
        assert messages[0]["content"] == "What is courage?"

    
    def test_rolling_window_included_before_query(self):
        prior_turns = [
            {"role": "user", "content": "What is justice?"},
            {"role": "assistant", "content": "Tell me what you mean by justice."},
        ]
        system, messages = assemble_prompt(
            system="You are Socrates.",
            rag_context="",
            session_summary="",
            open_threads="",
            rolling_window=prior_turns,
            user_query="What is courage?",
        )
        # Rolling window should appear before the final user query
        last = messages[-1]
        assert last["content"] == "What is courage?"
        # Prior turns should be in the messages
        contents = [m["content"] for m in messages]
        assert "What is justice?" in contents

    
    def test_low_confidence_warning_injected(self):
        system, messages = assemble_prompt(
            system="You are Socrates.",
            rag_context="Relevant passage.",
            session_summary="",
            open_threads="",
            rolling_window=[],
            user_query="What is courage?",
            confidence_warning="⚠ Retrieval confidence below threshold. State uncertainty.",
        )
        # Warning should appear in the rag context message
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "confidence" in all_content.lower()
