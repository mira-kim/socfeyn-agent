"""
agent/config_loader.py
-----------------------
Loads and validates all configuration and prompt files.
Everything is loaded once at startup and cached.

Does:    Load voices.json, load prompt text files, validate config.
Does NOT: Make API calls, write to database, contain prompt logic.
Depends on: Nothing (stdlib only).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution — relative to project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = Path(__file__).parent.parent
_CONFIG_FILE   = _PROJECT_ROOT / "config" / "voices.json"
_PROMPTS_DIR   = _PROJECT_ROOT / "config" / "prompts"


# ---------------------------------------------------------------------------
# AgentConfig — immutable, validated at construction time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable configuration loaded once at startup.
    frozen=True prevents accidental mutation mid-session.
    Validated at construction — bad config fails immediately, not on turn 47.
    """
    socrates_weight:      float
    feynman_weight:       float
    temperature:          float
    prompt_version:       str
    active_thinkers:      tuple[str, ...]

    # Thresholds with sensible defaults
    drift_threshold:      float = 0.75
    confidence_threshold: float = 0.72
    context_budget_tokens: int  = 2000
    rolling_window_turns: int   = 4

    # Token limits — hard caps on every API call
    max_response_tokens:   int  = 600
    max_scoring_tokens:    int  = 300
    max_extraction_tokens: int  = 400
    max_summary_tokens:    int  = 200

    def __post_init__(self) -> None:
        """Validate all values at construction. Fail loudly with clear messages."""
        if not (0 < self.socrates_weight <= 1):
            raise ValueError(
                f"socrates_weight must be in (0, 1], got {self.socrates_weight}"
            )
        if not (0 < self.feynman_weight <= 1):
            raise ValueError(
                f"feynman_weight must be in (0, 1], got {self.feynman_weight}"
            )
        total = self.socrates_weight + self.feynman_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Voice weights must sum to 1.0, got {total:.3f}. "
                f"Adjust socrates_weight and feynman_weight in voices.json."
            )
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(
                f"temperature must be in [0.0, 1.0], got {self.temperature}"
            )
        if self.context_budget_tokens < 500:
            raise ValueError(
                f"context_budget_tokens too small: {self.context_budget_tokens}. "
                f"Minimum 500."
            )
        if not self.active_thinkers:
            raise ValueError("active_thinkers cannot be empty.")
        if len(self.prompt_version) == 0:
            raise ValueError("prompt_version cannot be empty.")

    @classmethod
    def load(cls, path: Path | None = None) -> "AgentConfig":
        """
        Load config from voices.json.
        Raises FileNotFoundError if file missing.
        Raises ValueError if config is invalid.
        """
        config_path = path or _CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Create config/voices.json before starting the agent."
            )
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}") from e

        return cls(
            socrates_weight       = raw["socrates_weight"],
            feynman_weight        = raw["feynman_weight"],
            temperature           = raw["temperature"],
            prompt_version        = raw["prompt_version"],
            active_thinkers       = tuple(raw["active_thinkers"]),
            drift_threshold       = raw.get("drift_threshold", 0.75),
            confidence_threshold  = raw.get("confidence_threshold", 0.72),
            context_budget_tokens = raw.get("context_budget_tokens", 2000),
            rolling_window_turns  = raw.get("rolling_window_turns", 4),
            max_response_tokens   = raw.get("max_response_tokens", 600),
            max_scoring_tokens    = raw.get("max_scoring_tokens", 300),
            max_extraction_tokens = raw.get("max_extraction_tokens", 400),
            max_summary_tokens    = raw.get("max_summary_tokens", 200),
        )

    def as_dict(self) -> dict:
        """Snapshot-safe dict representation for saving to experiments."""
        return {
            "socrates_weight":  self.socrates_weight,
            "feynman_weight":   self.feynman_weight,
            "temperature":      self.temperature,
            "prompt_version":   self.prompt_version,
            "active_thinkers":  list(self.active_thinkers),
        }


# ---------------------------------------------------------------------------
# Prompt loading — files only, never string literals
# ---------------------------------------------------------------------------

# Module-level cache — load each file once per process
_prompt_cache: dict[str, str] = {}


def load_prompt(name: str, prompts_dir: Path | None = None) -> str:
    """
    Load a prompt from config/prompts/<name>.txt.
    Strips whitespace. Cached after first load.
    Raises FileNotFoundError with a clear message if file is missing.
    """
    if name in _prompt_cache:
        return _prompt_cache[name]

    prompt_path = (prompts_dir or _PROMPTS_DIR) / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Create config/prompts/{name}.txt before running the agent.\n"
            f"Required prompts: system, socrates, feynman"
        )

    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            f"Prompt file is empty: {prompt_path}\n"
            f"Add content to config/prompts/{name}.txt"
        )

    _prompt_cache[name] = content
    log.debug("prompt_loaded name=%s chars=%d", name, len(content))
    return content


def clear_prompt_cache() -> None:
    """
    Clears the prompt cache. Call this during testing or when
    prompt files are updated without restarting the process.
    """
    _prompt_cache.clear()
    log.debug("prompt_cache_cleared")


def load_all_prompts() -> dict[str, str]:
    """
    Loads all required prompts at startup. Fails immediately if any are missing.
    Call this once during application startup to surface missing files early.
    """
    required = ["system", "socrates", "feynman"]
    prompts = {}
    for name in required:
        prompts[name] = load_prompt(name)
        log.info("prompt_ready name=%s version=%s", name, "v1.0")
    return prompts


# ---------------------------------------------------------------------------
# Prompt assembly — positional order matters (Lost in the Middle)
# ---------------------------------------------------------------------------

def assemble_prompt(
    *,
    system: str,
    rag_context: str,
    session_summary: str,
    open_threads: str,
    rolling_window: list[dict],
    user_query: str,
    confidence_warning: str = "",
) -> tuple[str, list[dict]]:
    """
    Assembles the full prompt in correct positional order.
    Returns (system_prompt_str, messages_list) for the Anthropic API.

    Order (most important first and last — Lost in the Middle):
      system          → who the agent is (API system param, not messages)
      session_summary → what has been discussed (near top = well attended)
      open_threads    → what is unresolved (near top)
      rag_context     → source evidence (just before query)
      rolling_window  → recent turns verbatim (just before query)
      user_query      → always last

    Empty components are excluded — never inject empty context.
    """
    messages: list[dict] = []

    # Session context — near the top for strong attention
    if session_summary.strip():
        messages.append({
            "role": "user",
            "content": f"## What we have discussed so far\n{session_summary.strip()}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have this context in mind as we continue."
        })

    # Unresolved threads — near the top
    if open_threads.strip():
        messages.append({
            "role": "user",
            "content": f"## Unresolved threads from previous exchanges\n{open_threads.strip()}"
        })
        messages.append({
            "role": "assistant",
            "content": "Noted. I will return to these if they are relevant."
        })

    # RAG context — just before the query
    if rag_context.strip():
        warning_text = f"\n\n⚠ {confidence_warning}" if confidence_warning else ""
        messages.append({
            "role": "user",
            "content": f"## Relevant source passages\n{rag_context.strip()}{warning_text}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have reviewed these passages."
        })

    # Rolling window — verbatim recent turns just before the query
    messages.extend(rolling_window)

    # User query — always last
    messages.append({"role": "user", "content": user_query})

    return system, messages
