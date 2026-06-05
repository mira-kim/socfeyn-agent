"""
agent/skill_filter.py
----------------------
Topic-triggered soft-flag filter. Runs after dialogue generation.
Checks the response against loaded skill principles.
Never blocks — returns violations for display, never rewrites.

Pipeline position:
  run_dialogue_turn() generates response
      ↓
  skill_filter.check() runs (cheap Haiku call, only when relevant)
      ↓
  violations returned to traceability layer for display
  snapshot flagged if any violations found

Does:    Load skills from YAML, match topics, check principles via Haiku.
Does NOT: Rewrite responses, block output, make Sonnet calls.
Depends on: memory/retriever.py (embed), agent/runner.py (call_api_with_retry)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from memory.retriever import cosine_similarity

def _get_embed_fn():
    """
    Returns the embed function. Separated so tests can monkeypatch it
    without needing a live sentence-transformers model download.
    """
    from memory.retriever import embed
    return embed


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and model
# ---------------------------------------------------------------------------
_SKILLS_DIR  = Path(__file__).parent.parent / "skills"
FILTER_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principle:
    """A single checkable rule within a skill."""
    id:               str
    description:      str
    violation_signals: list[str]
    correction_hint:  str


@dataclass(frozen=True)
class Skill:
    """
    A loaded skill with all its metadata and principles.
    Immutable after loading — skills are read-only at runtime.
    """
    id:                  str
    name:                str
    version:             str
    description:         str
    triggers:            list[str]
    trigger_embeddings:  list[np.ndarray]   # pre-computed at load time
    relevance_threshold: float
    severity:            str
    principles:          list[Principle]


@dataclass
class SkillViolation:
    """A single principle violation detected in a response."""
    skill_id:        str
    skill_name:      str
    principle_id:    str
    excerpt:         str        # the offending phrase or sentence
    explanation:     str        # why this is a violation
    correction_hint: str
    severity:        str

    def as_dict(self) -> dict:
        return {
            "skill_id":        self.skill_id,
            "skill_name":      self.skill_name,
            "principle_id":    self.principle_id,
            "excerpt":         self.excerpt,
            "explanation":     self.explanation,
            "correction_hint": self.correction_hint,
            "severity":        self.severity,
        }


@dataclass
class FilterResult:
    """Everything the skill filter produces for one turn."""
    query_relevance:   dict[str, float]   # skill_id → relevance score
    skills_activated:  list[str]          # skill IDs that fired
    violations:        list[SkillViolation]
    cost_usd:          float = 0.0

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


# ---------------------------------------------------------------------------
# Skill loader — reads YAML, pre-computes trigger embeddings
# ---------------------------------------------------------------------------

_skill_cache: dict[str, Skill] = {}


def load_skills(skills_dir: Path | None = None) -> list[Skill]:
    """
    Loads all .yaml files from the skills directory.
    Caches loaded skills — only reads files once per process.
    Pre-computes trigger embeddings at load time for fast matching.
    """
    global _skill_cache
    if _skill_cache:
        return list(_skill_cache.values())

    directory = skills_dir or _SKILLS_DIR
    if not directory.exists():
        log.warning("skills_dir_missing path=%s", directory)
        return []

    yaml_files = list(directory.glob("*.yaml"))
    if not yaml_files:
        log.info("no_skill_files dir=%s", directory)
        return []

    loaded = []
    for yaml_file in yaml_files:
        try:
            skill = _load_skill_file(yaml_file)
            _skill_cache[skill.id] = skill
            loaded.append(skill)
            log.info(
                "skill_loaded id=%s principles=%d triggers=%d",
                skill.id, len(skill.principles), len(skill.triggers),
            )
        except Exception as e:
            log.error("skill_load_failed file=%s error=%s", yaml_file.name, e)

    return loaded


def _load_skill_file(yaml_file: Path) -> Skill:
    """Parses one YAML skill file and returns a Skill object."""
    raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))

    principles = [
        Principle(
            id               = p["id"],
            description      = p["description"].strip(),
            violation_signals = p.get("violation_signals", []),
            correction_hint  = p.get("correction_hint", "").strip(),
        )
        for p in raw.get("principles", [])
    ]

    triggers = raw.get("triggers", [])

    # Pre-compute trigger embeddings once at load time
    embed = _get_embed_fn()
    trigger_embeddings = [embed(t) for t in triggers]

    return Skill(
        id                  = raw["id"],
        name                = raw["name"],
        version             = raw.get("version", "1.0"),
        description         = raw.get("description", "").strip(),
        triggers            = triggers,
        trigger_embeddings  = trigger_embeddings,
        relevance_threshold = raw.get("relevance_threshold", 0.55),
        severity            = raw.get("severity", "medium"),
        principles          = principles,
    )


def clear_skill_cache() -> None:
    """Clears loaded skill cache. Used in testing."""
    global _skill_cache
    _skill_cache.clear()


# ---------------------------------------------------------------------------
# Topic matching — cosine similarity, no API call
# ---------------------------------------------------------------------------

def compute_relevance(query: str, skill: Skill) -> float:
    """
    Returns the maximum cosine similarity between the query embedding
    and any of the skill's trigger phrase embeddings.
    Pure local computation — free.
    """
    if not skill.trigger_embeddings:
        return 0.0

    embed = _get_embed_fn()
    query_emb = embed(query)
    scores = [cosine_similarity(query_emb, t) for t in skill.trigger_embeddings]
    return float(max(scores))


# ---------------------------------------------------------------------------
# Principle violation check — Haiku call
# ---------------------------------------------------------------------------

_CHECK_SYSTEM = """\
You are checking a philosophical dialogue response for a specific type of assumption.

You will be given:
1. A principle to check against (what the response must NOT do)
2. The response text to check

Your job: determine whether the response violates the principle.

Return ONLY valid JSON:
{{
  "violated": true/false,
  "excerpt": "the exact phrase or sentence that contains the violation, or empty string",
  "explanation": "one sentence explaining why this is a violation, or empty string"
}}

Be precise. Only flag a violation if the response clearly and specifically
violates the principle — not if it merely touches the topic.
No preamble. No markdown. JSON only."""


def _check_principle(
    response_text: str,
    principle: Principle,
    skill: Skill,
) -> Optional[SkillViolation]:
    """
    Checks one principle against the response using Haiku.
    Returns a SkillViolation if violated, None otherwise.

    First does a fast signal-word check to avoid unnecessary API calls.
    Only calls Haiku if at least one signal word is found.
    """
    # Fast pre-check: scan for signal words (free, no API)
    response_lower = response_text.lower()
    signal_found = any(
        signal.lower() in response_lower
        for signal in principle.violation_signals
    )

    if not signal_found:
        return None   # fast path — no signal words, skip API call

    # Signal found — use Haiku to verify it is an actual violation
    from agent.runner import call_api_with_retry

    prompt = (
        f"PRINCIPLE TO CHECK:\n{principle.description}\n\n"
        f"RESPONSE TO CHECK:\n{response_text}"
    )

    try:
        api_response = call_api_with_retry(
            model      = FILTER_MODEL,
            system     = _CHECK_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 250,
            temperature = 0.0,
        )

        raw = api_response.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])

        data = json.loads(raw)

        if not data.get("violated", False):
            return None

        return SkillViolation(
            skill_id        = skill.id,
            skill_name      = skill.name,
            principle_id    = principle.id,
            excerpt         = data.get("excerpt", ""),
            explanation     = data.get("explanation", ""),
            correction_hint = principle.correction_hint,
            severity        = skill.severity,
        )

    except Exception as e:
        log.error(
            "skill_check_failed skill=%s principle=%s error=%s",
            skill.id, principle.id, str(e),
        )
        return None


# ---------------------------------------------------------------------------
# Main filter entry point
# ---------------------------------------------------------------------------

def check(
    query: str,
    response_text: str,
    skills_dir: Path | None = None,
) -> FilterResult:
    """
    Runs all loaded skills against the response.

    Steps:
      1. Load skills (cached after first call)
      2. For each skill: compute topic relevance (free, local)
      3. If relevance > threshold: check each principle (Haiku)
      4. Collect violations
      5. Return FilterResult

    Never raises — on any error returns empty FilterResult.
    A filter failure must never interrupt the user's session.
    """
    try:
        skills = load_skills(skills_dir)
        if not skills:
            return FilterResult(
                query_relevance  = {},
                skills_activated = [],
                violations       = [],
            )

        query_relevance  = {}
        skills_activated = []
        violations       = []
        total_cost       = 0.0

        for skill in skills:
            relevance = compute_relevance(query, skill)
            query_relevance[skill.id] = round(relevance, 3)

            if relevance < skill.relevance_threshold:
                log.debug(
                    "skill_not_activated id=%s relevance=%.3f threshold=%.2f",
                    skill.id, relevance, skill.relevance_threshold,
                )
                continue

            log.info(
                "skill_activated id=%s relevance=%.3f threshold=%.2f",
                skill.id, relevance, skill.relevance_threshold,
            )
            skills_activated.append(skill.id)

            for principle in skill.principles:
                violation = _check_principle(response_text, principle, skill)
                if violation:
                    violations.append(violation)
                    log.warning(
                        "skill_violation skill=%s principle=%s excerpt=%r",
                        skill.id, principle.id, violation.excerpt[:60],
                    )

        return FilterResult(
            query_relevance  = query_relevance,
            skills_activated = skills_activated,
            violations       = violations,
            cost_usd         = total_cost,
        )

    except Exception as e:
        log.error("skill_filter_failed error=%s", str(e), exc_info=True)
        return FilterResult(
            query_relevance  = {},
            skills_activated = [],
            violations       = [],
        )
