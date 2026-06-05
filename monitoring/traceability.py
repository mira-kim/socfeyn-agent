"""
monitoring/traceability.py
---------------------------
Builds the traceability record shown in the monitoring panel after every turn.
Reads from the TurnResult produced by agent/runner.py — no API calls, no DB writes.

Does:    Format traceability data for UI display.
Does NOT: Make API calls, write to database, score responses.
Depends on: agent/runner.py (TurnResult), memory/retriever.py (RetrievalTrace)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.runner import TurnResult

log = logging.getLogger(__name__)


@dataclass
class TraceabilityRecord:
    """
    Everything needed for the Phase 1 monitoring panel.
    Populated after every turn from the TurnResult.
    """
    run_id:             str
    query:              str

    # Retrieval
    passages_retrieved: int
    passages_injected:  int
    top_passages:       list[dict]   # [{source, confidence, snippet}]
    below_threshold:    list[dict]   # passages that did not make the cut

    # Voice
    voice_rule_fired:   str
    socrates_word_pct:  float        # % of words spoken by Socrates
    feynman_word_pct:   float        # % of words spoken by Feynman

    # Confidence
    all_above_threshold: bool
    confidence_warning:  str

    # Scores
    auto_scores:         dict
    was_truncated:       bool
    cost_usd:            float

    # Failure flags
    flags:               list[str] = field(default_factory=list)


def build_trace(result: TurnResult) -> TraceabilityRecord:
    """
    Builds a TraceabilityRecord from a completed TurnResult.
    Called immediately after runner.run_dialogue_turn() returns.
    No I/O — pure data transformation.
    """
    trace  = result.trace
    scores = result.scores
    resp   = result.response

    # Format passage details for display
    top_passages = [
        {
            "source":     p.source,
            "thinker":    p.thinker,
            "confidence": round(p.confidence, 3),
            "snippet":    p.text[:120] + "..." if len(p.text) > 120 else p.text,
        }
        for p in trace.passages
    ]

    below_threshold = [
        {
            "source":     p.source,
            "confidence": round(p.confidence, 3),
        }
        for p in trace.below_threshold
    ]

    # Voice balance — count words per labelled voice section
    socrates_pct, feynman_pct = _measure_voice_balance(resp.text)

    # All passages above threshold?
    all_above = not bool(trace.confidence_warning)

    # Build failure flags
    flags = _detect_flags(
        response_text   = resp.text,
        scores          = scores,
        socrates_pct    = socrates_pct,
        feynman_pct     = feynman_pct,
        was_truncated   = resp.was_truncated,
        all_above       = all_above,
    )

    record = TraceabilityRecord(
        run_id              = result.run_id,
        query               = trace.query,
        passages_retrieved  = trace.candidates_found,
        passages_injected   = trace.passages_injected,
        top_passages        = top_passages,
        below_threshold     = below_threshold,
        voice_rule_fired    = result.voice_rule,
        socrates_word_pct   = socrates_pct,
        feynman_word_pct    = feynman_pct,
        all_above_threshold = all_above,
        confidence_warning  = trace.confidence_warning,
        auto_scores         = scores.as_dict(),
        was_truncated       = resp.was_truncated,
        cost_usd            = resp.cost_estimate_usd,
        flags               = flags,
    )

    if flags:
        log.warning("turn_flags run_id=%s flags=%s", result.run_id, flags)

    return record


def _measure_voice_balance(text: str) -> tuple[float, float]:
    """
    Measures the word-count ratio between Socrates and Feynman sections.
    Expects responses to use SOCRATES: and FEYNMAN: labels.
    Returns (socrates_pct, feynman_pct) as 0-100 floats.
    """
    lines     = text.split("\n")
    current   = None
    counts    = {"SOCRATES": 0, "FEYNMAN": 0}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SOCRATES:"):
            current = "SOCRATES"
            rest = stripped[len("SOCRATES:"):].strip()
            counts["SOCRATES"] += len(rest.split())
        elif stripped.startswith("FEYNMAN:"):
            current = "FEYNMAN"
            rest = stripped[len("FEYNMAN:"):].strip()
            counts["FEYNMAN"] += len(rest.split())
        elif current:
            counts[current] += len(stripped.split())

    total = counts["SOCRATES"] + counts["FEYNMAN"]
    if total == 0:
        return 50.0, 50.0

    s_pct = round(counts["SOCRATES"] / total * 100, 1)
    f_pct = round(counts["FEYNMAN"]  / total * 100, 1)
    return s_pct, f_pct


def _detect_flags(
    *,
    response_text: str,
    scores,
    socrates_pct: float,
    feynman_pct: float,
    was_truncated: bool,
    all_above: bool,
) -> list[str]:
    """
    Detects failure conditions and returns a list of flag strings.
    Each flag maps to a known failure mode from the rubric.
    """
    flags = []

    if "?" not in response_text:
        flags.append("NO_QUESTION: Response contains no question mark")

    if "SYNTHESIS:" not in response_text:
        flags.append("NO_SYNTHESIS: Response is missing the SYNTHESIS block")

    if scores.question_asked <= 2:
        flags.append(f"WEAK_QUESTION: score={scores.question_asked}/5")

    if feynman_pct > 55:
        flags.append(f"FEYNMAN_DOMINANT: {feynman_pct}% of words (threshold 55%)")

    if scores.voices_distinct <= 2:
        flags.append(f"VOICES_SIMILAR: score={scores.voices_distinct}/5")

    if scores.assumption_challenged <= 2:
        flags.append(f"WEAK_CHALLENGE: score={scores.assumption_challenged}/5")

    if not all_above and scores.uncertainty_acknowledged <= 2:
        flags.append(f"POOR_ABSTENTION: low-confidence retrieval but agent did not acknowledge uncertainty (score={scores.uncertainty_acknowledged}/5)")

    if was_truncated:
        flags.append("TRUNCATED: Response hit max_tokens limit")

    if not all_above:
        flags.append("LOW_CONFIDENCE: Some passages below confidence threshold")

    return flags


def format_for_display(record: TraceabilityRecord) -> dict:
    """
    Formats the record for Gradio display.
    Returns a dict of display strings for each panel section.
    """
    # Traceability section
    passage_lines = []
    for p in record.top_passages:
        passage_lines.append(
            f"  [{p['source']}] confidence={p['confidence']:.3f}\n"
            f"  \"{p['snippet']}\""
        )

    passages_text = "\n\n".join(passage_lines) if passage_lines else "  No passages retrieved"

    below_text = ""
    if record.below_threshold:
        below_text = "\nBelow threshold (not injected): " + ", ".join(
            f"{p['source']} ({p['confidence']:.2f})"
            for p in record.below_threshold
        )

    traceability = (
        f"Retrieved: {record.passages_retrieved} candidates → "
        f"{record.passages_injected} injected\n\n"
        f"{passages_text}"
        f"{below_text}\n\n"
        f"Voice rule: {record.voice_rule_fired}\n"
        f"Socrates: {record.socrates_word_pct}% words  "
        f"Feynman: {record.feynman_word_pct}% words"
    )

    if record.confidence_warning:
        traceability += f"\n\n⚠ {record.confidence_warning}"

    # Scores section
    s = record.auto_scores
    scores_text = (
        f"Question asked:          {s.get('question_asked', '?')}/5\n"
        f"Concrete example:        {s.get('example_given', '?')}/5\n"
        f"Voices distinct:         {s.get('voices_distinct', '?')}/5\n"
        f"Assumption challenged:   {s.get('assumption_challenged', '?')}/5\n"
        f"Uncertainty acknowledged:{s.get('uncertainty_acknowledged', '?')}/5\n"
        f"─────────────────────────\n"
        f"Turn total:              {s.get('total', '?')}/25\n"
        f"Cost this turn:          ${record.cost_usd:.4f}"
    )

    # Flags section
    if record.flags:
        flags_text = "\n".join(f"⚠ {f}" for f in record.flags)
    else:
        flags_text = "OK  No issues detected"

    return {
        "traceability": traceability,
        "scores":       scores_text,
        "flags":        flags_text,
        "run_id":       record.run_id,
    }
