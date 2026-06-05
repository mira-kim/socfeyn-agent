"""
agent/runner.py
---------------
Orchestrates a single dialogue turn end to end.

Sequence:
  1. Retrieve RAG context (free)
  2. Build prompt from all layers (free)
  3. Call Sonnet API (costs money)
  4. Auto-score with Haiku (cheap)
  5. Save snapshot (free)
  6. Return response + traceability trace

Does:    Turn orchestration, API calls, cost tracking.
Does NOT: Write to graph (deferred), manage sessions, build UI.
Depends on: agent/config_loader, agent/voices, memory/retriever, graph/graph_db
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import anthropic

from agent.config_loader import AgentConfig, assemble_prompt
from agent.voices import build_voice_prompt
from graph.graph_db import (
    add_turn,
    get_connection,
    get_open_threads,
    get_rolling_window,
    get_session,
    save_snapshot,
)
from memory.retriever import retrieve, RetrievalTrace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
DIALOGUE_MODEL   = "claude-sonnet-4-6"
SCORING_MODEL    = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# API client singleton
# ---------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """
    Singleton Anthropic client.
    Reads ANTHROPIC_API_KEY from environment.
    Never instantiate inside a per-turn function.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
        log.info("anthropic_client_initialized")
    return _client


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------

@dataclass
class DialogueResponse:
    """Validated output from a single dialogue turn."""
    text:          str
    stop_reason:   str
    input_tokens:  int
    output_tokens: int
    was_truncated: bool
    run_id:        str

    @property
    def cost_estimate_usd(self) -> float:
        input_cost  = (self.input_tokens  / 1000) * 0.003
        output_cost = (self.output_tokens / 1000) * 0.015
        return input_cost + output_cost

    @classmethod
    def from_api_response(cls, response, run_id: str) -> "DialogueResponse":
        return cls(
            text          = response.content[0].text,
            stop_reason   = response.stop_reason,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            was_truncated = response.stop_reason == "max_tokens",
            run_id        = run_id,
        )


@dataclass
class AutoScores:
    """Rubric scores from Haiku auto-scoring."""
    question_asked:           int = 0   # 1-5: did Socrates ask a genuine question
    example_given:            int = 0   # 1-5: did Feynman give a concrete example
    voices_distinct:          int = 0   # 1-5: are the two voices distinguishable
    assumption_challenged:    int = 0   # 1-5: was the user's assumption challenged
    uncertainty_acknowledged: int = 0   # 1-5: did the agent acknowledge what it doesn't know

    @property
    def total(self) -> int:
        return (self.question_asked + self.example_given +
                self.voices_distinct + self.assumption_challenged +
                self.uncertainty_acknowledged)

    def as_dict(self) -> dict:
        return {
            "question_asked":           self.question_asked,
            "example_given":            self.example_given,
            "voices_distinct":          self.voices_distinct,
            "assumption_challenged":    self.assumption_challenged,
            "uncertainty_acknowledged": self.uncertainty_acknowledged,
            "total":                    self.total,
        }

    @classmethod
    def empty(cls) -> "AutoScores":
        return cls()


@dataclass
class TurnResult:
    """Everything produced by a single turn — response + all metadata."""
    response:  DialogueResponse
    trace:     RetrievalTrace
    scores:    AutoScores
    run_id:    str
    voice_rule: str


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api_with_retry(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.5,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> anthropic.types.Message:
    """
    Every API call goes through here. Handles rate limits,
    overload errors, and timeouts with exponential backoff.
    Logs token usage on every successful call.
    """
    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            log.info(
                "api_call_complete model=%s input_tokens=%d output_tokens=%d "
                "stop_reason=%s cost_usd=%.4f",
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.stop_reason,
                estimate_cost(model, response.usage.input_tokens, response.usage.output_tokens),
            )

            if response.stop_reason == "max_tokens":
                log.warning(
                    "response_truncated model=%s max_tokens=%d — "
                    "increase limit or reduce context",
                    model, max_tokens,
                )

            return response

        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            log.warning("rate_limited attempt=%d/%d retry_in=%.1fs", attempt+1, max_retries, delay)
            if attempt < max_retries - 1:
                time.sleep(delay)

        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                delay = base_delay * (3 ** attempt) + random.uniform(0, 2)
                log.warning("api_overloaded attempt=%d/%d retry_in=%.1fs", attempt+1, max_retries, delay)
                if attempt < max_retries - 1:
                    time.sleep(delay)
            else:
                log.error("api_error status=%d message=%s", e.status_code, str(e))
                raise

        except anthropic.APITimeoutError:
            delay = base_delay * (2 ** attempt)
            log.warning("api_timeout attempt=%d/%d retry_in=%.1fs", attempt+1, max_retries, delay)
            if attempt < max_retries - 1:
                time.sleep(delay)

    raise RuntimeError(
        f"API call failed after {max_retries} retries. "
        f"Check logs for details."
    )


# ---------------------------------------------------------------------------
# Auto-scoring with Haiku
# ---------------------------------------------------------------------------

_SCORING_SYSTEM = """You are an evaluator for a Socratic dialogue agent.
Score the following response on five criteria, each from 1 to 5.

Return ONLY valid JSON with exactly these keys:
{
  "question_asked": <1-5>,
  "example_given": <1-5>,
  "voices_distinct": <1-5>,
  "assumption_challenged": <1-5>,
  "uncertainty_acknowledged": <1-5>
}

Scoring rubric:
question_asked:           5=genuine probing question targeting the specific claim,
                          3=question present but generic, 1=no question asked
example_given:            5=concrete physical analogy, 3=abstract example,
                          1=no example given
voices_distinct:          5=Socrates and Feynman clearly different styles,
                          3=somewhat similar, 1=indistinguishable
assumption_challenged:    5=specific hidden assumption exposed,
                          3=surface challenge only, 1=no challenge
uncertainty_acknowledged: 5=clearly states limits of knowledge, cites sources or
                          notes when evidence is weak, 3=some hedging but vague,
                          1=overconfident with no acknowledgment of uncertainty

No preamble. No markdown. JSON only."""


def auto_score(user_query: str, response_text: str, max_tokens: int = 300) -> AutoScores:
    """
    Scores a response using Haiku. Returns empty scores on any failure —
    scoring failure must never interrupt the user's session.
    """
    try:
        api_response = call_api_with_retry(
            model=SCORING_MODEL,
            system=_SCORING_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"User said: {user_query}\n\nAgent responded:\n{response_text}"
            }],
            max_tokens=max_tokens,
            temperature=0.0,  # deterministic scoring
        )

        raw = api_response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        data = json.loads(raw)
        return AutoScores(
            question_asked           = int(data.get("question_asked", 0)),
            example_given            = int(data.get("example_given", 0)),
            voices_distinct          = int(data.get("voices_distinct", 0)),
            assumption_challenged    = int(data.get("assumption_challenged", 0)),
            uncertainty_acknowledged = int(data.get("uncertainty_acknowledged", 0)),
        )

    except Exception as e:
        log.error("auto_score_failed error=%s", str(e))
        return AutoScores.empty()


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def fits_budget(
    components: dict[str, str],
    budget: int = 2000,
) -> dict[str, str]:
    """
    Enforces context token budget. Drops least-critical components first.
    Priority: rolling_window > rag_context > session_summary > open_threads
    """
    def token_estimate(text: str) -> int:
        return len(text) // 4  # rough: 1 token ≈ 4 chars

    priority = ["rolling_window", "rag_context", "session_summary", "open_threads"]
    result, used = {}, 0

    for key in priority:
        if key not in components:
            continue
        cost = token_estimate(components[key])
        if used + cost <= budget:
            result[key] = components[key]
            used += cost
        else:
            log.warning(
                "context_budget_exceeded dropping=%s cost_tokens=%d used=%d budget=%d",
                key, cost, used, budget
            )
    return result


# ---------------------------------------------------------------------------
# Main turn function
# ---------------------------------------------------------------------------

def run_dialogue_turn(
    *,
    user_query: str,
    session_id: str,
    config: AgentConfig,
    db_path=None,
) -> TurnResult:
    """
    Executes one complete dialogue turn.

    Args:
        user_query:  What the user just said.
        session_id:  Current session identifier.
        config:      Loaded AgentConfig.
        db_path:     Optional path to SQLite DB (uses default if None).

    Returns:
        TurnResult with response, trace, scores, and metadata.

    Side effects (after response returned):
        - Saves snapshot to database
        - Logs token counts and cost

    Raises:
        ValueError: if user_query is empty
        RuntimeError: if API call fails after all retries
    """
    if not user_query.strip():
        raise ValueError("user_query cannot be empty")

    run_id = str(uuid.uuid4())[:8]
    conn   = get_connection(db_path)

    try:
        # ── Step 1: Build voice prompt ───────────────────────────────────
        voice = build_voice_prompt(config)

        # ── Step 2: Retrieve RAG context ─────────────────────────────────
        rag_context, trace = retrieve(
            user_query,
            confidence_threshold = config.confidence_threshold,
            db_path              = db_path,
        )

        # ── Step 3: Get rolling window from DB ───────────────────────────
        rolling = get_rolling_window(conn, session_id, n=config.rolling_window_turns)

        # ── Step 4: Get session context ───────────────────────────────────
        session = get_session(conn, session_id)
        summary = session["summary"] if session and session["summary"] else ""

        thread_rows = get_open_threads(conn, session_id=session_id, limit=3)
        open_threads_text = ""
        if thread_rows:
            open_threads_text = "\n".join(
                f"- {row['claim']} (unresolved since {row['created_at'][:10]})"
                for row in thread_rows
            )

        # ── Step 5: Enforce context budget ───────────────────────────────
        components = {}
        if rolling:
            components["rolling_window"] = str(rolling)
        if rag_context:
            components["rag_context"] = rag_context
        if summary:
            components["session_summary"] = summary
        if open_threads_text:
            components["open_threads"] = open_threads_text

        budgeted = fits_budget(components, budget=config.context_budget_tokens)

        # ── Step 6: Assemble prompt ───────────────────────────────────────
        system, messages = assemble_prompt(
            system            = voice.system_with_voices,
            rag_context       = budgeted.get("rag_context", ""),
            session_summary   = budgeted.get("session_summary", ""),
            open_threads      = budgeted.get("open_threads", ""),
            rolling_window    = rolling,
            user_query        = user_query,
            confidence_warning = trace.confidence_warning,
        )

        # ── Step 7: Call Sonnet ───────────────────────────────────────────
        api_response = call_api_with_retry(
            model      = DIALOGUE_MODEL,
            system     = system,
            messages   = messages,
            max_tokens = config.max_response_tokens,
            temperature = config.temperature,
        )

        dialogue = DialogueResponse.from_api_response(api_response, run_id)

        # Save turns so get_rolling_window() has content on subsequent turns
        add_turn(conn, turn_id=f"{run_id}_u", session_id=session_id,
                 role="user",      content=user_query)
        add_turn(conn, turn_id=f"{run_id}_a", session_id=session_id,
                 role="assistant", content=dialogue.text)

        log.info(
            "turn_complete run_id=%s session=%s tokens_in=%d tokens_out=%d "
            "cost_usd=%.4f",
            run_id, session_id,
            dialogue.input_tokens, dialogue.output_tokens,
            dialogue.cost_estimate_usd,
        )

        # ── Step 8: Auto-score ────────────────────────────────────────────
        scores = auto_score(
            user_query,
            dialogue.text,
            max_tokens=config.max_scoring_tokens,
        )

        log.info(
            "auto_scores run_id=%s q=%d ex=%d dist=%d challenge=%d total=%d",
            run_id,
            scores.question_asked,
            scores.example_given,
            scores.voices_distinct,
            scores.assumption_challenged,
            scores.total,
        )

        # ── Step 9: Save snapshot ─────────────────────────────────────────
        save_snapshot(
            conn,
            run_id     = run_id,
            session_id = session_id,
            scenario   = user_query,
            config     = config.as_dict(),
            response   = dialogue.text,
            auto_scores = scores.as_dict(),
        )

        write_run_json(
            run_id      = run_id,
            session_id  = session_id,
            scenario    = user_query,
            config      = config.as_dict(),
            response    = dialogue.text,
            auto_scores = scores.as_dict(),
        )
        log.info("run_json_written run_id=%s", run_id)

        return TurnResult(
            response   = dialogue,
            trace      = trace,
            scores     = scores,
            run_id     = run_id,
            voice_rule = voice.voice_rule_fired,
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cost utilities — exported for testing and session tracking
# ---------------------------------------------------------------------------

_RUNS_DIR = Path(__file__).parent.parent / "experiments" / "runs"


def write_run_json(
    *,
    run_id: str,
    session_id: str,
    scenario: str,
    config: dict,
    response: str,
    auto_scores: dict,
    human_scores: dict | None = None,
    combined_score: float | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    promoted: bool = False,
    flagged: bool = False,
) -> Path:
    """Write (or overwrite) the JSON snapshot for a run to experiments/runs/."""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id":     run_id,
        "session_id": session_id,
        "timestamp":  datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario":   scenario,
        "config":     config,
        "response":   response,
        "scores": {
            "auto":     auto_scores,
            "human":    human_scores or {},
            "combined": combined_score,
        },
        "tags":     tags or [],
        "notes":    notes,
        "promoted": promoted,
        "flagged":  flagged,
    }
    path = _RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


COST_PER_1K_INPUT  = {
    "claude-sonnet-4-6":        0.003,
    "claude-haiku-4-5-20251001": 0.001,
}
COST_PER_1K_OUTPUT = {
    "claude-sonnet-4-6":        0.015,
    "claude-haiku-4-5-20251001": 0.005,
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Returns estimated cost in USD for a single API call."""
    input_cost  = (input_tokens  / 1000) * COST_PER_1K_INPUT.get(model, 0.003)
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT.get(model, 0.015)
    return input_cost + output_cost


class SessionCostTracker:
    """Tracks cumulative API cost across all calls in a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.total_usd  = 0.0
        self.call_count = 0
        self._log       = logging.getLogger(f"{__name__}.cost")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_usd  += cost
        self.call_count += 1
        self._log.debug(
            "session=%s call=%d cost=$%.4f total=$%.4f",
            self.session_id, self.call_count, cost, self.total_usd,
        )

    def warn_if_expensive(self, threshold_usd: float = 0.10) -> None:
        if self.total_usd > threshold_usd:
            self._log.warning(
                "Session %s has spent $%.3f (threshold $%.2f).",
                self.session_id, self.total_usd, threshold_usd,
            )
