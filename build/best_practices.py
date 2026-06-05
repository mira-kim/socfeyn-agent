"""
philosopher-agent/build/best_practices.py
==========================================
Senior Engineering Reference — Generative AI Application

This file is not executed. It is read.
Every pattern here exists because someone built it wrong first.
Written as if you will not remember why you made a decision,
because six months from now you will not.

Author:  Sr. Engineer reference document
Version: 1.0
Updated: May 2026
"""
# ruff: noqa: E402

# =============================================================================
# TABLE OF CONTENTS
# =============================================================================
#
#  1.  CORE PHILOSOPHY
#  2.  PROJECT STRUCTURE RULES
#  3.  API CALL PATTERNS (ANTHROPIC)
#  4.  PROMPT ENGINEERING IN CODE
#  5.  MEMORY AND CONTEXT MANAGEMENT
#  6.  GRAPH AND DATABASE PATTERNS
#  7.  RAG AND RETRIEVAL PATTERNS
#  8.  ERROR HANDLING AND RESILIENCE
#  9.  COST CONTROL
# 10.  STREAMING AND PERFORMANCE
# 11.  CONFIGURATION MANAGEMENT
# 12.  LOGGING AND OBSERVABILITY
# 13.  TESTING — MINIMUM VIABLE TEST SUITE
# 14.  WHAT NOT TO DO (ANTI-PATTERNS)
# 15.  PHASE-BY-PHASE CHECKLIST
#
# =============================================================================


# =============================================================================
# 1. CORE PHILOSOPHY
# =============================================================================
"""
PRINCIPLE 1 — BUILD DUMB BEFORE SMART
  The most common mistake in generative AI is over-engineering before
  you have empirical signal. Build the simplest thing that produces
  a conversation. Add complexity only when a specific, measured failure
  demands it. Every abstraction you add before you need it is a liability.

PRINCIPLE 2 — ONE THING AT A TIME
  Change one variable per experiment. One prompt tweak, one temperature
  adjustment, one weight change. If you change two things and the score
  improves you have learned nothing. If it gets worse you do not know
  what broke it.

PRINCIPLE 3 — THE SYSTEM IS THE PROMPT
  For generative AI applications, the system prompt is more important
  than any code you write. A bad prompt cannot be fixed with clever code.
  A great prompt with simple code beats a mediocre prompt with complex code
  every time. Invest in the prompt first.

PRINCIPLE 4 — EVERY CALL IS A TEST
  There is no "this is just a quick check" API call. Every call costs
  money and latency. Batch where possible. Cache aggressively. Rate limit
  yourself before the API does it for you.

PRINCIPLE 5 — FAIL LOUDLY AND EARLY
  Generative AI failures are silent by default. The model returns
  something plausible-sounding that is wrong. Build explicit checks
  at every boundary. Validate outputs. Assert invariants. Log everything.
  Silent failures compound into hard-to-debug disasters.

PRINCIPLE 6 — THE DATABASE IS THE SOURCE OF TRUTH
  Not the in-memory state. Not the running session. The SQLite file.
  If the process crashes and restarts, the system should recover
  gracefully from the database with no data loss. Design for crash
  recovery from day one.

PRINCIPLE 7 — SIMPLICITY IS A FEATURE
  Every junior engineer wants to add an abstraction layer. Every senior
  engineer has learned to delete them. If you cannot explain a function
  in one sentence, it is doing too many things. If a module has more
  than one reason to change, split it.
"""


# =============================================================================
# 2. PROJECT STRUCTURE RULES
# =============================================================================
"""
RULE: One responsibility per module.

  agent/      →  only dialogue generation
  memory/     →  only memory read/write
  graph/      →  only graph operations
  monitoring/ →  only observability output
  experiments/→  only snapshot save/load/score
  scripts/    →  only one-time operations (ingest)
  config/     →  only configuration, no logic

  If a module imports from more than three other modules,
  it is doing too much. Refactor.

RULE: No circular imports. Ever.
  The dependency graph is:
  app.py → agent/ → memory/ → graph/ → (nothing)
  monitoring/ reads from experiments/ and graph/
  experiments/ reads from agent/ outputs only
  scripts/ is standalone — imports nothing from the app

RULE: Constants at the top, logic in the middle, I/O at the edges.
  Functions that call the API live at the module boundary.
  Pure functions (no I/O, no API) live in the middle.
  Config and constants never live inside functions.

RULE: Every file starts with a module docstring.
  What it does. What it does NOT do. What it depends on.
  Three lines minimum. Written before the first line of code.
"""

# Example of correct module structure:
_EXAMPLE_MODULE_STRUCTURE = '''
"""
agent/runner.py
---------------
Orchestrates a single dialogue turn: retrieves context,
assembles prompt, calls Sonnet, returns response.

Does NOT: score, save snapshots, update the graph.
Depends on: memory.retriever, agent.voices, agent.moderator
"""

from __future__ import annotations

# stdlib
import logging
from dataclasses import dataclass
from typing import Optional

# third party
import anthropic

# internal — explicit, never wildcard imports
from agent.config_loader import AgentConfig
from agent.voices import build_voice_prompt
from memory.retriever import retrieve_context

log = logging.getLogger(__name__)

# Module-level constants — never buried in functions
MAX_TOKENS = 600
DIALOGUE_MODEL = "claude-sonnet-4-6"
'''


# =============================================================================
# 3. API CALL PATTERNS (ANTHROPIC)
# =============================================================================
"""
PATTERN: Always set max_tokens explicitly.
  Never rely on the default. Unbounded responses are unbounded costs.
  For dialogue: 600 tokens. For scoring: 300 tokens. For extraction: 400.

PATTERN: Always use exponential backoff with jitter on retries.
  The API will rate limit you. It will timeout. It will return 529.
  Every API call must be wrapped in retry logic. Not "most". Every.

PATTERN: Separate the prompt-building function from the API call function.
  This makes prompts testable without spending money.
  prompt = build_prompt(query, context)   # pure, free, testable
  response = call_api(prompt)             # impure, costs money
  result = parse_response(response)       # pure, free, testable

PATTERN: Always check response.stop_reason.
  "end_turn"     →  normal completion
  "max_tokens"   →  response was cut off — increase max_tokens or
                    reduce context. Never silently accept a truncated
                    philosophical argument as complete.
  "stop_sequence" → expected if you use stop sequences
  anything else  →  log and investigate

PATTERN: Log input and output token counts on every call.
  Not just errors. Every call. Token counts are your cost signal
  and your context budget signal. You cannot optimize what you
  do not measure.
"""

import time
import random
import anthropic
from typing import Optional

_client: Optional[anthropic.Anthropic] = None

def get_client() -> anthropic.Anthropic:
    """
    Singleton client. One client, one connection pool.
    Never instantiate anthropic.Anthropic() inside a function
    that gets called per turn — connection overhead compounds.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def call_api_with_retry(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> anthropic.types.Message:
    """
    Every API call goes through this function. No exceptions.

    Handles:
      - Rate limits (429) with exponential backoff + jitter
      - Overload errors (529) with longer backoff
      - Timeouts with retry
      - Logs token usage on every successful call

    Raises:
      RuntimeError after max_retries exhausted
    """
    import logging
    log = logging.getLogger(__name__)
    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
            )

            # Always log token usage — this is your cost signal
            log.info(
                "api_call model=%s input_tokens=%d output_tokens=%d "
                "stop_reason=%s",
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.stop_reason,
            )

            # Warn on truncation — never silently accept it
            if response.stop_reason == "max_tokens":
                log.warning(
                    "Response truncated at max_tokens=%d. "
                    "Consider increasing limit or reducing context.",
                    max_tokens,
                )

            return response

        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            log.warning("Rate limited. Retry %d/%d in %.1fs", attempt + 1, max_retries, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # overloaded
                delay = base_delay * (3 ** attempt) + random.uniform(0, 2)
                log.warning("API overloaded. Retry %d/%d in %.1fs", attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                raise  # don't retry unknown errors

        except anthropic.APITimeoutError:
            delay = base_delay * (2 ** attempt)
            log.warning("API timeout. Retry %d/%d in %.1fs", attempt + 1, max_retries, delay)
            time.sleep(delay)

    raise RuntimeError(f"API call failed after {max_retries} retries")


# =============================================================================
# 4. PROMPT ENGINEERING IN CODE
# =============================================================================
"""
PATTERN: Prompts are data, not strings.
  Every prompt component lives in config/prompts/ as a text file.
  Python code assembles them. Python code never hard-codes
  philosophical instructions. Prompts change more often than code.
  Keep them out of the code.

PATTERN: Build prompts in layers. Each layer has one job.
  layer_1 = load_system_prompt()          # who the agent is
  layer_2 = load_method_prompt(voice)     # how it behaves
  layer_3 = format_rag_context(passages)  # what it knows
  layer_4 = format_memory(session)        # what it remembers
  layer_5 = format_query(user_input)      # what to respond to
  full_prompt = assemble(1, 2, 3, 4, 5)

PATTERN: Important context goes FIRST and LAST in the prompt.
  Models attend poorly to the middle of long contexts.
  (See: Lost in the Middle, Liu et al. 2023)
  Put the system rules first. Put the user query last.
  Put session memory near the top. Put retrieved passages
  just before the query — highest positional relevance.

PATTERN: Version your prompts like code.
  Every change to a prompt is a new version: v1.0, v1.1, v2.0.
  Every version change is logged in experiments/changelog.md
  with the reason, the run IDs that motivated it, and the
  score delta it produced. If you cannot articulate why you
  changed a prompt, do not change it.

PATTERN: Self-check instruction at the end of every voice prompt.
  The last instruction in every voice prompt:
  "Before responding, verify:
   1. Did Socrates ask at least one genuine question? If not, add one.
   2. Did Feynman give a concrete example? If not, add one.
   3. Are the two voices distinguishable? If not, sharpen them."
  This costs a few tokens per turn. It prevents the most common
  failure modes without any post-processing code.
"""

def load_prompt(name: str) -> str:
    """
    Load a prompt from config/prompts/. Never from a string literal.
    Strips leading/trailing whitespace. Fails loudly if file missing.
    """
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent / "config" / "prompts" / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Create config/prompts/{name}.txt before calling this function."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


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
    Assembles the full prompt in the correct positional order.
    Returns (system_prompt, messages_list) ready for the API.

    Order rationale (Lost in the Middle):
      system          → model identity and rules (always first)
      session_summary → what has been discussed (near top)
      open_threads    → what is unresolved (near top)
      rag_context     → source text evidence (just before query)
      rolling_window  → recent verbatim turns (just before query)
      user_query      → the actual question (always last)
    """
    # System prompt is passed directly to the API system param
    # not injected into messages — this gets special positional handling
    full_system = system

    # Messages list — order matters for attention
    messages = []

    if session_summary:
        messages.append({
            "role": "user",
            "content": f"## Session context\n{session_summary}"
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have this context in mind."
        })

    if open_threads:
        messages.append({
            "role": "user",
            "content": f"## Unresolved threads\n{open_threads}"
        })
        messages.append({
            "role": "assistant",
            "content": "Noted. I will return to these if relevant."
        })

    if rag_context:
        warning = f"\n\n{confidence_warning}" if confidence_warning else ""
        messages.append({
            "role": "user",
            "content": f"## Source texts\n{rag_context}{warning}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have reviewed the relevant passages."
        })

    # Rolling window — verbatim recent turns
    messages.extend(rolling_window)

    # User query is always last
    messages.append({"role": "user", "content": user_query})

    return full_system, messages


# =============================================================================
# 5. MEMORY AND CONTEXT MANAGEMENT
# =============================================================================
"""
RULE: Never exceed your context budget.
  Set a hard token limit for context components: 2000 tokens total.
  If you are over budget, cut the least relevant component first.
  Never cut the user query. Never cut the system prompt.
  Cut in this order: open_threads → rag_context → session_summary
  → rolling_window (reduce from n=4 to n=2 before cutting entirely)

RULE: Rolling window is verbatim. Summaries are compressed.
  Never mix them. Rolling window is the last N turns word for word.
  Summary is a compressed representation of everything before that.
  Compressing the rolling window defeats its purpose — you need
  exact wording for the agent to track conversational continuity.

RULE: Rebuild summaries on drift, not on a timer.
  A timer-based summary rebuild is wasteful and often useless.
  (Turn 6 of a 20-turn ethics discussion does not need a new summary.)
  Trigger a rebuild when cosine similarity between the current query
  and the session centroid drops below 0.75. That is when the topic
  has actually shifted enough to matter.

RULE: Summaries are incremental, not full rebuilds.
  Never re-read the full conversation to rebuild a summary.
  Read the existing summary + the new turns since last rebuild.
  Update the summary by appending the delta. This is a tiny Haiku
  call instead of a full context read.

RULE: Memory writes happen after the response is returned.
  Never block the user's response on a memory write.
  Return the dialogue response immediately.
  Write to graph, update snapshot, rebuild summary — all async
  or deferred. The user should never wait for a database write.

RULE: The four layers never cross-contaminate.
  Semantic layer is read-only at runtime. Always.
  No conversation can write to the semantic layer.
  The procedural layer (method nodes) is set at startup and never
  changes during a session. Episodic and working layers are the
  only layers that change during a conversation.
"""

# Context budget constants — define once, reference everywhere
CONTEXT_BUDGET_TOKENS = 2000
ROLLING_WINDOW_TURNS  = 4
SUMMARY_MAX_TOKENS    = 200
RAG_PASSAGES          = 3
DRIFT_THRESHOLD       = 0.75
CONFIDENCE_THRESHOLD  = 0.72


def fits_budget(components: dict[str, str], budget: int = CONTEXT_BUDGET_TOKENS) -> dict[str, str]:
    """
    Enforce context budget. Removes least-critical components first.
    Never removes system prompt or user query.
    Returns a filtered dict that fits within budget.

    Priority order (keep if budget allows):
      rolling_window > rag_context > session_summary > open_threads
    """
    # Rough token estimate: 1 token ≈ 4 characters
    def token_estimate(text: str) -> int:
        return len(text) // 4

    priority = ["rolling_window", "rag_context", "session_summary", "open_threads"]
    result = {}
    used = 0

    for key in priority:
        if key not in components:
            continue
        cost = token_estimate(components[key])
        if used + cost <= budget:
            result[key] = components[key]
            used += cost
        else:
            import logging
            logging.getLogger(__name__).warning(
                "Context budget exceeded. Dropping '%s' (%d tokens). "
                "Used: %d/%d", key, cost, used, budget
            )

    return result


# =============================================================================
# 6. GRAPH AND DATABASE PATTERNS
# =============================================================================
"""
RULE: One database file. philosopher.db. That is the system.
  No separate vector database. No separate session store.
  No JSON files that duplicate what the database contains.
  sqlite-vec handles vectors. SQLite handles everything else.
  One file to back up. One file to inspect. One file to corrupt
  and lose everything — back it up.

RULE: Always use parameterized queries. No exceptions.
  cursor.execute("SELECT * FROM nodes WHERE label = ?", (label,))
  Never: cursor.execute(f"SELECT * FROM nodes WHERE label = '{label}'")
  This is not about security (it is a local app). It is about
  correctness — user input with apostrophes will break f-string queries.

RULE: Use transactions for multi-step writes.
  If a turn extraction writes 3 nodes and 5 edges, that is one
  transaction. If it fails halfway, you want a rollback, not a
  partial write that leaves the graph in an inconsistent state.
  with conn:   ← this is a transaction context manager in sqlite3
      conn.execute(...)
      conn.execute(...)

RULE: Index what you query. Always.
  At minimum: nodes.type, nodes.label, edges.from_id, edges.to_id,
  edges.relation, threads.status, threads.session_id.
  Unindexed queries on a growing graph will become slow.
  Add the index when you create the table, not when queries get slow.

RULE: Embeddings are stored as BLOB. Retrieved as numpy arrays.
  import numpy as np
  # Store:   embedding_bytes = embedding.astype(np.float32).tobytes()
  # Retrieve: embedding = np.frombuffer(row[0], dtype=np.float32)
  Never store embeddings as JSON arrays. 10x the storage, 100x slower.

RULE: Close connections properly. Use context managers.
  with sqlite3.connect(DB_PATH) as conn:
  Never leave connections open between requests. SQLite does not
  handle concurrent writes well. Serialize writes. Parallelize reads.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "philosopher.db"

def get_connection() -> sqlite3.Connection:
    """
    Returns a configured connection. Row factory set for dict-like access.
    WAL mode enabled for better concurrent read performance.
    Call close() when done, or use as context manager.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # access columns by name: row["label"]
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")    # enforce FK constraints
    conn.execute("PRAGMA cache_size=-64000")  # 64MB page cache
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Creates all tables and indexes if they do not exist.
    Idempotent — safe to call on every startup.
    Call this once at application start, not per request.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            label       TEXT NOT NULL,
            source      TEXT,
            metadata    TEXT,  -- JSON
            embedding   BLOB,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS edges (
            id          TEXT PRIMARY KEY,
            from_id     TEXT NOT NULL REFERENCES nodes(id),
            to_id       TEXT NOT NULL REFERENCES nodes(id),
            relation    TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            session_id  TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS threads (
            id          TEXT PRIMARY KEY,
            claim       TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'open',
            from_node   TEXT REFERENCES nodes(id),
            session_id  TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            summary     TEXT,
            topics      TEXT,  -- JSON array
            open_claims TEXT,  -- JSON array
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS turns (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            topics      TEXT,  -- JSON array
            embedding   BLOB,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes — create at schema init, never after the fact
        CREATE INDEX IF NOT EXISTS idx_nodes_type    ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_label   ON nodes(label);
        CREATE INDEX IF NOT EXISTS idx_edges_from    ON edges(from_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to      ON edges(to_id);
        CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
        CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
        CREATE INDEX IF NOT EXISTS idx_turns_session  ON turns(session_id);
    """)


# =============================================================================
# 7. RAG AND RETRIEVAL PATTERNS
# =============================================================================
"""
PATTERN: Retrieve wide, re-rank, inject narrow.
  Step 1: vector_search(query, top_k=20)    — cast wide net
  Step 2: cross_encoder_rerank(query, results)  — re-score on relevance
  Step 3: inject top_k=3 into prompt         — only what matters

  Never skip the re-ranking step. Vector similarity finds related
  passages. The cross-encoder finds relevant passages. These are
  different things. For philosophical text the difference is large.

PATTERN: Gate on confidence before injecting.
  If the top-scoring passage has similarity < 0.72, tell the agent:
  "No high-confidence source text found for this query. Reason from
  general knowledge and state your uncertainty explicitly."
  Never inject low-confidence passages silently — they mislead more
  than they help.

PATTERN: Tag every retrieved passage with its metadata.
  When you inject a passage, inject its metadata too:
  "[Republic, Book I, Jowett translation]
   <passage text>"
  This is what enables the traceability panel and the citation
  requirement. Without metadata in the context, the agent cannot cite.

PATTERN: Embed the hypothetical answer, not the question. (HyDE)
  Short philosophical queries ("what is justice?") match poorly
  against long rich passages in vector space.
  Instead: generate a hypothetical answer first (one sentence),
  embed that, use that embedding for retrieval.
  The hypothetical answer occupies the same semantic space as the
  passage you are looking for.

  hyp_answer = haiku_hypothetical("what is justice?")
  # → "Justice is the harmonious functioning of each part of
  #    society doing what it is best suited to do."
  embedding = embed(hyp_answer)   ← matches Republic passage well
  # vs
  embedding = embed("what is justice?")  ← matches poorly
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Load once at module level — never inside a function that runs per turn
_EMBED_MODEL: SentenceTransformer | None = None
_RERANK_MODEL: CrossEncoder | None = None

def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

def get_rerank_model() -> CrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK_MODEL

def embed(text: str) -> np.ndarray:
    """Single text embedding. Returns float32 array."""
    return get_embed_model().encode(text, normalize_embeddings=True)

def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Batch embedding. Always prefer this over calling embed() in a loop.
    10x faster for ingest. Same cost per token.
    """
    return get_embed_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized vectors.
    If vectors are already normalized (normalize_embeddings=True),
    this is just a dot product. Fast.
    """
    return float(np.dot(a, b))


# =============================================================================
# 8. ERROR HANDLING AND RESILIENCE
# =============================================================================
"""
RULE: Never let an API failure crash the user session.
  If the scoring call fails, log it and continue — the user should
  still get their response. If the memory write fails, log it and
  continue — the session should not end because a graph write timed out.
  Critical path: API dialogue call. Degrade gracefully on everything else.

RULE: Validate LLM outputs before using them.
  The model will return malformed JSON. It will return empty responses.
  It will return a response that does not contain the expected structure.
  Every LLM output that feeds downstream logic must be validated.
  Use a schema. Raise a specific exception on validation failure.
  Log the raw output when validation fails — you will need it to debug.

RULE: Never silence exceptions with bare except.
  except Exception: pass   ← this is how bugs hide for months
  Catch specific exceptions. Log them. Re-raise or handle explicitly.
  If you genuinely do not care about an exception, write a comment
  explaining why, then catch it specifically.

RULE: Use dataclasses for structured outputs. Not dicts.
  A dict return from a function has no schema.
  A dataclass return has a schema, type hints, and IDE support.
  When the LLM extraction returns {"concepts": [], "claims": []},
  parse it into an ExtractionResult dataclass immediately.
  Never pass raw dicts between modules.
"""

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured output from Haiku turn extraction."""
    concepts:      list[str]         = field(default_factory=list)
    claims:        list[dict]        = field(default_factory=list)
    relationships: list[dict]        = field(default_factory=list)
    unresolved:    list[str]         = field(default_factory=list)

    @classmethod
    def from_llm_output(cls, raw: str) -> "ExtractionResult":
        """
        Parse LLM output into ExtractionResult.
        Handles the model returning markdown fences, preamble text,
        and missing keys — all of which happen in practice.
        """
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.error(
                "Failed to parse LLM extraction output as JSON. "
                "Error: %s. Raw output: %r", e, raw[:500]
            )
            return cls()  # return empty result, do not crash

        return cls(
            concepts      = data.get("concepts", []),
            claims        = data.get("claims", []),
            relationships = data.get("relationships", []),
            unresolved    = data.get("unresolved", []),
        )


@dataclass
class DialogueResponse:
    """Structured output from a single dialogue turn."""
    text:           str
    stop_reason:    str
    input_tokens:   int
    output_tokens:  int
    was_truncated:  bool

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate for this turn. Sonnet pricing."""
        return (self.input_tokens * 0.003 + self.output_tokens * 0.015) / 1000

    @classmethod
    def from_api_response(cls, response) -> "DialogueResponse":
        return cls(
            text          = response.content[0].text,
            stop_reason   = response.stop_reason,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            was_truncated = response.stop_reason == "max_tokens",
        )


# =============================================================================
# 9. COST CONTROL
# =============================================================================
"""
RULE: Every API call has a hard max_tokens limit. No exceptions.
  Dialogue:    600 tokens  — Socrates should be concise
  Extraction:  400 tokens  — structured JSON only
  Scoring:     300 tokens  — scores and one-line rationale
  Summarize:   200 tokens  — compressed summary only

RULE: Use prompt caching for static content.
  Your system prompt is identical on every turn.
  Your source text passages are identical for repeated queries.
  Mark them with cache_control: {"type": "ephemeral"}
  Cache hits cost 10% of normal input token cost.
  On a 100-turn session with a 1000-token system prompt:
  Without caching: 100 * 1000 * $0.003/1k = $0.30 just for system prompt
  With caching:    100 * 1000 * $0.0003/1k = $0.03
  Same session. Ten times cheaper.

RULE: Track cumulative cost per session.
  Every DiologueResponse has a cost_estimate_usd property.
  Sum them. Log the session total when the session ends.
  This is how you know if a session is costing $0.05 or $0.50.

RULE: Batch Haiku calls where possible.
  If you need to score AND extract in the same turn, combine them
  into one Haiku call with a structured JSON output schema.
  Two tasks, one call, half the cost and latency.
"""

# Cost constants — update when Anthropic pricing changes
COST_PER_1K_INPUT  = {"claude-sonnet-4-6": 0.003, "claude-haiku-4-5-20251001": 0.001}
COST_PER_1K_OUTPUT = {"claude-sonnet-4-6": 0.015, "claude-haiku-4-5-20251001": 0.005}
CACHE_DISCOUNT     = 0.10  # cache hits cost 10% of normal input


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Returns estimated cost in USD for a single API call."""
    input_cost  = (input_tokens  / 1000) * COST_PER_1K_INPUT.get(model, 0.003)
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT.get(model, 0.015)
    return input_cost + output_cost


class SessionCostTracker:
    """
    Tracks cumulative API cost for a session.
    Instantiate once per session. Pass to every API call wrapper.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.total_usd  = 0.0
        self.call_count = 0
        self._log = logging.getLogger(f"{__name__}.cost")

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
                "Session %s has spent $%.3f (threshold $%.2f). "
                "Check for context window growth or redundant calls.",
                self.session_id, self.total_usd, threshold_usd,
            )


# =============================================================================
# 10. STREAMING AND PERFORMANCE
# =============================================================================
"""
RULE: Load models once. Never per request.
  sentence-transformers models take 1-3 seconds to load.
  The cross-encoder takes another second.
  Load them at application startup as module-level singletons.
  (See: get_embed_model(), get_rerank_model() above.)
  Loading per request makes the first turn of every session terrible.

RULE: Ingest is offline. Never online.
  scripts/ingest.py runs once. It is slow and expensive.
  Never let ingest logic run inside a request handler.
  If you need to add a new thinker, run ingest.py manually,
  then restart the app.

RULE: Memory writes are deferred. Never blocking.
  The sequence is:
    1. Generate response         ← user waits for this
    2. Return response to user   ← user sees output immediately
    3. Write to graph            ← happens after, user does not wait
    4. Update snapshot           ← happens after
    5. Rebuild summary if needed ← happens after
  Use Python's concurrent.futures.ThreadPoolExecutor for deferred writes.
  Or simply: fire-and-forget with threading.Thread(target=write_fn).start()

RULE: Streaming responses are better UX but more complex state.
  For Phase 1, use non-streaming (simpler).
  For Phase 2+, consider streaming via response.stream() for long responses.
  Never start the memory write until the stream is complete.
"""


# =============================================================================
# 11. CONFIGURATION MANAGEMENT
# =============================================================================
"""
RULE: All configuration lives in config/voices.json and .env.
  No magic numbers buried in code.
  No API keys in source files.
  No temperature values hard-coded in functions.

RULE: .env for secrets. voices.json for tuning parameters.
  .env:          ANTHROPIC_API_KEY, DB_PATH, LOG_LEVEL
  voices.json:   weights, temperatures, active thinkers, thresholds

RULE: Never read config inside a hot path.
  Config is read once at startup and cached in a dataclass.
  If you call json.load() inside a function that runs every turn,
  you are doing 60 file reads per minute for no reason.

RULE: Validate config at startup. Fail immediately if invalid.
  Do not discover that socrates_weight: 1.5 is out of range
  on turn 47 of a session. Validate all config values at startup
  and raise a clear error with the fix.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable configuration loaded once at startup.
    frozen=True prevents accidental mutation during a session.
    """
    socrates_weight:    float
    feynman_weight:     float
    temperature:        float
    prompt_version:     str
    active_thinkers:    tuple[str, ...]
    drift_threshold:    float  = DRIFT_THRESHOLD
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    context_budget:     int   = CONTEXT_BUDGET_TOKENS
    rolling_window_n:   int   = ROLLING_WINDOW_TURNS

    def __post_init__(self):
        """Validate at construction time. Fail loudly with clear messages."""
        assert 0 < self.socrates_weight <= 1, \
            f"socrates_weight must be in (0, 1], got {self.socrates_weight}"
        assert 0 < self.feynman_weight <= 1, \
            f"feynman_weight must be in (0, 1], got {self.feynman_weight}"
        assert abs(self.socrates_weight + self.feynman_weight - 1.0) < 0.01, \
            f"Voice weights must sum to 1.0, got {self.socrates_weight + self.feynman_weight}"
        assert 0.0 <= self.temperature <= 1.0, \
            f"temperature must be in [0, 1], got {self.temperature}"
        assert self.context_budget > 500, \
            f"context_budget too small: {self.context_budget}"

    @classmethod
    def load(cls, path: Path | None = None) -> "AgentConfig":
        config_path = path or Path(__file__).parent.parent / "config" / "voices.json"
        raw = json.loads(config_path.read_text())
        return cls(
            socrates_weight  = raw["socrates_weight"],
            feynman_weight   = raw["feynman_weight"],
            temperature      = raw["temperature"],
            prompt_version   = raw["prompt_version"],
            active_thinkers  = tuple(raw["active_thinkers"]),
        )


# =============================================================================
# 12. LOGGING AND OBSERVABILITY
# =============================================================================
"""
RULE: Structured logging. Every log line is machine-parseable.
  Use key=value pairs in log messages. Not prose.
  Good:  log.info("turn_complete session=%s tokens=%d cost=$%.4f", ...)
  Bad:   log.info("Turn completed for session abc123 using 450 tokens")

RULE: Log at the right level.
  DEBUG:   every function call, every token count, every cache hit
  INFO:    turn complete, session start/end, ingest progress
  WARNING: truncated responses, low-confidence retrieval, high cost
  ERROR:   API failures, parse failures, DB write failures
  CRITICAL: startup failures, config validation failures

RULE: Never log the full prompt or response to INFO.
  They are too long. They fill up log files. Log them at DEBUG only,
  and only when a debug flag is explicitly set.
  Do log the first 100 characters as a preview for debugging.

RULE: Separate log streams for cost, performance, and errors.
  file: logs/cost.log    → every API call with token counts and cost
  file: logs/turns.log   → every turn with session, scores, latency
  file: logs/errors.log  → every exception with full traceback
  stderr:                → WARNING and above for terminal visibility
"""

import logging
import logging.handlers
from pathlib import Path

def configure_logging(log_dir: Path | None = None) -> None:
    """
    Call once at application startup. Sets up structured logging
    to rotating files + stderr.
    """
    log_dir = log_dir or Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file — keep last 10 files of 10MB each
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=10_000_000, backupCount=10
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stderr — warnings and above only
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)


# =============================================================================
# 13. TESTING — MINIMUM VIABLE TEST SUITE
# =============================================================================
"""
PHILOSOPHY ON TESTING GENERATIVE AI
=====================================
You cannot unit test a response quality. You can unit test everything
around it. The goal of this test suite is not to verify that Socrates
sounds like Socrates — that is what your rubric and human scoring are for.

The goal is to verify that:
  - The plumbing works (config loads, DB initializes, prompts build)
  - The expensive code is never called in tests (no API calls in unit tests)
  - The contract between modules is respected (input/output schemas)
  - Failure modes are handled correctly (bad JSON, missing files, etc.)
  - Cost-critical logic is correct (token counting, budget enforcement)

TEST CATEGORIES
  Unit tests:         Pure functions. No I/O. No API. Fast.
  Integration tests:  DB operations. File I/O. No API. Medium speed.
  Smoke tests:        One real API call. Validates end-to-end plumbing.
                      Run manually before a deploy, not in CI.
  Evaluation tests:   Run scenarios, check rubric scores. Expensive.
                      Run weekly or before a prompt version change.

RUN TESTS WITH:  pytest build/tests/ -v
                 pytest build/tests/ -v -m "not smoke"   ← skip API calls
                 pytest build/tests/ -v -m "smoke"       ← only smoke tests
"""

# To run these tests:
# pip install pytest pytest-mock
# pytest build/best_practices.py::test_* -v   ← run inline tests below
# or place tests in build/tests/ directory

import pytest
from unittest.mock import patch
from dataclasses import dataclass


# --- CONFIG TESTS ---

class TestAgentConfig:
    """Config loads correctly and validates its own invariants."""

    def test_valid_config_loads(self, tmp_path):
        config_file = tmp_path / "voices.json"
        config_file.write_text(json.dumps({
            "socrates_weight": 0.6,
            "feynman_weight": 0.4,
            "temperature": 0.5,
            "prompt_version": "v1.0",
            "active_thinkers": ["socrates", "feynman"],
        }))
        config = AgentConfig.load(config_file)
        assert config.socrates_weight == 0.6
        assert config.feynman_weight == 0.4
        assert config.temperature == 0.5

    def test_weights_must_sum_to_one(self):
        with pytest.raises(AssertionError, match="sum to 1.0"):
            AgentConfig(
                socrates_weight=0.6,
                feynman_weight=0.6,  # sums to 1.2 — invalid
                temperature=0.5,
                prompt_version="v1.0",
                active_thinkers=("socrates",),
            )

    def test_temperature_out_of_range(self):
        with pytest.raises(AssertionError, match="temperature"):
            AgentConfig(
                socrates_weight=0.6,
                feynman_weight=0.4,
                temperature=1.5,  # invalid
                prompt_version="v1.0",
                active_thinkers=("socrates",),
            )

    def test_config_is_immutable(self):
        config = AgentConfig(
            socrates_weight=0.6, feynman_weight=0.4, temperature=0.5,
            prompt_version="v1.0", active_thinkers=("socrates",),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.temperature = 0.9  # frozen dataclass should raise


# --- EXTRACTION TESTS ---

class TestExtractionResult:
    """LLM output parsing handles all the ways the model can misbehave."""

    def test_valid_json_parses_correctly(self):
        raw = json.dumps({
            "concepts": ["justice", "equality"],
            "claims": [{"speaker": "user", "claim": "justice = equality"}],
            "relationships": [{"from": "justice", "edge": "requires", "to": "equality"}],
            "unresolved": ["whether equality is necessary for justice"],
        })
        result = ExtractionResult.from_llm_output(raw)
        assert "justice" in result.concepts
        assert len(result.claims) == 1
        assert len(result.unresolved) == 1

    def test_markdown_fences_stripped(self):
        raw = "```json\n{\"concepts\": [\"justice\"]}\n```"
        result = ExtractionResult.from_llm_output(raw)
        assert "justice" in result.concepts

    def test_malformed_json_returns_empty_not_crash(self):
        """The agent must never crash because the model returned bad JSON."""
        raw = "Sorry, I could not extract structured data from this turn."
        result = ExtractionResult.from_llm_output(raw)
        assert result.concepts == []
        assert result.claims == []
        assert result.unresolved == []

    def test_missing_keys_use_defaults(self):
        raw = json.dumps({"concepts": ["virtue"]})  # no claims, no relationships
        result = ExtractionResult.from_llm_output(raw)
        assert result.concepts == ["virtue"]
        assert result.claims == []       # default, not KeyError
        assert result.relationships == []


# --- CONTEXT BUDGET TESTS ---

class TestContextBudget:
    """Budget enforcement drops the right components in the right order."""

    def test_all_components_fit(self):
        components = {
            "rolling_window": "a" * 100,
            "rag_context":    "b" * 100,
            "session_summary": "c" * 100,
            "open_threads":   "d" * 100,
        }
        result = fits_budget(components, budget=500)  # 400 chars ≈ 100 tokens, fits
        assert len(result) == 4

    def test_least_critical_dropped_first(self):
        components = {
            "rolling_window": "a" * 2000,   # ~500 tokens — high priority
            "rag_context":    "b" * 2000,   # ~500 tokens
            "session_summary": "c" * 2000,  # ~500 tokens
            "open_threads":   "d" * 2000,   # ~500 tokens — lowest priority
        }
        result = fits_budget(components, budget=1200)
        # open_threads should be dropped first
        assert "open_threads" not in result
        assert "rolling_window" in result

    def test_empty_components_handled(self):
        result = fits_budget({}, budget=2000)
        assert result == {}


# --- DATABASE TESTS ---

class TestDatabase:
    """Schema initializes correctly. Basic read/write works."""

    def test_schema_initializes(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        init_schema(conn)
        # Verify all tables exist
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "nodes" in tables
        assert "edges" in tables
        assert "threads" in tables
        assert "sessions" in tables
        assert "turns" in tables
        conn.close()

    def test_schema_is_idempotent(self, tmp_path):
        """Calling init_schema twice should not raise."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        init_schema(conn)
        init_schema(conn)  # second call should be a no-op
        conn.close()

    def test_node_insert_and_read(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        with conn:
            conn.execute(
                "INSERT INTO nodes (id, type, label) VALUES (?, ?, ?)",
                ("test-001", "concept", "justice")
            )
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", ("test-001",)).fetchone()
        assert row["label"] == "justice"
        assert row["type"] == "concept"
        conn.close()

    def test_foreign_key_enforced(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA foreign_keys=ON")
        init_schema(conn)
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "INSERT INTO edges (id, from_id, to_id, relation) VALUES (?, ?, ?, ?)",
                    ("e001", "nonexistent-node", "also-nonexistent", "related_to")
                )
        conn.close()


# --- PROMPT LOADING TESTS ---

class TestPromptLoading:
    """Prompt files load correctly. Missing files fail loudly."""

    def test_missing_prompt_file_raises_clearly(self, tmp_path):
        """Missing prompt should raise FileNotFoundError, not AttributeError."""
        with pytest.raises(FileNotFoundError, match="nonexistent.txt"):
            # Monkey-patch the path to tmp_path
            with patch("pathlib.Path.__truediv__", return_value=tmp_path / "nonexistent.txt"):
                load_prompt("nonexistent")

    def test_prompt_strips_whitespace(self, tmp_path):
        prompt_dir = tmp_path / "config" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "test.txt").write_text("   \nYou are Socrates.\n   ")
        # Would need to patch the path — shows the pattern
        # In practice: create fixture that sets up config directory


# --- COST TESTS ---

class TestCostTracking:
    """Cost estimates are sane and cumulative tracking works."""

    def test_cost_estimate_sonnet(self):
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        # 1000 * 0.003/1k + 500 * 0.015/1k = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 0.001

    def test_session_cost_accumulates(self):
        tracker = SessionCostTracker("test-session")
        tracker.record("claude-sonnet-4-6", 1000, 500)
        tracker.record("claude-haiku-4-5-20251001", 500, 200)
        assert tracker.total_usd > 0
        assert tracker.call_count == 2

    def test_cost_warning_threshold(self, caplog):
        import logging
        tracker = SessionCostTracker("expensive-session")
        # Simulate an expensive session
        for _ in range(20):
            tracker.record("claude-sonnet-4-6", 2000, 800)
        with caplog.at_level(logging.WARNING):
            tracker.warn_if_expensive(threshold_usd=0.10)
        assert any("spent" in r.message for r in caplog.records)


# --- SMOKE TESTS (marked, skipped in CI) ---

@pytest.mark.smoke
class TestSmokeAPI:
    """
    Real API calls. Run manually before deploying prompt changes.
    Skip in automated CI: pytest -m "not smoke"
    """

    def test_haiku_returns_valid_json_extraction(self):
        """
        Verify that the Haiku extraction prompt returns parseable JSON.
        If this fails, the extraction prompt needs updating.
        """
        client = get_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=(
                "Extract from this conversation turn. "
                "Return ONLY valid JSON with keys: "
                "concepts (list), claims (list), "
                "relationships (list), unresolved (list). "
                "No preamble. No markdown."
            ),
            messages=[{
                "role": "user",
                "content": (
                    "User: I think justice means treating everyone equally.\n"
                    "Socrates: But tell me — if a surgeon and a child both need "
                    "medicine and you have only one dose, does equal treatment serve justice?"
                )
            }],
        )
        result = ExtractionResult.from_llm_output(response.content[0].text)
        # Should have extracted something — not empty
        assert len(result.concepts) > 0 or len(result.claims) > 0, (
            f"Extraction returned nothing. Raw output: {response.content[0].text}"
        )

    def test_sonnet_responds_with_question(self):
        """
        Verify the dialogue model actually asks a question when prompted.
        If this fails, the Socratic system prompt needs strengthening.
        """
        system = (
            "You are Socrates. You always respond with a probing question "
            "that challenges the user's assumption. You never lecture. "
            "You never give direct answers. Ask one question only."
        )
        client = get_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": "Justice means treating everyone equally."}],
        )
        text = response.content[0].text
        assert "?" in text, (
            f"Socratic response should contain a question mark. Got: {text}"
        )


# =============================================================================
# 14. WHAT NOT TO DO (ANTI-PATTERNS)
# =============================================================================
"""
ANTI-PATTERN: Instantiating the API client per call
  ❌  def generate(prompt):
          client = anthropic.Anthropic()   ← new client every call
          return client.messages.create(...)
  ✓   Use get_client() singleton above

ANTI-PATTERN: Loading models inside the turn handler
  ❌  def retrieve(query):
          model = SentenceTransformer("all-MiniLM-L6-v2")  ← 2 second load
          return model.encode(query)
  ✓   Use get_embed_model() singleton above

ANTI-PATTERN: F-string SQL queries
  ❌  conn.execute(f"SELECT * FROM nodes WHERE label = '{label}'")
  ✓   conn.execute("SELECT * FROM nodes WHERE label = ?", (label,))

ANTI-PATTERN: Storing the full conversation history in memory
  ❌  self.history = []
      self.history.append(every_turn_ever)
  ✓   Store turns in SQLite. Load rolling window (last 4) on each turn.

ANTI-PATTERN: Catching and silencing all exceptions
  ❌  try:
          result = call_api(...)
      except:
          pass
  ✓   Catch specific exceptions. Log them. Return a graceful fallback.

ANTI-PATTERN: Hard-coding prompt content in Python strings
  ❌  system = "You are Socrates. Always ask questions. Never lecture..."
                (100 lines of philosophy in a Python string)
  ✓   system = load_prompt("system")  ← reads from config/prompts/system.txt

ANTI-PATTERN: Not validating LLM output before using it
  ❌  data = json.loads(response.content[0].text)  ← crashes on bad JSON
      concepts = data["concepts"]                  ← KeyError possible
  ✓   result = ExtractionResult.from_llm_output(response.content[0].text)

ANTI-PATTERN: Changing two things at once during tuning
  ❌  "I changed the temperature AND the Socrates weight AND updated
      the system prompt. The score went up! Great!"
  ✓   Change one thing. Record the before/after scores. Document the delta.

ANTI-PATTERN: Writing to the semantic layer at runtime
  ❌  # User mentioned Aristotle in conversation
      graph.add_node(type="concept", label="aristotle")   ← corrupts semantic layer
  ✓   Semantic layer is read-only at runtime. Always.

ANTI-PATTERN: Blocking the response on memory writes
  ❌  def handle_turn(query):
          response = generate(query)
          write_to_graph(response)        ← user waits for this
          update_snapshot(response)       ← and this
          rebuild_summary(session)        ← and this
          return response
  ✓   Return the response first. Defer writes with threading.Thread.

ANTI-PATTERN: Using the same model for everything
  ❌  All tasks → claude-sonnet-4-6
  ✓   Dialogue        → claude-sonnet-4-6   (quality critical)
      Extraction      → claude-haiku-4-5   (fast + cheap)
      Scoring         → claude-haiku-4-5   (fast + cheap)
      Summarization   → claude-haiku-4-5   (fast + cheap)
"""


# =============================================================================
# 15. PHASE-BY-PHASE CHECKLIST
# =============================================================================
"""
Use this checklist before marking a phase complete.
Do not start the next phase until every item is checked.

PHASE 1 — WORKING DIALOGUE + TRACEABILITY
  [ ] AgentConfig loads and validates without error
  [ ] All prompt files exist in config/prompts/
  [ ] ingest.py runs to completion, philosopher.db exists
  [ ] At least 5 passage nodes and 3 concept nodes in DB
  [ ] Single turn produces a response with a question mark
  [ ] Traceability panel shows retrieved passages with confidence scores
  [ ] Traceability panel shows which voice rule fired
  [ ] Low confidence warning fires when similarity < 0.72
  [ ] API client is a singleton (check with: grep -r "Anthropic()" agent/)
  [ ] Models loaded at startup (check startup logs for load messages)
  [ ] Unit tests pass: pytest build/tests/ -m "not smoke"
  [ ] Smoke test passes: pytest build/tests/ -m "smoke"

PHASE 2 — SESSION MEMORY + SCORE DISTRIBUTION
  [ ] Session node created on session start
  [ ] Rolling window limited to last 4 turns
  [ ] Incremental summary fires on topic drift, not timer
  [ ] Summary is < 200 tokens (check logs)
  [ ] Memory write is non-blocking (response returns before write completes)
  [ ] Score trend panel shows moving average over last 10 turns
  [ ] Voice balance percentage displayed and correct
  [ ] Context budget enforced — no turn exceeds 2000 context tokens
  [ ] Return to a topic mid-session: agent picks up the thread
  [ ] All Phase 1 checks still pass

PHASE 3 — KNOWLEDGE GRAPH + FAILURE DETECTION
  [ ] Concept nodes populated from ingest (semantic layer)
  [ ] Session nodes created with episodic edges
  [ ] Thread nodes created for unresolved claims
  [ ] Cross-session recall working: agent references past session
  [ ] Failure detector flags when a tag exceeds 30% of recent runs
  [ ] Changelog.md started with first prompt version entry
  [ ] All Phase 1-2 checks still pass

PHASE 4 — TUNING SYSTEM + A/B RUNNER
  [ ] Every run saved as JSON snapshot
  [ ] Human feedback panel works (stars clickable, saves to snapshot)
  [ ] Score thresholds correctly promote/flag runs
  [ ] A/B runner compares two configs on same scenario
  [ ] Score diff calculated and displayed
  [ ] At least 20 scored runs in experiments/runs/
  [ ] At least 3 runs promoted to golden examples
  [ ] All Phase 1-3 checks still pass

PHASE 5 — FINE-TUNING + CAUSAL ANALYSIS
  [ ] 50+ golden examples in experiments/golden/
  [ ] Fine-tuning dataset exported in correct Anthropic format
  [ ] Causal analysis identifies at least one confounded variable
  [ ] Fine-tuned model evaluated against base model on test scenarios
  [ ] All Phase 1-4 checks still pass

BEFORE EVERY PROMPT CHANGE
  [ ] Document the motivation in experiments/changelog.md
  [ ] Record the current average score (before change)
  [ ] Change exactly ONE thing
  [ ] Run 5 scenarios with the new prompt
  [ ] Record the new average score
  [ ] If score did not improve, revert and document why

BEFORE ADDING A LEVEL 2 THINKER
  [ ] Identify the specific gap that motivated the addition
  [ ] Document the gap in experiments/changelog.md
  [ ] Run ingest.py with --role neighbor for the new thinker
  [ ] Verify new concept nodes appear in DB without replacing existing ones
  [ ] Run 5 scenarios that exercise the gap
  [ ] Verify the gap is addressed without degrading other topics
"""


# =============================================================================
# END OF BEST PRACTICES REFERENCE
# =============================================================================
#
# This file is a living document.
# When you discover a new pattern that matters — add it here.
# When a pattern proves wrong — correct it here with a note explaining why.
# When you spend more than 30 minutes debugging something — add the
# diagnosis and fix here so you never spend that 30 minutes again.
#
# Last updated: May 2026
# =============================================================================
