"""
memory/retriever.py
--------------------
RAG retrieval pipeline: embed query → vector search → rerank → format context.

Pattern: retrieve wide (top_k=20), rerank locally, inject narrow (top_k=3).
Irrelevant context actively hurts quality — be ruthless about what gets in.

Does:    Embed queries, search nodes, rerank, format passages for prompt injection.
Does NOT: Make API calls, write to database, assemble full prompt.
Depends on: graph/graph_db.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from graph.graph_db import vector_search, get_connection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singletons — loaded ONCE at module import, never per request
# ---------------------------------------------------------------------------
_EMBED_MODEL:  SentenceTransformer | None = None
_RERANK_MODEL: CrossEncoder | None = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        log.info("loading_embed_model model=all-MiniLM-L6-v2")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def get_rerank_model() -> CrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        log.info("loading_rerank_model model=cross-encoder/ms-marco-MiniLM-L-6-v2")
        _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK_MODEL


def embed(text: str) -> np.ndarray:
    """Single text → normalized float32 embedding."""
    return get_embed_model().encode(text, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized vectors.
    Vectors normalized at embed time → dot product is sufficient.
    """
    return float(np.dot(a, b))


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Batch embedding — always prefer over calling embed() in a loop.
    Returns 2D array: (n_texts, embedding_dim).
    """
    return get_embed_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


# ---------------------------------------------------------------------------
# Retrieval result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPassage:
    """A single retrieved and reranked passage."""
    node_id:    str
    text:       str
    source:     str          # e.g. "Republic, Book I"
    thinker:    str          # e.g. "plato"
    similarity: float        # vector similarity score (0-1)
    rerank_score: float      # cross-encoder rerank score
    confidence: float        # final confidence used for threshold check

    @property
    def citation(self) -> str:
        """Formatted citation for prompt injection."""
        return f"[{self.source}]"

    def format_for_prompt(self) -> str:
        """Formats the passage for injection into the RAG context block."""
        return f"{self.citation}\n{self.text}"


@dataclass
class RetrievalTrace:
    """
    Traceability record for the monitoring panel.
    Populated after every retrieval — never None.
    """
    query:              str
    candidates_found:   int
    passages_injected:  int
    passages:           list[RetrievedPassage] = field(default_factory=list)
    below_threshold:    list[RetrievedPassage] = field(default_factory=list)
    confidence_warning: str = ""
    used_hyde:          bool = False


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    *,
    confidence_threshold: float = 0.72,
    top_k_candidates: int = 20,
    top_k_inject: int = 3,
    db_path=None,
) -> tuple[str, RetrievalTrace]:
    """
    Full retrieval pipeline for a single query.

    Steps:
      1. Embed query
      2. Vector search top_k_candidates passages
      3. Rerank by cross-encoder relevance
      4. Filter by confidence threshold
      5. Format top_k_inject passages for prompt injection

    Returns:
      (rag_context_string, RetrievalTrace)

    The rag_context_string is ready to inject into the prompt.
    The trace is for the monitoring panel.
    """
    conn = get_connection(db_path)
    trace = RetrievalTrace(query=query, candidates_found=0, passages_injected=0)

    try:
        # Step 1 — embed
        query_embedding = embed(query)

        # Step 2 — vector search, cast wide
        candidates = vector_search(
            conn,
            query_embedding=query_embedding,
            node_type="passage",
            top_k=top_k_candidates,
        )
        trace.candidates_found = len(candidates)

        if not candidates:
            log.warning("retrieval_empty query=%r", query[:80])
            trace.confidence_warning = (
                "No source passages found for this query. "
                "Reasoning from general knowledge."
            )
            return "", trace

        # Step 3 — rerank by actual query relevance
        rerank_inputs = [(query, row["label"]) for row, _ in candidates]
        rerank_scores = get_rerank_model().predict(rerank_inputs, show_progress_bar=False)

        # Step 4 — build RetrievedPassage objects, sort by rerank score
        passages_all = []
        for (row, sim_score), rerank_score in zip(candidates, rerank_scores):
            confidence = float(rerank_score)   # cross-encoder score is the confidence
            p = RetrievedPassage(
                node_id      = row["id"],
                text         = row["label"],    # passage text stored in label
                source       = row["source"] or "Unknown source",
                thinker      = row["thinker"] or "unknown",
                similarity   = sim_score,
                rerank_score = float(rerank_score),
                confidence   = confidence,
            )
            passages_all.append(p)

        passages_all.sort(key=lambda p: p.rerank_score, reverse=True)

        # Step 5 — split into above/below threshold
        above = [p for p in passages_all if p.confidence >= confidence_threshold]
        below = [p for p in passages_all if p.confidence < confidence_threshold]

        trace.below_threshold = below[:3]  # keep a few for monitoring visibility

        if not above:
            log.warning(
                "retrieval_low_confidence query=%r best_score=%.3f threshold=%.2f",
                query[:80],
                passages_all[0].confidence if passages_all else 0.0,
                confidence_threshold,
            )
            trace.confidence_warning = (
                f"Retrieved passages have low confidence "
                f"(best: {passages_all[0].confidence:.2f}, threshold: {confidence_threshold}). "
                f"State your uncertainty explicitly."
            )
            # Still inject top passage but flag it
            inject = passages_all[:top_k_inject]
        else:
            inject = above[:top_k_inject]

        trace.passages          = inject
        trace.passages_injected = len(inject)

        # Step 6 — format for prompt
        formatted = "\n\n".join(p.format_for_prompt() for p in inject)

        log.info(
            "retrieval_complete query=%r candidates=%d injected=%d "
            "top_confidence=%.3f",
            query[:80],
            trace.candidates_found,
            trace.passages_injected,
            inject[0].confidence if inject else 0.0,
        )

        return formatted, trace

    finally:
        conn.close()
