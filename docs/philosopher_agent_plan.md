# Philosopher Agent — Master Plan & Architecture Review

---

## What We Are Building

A **personal thinking partner** that challenges assumptions across any topic.
Not a chatbot. Not a search engine. Not a philosophy quiz.

A system that:
- Interrogates your claims the way Socrates interrogated Athens
- Grounds every challenge in actual source texts
- Remembers your intellectual journey across sessions
- Builds a living map of your thinking over time
- Gets sharper the more you use it
- Makes every response traceable — you can always see why it said what it said
- Surfaces failure patterns so you can fix them systematically

---

## What This Is Not

| Common Misconception | Reality |
|---|---|
| Training an LLM | We are building an application on top of Claude |
| A fine-tuned model | Fine-tuning is Phase 5 — the last step, not the first |
| A simple chatbot | Four distinct memory systems working in concert |
| A search engine over Plato | A structured knowledge graph with typed relationships |
| A static system | Learns and sharpens with every conversation |
| A black box | Every response is traceable to source nodes and graph activations |

---

## The Goal Stated Precisely

> A personal thinking partner that challenges assumptions
> across any topic, drawing from Level 1 philosophical and
> scientific source texts, remembering the full history of
> your intellectual journey, and building structured
> relationships between ideas across sessions.

Every architectural decision traces back to this sentence.

---

## Why This Architecture

### The Core Problem With Simpler Approaches

**Simple chatbot (no memory, no RAG)**
Claude already knows Plato. A simple chatbot works for one session.
The moment you close the window, everything is gone. No thread
continuity. No accumulated understanding. No growing relationship
between ideas. Philosophically interesting for five minutes.
Useless as a thinking partner.

**RAG without graph**
Better — now responses are grounded in source texts. But still
no memory of your specific journey. Every session starts cold.
Cannot track which claims you left unresolved. Cannot notice
when you contradict something you said three sessions ago.
Retrieval without structure.

**Graph without RAG**
The graph knows relationships between concepts but has no
passage-level grounding. Claims cannot be traced to specific
texts. The agent invents connections that feel authentic but
are not in the source material. Philosophically dangerous —
this is how you get a confident but fabricated Socrates.

**What the architecture solves**
All four memory systems working together give you:
- Source grounding (semantic layer — RAG)
- Behavioral consistency (procedural layer — prompts + graph)
- Conversation continuity (episodic layer — session graph)
- Live relationship building (working layer — turn extraction)

---

## The Four Memory Systems

### Why human memory as the model

Human memory is the right model because it is the only proven
system for maintaining coherent reasoning across an unbounded
number of topics over an unbounded time period. The brain does
not store memories as rows in a database. It stores relationships
between concepts and reconstructs memories by traversing those
relationships at recall time. A knowledge graph replicates this
structure explicitly.

---

### Memory System 1 — Semantic Memory
**What it is:** Everything the thinkers knew.
**Human analog:** Knowing what justice means, what first principles are, what the Sophists argued.
**Implementation:** Embedded text chunks + concept nodes in SQLite.
**When it runs:** Built once at ingest. Queried at every turn. Never modified.
**Cost:** ~$3–5 to build. $0.00 forever after.

```
Source texts → chunks → local embeddings → concept nodes → graph edges
     ↓
Permanent read-only foundation
Every concept traceable to a specific passage
Every passage tagged with thinker + source + topic
```

---

### Memory System 2 — Procedural Memory
**What it is:** How each thinker behaves and responds.
**Human analog:** Knowing how to ride a bike — fires automatically, not consciously recalled.
**Implementation:** Method nodes in graph + system prompt rules.
**When it runs:** Loaded into every prompt. Never changes unless you retune.
**Cost:** $0.00 — pure prompt and graph lookup.

```
Socrates method node:
  always questions · never lectures
  opens + closes every exchange
  exposes contradiction before offering alternative
  feigns ignorance · demands precise definitions

Feynman method node:
  first principles only · plain language always
  concrete analogy required · no jargon permitted
  rebuilds from scratch rather than accepting framing
  admits uncertainty explicitly
```

---

### Memory System 3 — Episodic Memory
**What it is:** What happened in past sessions.
**Human analog:** Remembering that last Tuesday you argued about justice and left something unresolved.
**Implementation:** Session nodes in graph with typed edges between sessions.
**When it runs:** Retrieved at session start + when topic drift detected.
**Cost:** $0.00 — local graph traversal.

```
Session 014 node:
  topics: [ethics, justice, equality]
  key claims: ["justice requires equal treatment"]
  challenged by: Socrates (surgeon example)
  status: UNRESOLVED
  linked to: Session 015 (drifted to morals)
             Session 016 (returned to ethics)
```

---

### Memory System 4 — Working Memory
**What it is:** What is alive right now — this turn, this session.
**Human analog:** The four or five things actively in your mind during a conversation.
**Implementation:** Rolling window (last 4 turns verbatim) + live graph extraction each turn.
**When it runs:** Every single turn.
**Cost:** ~$0.001 per turn (Haiku extraction).

```
Each turn:
  1. Add turn to rolling window
  2. Haiku extracts: concepts · claims · relationships
  3. New nodes + edges written to graph
  4. Open threads updated
  5. Topic drift checked → episodic recall if needed
```

---

## The Knowledge Graph

### Why a graph and not a flat database

A flat database stores facts. A graph stores relationships.
The difference matters enormously for a thinking partner.

Flat database answer to "what did we discuss about justice?":
> "Session 014: justice. Session 016: justice."

Graph answer to the same query:
> "In session 014 you claimed justice requires equal treatment.
> Socrates challenged this with the surgeon example. You never
> resolved it. In session 016 you returned to justice via morality.
> The unresolved thread is still open."

The graph answer is what makes this a thinking partner rather than a search engine.

---

### Node Types

| Type | Role | Example | Layer |
|---|---|---|---|
| concept | What things mean | "justice", "first principles" | Semantic |
| passage | Source text chunk | Republic Bk I passage 47 | Semantic |
| method | How thinkers behave | Socrates method rules | Procedural |
| session | Past conversations | Session 014 — ethics | Episodic |
| claim | User's stated positions | "justice = equality" | Episodic/Working |
| thread | Unresolved questions | surgeon example unresolved | Working |

---

### Edge Types

| Relation | Meaning | Example |
|---|---|---|
| source_text_for | Passage grounds concept | Republic passage → justice concept |
| related_to | Concepts connect | justice ↔ equality |
| contradicts | Tension between ideas | justice ↔ "advantage of the stronger" |
| voiced_by | Concept raised by thinker | justice concept → Socrates |
| challenges | Claim under examination | user claim → Socrates question |
| drifted_to | Topic shift | session 014 → session 015 |
| returns_via | User came back | session 016 → session 014 |
| unresolved_in | Claim never settled | surgeon claim → session 014 |

---

### Edge Weights

Edges gain weight every time they are activated. After five sessions
where justice and equality co-occur, the edge weight is 5.0. This is
your graph learning which connections matter most to **you specifically**
— not just what is philosophically true in general. A well-used graph
becomes a personalized map of your intellectual landscape.

---

## The Corpus Architecture

### Level 0 — Primary Voices (speak directly)

**Socrates via Plato**
```
Apology          Socrates defending his method — essential
Meno             Virtue and knowledge — can they be taught
Republic         Justice and the ideal — most cited
Phaedo           Knowledge and the soul
Theaetetus       What is knowledge — most important for your goal
Gorgias          Rhetoric vs philosophy
Protagoras       Virtue and the Sophists
Euthyphro        Definition and the method in pure form
Phaedrus         Love, rhetoric, the nature of knowledge
Symposium        Love and the examined life
```

**Socrates via Xenophon**
```
Memorabilia      Socrates in daily life — different portrait
Apology          Cross-reference with Plato's version
```

**Feynman**
```
Feynman Lectures on Physics      First principles method in action
The Character of Physical Law    How science actually works
Surely You're Joking Mr Feynman  Reasoning style in plain language
What Do You Care What Others Think  Curiosity as a way of life
```

---

### Level 1 — Direct Context (enrich primary voices, never speak)

**Challengers — who Socrates argued against**
```
Aristotle Nicomachean Ethics     Directly challenges Plato on virtue
Aristotle Metaphysics Book 1     Challenges the theory of forms
Protagoras fragments             Man is the measure of all things
Gorgias fragments                Nothing can be known or communicated
Thrasymachus fragments           Justice is the advantage of the stronger
```

**Intellectual descendants**
```
Epictetus Enchiridion            Socratic method in daily life
Marcus Aurelius Meditations      Practical Socratic reasoning
Zeno of Citium fragments         Foundation of Stoic thought
```

**Feynman's scientific context**
```
Einstein Relativity (popular)    What Feynman built on
Bohr selected papers             What Feynman argued against
Wheeler Geons Black Holes        Feynman's direct collaborator
Dirac Principles of QM           What Feynman's QED extends
```

---

### Level 2 — Selective Expansion (add when a specific gap is hit)

| Gap | Thinkers | Add When |
|---|---|---|
| Logic gap | Russell, Wittgenstein, Frege | User hits logical structure walls |
| Epistemology gap | Popper, Hume, Descartes | Falsifiability challenges arise |
| Modern ethics gap | Mill, Kant, Nietzsche | Rights/duties/consequences dominate |
| Practical gap | Munger, Taleb, Kahneman | User needs applied decision lens |

**Design principle:** Each Level 2 addition is one command:
```bash
python scripts/ingest.py \
  --file texts/popper_logic_discovery.txt \
  --thinker "Popper" \
  --role "neighbor" \
  --topics "knowledge,science,falsifiability"
```

---

### Source Texts — Where to Get Them Free

| Source | URL | What to get |
|---|---|---|
| Project Gutenberg | gutenberg.org | All Plato (Jowett), Xenophon, Aristotle, Epictetus, Marcus Aurelius |
| Perseus Digital Library | perseus.tufts.edu | Greek originals + translations, Sophist fragments |
| Feynman Lectures | feynmanlectures.caltech.edu | Complete lectures free online |
| Internet Archive | archive.org | Feynman books, Wheeler, Bohr papers |

**Translation recommendation:** Jowett for Plato. Public domain, free,
philosophically faithful. Cooper (Hackett) is more precise if you can
source the text — better for concept extraction.

---

## The Ingest Pipeline (runs once)

```
Step 1   Read source files
         Pure Python file I/O — $0.00

Step 2   Chunk into passages
         500 tokens, 50 token overlap
         Tagged: thinker · source · topic · role
         Pure Python — $0.00

Step 3   Embed locally
         sentence-transformers all-MiniLM-L6-v2
         Runs on your machine
         384-dimension vector per chunk
         No API call — $0.00

Step 4   Extract concept nodes
         One Haiku batch job
         Extracts: concepts · claims · definitions · relationships
         Writes nodes + edges to SQLite
         Cost: ~$3–5 once, never again

Step 5   Build semantic graph edges
         related_to · contradicts · defined_as · argued_by
         Pure Python graph operations — $0.00

Result   Permanent semantic foundation
         ~4,000 passage nodes
         ~800 concept nodes
         ~3,000 typed edges
         Queryable forever at zero cost
```

---

## The Runtime Architecture

### Per-Turn Flow

```
User query
    ↓
[1] Topic drift check          free — cosine similarity local
    ↓ (if drift detected)
[2] Episodic recall            free — graph traversal
    ↓
[3] Semantic retrieval         free — vector search local
    Retrieve top 20
    Cross-encoder re-rank      free — local model
    Inject top 3
    ↓
[4] Context budget assembly    free — Python
    Gate each component by relevance > 0.6
    Fit to 2000 token budget
    Most relevant wins if over budget
    ↓
[5] Prompt assembly            free — Python string ops
    system prompt
    + method nodes (procedural)
    + session summary (episodic)
    + retrieved passages (semantic)
    + open threads (working)
    + rolling window 4 turns (working)
    + user query
    ↓
[6] Sonnet dialogue turn       ~$0.005 — only paid step
    ↓
[7] Haiku extraction           ~$0.001
    Extract: concepts · claims · relationships · threads
    ↓
[8] Graph write                free — SQLite
    New nodes + edges
    Update thread status
    Update edge weights
    ↓
[9] Auto-score                 ~$0.001 — Haiku
    Rubric: question · example · distinct · challenged
    ↓
[10] Save snapshot             free — SQLite write
     config · scores · output · tags
    ↓
Agent response + scores shown to user
User adds human feedback scores
```

**Total cost per turn: ~$0.007**
**100 full sessions: ~$0.70**

---

## The Tuning System

### Automated scoring (runs every turn)

| Criterion | What it checks | Model |
|---|---|---|
| Genuine question asked | Did Socrates ask rather than tell | Haiku |
| Concrete example given | Did Feynman ground it physically | Haiku |
| Voices are distinct | Do they sound different from each other | Haiku |
| Assumption challenged | Did the agent actually push back | Haiku |

### Human scoring (user adds after reading)

| Criterion | What it checks |
|---|---|
| Challenged me specifically | Not generic — targeted at my actual claim |
| Felt authentic | Socrates or a chatbot performing Socrates |
| Produced new insight | The whole point — did it move my thinking |

### Score thresholds

| Score | Action |
|---|---|
| ≥ 28/35 | Auto-promote to golden examples folder |
| 20–27 | Log and continue |
| 12–19 | Flag for prompt debugging |
| < 12 | Discard — exclude from all training data |

### Failure mode tags (user selects)

```
Feynman too dominant
Socrates lectured
Both voices agreed too easily
Too abstract — no grounding
Too surface level
Ignored my actual premise
Great — no issues
```

### Experiment snapshots

Every run saves:
```json
{
  "run_id": "run_042",
  "timestamp": "2026-05-17T14:32:00Z",
  "scenario": "What is justice?",
  "config": {
    "socrates_weight": 0.6,
    "feynman_weight": 0.4,
    "temperature": 0.5,
    "prompt_version": "v1.3"
  },
  "scores": {
    "auto": { "question": 4, "example": 3, "distinct": 5, "challenged": 4 },
    "human": { "challenged_me": 3, "authentic": 4, "new_insight": 5 },
    "combined": 24
  },
  "tags": ["Too abstract"],
  "notes": "Feynman gave great analogy but Socrates never returned to challenge it",
  "promoted": false,
  "flagged": false
}
```

---

## The Feedback Flywheel

```
Run scenario
    ↓
Save snapshot
    ↓
Score (auto + human)
    ↓
Low score → diagnose → fix one prompt variable → rerun
High score → promote to golden examples folder
    ↓
50+ golden examples → fine-tune (Phase 5)
```

**The compounding effect:**
Each promoted example makes future runs more likely to score highly.
Each fixed prompt variable eliminates a failure mode permanently.
Each fine-tuning cycle bakes quality into the model itself.

---

## The Monitoring Layer

### Origin and Philosophy

Derived from Microsoft's Responsible AI Toolbox concepts, translated
from traditional ML framing into generative AI terms. The RAI Toolbox
was designed for trained classification models — but six of its core
concepts map directly onto what your agent needs. The implementation
is entirely custom, built on your existing snapshot system.

The goal is not compliance or safety theatre. The goal is
**systematic visibility** — knowing why the agent behaved as it did,
detecting failure patterns before they compound, and making tuning
decisions based on evidence rather than intuition.

---

### The Six Monitoring Concepts and Their Translations

**1. Model Statistics → Response Distribution Monitoring**
RAI original: Track prediction distribution across metrics and subgroups.
Your version: Track how outputs distribute across rubric dimensions
over time — is the question-asking rate trending up or down, which
topics produce the lowest insight scores, is voice balance drifting.

**2. Data Explorer → Conversation Pattern Explorer**
RAI original: Visualize how data clusters, find underrepresented groups.
Your version: Explore which topic + config combinations produce high
scores vs low scores. Find the conditions under which the agent
consistently underperforms.

**3. Interpretability → Response Traceability**
RAI original: Explain which features drove a prediction, globally and locally.
Your version: For every response, surface exactly what was retrieved,
which graph nodes were activated, which voice rule fired, which thread
was recalled. Every claim traceable to a source node. This is already
in your architecture as the hard grounding requirement — monitoring
makes it visible.

**4. Error Analysis → Failure Pattern Detection**
RAI original: Find cohorts of data where the model systematically fails.
Your version: Find patterns in low-scoring runs. Not random failures
but systematic ones — which configuration combinations, topic types,
and session lengths consistently produce bad output.

**5. Counterfactual / What-If → Prompt Perturbation Testing**
RAI original: What would the model predict if this input changed?
Your version: Run the same scenario through two different
configurations and compare outputs and scores side by side. The A/B
layer for your tuning system.

**6. Causal Inference → Tuning Impact Analysis**
RAI original: Does this intervention actually cause a change in outcome?
Your version: Did changing temperature actually cause score improvement,
or was it the prompt version change that happened at the same time?
Isolate variables across your experiment log.

---

### The MVP Monitoring Panel

A single panel in the Gradio UI, shown alongside every response.
Built entirely from data the pipeline already produces.
No third-party tool. No Azure dependency. Pure Python.

```
┌─ TRACEABILITY ──────────────────────────────────────────────┐
│  Retrieved:   Republic Bk I p.47 (confidence 0.89)         │
│               Feynman Lectures Ch.1 (confidence 0.81)      │
│  Graph nodes: justice → equality (edge weight 3.2)         │
│  Thread:      Session 014 unresolved claim recalled        │
│  Voice rule:  Socrates — expose contradiction              │
│  Uncertainty: all passages above 0.72 threshold ✓         │
└─────────────────────────────────────────────────────────────┘

┌─ SCORES ────────────────────────────────────────────────────┐
│  Auto:   question 4  example 3  distinct 5  challenged 4   │
│  Trend (last 10 runs): 23.4 avg  ↑ from 21.2              │
│  This topic avg: 21.2  vs overall avg: 22.8               │
└─────────────────────────────────────────────────────────────┘

┌─ FAILURE FLAGS ─────────────────────────────────────────────┐
│  ⚠ Voice balance: Feynman 58% of words (threshold 50%)    │
│  ✓ Question asked                                          │
│  ✓ Assumption challenged                                   │
│  ✓ Source cited                                            │
│  ✓ Confidence above threshold                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Monitoring Build Order

Monitoring is added incrementally alongside the agent phases.
Each monitoring capability requires the agent phase beneath it to
exist first — you cannot monitor what has not been built.

```
Phase 1  Traceability panel
         Show retrieved passages + confidence scores
         Show which voice rule fired
         Show uncertainty flag when confidence < 0.72
         Most immediately useful — verifies RAG is working

Phase 2  Score distribution panel
         Running average per topic
         Trend line across last N runs
         Voice balance percentage per session
         Needs 20+ runs to be meaningful

Phase 3  Failure pattern detection
         Automated cohort flagging from tagged runs
         Alert when a failure tag exceeds 30% threshold
         Systematic pattern surfacing across experiment log
         Needs 30+ tagged runs

Phase 4  Counterfactual A/B runner
         Same query, two configs, side-by-side comparison
         Score diff automatically calculated
         Needs working tuning system and experiment tracker

Phase 5  Causal analysis
         Variable isolation across experiment log
         Did prompt v1.4 cause improvement independent of topic
         Needs 50+ scored runs to be statistically meaningful
```

---

### What Monitoring Does NOT Do

Important boundaries to maintain:

- It does not make decisions — it surfaces information for you to act on
- It does not auto-fix prompts — human judgment required for all changes
- It does not replace human scoring — automated scores are signals not verdicts
- It does not monitor for safety/harm — your use case does not require that
- It is not a compliance dashboard — it is a quality improvement tool

---

## The File Structure

```
philosopher-agent/
│
├── scripts/
│   └── ingest.py              # run once to build semantic layer
│
├── config/
│   ├── voices.json            # weights, temperature, active thinkers
│   └── prompts/
│       ├── system.txt         # master rules
│       ├── socrates.txt       # Socratic method instructions
│       └── feynman.txt        # first principles instructions
│
├── agent/
│   ├── runner.py              # main conversation loop
│   ├── moderator.py           # turn order + steelman trigger
│   ├── voices.py              # weight application
│   ├── auto_score.py          # rubric checks post-response
│   └── config_loader.py       # reads config + prompts
│
├── memory/
│   ├── memory_manager.py      # coordinates all four layers
│   ├── summarizer.py          # incremental summary on drift
│   ├── extractor.py           # Haiku turn extraction
│   └── retriever.py           # retrieve + rerank pipeline
│
├── graph/
│   ├── graph_db.py            # SQLite graph operations
│   ├── nodes.py               # node type definitions
│   └── edges.py               # edge type definitions
│
├── monitoring/
│   ├── traceability.py        # Phase 1 — retrieval + graph visibility
│   ├── distributor.py         # Phase 2 — score trends + voice balance
│   ├── failure_detector.py    # Phase 3 — pattern detection + alerts
│   ├── ab_runner.py           # Phase 4 — counterfactual config testing
│   ├── causal.py              # Phase 5 — variable isolation analysis
│   └── panel.py               # Gradio panel component (all phases)
│
├── experiments/
│   ├── tracker.py             # saves every run as snapshot
│   ├── scorer.py              # auto + human scoring
│   ├── changelog.md           # prompt version history + rationale
│   └── runs/                  # one JSON per run
│
├── texts/                     # your source texts
│   ├── plato/
│   ├── feynman/
│   ├── context/               # Level 1 context texts
│   └── neighbors/             # Level 2 additions over time
│
├── philosopher.db             # single SQLite file — everything
├── app.py                     # Gradio UI entry point
└── requirements.txt
```

---

## The Technology Stack

| Layer | Library | Why | Cost |
|---|---|---|---|
| LLM dialogue | anthropic SDK | Claude Sonnet — quality | ~$0.005/turn |
| Scoring + extraction | anthropic SDK | Claude Haiku — cheap | ~$0.001/call |
| Local embeddings | sentence-transformers | No API cost | $0.00 |
| Vector + relational | sqlite-vec | One file, zero infra | $0.00 |
| Re-ranking | sentence-transformers cross-encoder | Local, no API | $0.00 |
| UI + monitoring panel | Gradio | Fast, runs locally, tabbed UI | $0.00 |
| Text chunking | nltk | Sentence-aware chunking | $0.00 |
| Monitoring charts | Plotly (via Gradio) | Score trends, distributions | $0.00 |
| Experiment tracking | Custom JSON + SQLite | Already in architecture | $0.00 |

**Total infrastructure cost: $0.00**
**Only cost is Anthropic API tokens**

### Model Routing

```python
DIALOGUE_MODEL   = "claude-sonnet-4-6"        # quality critical
EXTRACTION_MODEL = "claude-haiku-4-5-20251001" # fast + cheap
SCORING_MODEL    = "claude-haiku-4-5-20251001" # fast + cheap
SUMMARY_MODEL    = "claude-haiku-4-5-20251001" # fast + cheap
```

Sonnet for dialogue. Haiku for all background tasks.
This separation keeps quality high where it matters and
cost low everywhere else.

---

## The Build Phases

### Phase 1 — Working Dialogue + Traceability
*Goal: Agent speaks, conversation flows, every response is traceable*

Deliverables:
- scripts/ingest.py (semantic layer built once)
- agent/runner.py (conversation loop)
- agent/voices.py (Socrates + Feynman with weights)
- config/prompts/ (system + voice prompts)
- app.py (Gradio UI with two panels: dialogue + monitoring)
- monitoring/traceability.py (retrieved passages, graph nodes, voice rule, confidence flag)
- monitoring/panel.py (Gradio panel component, Phase 1 version)

Success criterion: 10 turns of authentic Socratic dialogue
                   grounded in source texts. After each turn,
                   monitoring panel shows exactly what was
                   retrieved and which voice rule fired.

---

### Phase 2 — Session Memory + Score Distribution
*Goal: Agent remembers this conversation end to end. Monitoring shows trends.*

Deliverables:
- memory/memory_manager.py
- memory/summarizer.py (incremental, drift-triggered)
- memory/extractor.py (Haiku turn extraction)
- graph/graph_db.py (session + claim nodes)
- monitoring/distributor.py (score trends, voice balance, per-topic averages)
- monitoring/panel.py updated (Phase 2 — adds score trend tab)

Success criterion: Return to a topic mid-session and have agent
                   pick up the thread. Monitoring panel shows
                   voice balance percentage and score trend
                   across the session.

---

### Phase 3 — Knowledge Graph + Failure Pattern Detection
*Goal: Full four-layer memory, cross-session continuity. Monitoring detects systematic failures.*

Deliverables:
- graph/nodes.py + edges.py (full type system)
- memory/retriever.py (retrieve wide, rerank, inject narrow)
- Episodic session nodes with typed edges
- Open thread tracker
- monitoring/failure_detector.py (cohort flagging, threshold alerts)
- monitoring/panel.py updated (Phase 3 — adds failure patterns tab)
- experiments/changelog.md (prompt version history starts here)

Success criterion: Return to ethics after discussing morals and
                   have agent cite the specific unresolved claim
                   from a previous session. Monitoring flags when
                   any failure tag exceeds 30% of recent runs.

---

### Phase 4 — Tuning System + Counterfactual A/B Runner
*Goal: Systematic improvement with tracked experiments. Side-by-side config comparison.*

Deliverables:
- experiments/tracker.py
- experiments/scorer.py
- UI scoring panel (human feedback)
- Changelog system for prompt versions
- monitoring/ab_runner.py (same query, two configs, score diff)
- monitoring/panel.py updated (Phase 4 — adds A/B comparison tab)

Success criterion: 20 scored runs, at least 3 promoted to golden
                   examples. A/B runner confirms which config
                   change caused score improvement.

---

### Phase 5 — Fine-Tuning + Causal Analysis
*Goal: Bake quality into the model itself. Verify that interventions caused improvements.*

Deliverables:
- 50+ golden example runs
- Fine-tuning dataset formatted for Anthropic API
- Evaluation suite comparing base vs fine-tuned
- monitoring/causal.py (variable isolation across experiment log)
- monitoring/panel.py updated (Phase 5 — adds causal analysis tab)

Success criterion: Fine-tuned model scores higher on rubric without
                   heavy system prompt. Causal analysis confirms
                   which prompt changes independently caused
                   score improvement vs which were confounded.

---

## The Five Principles From Best-In-Class Systems

Learned from Perplexity, Cursor, NotebookLM, Mem0:

**1. Retrieve wide, inject narrow**
Cast top-20 vector search. Re-rank locally. Inject top-3.
Irrelevant context actively hurts quality.

**2. Budget your context window**
Hard 2000 token limit for context components.
Gate each by relevance score > 0.6 before injection.

**3. Hard grounding over soft blending**
Every claim cites a source node.
If confidence < 0.72, state uncertainty explicitly.
Never invent connections between thinkers.

**4. Memory is selective not comprehensive**
Extract only what changes future behavior.
If nothing is worth remembering, store nothing.
Noise in memory is worse than gaps.

**5. Separation of concerns**
Retrieval, memory, dialogue, scoring, and monitoring are five systems.
Clean interfaces between them.
Never one tangled pipeline.

**6. Monitor what you cannot see**
Every response should be explainable after the fact.
Traceability is not optional — it is how you know the system
is working and how you fix it when it is not.
Build monitoring alongside the feature it monitors, not after.

---

## Cost Summary

| Activity | Cost |
|---|---|
| Full semantic ingest (once ever) | ~$3–5 |
| Each Level 2 thinker addition | ~$0.50–2 |
| Per dialogue turn (Sonnet) | ~$0.005 |
| Per turn memory extraction (Haiku) | ~$0.001 |
| Per turn auto-scoring (Haiku) | ~$0.001 |
| Session summary rebuild (on drift) | ~$0.001 |
| Traceability panel (Phase 1) | $0.00 — reads existing data |
| Score distribution panel (Phase 2) | $0.00 — aggregates snapshots |
| Failure pattern detection (Phase 3) | $0.00 — pure Python |
| A/B runner (Phase 4) | ~$0.010 per comparison run |
| Causal analysis (Phase 5) | $0.00 — statistical ops on logs |
| All storage, retrieval, graph ops | $0.00 |
| **100 full conversation sessions** | **~$0.70** |
| **1000 full conversation sessions** | **~$7.00** |

Monitoring adds zero ongoing cost to normal usage.
Only the A/B runner costs extra — you control when it runs.

---

## Literature and Reference Reading

See the Reading List section below for everything worth studying
before and during the build.

---

# Reading List

Organized by what each work teaches you and when to read it.

---

## Read Before Writing Any Code

### On RAG Architecture

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
Lewis et al., 2020 — the original RAG paper
*What it teaches:* The foundational architecture. Every RAG system
is a variation of what this paper describes. Read the abstract and
sections 2-3. The math is optional — the concepts are essential.
https://arxiv.org/abs/2005.11401

**Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)**
Gao et al., 2022
*What it teaches:* Instead of embedding the query directly, generate
a hypothetical answer first then embed that. Dramatically improves
retrieval quality for philosophical questions where the query is
short but the relevant passage is rich. Directly applicable to your
semantic layer.
https://arxiv.org/abs/2212.10496

**REALM: Retrieval-Augmented Language Model Pre-Training**
Guu et al., 2020 — Google Brain
*What it teaches:* How retrieval and generation interact at a deep
level. Builds intuition for why retrieval quality cascades into
response quality.
https://arxiv.org/abs/2002.08909

---

### On Knowledge Graphs for LLMs

**From Local to Global: A Graph RAG Approach**
Edge et al., 2024 — Microsoft Research
*What it teaches:* GraphRAG — the most important paper for your
specific architecture. Microsoft built exactly what we are designing:
a knowledge graph extracted from source texts used to ground LLM
responses. Their entity extraction prompt and community summarization
approach are directly applicable. Read this carefully.
https://arxiv.org/abs/2404.16130

**Think-on-Graph: Deep and Responsible Reasoning of LLMs on Knowledge Graphs**
Sun et al., 2023
*What it teaches:* How to traverse a knowledge graph during inference
rather than just at retrieval time. Relevant for Phase 3 when your
graph is large enough to reason across.
https://arxiv.org/abs/2307.07697

**KGRAG: Knowledge Graph Enhanced Retrieval Augmented Generation**
Various, 2024
*What it teaches:* Practical patterns for combining vector retrieval
with graph traversal. Directly relevant to your retrieve + rerank
pipeline in Phase 4.
Search: "KGRAG knowledge graph retrieval augmented generation 2024"

---

### On Memory for LLMs

**MemGPT: Towards LLMs as Operating Systems**
Packer et al., 2023 — UC Berkeley
*What it teaches:* The hierarchical memory architecture that inspired
our four-layer design. Main memory (working), external storage
(episodic + semantic), and the paging mechanism between them. This
paper is the closest published architecture to what we are building.
Read the full paper.
https://arxiv.org/abs/2310.08560

**Cognitive Architectures for Language Agents (CoALA)**
Sumers et al., 2023
*What it teaches:* A unified framework mapping cognitive science
memory systems (semantic, episodic, procedural, working) onto LLM
architectures. This is the academic grounding for the exact four-layer
design we chose. Validates that the human memory analogy is not just
poetic — it is architecturally sound.
https://arxiv.org/abs/2309.02427

**Mem0: The Memory Layer for AI**
*What it teaches:* Production-grade memory extraction and storage.
Their selective extraction philosophy — ruthlessly filter what is
worth remembering — is directly applicable. Read their GitHub README
and their blog post on memory architecture.
https://github.com/mem0ai/mem0
https://mem0.ai/blog/memory-for-llm-applications

---

### On Context Management

**Lost in the Middle: How Language Models Use Long Contexts**
Liu et al., 2023 — Stanford
*What it teaches:* The empirical proof that models attend poorly to
information in the middle of long contexts. Relevant information
placed at the beginning or end of the context window is recalled
significantly better. This changes how you assemble your prompt —
most important information goes first and last, not middle.
https://arxiv.org/abs/2307.03172

**Efficient Streaming Language Models with Attention Sinks**
Xiao et al., 2023 — MIT + Meta
*What it teaches:* Why rolling windows work and where they break.
Builds intuition for why we keep the rolling window short (4 turns)
rather than expanding it indefinitely.
https://arxiv.org/abs/2309.17453

---

## Read During Phase 1 Build

### On Prompting

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
Wei et al., 2022 — Google Brain
*What it teaches:* The empirical basis for asking models to think
step by step. Relevant for your self-check mechanism where the agent
reviews its own output before responding.
https://arxiv.org/abs/2201.11903

**ReAct: Synergizing Reasoning and Acting in Language Models**
Yao et al., 2022 — Princeton + Google
*What it teaches:* The reasoning-action loop that underlies agent
architectures. Relevant for how your moderator coordinates between
Socrates and Feynman.
https://arxiv.org/abs/2210.03629

**Constitutional AI: Harmlessness from AI Feedback**
Anthropic, 2022
*What it teaches:* How to build self-critique into the model's
response loop. Directly applicable to your self-check prompt that
asks the agent to verify its own output before returning it.
https://arxiv.org/abs/2212.08073

---

### On Embeddings and Retrieval

**Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**
Reimers and Gurevych, 2019
*What it teaches:* The model you will use for local embeddings.
Understanding how it works builds intuition for why some passages
retrieve well and others do not. Read sections 1-3.
https://arxiv.org/abs/1908.10084

**Improving Text Embeddings with Large Language Models**
Wang et al., 2024 — Microsoft
*What it teaches:* Modern embedding approaches that outperform
sentence-transformers for philosophical text. Relevant when you
want to upgrade the embedding model in Phase 4.
https://arxiv.org/abs/2401.00368

---

## Read During Phase 3 Build (Graph)

### On Graph Construction

**REBEL: Relation Extraction By End-to-end Language Generation**
Cabot and Navigli, 2021
*What it teaches:* How to extract typed relationships from text
automatically. Relevant for your Haiku extraction step in the
ingest pipeline — specifically how to extract the edge types
(related_to, contradicts, defined_as) reliably.
https://aclanthology.org/2021.findings-emnlp.204/

**Language Models as Knowledge Bases?**
Petroni et al., 2019 — Facebook AI
*What it teaches:* What LLMs already know vs what needs to be in
your graph. Builds intuition for why we are not re-teaching Claude
about Plato — we are adding structure and precision that the model
cannot retrieve on its own.
https://arxiv.org/abs/1909.01066

---

## Practical Code References

### Open Source Architectures Worth Reading

**LlamaIndex source code**
https://github.com/run-llama/llama_index
*What to read:* KnowledgeGraphIndex, VectorStoreIndex,
BaseMemory, ContextChatEngine. Not to copy — to understand
the abstractions. Each class represents a solved problem.

**LangChain source code**
https://github.com/langchain-ai/langchain
*What to read:* ConversationKGMemory (knowledge graph memory),
ConversationSummaryBufferMemory (our summarizer pattern),
ContextualCompressionRetriever (our rerank pattern).

**Mem0 source code**
https://github.com/mem0ai/mem0
*What to read:* The Memory class and its add/search/update methods.
Their extraction prompt is particularly valuable — copy the
philosophy not the code.

**Microsoft GraphRAG**
https://github.com/microsoft/graphrag
*What to read:* The entity extraction pipeline and the community
summarization approach. Most directly relevant to your semantic
layer ingest design.

---

### Blog Posts Worth Reading

**Anthropic: Building Effective Agents**
https://www.anthropic.com/research/building-effective-agents
*What it teaches:* Anthropic's own guidance on agent architecture.
How to structure multi-step reasoning. When to use tools vs context.
Read before writing any agent code.

**Pinecone: RAG Guide**
https://www.pinecone.io/learn/retrieval-augmented-generation/
*What it teaches:* Practical chunking strategies, embedding model
comparison, retrieval evaluation. The chunking section is directly
applicable to your ingest pipeline.

**Weaviate: Vector Database Concepts**
https://weaviate.io/blog/distance-metrics-in-vector-search
*What it teaches:* Why cosine similarity for philosophical text,
when to use other distance metrics. Builds intuition for your
re-ranking threshold choices.

**Lilian Weng: LLM Powered Autonomous Agents**
https://lilianweng.github.io/posts/2023-06-23-agent/
*What it teaches:* The best single overview of agent architectures
by OpenAI's head of safety research. Memory, planning, tool use —
all the components of what we are building explained clearly.
Read this early and return to it often.

**Eugene Yan: Patterns for Building LLM-Based Systems**
https://eugeneyan.com/writing/llm-patterns/
*What it teaches:* Production patterns from a practitioner who has
built these systems at Amazon. Evals, RAG, fine-tuning, caching.
Directly applicable to Phases 1-5.

---

### On Feynman's Everyday Reasoning

**Solving Feynman's Formula for Eating Well, Parking Your Car, and Finding a Mate**
Kristen French, Nautilus, June 2026
*What it teaches:* Feynman's unpublished mathematical treatment of
the explore-vs-exploit dilemma — when to stick with what you know
vs. try something new. He modeled it as an optimal stopping problem
and showed that experimenting more early pays off when you have more
time ahead of you. Directly relevant to how the Feynman voice reasons
about decisions in the agent: first principles applied to everyday
choice, not just physics. Also useful for grounding the Feynman voice
in practical analogy rather than letting it drift toward abstraction.
https://nautil.us/solving-feynmans-formula-for-eating-well-parking-your-car-and-finding-a-mate-1281700

---

### Philosophical Texts on Knowledge and Memory
*(For understanding what you are building at a conceptual level)*

**Plato: Meno**
*What it teaches:* Anamnesis — the idea that learning is recollection.
The philosophical grounding for why a system that builds on prior
knowledge is more than just a technical convenience. The graph is
implementing Plato's theory of knowledge.

**Plato: Theaetetus**
*What it teaches:* What is knowledge? The dialogue most directly
relevant to your goal. Socrates examines and refutes three
definitions of knowledge. Every conversation your agent has will
eventually reach this territory.

**Aristotle: De Memoria et Reminiscentia**
*What it teaches:* Aristotle's theory of memory as association —
remarkably close to what we now call a knowledge graph. The first
serious philosophical treatment of how memory works through
connections between ideas.

**William James: The Principles of Psychology, Chapter 16**
*What it teaches:* The psychological basis for associative memory.
James describes memory as a network of associations — the direct
predecessor of modern graph memory theory. Short chapter, profound
implications for your architecture.

---

## Evaluation Papers
*(Read before Phase 4 — tuning)*

**RAGAS: Automated Evaluation of Retrieval Augmented Generation**
Es et al., 2023
*What it teaches:* How to evaluate RAG quality systematically.
Their faithfulness, answer relevancy, and context precision metrics
map directly onto your rubric criteria.
https://arxiv.org/abs/2309.15217

**G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**
Liu et al., 2023
*What it teaches:* Using LLMs to score LLM outputs — the basis for
your Haiku auto-scoring system. Their chain-of-thought evaluation
prompt structure is directly applicable.
https://arxiv.org/abs/2303.16634

---

## Monitoring and Observability References
*(Read before Phase 1 — monitoring is built alongside the agent)*

**Microsoft Responsible AI Toolbox — Concept Documentation**
https://responsibleaitoolbox.ai/introducing-responsible-ai-dashboard/
*What it teaches:* The six monitoring concepts that translate into
your monitoring layer: model statistics, data explorer, interpretability,
error analysis, counterfactual what-if, and causal inference.
Read for concepts only — do not use the implementation, which is
designed for trained ML models not generative AI applications.
The translation is: statistics → distribution, interpretability →
traceability, error analysis → failure pattern detection,
counterfactual → A/B runner, causal inference → variable isolation.

**Weights and Biases: A Developer's Guide to LLM Monitoring**
https://wandb.ai/fully-connected/blog/llm-monitoring
*What it teaches:* What production LLM monitoring looks like in
practice. Use as a reference for what your monitoring panel should
surface — not as a tool to integrate (too heavy for your scale).

**Arize AI: LLM Observability Guide**
https://arize.com/blog/llm-observability/
*What it teaches:* The four pillars of LLM observability — retrieval
quality, response quality, drift detection, and cost tracking.
Each pillar maps onto a panel in your monitoring layer.

**Honeyhive: Tracing and Evaluating LLM Applications**
https://docs.honeyhive.ai/
*What it teaches:* How production teams implement traceability —
logging every step of the pipeline, from retrieval through generation,
in a way that makes debugging fast. The traceability panel in Phase 1
implements this concept at minimal scale.

**HELM: Holistic Evaluation of Language Models**
Liang et al., Stanford 2022
*What it teaches:* How to define a comprehensive evaluation framework
for language models. Their scenario, metric, and aggregation structure
informs how to grow your rubric from Phase 1 basics to Phase 4 depth.
https://arxiv.org/abs/2211.09110

---

## Reference Implementations to Study

| Project | URL | What to learn |
|---|---|---|
| PrivateGPT | github.com/zylon-ai/private-gpt | Full local RAG pipeline |
| Quivr | github.com/QuivrHQ/quivr | Memory architecture |
| Danswer | github.com/danswer-ai/danswer | Production RAG + citation patterns |
| Khoj | github.com/khoj-ai/khoj | Personal knowledge assistant |
| Storm (Stanford) | github.com/stanford-oval/storm | Multi-perspective synthesis |
| Phoenix (Arize) | github.com/Arize-ai/phoenix | LLM tracing + observability |

Storm is particularly relevant — Stanford built a system that
synthesizes multiple perspectives on a topic, which is exactly
the multi-voice architecture we are building.

Phoenix is worth looking at specifically for the traceability panel —
it is open-source LLM observability designed for RAG pipelines and
shows exactly what a production traceability implementation looks like.

---

## Reading Order Recommendation

**Week 1 — Foundation**
1. CoALA paper (cognitive architectures) — validates our four-layer design
2. MemGPT paper — our memory architecture in detail
3. GraphRAG paper (Microsoft) — our semantic layer design
4. Lilian Weng agent post — best overview of the full stack
5. Plato Theaetetus — understand what knowledge means to Socrates

**Week 2 — Retrieval and Context**
6. Original RAG paper — foundational
7. Lost in the Middle paper — changes how you write prompts
8. HyDE paper — upgrade to retrieval quality
9. Eugene Yan patterns post — practitioner perspective
10. Mem0 README and blog — memory extraction philosophy

**Week 3 — Graph, Evaluation, and Monitoring**
11. Think-on-Graph paper — graph traversal during inference
12. REBEL paper — relationship extraction for ingest
13. RAGAS paper — how to evaluate what you build
14. G-Eval paper — auto-scoring design
15. RAI Toolbox concepts page — monitoring layer concepts
16. Arize LLM observability guide — what to monitor and why
17. Anthropic building effective agents — before writing agent code

---

*Document version 1.1 — May 2026*
*Added: Monitoring layer (RAI Toolbox concepts translated to generative AI),*
*updated phases with monitoring deliverables, updated file structure,*
*technology stack, cost summary, and reading list.*
*Review and annotate before beginning Phase 1 build*
