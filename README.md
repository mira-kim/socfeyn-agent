# socfeyn-agent

A personal thinking partner that challenges your assumptions — Socrates asks the questions, Feynman grounds them in reality. Built on Claude, a local knowledge graph, and RAG over primary source texts.

---

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Install

```bash
pip install -r requirements.txt
```

## Configure

Copy `.env` and add your key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=your_key_here
```

## Build the knowledge base (run once)

```bash
python scripts/ingest.py
```

This embeds all source texts locally and builds the graph. Takes a few minutes, costs ~$3–5 in API tokens, never needs to run again.

## Run

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. Left panel is the dialogue, right panel shows what was retrieved, auto-scores, and failure flags after each turn.

## Run tests

```bash
pytest
```
