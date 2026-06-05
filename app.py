"""
app.py
------
Gradio UI entry point for the philosopher agent.
Two-panel interface: dialogue on the left, monitoring on the right.

Run with: python app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr

# Load .env before any other imports that need the API key
# Use an absolute path so .env is found regardless of the shell's cwd
load_dotenv(Path(__file__).parent / ".env", override=True)

from agent.config_loader import AgentConfig, load_all_prompts  # noqa: E402
from agent.runner import run_dialogue_turn, write_run_json  # noqa: E402
from memory.retriever import get_embed_model, get_rerank_model  # noqa: E402
from graph.graph_db import (  # noqa: E402
    get_connection, init_schema, create_session, update_human_scores,
    get_recent_snapshots,
)
from monitoring.traceability import build_trace, format_for_display  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup — load config and validate everything before the UI starts
# ---------------------------------------------------------------------------

def startup() -> tuple[AgentConfig, str]:
    """
    Validates the full environment at startup.
    Fails loudly if anything is missing — better than failing on first turn.
    Returns (config, session_id).
    """
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Add it to .env or set it in your environment:\n"
            "  export ANTHROPIC_API_KEY=your_key_here"
        )

    # Load and validate config
    config = AgentConfig.load()
    log.info(
        "config_loaded version=%s socrates=%.1f feynman=%.1f temp=%.1f",
        config.prompt_version,
        config.socrates_weight,
        config.feynman_weight,
        config.temperature,
    )

    # Validate all prompt files exist and are non-empty
    load_all_prompts()
    log.info("prompts_validated")

    # Initialize database schema
    conn = get_connection()
    init_schema(conn)
    conn.close()
    log.info("database_ready")

    # Warm local ML models so the first query has no cold-start delay
    log.info("warming_embed_model")
    get_embed_model()
    log.info("warming_rerank_model")
    get_rerank_model()
    log.info("models_ready")

    # Create a new session ID for this app run
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    create_session(conn, session_id)
    conn.close()
    log.info("session_created session_id=%s", session_id)

    return config, session_id


# ---------------------------------------------------------------------------
# Turn handler — called by Gradio on every user message
# ---------------------------------------------------------------------------

def handle_turn(
    user_message: str,
    history: list[dict],
    session_state: dict,
):
    """
    Processes one dialogue turn and returns updated UI state.

    Returns:
        (cleared_input, updated_history, session_state,
         traceability_text, scores_text, flags_text, run_id,
         score_challenged, score_authentic, score_insight,
         score_tags, score_notes, score_status)
    """
    if not user_message.strip():
        return "", history, session_state, "", "", "", "", 3, 3, 3, [], "", ""

    config     = session_state["config"]
    session_id = session_state["session_id"]

    try:
        # Run the turn
        result = run_dialogue_turn(
            user_query = user_message,
            session_id = session_id,
            config     = config,
        )

        # Build monitoring trace
        trace_record = build_trace(result)
        display      = format_for_display(trace_record)

        # Update chat history (Gradio 6 messages format)
        history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": result.response.text},
        ]

        log.info(
            "ui_turn_complete run_id=%s score=%d/25",
            result.run_id,
            result.scores.total,
        )

        return (
            "",                          # clear input box
            history,
            session_state,
            display["traceability"],
            display["scores"],
            display["flags"],
            result.run_id,               # for human scoring panel
            3, 3, 3, [], "", "",         # reset Your Scores panel
        )

    except Exception as e:
        log.error("turn_failed error=%s", str(e), exc_info=True)
        error_msg = f"[Error]: {str(e)}\n\nCheck logs for details."
        history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": error_msg},
        ]
        return "", history, session_state, "", f"Error: {str(e)}", "", "", 3, 3, 3, [], "", ""


def handle_new_conversation(
    session_state: dict,
) -> tuple[list, dict, str, str, str, str]:
    """
    Clears the chat and starts a fresh session in the DB.
    Called by the New Conversation button.
    """
    config         = session_state["config"]
    new_session_id = f"session_{uuid.uuid4().hex[:8]}"
    conn           = get_connection()
    create_session(conn, new_session_id)
    conn.close()
    log.info("new_conversation session_id=%s", new_session_id)

    new_state   = {"config": config, "session_id": new_session_id}
    footer_text = (
        f"*Session: `{new_session_id}` · "
        f"DB: `philosopher.db` · "
        f"Snapshots saved to `experiments/runs/`*"
    )
    return [], new_state, "", "", "", footer_text


def handle_human_score(
    run_id: str,
    challenged_me: int,
    felt_authentic: int,
    new_insight: int,
    tags: list[str],
    notes: str,
    session_state: dict,
) -> str:
    """
    Saves human feedback scores to the snapshot in the database.
    Returns a status message.
    """
    if not run_id:
        return "No run to score yet — make a query first."

    human_scores = {
        "challenged_me":  challenged_me,
        "felt_authentic": felt_authentic,
        "new_insight":    new_insight,
    }

    # Combined score = auto total (from DB) + human total
    # Human max is 15 (3 criteria × 5), auto max is 25
    human_total    = challenged_me + felt_authentic + new_insight
    combined_score = float(human_total)   # optimizer will merge with auto

    promoted = combined_score >= 12   # all three criteria ≥ 4
    flagged  = combined_score <= 6    # all three criteria ≤ 2

    try:
        conn = get_connection()
        update_human_scores(
            conn,
            run_id         = run_id,
            human_scores   = human_scores,
            combined_score = combined_score,
            tags           = tags or [],
            notes          = notes or "",
            promoted       = promoted,
            flagged        = flagged,
        )
        conn.close()

        # Update the JSON file with human scores
        from agent.runner import _RUNS_DIR
        run_path = _RUNS_DIR / f"{run_id}.json"
        if run_path.exists():
            data = json.loads(run_path.read_text(encoding="utf-8"))
            data["scores"]["human"]    = human_scores
            data["scores"]["combined"] = combined_score
            data["tags"]     = tags or []
            data["notes"]    = notes or ""
            data["promoted"] = promoted
            data["flagged"]  = flagged
            run_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            write_run_json(
                run_id         = run_id,
                session_id     = session_state.get("session_id", ""),
                scenario       = "",
                config         = {},
                response       = "",
                auto_scores    = {},
                human_scores   = human_scores,
                combined_score = combined_score,
                tags           = tags or [],
                notes          = notes or "",
                promoted       = promoted,
                flagged        = flagged,
            )

        status = f"✓ Saved scores for run {run_id[:8]}"
        if promoted:
            status += " — promoted to golden examples ⭐"
        elif flagged:
            status += " — flagged for review ⚠"
        log.info("human_score_saved run_id=%s combined=%.1f promoted=%s",
                 run_id, combined_score, promoted)
        return status

    except Exception as e:
        log.error("human_score_save_failed run_id=%s error=%s", run_id, str(e))
        return f"Error saving scores: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(config: AgentConfig, session_id: str) -> gr.Blocks:
    """
    Builds the two-panel Gradio interface.
    Left panel: dialogue. Right panel: monitoring.
    """
    with gr.Blocks(title="Philosopher Agent") as demo:

        # Session state — passed through every callback
        session_state = gr.State({
            "config":     config,
            "session_id": session_id,
        })

        # Tracks the run_id of the most recent turn for human scoring
        current_run_id = gr.State("")

        # ── Header ───────────────────────────────────────────────────────
        gr.Markdown(
            f"## Philosopher Agent\n"
            f"*Socrates {int(config.socrates_weight*100)}% · "
            f"Feynman {int(config.feynman_weight*100)}% · "
            f"Temp {config.temperature} · "
            f"Prompt {config.prompt_version}*"
        )

        with gr.Row():

            # ── Left: Dialogue ───────────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### Dialogue", elem_classes="panel-header")
                chatbot = gr.Chatbot(
                    label      = "",
                    height     = 500,
                    show_label = False,
                    layout     = "bubble",
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder = "Make a claim or ask a question...",
                        label       = "",
                        scale       = 4,
                        show_label  = False,
                        lines       = 2,
                    )
                    submit_btn = gr.Button("→", scale=1, variant="primary")
                with gr.Row():
                    new_conv_btn = gr.Button(
                        "New Conversation",
                        variant = "secondary",
                        size    = "sm",
                    )

            # ── Right: Monitoring ────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Monitoring", elem_classes="panel-header")
                with gr.Tabs():

                    with gr.Tab("Traceability"):
                        trace_box = gr.Textbox(
                            label      = "What was retrieved and why",
                            lines      = 12,
                            interactive = False,
                            show_label  = True,
                        )

                    with gr.Tab("Scores"):
                        scores_box = gr.Textbox(
                            label      = "Auto-scores this turn",
                            lines      = 8,
                            interactive = False,
                            show_label  = True,
                        )

                    with gr.Tab("Flags"):
                        flags_box = gr.Textbox(
                            label      = "Failure flags and skill violations",
                            lines      = 8,
                            interactive = False,
                            show_label  = True,
                        )

                    with gr.Tab("Your Scores"):
                        gr.Markdown("*Rate this response — scores are saved to the experiment log*")
                        score_challenged = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Did it challenge YOU specifically? (1–5)",
                        )
                        score_authentic = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Did it feel authentic — Socrates or a performance? (1–5)",
                        )
                        score_insight = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Did it produce new insight? (1–5)",
                        )
                        score_tags = gr.CheckboxGroup(
                            choices=[
                                "Feynman too dominant",
                                "Socrates lectured",
                                "Both voices agreed too easily",
                                "Too abstract",
                                "Too surface level",
                                "Ignored my actual premise",
                                "Great — no issues",
                            ],
                            label="What went wrong? (select all that apply)",
                        )
                        score_notes = gr.Textbox(
                            placeholder="Optional: describe what felt off or what made this great...",
                            label="Notes",
                            lines=3,
                        )
                        save_score_btn = gr.Button("Save scores", variant="primary", size="sm")
                        score_status   = gr.Textbox(
                            label="",
                            lines=1,
                            interactive=False,
                            show_label=False,
                        )

        # ── Footer ───────────────────────────────────────────────────────
        footer_md = gr.Markdown(
            f"*Session: `{session_id}` · "
            f"DB: `philosopher.db` · "
            f"Snapshots saved to `experiments/runs/`*",
        )

        # ── Event handlers ───────────────────────────────────────────────
        submit_inputs  = [user_input, chatbot, session_state]
        submit_outputs = [user_input, chatbot, session_state,
                          trace_box, scores_box, flags_box, current_run_id,
                          score_challenged, score_authentic, score_insight,
                          score_tags, score_notes, score_status]

        submit_btn.click(
            fn      = handle_turn,
            inputs  = submit_inputs,
            outputs = submit_outputs,
        )
        user_input.submit(
            fn      = handle_turn,
            inputs  = submit_inputs,
            outputs = submit_outputs,
        )

        new_conv_btn.click(
            fn      = handle_new_conversation,
            inputs  = [session_state],
            outputs = [chatbot, session_state, trace_box, scores_box, flags_box, footer_md],
        )

        save_score_btn.click(
            fn      = handle_human_score,
            inputs  = [
                current_run_id,
                score_challenged,
                score_authentic,
                score_insight,
                score_tags,
                score_notes,
                session_state,
            ],
            outputs = [score_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        config, session_id = startup()
    except (EnvironmentError, FileNotFoundError, ValueError) as e:
        print(f"\nSTARTUP FAILED:\n{e}\n")
        sys.exit(1)

    demo = build_ui(config, session_id)

    log.info("starting_ui session_id=%s", session_id)
    demo.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        share       = False,
        show_error  = True,
        theme       = gr.themes.Soft(),
        css         = """
            .panel-header { font-size: 13px; font-weight: 500;
                            color: #666; margin-bottom: 4px; }
            .flag-warning { color: #c0392b; }
            .flag-ok      { color: #27ae60; }
        """,
    )


if __name__ == "__main__":
    main()
