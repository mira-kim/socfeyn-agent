"""
agent/voices.py
---------------
Builds the combined voice prompt from Socrates + Feynman weights.
Applies weight ratios as structural instructions, not random sampling.

Does:    Combine voice prompts according to config weights.
Does NOT: Make API calls, read from database, assemble the full prompt.
Depends on: agent/config_loader.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.config_loader import AgentConfig, load_prompt

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoicePrompt:
    """
    The combined voice instructions for a single turn.
    Immutable — built fresh each session from config.
    """
    system_with_voices: str    # system prompt + voice instructions
    socrates_weight:    float
    feynman_weight:     float
    voice_rule_fired:   str    # which structural rule is active this turn


def build_voice_prompt(config: AgentConfig) -> VoicePrompt:
    """
    Builds the combined voice prompt from config weights.

    Weight ratios translate into structural rules:
      socrates_weight > 0.65  → Socrates dominates, Feynman interjects rarely
      socrates_weight 0.5-0.65 → balanced, both contribute each turn
      socrates_weight < 0.5   → Feynman leads on mechanism, Socrates closes

    The voice rule is logged to traceability on every turn.
    """
    system   = load_prompt("system")
    socrates = load_prompt("socrates")
    feynman  = load_prompt("feynman")

    sw = config.socrates_weight
    fw = config.feynman_weight

    if sw >= 0.65:
        rule = "socrates_dominant"
        structure = (
            f"VOICE STRUCTURE FOR THIS TURN:\n"
            f"Socrates leads ({int(sw*100)}%) — speaks first and last, "
            f"drives the questioning throughout.\n"
            f"Feynman ({int(fw*100)}%) — interjects only if the topic requires "
            f"a concrete mechanism or scientific grounding. "
            f"If not needed, Feynman stays silent this turn.\n"
        )
    elif sw >= 0.5:
        rule = "balanced"
        structure = (
            f"VOICE STRUCTURE FOR THIS TURN:\n"
            f"Socrates ({int(sw*100)}%) — opens with a probing question, "
            f"closes with a question.\n"
            f"Feynman ({int(fw*100)}%) — contributes one concrete example "
            f"or first-principles reframe in the middle.\n"
            f"Both voices must be present and distinct.\n"
        )
    else:
        rule = "feynman_grounding"
        structure = (
            f"VOICE STRUCTURE FOR THIS TURN:\n"
            f"Feynman ({int(fw*100)}%) — leads with first-principles analysis.\n"
            f"Socrates ({int(sw*100)}%) — closes with a question that builds "
            f"on Feynman's grounding.\n"
            f"Use this structure when precision is more urgent than aporia.\n"
        )

    combined = "\n\n---\n\n".join([
        system,
        f"## SOCRATES — HOW YOU SPEAK\n{socrates}",
        f"## FEYNMAN — HOW YOU SPEAK\n{feynman}",
        structure,
    ])

    log.debug("voice_prompt_built rule=%s sw=%.2f fw=%.2f", rule, sw, fw)

    return VoicePrompt(
        system_with_voices = combined,
        socrates_weight    = sw,
        feynman_weight     = fw,
        voice_rule_fired   = rule,
    )
