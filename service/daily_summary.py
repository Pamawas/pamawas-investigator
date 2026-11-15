"""Prose summary of a day's incidents for the daily report.

The reporter asks the investigator for this while rendering the digest, so the
summary is part of the message that goes out rather than a follow-up. All LLM
access lives in this service; the reporter holds no model credentials.

The summary is an addition to the digest, never a dependency: every failure path
returns an empty string so the report still delivers with its deterministic
incident lists.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# A morning digest gets read in ten seconds or not at all.
MAX_SENTENCES = 3

SYSTEM_PROMPT = (
    "You summarise an overnight infrastructure incident report for an on-call "
    "engineer reading it first thing in the morning.\n\n"
    "Rules:\n"
    "- At most three sentences. No bullet points, no headings, no preamble.\n"
    "- Lead with whether anything is still broken and needs action today.\n"
    "- Then say what broke and recovered overnight, in plain language.\n"
    "- Use only the facts given. Do not speculate about causes that are not "
    "stated, and do not invent numbers.\n"
    "- If nothing happened, say that in one sentence.\n"
    "- Write in the past tense, plainly, without hype."
)


def build_user_prompt(facts: dict[str, Any]) -> str:
    """Render the incident fact sheet the model may use."""
    incidents = facts.get("incidents") or []
    lines = [
        f"Period: {facts.get('period_start', '')} to {facts.get('period_end', '')} "
        f"({facts.get('timezone', 'UTC')})",
        f"Incidents: {len(incidents)}",
    ]
    for inc in incidents:
        line = (
            f"- \"{inc.get('title', '')}\" | severity={inc.get('severity', '')} "
            f"| status={inc.get('status', '')} | affected={inc.get('affected', '')} "
            f"| started={inc.get('started_at', '')}"
        )
        if inc.get("resolved_at"):
            line += f" | resolved={inc['resolved_at']} | duration={inc.get('duration', '')}"
        lines.append(line)

    lines.append("")
    lines.append("Write the summary.")
    return "\n".join(lines)


def clamp_sentences(text: str, n: int = MAX_SENTENCES) -> str:
    """Trim a reply to n sentences.

    A chatty model must not turn the digest back into the wall of text the
    digest format exists to replace.
    """
    text = " ".join((text or "").split())
    if not text:
        return ""

    count = 0
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] == " "):
            count += 1
            if count == n:
                return text[: i + 1]
    return text


def write_day_summary(llm, facts: dict[str, Any]) -> str:
    """Ask the model for the summary. Returns "" on any failure."""
    if not llm or not facts.get("incidents"):
        return ""

    try:
        response = llm.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(facts)},
            ],
            # No tools: this is a writing task, not an investigation.
            tools=None,
            max_tokens=300,
        )
    except Exception as e:  # noqa: BLE001 - the digest must never fail for this
        logger.warning("Daily summary generation failed: %s", e)
        return ""

    content = getattr(response, "content", "") or ""
    return clamp_sentences(content)
