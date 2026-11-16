"""Tests for the daily report's prose summary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from service.daily_summary import (
    MAX_SENTENCES,
    build_user_prompt,
    clamp_sentences,
    write_day_summary,
)

FACTS = {
    "period_start": "2026-11-15T00:00:00Z",
    "period_end": "2026-11-16T00:00:00Z",
    "timezone": "Asia/Jakarta",
    "incidents": [
        {
            "title": "Monitoring Target Down",
            "severity": "critical",
            "status": "open",
            "affected": "10.41.126.189:8080",
            "started_at": "2026-11-15T13:02:20Z",
        },
        {
            "title": "RAM",
            "severity": "warning",
            "status": "resolved",
            "affected": "koba-labs",
            "started_at": "2026-11-15T12:52:10Z",
            "resolved_at": "2026-11-15T12:57:10Z",
            "duration": "5m0s",
        },
    ],
}


def test_prompt_carries_the_facts_and_no_transcripts():
    """The summary is only as good as its facts; it must not get raw logs."""
    prompt = build_user_prompt(FACTS)

    assert "Monitoring Target Down" in prompt
    assert "severity=critical" in prompt
    assert "status=open" in prompt
    assert "duration=5m0s" in prompt
    assert "Incidents: 2" in prompt
    assert "Asia/Jakarta" in prompt


def test_clamp_sentences_limits_the_summary():
    """A chatty model must not turn the digest back into a wall of text."""
    text = "One broke. It recovered. All quiet. Extra sentence nobody needs."
    assert clamp_sentences(text) == "One broke. It recovered. All quiet."


def test_clamp_sentences_keeps_decimals_intact():
    """A dot inside a number is not a sentence boundary."""
    assert clamp_sentences("Only 1.5 minutes of downtime. Fine.") == (
        "Only 1.5 minutes of downtime. Fine."
    )


def test_clamp_sentences_flattens_markdown_lists():
    """A list-shaped reply becomes prose, since the card has no bullets here."""
    assert clamp_sentences("- One.\n- Two.") == "- One. - Two."


def test_write_day_summary_returns_empty_on_llm_failure():
    """The digest must still deliver when the model is unreachable."""
    llm = MagicMock()
    llm.chat_completion.side_effect = RuntimeError("provider down")

    assert write_day_summary(llm, FACTS) == ""


def test_write_day_summary_returns_empty_without_incidents():
    llm = MagicMock()
    assert write_day_summary(llm, {"incidents": []}) == ""
    llm.chat_completion.assert_not_called()


def test_write_day_summary_asks_for_no_tools():
    """This is a writing task; sending tools would invite tool calls."""
    llm = MagicMock()
    llm.chat_completion.return_value = SimpleNamespace(content="All quiet overnight.")

    summary = write_day_summary(llm, FACTS)

    assert summary == "All quiet overnight."
    kwargs = llm.chat_completion.call_args.kwargs
    assert kwargs["tools"] is None
    assert "tool_choice" not in kwargs
    assert MAX_SENTENCES == 3


def test_prompt_echo_is_dropped():
    """A model that restates its instructions must not reach the digest.

    A delivered report opened with "We need answer user's request. Need produce
    final summary at most three sentences, no bullets/headings, no preamble..."
    — instructions, not a summary, sitting above the incident list where a
    reader expects the state of production.
    """
    echo = (
        "We need answer user's request. Need produce final summary at most "
        "three sentences, no bullets/headings/preamble. Lead with whether "
        "anything still broken and needs action today."
    )
    llm = MagicMock()
    llm.chat_completion.return_value = SimpleNamespace(content=echo)

    assert write_day_summary(llm, FACTS) == ""


def test_real_summary_survives_the_echo_filter():
    summary = (
        "Nothing is broken this morning; all incidents recovered overnight. "
        "Two monitoring conditions kept recurring, on last-labs and koba-labs. "
        "No action is needed today."
    )
    llm = MagicMock()
    llm.chat_completion.return_value = SimpleNamespace(content=summary)

    assert write_day_summary(llm, FACTS) == summary


def test_looks_like_prompt_echo_requires_two_markers():
    """One incidental phrase must not discard an otherwise good summary."""
    from service.daily_summary import looks_like_prompt_echo

    assert not looks_like_prompt_echo("The node ran out of memory at 02:14.")
    assert not looks_like_prompt_echo("")
    assert looks_like_prompt_echo("No bullets, no headings. Write the summary in the past tense.")
