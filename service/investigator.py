"""Main Investigator Service."""

import asyncio
import json
import logging
import re
import time
from dataclasses import replace
from datetime import datetime
from datetime import timezone as tz
from typing import Any

import openai
import psycopg2
from openai import OpenAI

from adapters import (
    DeploymentAdapterError,
    LokiError,
    PrometheusError,
    RelatedIncidentsError,
)
from config import Config
from metrics import (
    increment_db_errors,
    increment_deadline_exceeded,
    increment_findings_by_type,
    increment_forced_submission,
    increment_investigations,
    increment_max_tool_calls_reached,
)
from models import (
    EvidenceSource,
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    ToolResult,
)
from service.daily_summary import write_day_summary as _write_day_summary
from service.loki_fetcher import LogsFetcher, LokiFetcher
from service.persistence import InvestigationPersistence
from service.prometheus_fetcher import MetricsFetcher, PrometheusFetcher
from tools import InvestigatorTools, ToolRegistry

logger = logging.getLogger(__name__)

# Use datetime.UTC from Python 3.11+, fallback for older versions
try:
    from datetime import UTC
except ImportError:
    UTC = tz.utc


# Openers that mean the model is narrating its process rather than stating a
# conclusion. A text-only reply that begins with one of these is working notes,
# not a finding: rendered under "🎯 Likely cause" on a critical alert card it
# read as a wall of chain-of-thought ("Let me analyze the data more carefully.
# I need to convert the Unix timestamps to understand the timeline...").
_REASONING_OPENERS = (
    "let me",
    "i need to",
    "i should",
    "i will",
    "first, i",
    "now let me",
    "looking at",
    "let's",
    "to understand",
    "to determine",
    "i'll",
    "we need to",
)


# Longest text that can still be a finding. A finding is a claim — a sentence or
# two — not an essay. This also bounds what a report has to render.
MAX_FINDING_CHARS = 1200

# Phrases that appear when the model narrates its process rather than stating a
# result: "Let me check...", "Wait —", "Actually, let me reconsider".
_PROCESS_MARKERS = (
    "let me",
    "let's ",
    "i need to",
    "i should ",
    "i'll ",
    "i will ",
    "wait —",
    "wait,",
    "hmm",
    "actually,",
    "but actually",
    "let me step back",
    "the key question is",
    "i already ",
    "let me think",
    "let me reconsider",
    "to summarize what i",
    "checking ",
)


def _is_reasoning_preamble(text: str) -> bool:
    """True when a reply is process narration rather than a stated conclusion.

    The first version of this only looked at the opening words and then gave up
    if the text mentioned "likely cause" anywhere. That was trivially defeated:
    a 6,292-character essay of working notes began with "Let me organize what I
    know so far:" and contained the phrase "what the most likely cause is", so
    the conclusion check passed and the whole essay was stored as a hypothesis
    and published as the likely cause on a critical card.

    Length is the reliable signal. A finding is one or two sentences; anything
    long enough to contain a reconsidered line of reasoning is reasoning, no
    matter which phrases it happens to include.
    """
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return False

    # An essay is not a finding. This alone catches the case above.
    if len(lowered) > MAX_FINDING_CHARS:
        return True

    opens_with_narration = any(lowered.startswith(opener) for opener in _REASONING_OPENERS)
    marker_hits = sum(1 for marker in _PROCESS_MARKERS if marker in lowered)

    if opens_with_narration:
        # Narration followed by a conclusion is still worth reading, but only
        # when it stays short; several process markers mean the model never
        # reached a conclusion at all.
        return marker_hits >= 2 or len(lowered) > 400

    # Not opening with narration, but visibly reasoning throughout.
    return marker_hits >= 3


def _plain_typography(text: str) -> str:
    """Replace the punctuation the model reaches for with plain punctuation.

    Em and en dashes are the clearest writing tell there is: they show up in
    generated prose far more often than in anything a person types, and they
    render badly in a proportional-font chat card. The prompt asks for plain
    sentences, but a style instruction is a preference, not a constraint, so the
    typography is also normalized where every finding passes through.
    """
    if not text:
        return text
    cleaned = re.sub(r"\s*[\N{EM DASH}\N{EN DASH}\N{HORIZONTAL BAR}]\s*", ", ", text)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _normalize_finding(finding: Finding) -> Finding:
    """Apply the rules every finding must satisfy before it is stored.

    Typography: no em or en dashes, which are the clearest generated-prose tell
    and render as a stray line in a proportional font.

    Confidence on an UNKNOWN: zero. The model was submitting `unknown` findings
    at 0.9 confidence — "I am 90% confident that I do not know" — and the report
    printed `UNKNOWN (90%)`, which reads as a strong claim about a non-answer.
    Confidence belongs to a claim; an UNKNOWN is the absence of one.
    """
    content = _plain_typography(finding.content)
    confidence = finding.confidence
    if finding.type == EvidenceType.UNKNOWN:
        confidence = 0.0
    if content == finding.content and confidence == finding.confidence:
        return finding
    return replace(finding, content=content, confidence=confidence)


def _is_placeholder_or_dummy(text: str) -> bool:
    """True when text is a placeholder, dummy value, or lacks substantive content."""
    cleaned = text.strip().lower()
    dummy_exact = {
        "placeholder",
        "todo",
        "tbd",
        "none",
        "null",
        "n/a",
        "na",
        "test",
        "asdf",
        "unknown",
        "...",
        "-",
    }
    if cleaned in dummy_exact:
        return True
    if (
        cleaned.startswith("placeholder")
        or cleaned.startswith("todo:")
        or cleaned.startswith("tbd:")
    ):
        return True
    return False


def _sanitize_finding(finding: Finding) -> Finding:
    """Keep working notes and placeholder text out of a finding, whichever path produced it.

    A model that narrates its reasoning does so through both exits of the loop:
    as a plain text reply, and as a `submit_findings` call whose content is the
    transcript rather than a conclusion. The text path rejects that; this covers
    the tool path, so an essay cannot reach the report by being submitted
    instead of typed.

    Additionally, lazy placeholder submissions like 'placeholder', 'TBD', or 'TODO'
    are rejected and replaced with honest terminal claims.
    """
    content = _plain_typography(finding.content)
    if _is_placeholder_or_dummy(content):
        logger.warning(
            "Discarding a %s finding with placeholder content: %s",
            finding.type,
            content[:120],
        )
        return Finding(
            type=EvidenceType.UNKNOWN,
            content=(
                "Investigation ended without a substantive conclusion; the model "
                "submitted placeholder text."
            ),
            source=EvidenceSource.SYSTEM,
            confidence=0.0,
        )

    if not _is_reasoning_preamble(content):
        if content != finding.content:
            return replace(finding, content=content)
        return finding
    logger.warning(
        "Discarding a %s finding that was working notes (%d chars): %s",
        finding.type,
        len(finding.content),
        " ".join(finding.content.split())[:120],
    )
    return Finding(
        type=EvidenceType.UNKNOWN,
        content=(
            "Investigation ended without a stated conclusion; the model submitted "
            "its working notes instead of a finding."
        ),
        source=EvidenceSource.SYSTEM,
        confidence=0.0,
    )


def unwrap_tool_call_envelope(name: str, args: dict) -> tuple[str, dict]:
    """Unwrap a tool call that arrived wrapped in a second function envelope.

    Some [OI]-compatible endpoints return the whole tool call as the arguments
    of a generic `functions` call:

        {"name": "functions", "arguments": {"name": "query_loki",
                                            "arguments": {"logql": "..."}}}

    Treated literally, every such call is an unknown tool and the model gets an
    error back for a call it made correctly — production runs burned ten calls
    this way, each one a round trip that produced nothing.
    """
    if name != "functions" or not isinstance(args, dict):
        return name, args

    inner_name = args.get("name")
    if not isinstance(inner_name, str) or not inner_name:
        return name, args

    inner_args = args.get("arguments")
    if inner_args is None:
        # A no-argument tool (get_loki_labels) arrives as {"name": "..."} with
        # no arguments key at all.
        logger.warning("Unwrapped nested tool call envelope: functions -> %s", inner_name)
        return inner_name, {}
    if isinstance(inner_args, str):
        try:
            inner_args = json.loads(inner_args)
        except json.JSONDecodeError:
            return name, args
    if not isinstance(inner_args, dict):
        return name, args

    logger.warning("Unwrapped nested tool call envelope: functions -> %s", inner_name)
    return inner_name, inner_args


def _resolve_tool_call(tool_call) -> tuple[str, dict]:
    """Return the real tool name and arguments for a tool call."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    return unwrap_tool_call_envelope(name, args)


def _resolved_tool_name(tool_call) -> str:
    """The tool name after unwrapping a nested envelope, without parsing args."""
    name = tool_call.function.name
    if name != "functions":
        return name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        return name
    inner = args.get("name") if isinstance(args, dict) else None
    return inner if isinstance(inner, str) and inner else name


# Floor for a single LLM call. Below this a call cannot return anything useful,
# so failing fast and letting the investigation submit what it has beats opening
# another connection.
_MIN_CALL_TIMEOUT_SECONDS = 5.0

# Ceiling for a single LLM call. It exists so a hung endpoint cannot consume the
# whole investigation on one attempt, not to say how long a healthy call takes.
MAX_CALL_TIMEOUT_SECONDS = 180.0


def _call_timeout(
    request_timeout: float, budget_seconds: float | None, attempts_left: int = 1
) -> float:
    """How long one LLM request may take, given what is left of the budget.

    This used to be `min(request_timeout, budget)`, which capped every call at
    30 seconds regardless of how much budget remained. That is a paid-API
    assumption: a free endpoint generating a tool-call response routinely needs
    60 to 120 seconds, so every call was cut off mid-generation, retried against
    the same slow endpoint, and eventually failed over to a second slow endpoint
    that timed out the same way. Production logs show exactly that — six
    consecutive `Request timed out` lines for one incident — and the run ended
    with no findings after burning its whole budget on timeouts.

    The budget, not a fixed number, is the authority. Each attempt gets an equal
    share of what is left, which lets a slow model finish while still leaving
    room for a retry. `request_timeout` is now a floor for the normal case, not a
    cap.

    Two bounds keep the share honest. It never exceeds what is left, so a call
    cannot promise time the investigation does not have; and it never exceeds
    `MAX_CALL_TIMEOUT_SECONDS`, so one hung endpoint cannot consume the whole
    run on a single attempt. The only exception to the first bound is the
    floor: with seconds left, a 5-second attempt is still worth making.
    """
    if budget_seconds is None:
        return request_timeout
    share = max(request_timeout, budget_seconds / max(1, attempts_left))
    return max(
        _MIN_CALL_TIMEOUT_SECONDS,
        min(share, MAX_CALL_TIMEOUT_SECONDS, budget_seconds),
    )


def _remaining_budget(started: float, budget_seconds: float | None) -> float | None:
    """Seconds left of the budget, measured from when the call began."""
    if budget_seconds is None:
        return None
    import time as _time

    return max(0.0, budget_seconds - (_time.monotonic() - started))


def _is_timeout(error: Exception) -> bool:
    """True when the endpoint accepted the request but did not answer in time."""
    return isinstance(error, (openai.APITimeoutError, TimeoutError))


def _claim_recovered(text: str) -> bool:
    """True when a finding states the condition is no longer failing.

    A verification only closes an incident on a positive claim of recovery.
    Anything phrased as failure, ongoing alert, negative, or uncertainty must not count.
    """
    if not text:
        return False
    lowered = " ".join(text.lower().split())

    disqualifiers = (
        "unreachable",
        "not reachable",
        "still down",
        "still firing",
        "still failing",
        "remains down",
        "remains unreachable",
        "no recovery",
        "not recovered",
        "has not recovered",
        "did not recover",
        "did not confirm",
        "cannot reach",
        "unable to reach",
        "cannot conclude",
        "could not verify",
        "unconfirmed",
        "not confirmed",
        "may still be down",
        "up is 0",
        "up=0",
        "up == 0",
    )
    if any(d in lowered for d in disqualifiers):
        return False

    positive_recovery_patterns = (
        r"\brecovered\b",
        r"\brecovery\b",
        r"\bback up\b",
        r"\bbackup\b",
        r"\bis up\b",
        r"\bis back\b",
        r"\bno longer down\b",
        r"\bno longer failing\b",
        r"\breturned\b",
        r"\breachable\b",
        r"\bis alive\b",
        r"\brestored\b",
        r"\bself-resolved\b",
    )
    return any(re.search(pat, lowered) for pat in positive_recovery_patterns)


def _out_of_time(remaining: float | None, attempt: int) -> bool:
    """True when a further retry cannot finish inside what is left.

    The first attempt always runs: with the budget gone it still gets the floor
    timeout, because returning an error without trying is strictly worse.
    """
    if remaining is None or attempt == 0:
        return False
    return remaining <= _MIN_CALL_TIMEOUT_SECONDS


class InvestigatorLLM:
    """OpenAI-compatible LLM client with endpoint redundancy for investigator."""

    # Bounded retry policy: transient network/5xx errors are retried with
    # exponential backoff; anything else fails fast.
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 0.5
    MAX_BACKOFF_SECONDS = 4.0

    # How long to back off when the LLM endpoint rate-limits us (HTTP 429)
    # and no other endpoint is available to fail over to.
    #
    # Previously this was 30 s regardless of the Retry-After header value
    # the endpoint sends.  With two concurrent investigations + a daily-
    # summary call each sleeping 30-90 s, all three threads held their full
    # messages[] history in RAM simultaneously and pushed the pod past its
    # 1 Gi limit (observed: exit code 137, restart #39/#40, 2026-11-22).
    #
    # The new policy:
    #   1. Honour Retry-After when the response carries one, capped at this
    #      constant so a misconfigured endpoint cannot park a thread forever.
    #   2. Fail over to the next endpoint immediately on 429 when one exists
    #      instead of sleeping - the other endpoint is not rate-limited yet.
    #   3. Only sleep (up to this cap) when we are truly out of endpoints.
    RATE_LIMIT_BACKOFF_SECONDS = 30.0

    def __init__(self, config: Config):
        # Floor for a single request, from LLM_REQUEST_TIMEOUT_SECONDS. It is a
        # floor rather than a cap: `_call_timeout` gives each attempt a share of
        # the remaining investigation budget, so a slow free endpoint is allowed
        # the time it actually needs. Tuning this below the model's real latency
        # is what made every call time out.
        self.request_timeout = float(getattr(config, "llm_request_timeout_seconds", 0) or 60.0)

        # Build ordered endpoint list from plural env vars only.
        endpoints: list[tuple[str, str | None, str | None]] = []
        if config.llm_base_urls:
            keys = config.llm_api_keys or []
            models = config.llm_models or []
            for i, url in enumerate(config.llm_base_urls):
                endpoints.append(
                    (
                        url,
                        keys[i] if i < len(keys) else None,
                        models[i] if i < len(models) else None,
                    )
                )

        self.endpoints = endpoints
        # Each endpoint gets its own OpenAI client. Endpoints without a
        # configured API key are skipped with a warning so the service can
        # still start when LLM credentials are missing (investigations will
        # fail gracefully at call time rather than crashing on startup).
        self.clients: list[OpenAI] = []
        self.models: list[str] = []
        for url, key, model in endpoints:
            try:
                self.clients.append(
                    OpenAI(base_url=url, api_key=key or "", timeout=self.request_timeout)
                )
                self.models.append(model or config.llm_model)
            except openai.OpenAIError as e:
                logger.warning(
                    "Skipping LLM endpoint %s: missing or invalid credentials (%s)", url, e
                )

        if not self.clients:
            logger.warning(
                "No LLM endpoints with valid credentials configured; "
                "investigations will fail until an API key is provided"
            )

    def _is_retryable(self, error: Exception) -> bool:
        """Return True for transient errors worth retrying / failing over."""
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIConnectionError):
            return True
        if isinstance(error, openai.APIStatusError):
            # 429 Too Many Requests: the LLM endpoint is rate-limiting us.
            # Back off (respecting Retry-After when advertised) instead of
            # failing fast so an investigation can complete once the quota
            # resets.
            if error.status_code == 429:
                return True
            # 5xx server errors are transient; retry them too.
            if error.status_code >= 500:
                return True
        return False

    def _should_fail_over(
        self,
        error: Exception,
        endpoint_index: int,
        attempt: int,
        call_timeout: float,
        endpoint_label: str,
        session: str,
    ) -> bool:
        """Decide whether to abandon this endpoint and try the next one.

        A timeout is a capacity signal, not a blip: the endpoint accepted the
        request and could not answer inside the time it was given. Retrying the
        same slow endpoint spends the budget waiting for the same result, so the
        next endpoint is tried instead. Retries stay for connection errors and
        5xx responses, which really are transient.

        A 429 from a free aggregator (freellmapi) means every model on that
        endpoint is exhausted right now.  Sleeping 30 s and retrying the same
        endpoint wastes the budget and keeps the messages[] list in RAM the
        whole time.  When another endpoint is available, fail over immediately;
        only sleep when we have no other choice.

        With only one endpoint configured there is nothing to fail over to, so
        the normal retry policy applies and the endpoint gets another attempt.
        """
        retryable = self._is_retryable(error)
        has_another_endpoint = endpoint_index < len(self.clients) - 1

        if _is_timeout(error) and has_another_endpoint:
            logger.warning(
                "LLM call timed out on %s after %.0fs; failing over instead of "
                "retrying the same endpoint (session=%s)",
                endpoint_label,
                call_timeout,
                session,
            )
            return True

        # 429 with another endpoint available: fail over immediately instead of
        # sleeping.  Sleeping 30 s per attempt while holding the full message
        # history caused the pod to OOM when several investigations ran at once.
        is_rate_limited = (
            isinstance(error, openai.APIStatusError) and error.status_code == 429
        )
        if is_rate_limited and has_another_endpoint:
            logger.warning(
                "LLM endpoint %s returned 429; failing over immediately (session=%s)",
                endpoint_label,
                session,
            )
            return True

        if retryable and attempt < self.MAX_RETRIES - 1:
            return False

        if retryable:
            logger.warning(
                "LLM retries exhausted on %s (%d); failing over",
                endpoint_label,
                self.MAX_RETRIES,
            )
        else:
            logger.error("LLM call failed (non-retryable, session=%s): %s", session, error)
        return True

    def _compute_backoff(
        self,
        error: Exception,
        attempt: int,
        started: float,
        budget_seconds: float | None,
    ) -> float:
        """Calculate backoff duration for a failed LLM call."""
        if isinstance(error, openai.APIStatusError) and error.status_code == 429:
            retry_after: float | None = None
            try:
                ra = error.response.headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
            except (AttributeError, ValueError):
                pass
            fallback = self.RATE_LIMIT_BACKOFF_SECONDS
            backoff = min(
                retry_after if retry_after is not None else fallback,
                fallback,
            )
            # Never sleep past the remaining budget - there is no point waiting
            # 30 s for a rate limit to clear if the investigation deadline is in 10 s.
            remaining_now = _remaining_budget(started, budget_seconds)
            if remaining_now is not None:
                floor_reserve = remaining_now - _MIN_CALL_TIMEOUT_SECONDS
                backoff = min(backoff, max(0.0, floor_reserve))
            return backoff

        return min(
            self.INITIAL_BACKOFF_SECONDS * (2**attempt),
            self.MAX_BACKOFF_SECONDS,
        )

    def _call_endpoint(
        self,
        client: OpenAI,
        model: str,
        endpoint_index: int,
        endpoint_label: str,
        messages: list[dict],
        optional: dict,
        max_tokens: int,
        session: str,
        started: float,
        budget_seconds: float | None,
    ) -> tuple[Any | None, Exception | None]:
        """Try calling an endpoint up to MAX_RETRIES times."""
        import time as _time

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            remaining = _remaining_budget(started, budget_seconds)
            if _out_of_time(remaining, attempt):
                break
            call_timeout = _call_timeout(
                self.request_timeout, remaining, self.MAX_RETRIES - attempt
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for deterministic output
                    max_tokens=max_tokens,
                    timeout=call_timeout,
                    **optional,
                )
                if endpoint_index > 0:
                    logger.warning(
                        "LLM failover succeeded on %s (session=%s)",
                        endpoint_label,
                        session,
                    )
                return response.choices[0].message, None
            except (
                openai.OpenAIError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
            ) as e:
                last_error = e
                if self._should_fail_over(
                    e, endpoint_index, attempt, call_timeout, endpoint_label, session
                ):
                    break
                backoff = self._compute_backoff(e, attempt, started, budget_seconds)
                if backoff <= 0:
                    break
                logger.warning(
                    "LLM call failed (attempt %d/%d) on %s, retrying in %.1fs (session=%s): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    endpoint_label,
                    backoff,
                    session,
                    e,
                )
                _time.sleep(backoff)

        return None, last_error

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 2000,
        session: str = "",
        budget_seconds: float | None = None,
    ):
        """Call the LLM with optional tool use; failing over across endpoints
        and raising the last error.

        `session` labels the call in logs (the incident id). The client is
        stateless — it holds no conversation — so the label exists to make a
        log line traceable to exactly one investigation when several run at
        once, not because any state is attached to it.
        """
        import time as _time

        # Tool kwargs are only sent when there are tools. Passing
        # tool_choice="none" alongside tools=None is rejected by some
        # OpenAI-compatible endpoints, and tool-free callers (the daily
        # summary) must not depend on that being tolerated.
        optional = {}
        if tools is not None:
            optional = {"tools": tools, "tool_choice": tool_choice}

        # Counted down across every attempt and every endpoint. A snapshot
        # would let 3 retries x 30s run 90 seconds past the point where the
        # investigation needed its reserve, which is exactly how a run with 13
        # successful tool calls ended with no conclusion.
        started = _time.monotonic()

        last_error: Exception | None = None
        for idx, (client, model) in enumerate(zip(self.clients, self.models)):
            endpoint_label = f"endpoint {idx + 1}/{len(self.endpoints)}"
            message, err = self._call_endpoint(
                client=client,
                model=model,
                endpoint_index=idx,
                endpoint_label=endpoint_label,
                messages=messages,
                optional=optional,
                max_tokens=max_tokens,
                session=session,
                started=started,
                budget_seconds=budget_seconds,
            )
            if message is not None:
                return message
            if err is not None:
                last_error = err

        # All endpoints exhausted.
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM call failed after all endpoints exhausted")  # pragma: no cover


class PamawasInvestigator:
    """Main investigator orchestrator."""

    # How many turns in a row may produce prose without a tool call before the
    # loop stops asking the model to continue. A model that replies with text
    # three times running has stopped investigating; asking it a fourth time
    # only burns wall-clock budget that a paged engineer is waiting on.
    MAX_CONSECUTIVE_TEXT_TURNS = 3

    # Wall-clock budget reserved at the end of an investigation for the model to
    # write its findings and for them to be persisted.
    #
    # Without a reserve the deadline was only noticed at the top of a turn, so a
    # run that spent its last minutes inside tool calls discovered it had no
    # time left only after the model had gathered evidence it never got to
    # report. Runs of 10m21s ended with "deadline exceeded without submitting
    # findings" despite every Prometheus and Loki query having succeeded.
    DEADLINE_RESERVE_FRACTION = 0.25
    DEADLINE_RESERVE_MIN_SECONDS = 90.0

    # How long before the reserve begins the model is told to wrap up. It gets a
    # chance to synthesise what it has rather than being cut off mid-exploration.
    DEADLINE_WARN_FRACTION = 0.5

    def __init__(self, config: Config):
        self.config = config
        self.llm = InvestigatorLLM(config)
        self.tools = InvestigatorTools(config)

        # Initialize fetchers for evidence enrichment
        self.metrics_fetcher: MetricsFetcher = PrometheusFetcher(
            base_url=config.prometheus_url,
            service_label=config.prometheus_service_label,
            timeout_seconds=config.prometheus_timeout_seconds,
        )
        self.logs_fetcher: LogsFetcher = LokiFetcher(
            base_url=config.loki_url,
            service_label=config.loki_service_label,
            timeout_seconds=config.loki_timeout_seconds,
            max_lines=config.loki_max_log_lines,
        )

        self.db_conn = None
        self.persistence = None
        self._connect_db()

    def _connect_db(self):
        """Establish database connection with timeout."""
        if not self.config.database_url:
            logger.warning("No DATABASE_URL provided, database features disabled")
            return

        try:
            self.db_conn = psycopg2.connect(
                self.config.database_url,
                connect_timeout=5,  # 5-second timeout to prevent startup block
            )
            # This connection is used only for read-only lookups (incident
            # context, health probes). autocommit keeps every statement out of
            # an explicit transaction so reads never leak open transactions.
            self.db_conn.autocommit = True
            self.persistence = InvestigationPersistence(self.config.database_url)
            logger.info("Connected to database")
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to connect to database: %s", e)
            increment_db_errors()
            self.db_conn = None

    async def close(self):
        """Close investigator, tools, fetchers, and database connections."""
        await self.tools.close()
        await self.metrics_fetcher.close()
        await self.logs_fetcher.close()
        if self.db_conn and not self.db_conn.closed:
            self.db_conn.close()
        if self.persistence:
            self.persistence.close()

    def _get_incident_context(self, incident_id: str) -> IncidentContext:
        """Retrieve incident context from the database."""
        if not self.db_conn:
            logger.warning("No database connection, returning mock context")
            return IncidentContext(
                incident_id=incident_id,
                title=f"Incident {incident_id}",
                events=[],
                started_at="2026-08-13T02:00:00Z",
            )

        try:
            with self.db_conn.cursor() as cursor:
                # Get incident info
                cursor.execute(
                    """
                    SELECT id, title, status, started_at, resolved_at, severity, affected_services
                    FROM incidents WHERE id = %s
                """,
                    (incident_id,),
                )
                incident_row = cursor.fetchone()

                if not incident_row:
                    return IncidentContext(
                        incident_id=incident_id,
                        title=f"Incident {incident_id} not found",
                        events=[],
                        started_at="",
                        error=f"Incident {incident_id} not found",
                    )

                # Get events for this incident
                # NOTE: the column is `occurred_at`, not `timestamp`
                cursor.execute(
                    """
                    SELECT e.id, e.source, e.type, e.occurred_at, e.service, e.environment,
                           e.severity, e.title, e.status, e.labels, e.raw_payload
                    FROM events e
                    JOIN incident_events ie ON e.id = ie.event_id
                    WHERE ie.incident_id = %s
                    ORDER BY e.occurred_at
                """,
                    (incident_id,),
                )
                events = cursor.fetchall()

                return IncidentContext(
                    incident_id=incident_row[0],
                    title=incident_row[1],
                    status=incident_row[2],
                    started_at=str(incident_row[3]),
                    resolved_at=str(incident_row[4]) if incident_row[4] else None,
                    severity=incident_row[5],
                    affected_services=incident_row[6],
                    events=[
                        {
                            "id": e[0],
                            "source": e[1],
                            "type": e[2],
                            "timestamp": str(e[3]),
                            "service": e[4],
                            "environment": e[5],
                            "severity": e[6],
                            "title": e[7],
                            "status": e[8],
                            "labels": e[9],
                            "raw_payload": e[10],
                        }
                        for e in events
                    ],
                )
            # db_conn runs in autocommit mode, so each statement completes
            # immediately and no transaction is left open.
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to get incident context: %s", e)
            # Roll back the aborted transaction so the connection remains usable
            try:
                self.db_conn.rollback()
            except psycopg2.Error:
                pass
            return IncidentContext(
                incident_id=incident_id,
                title="Error loading incident",
                events=[],
                started_at="",
                error=str(e),
            )

    def _truncate_context(self, text: str, limit: int) -> str:
        """Truncate context to prevent excessive growth."""
        if len(text) <= limit:
            return text
        # Keep beginning and end, accounting for truncation marker
        marker = "\n... [TRUNCATED] ...\n"
        marker_len = len(marker)
        half = (limit - marker_len) // 2
        return text[:half] + marker + text[-half:]

    def _build_system_prompt(self, context: IncidentContext | None = None) -> str:
        """Build the system prompt for the investigator.

        The incident's identity is part of the system prompt, not only of the
        first user message. Every investigation gets a fresh conversation, but a
        model with no stated scope will happily explain the incident using
        another incident's evidence — the related-incidents tool hands it real
        neighbours, and Prometheus and Loki are queried by host, so a
        neighbouring host's symptoms are one query away. Naming the incident at
        the top and stating what is out of scope keeps the findings about this
        incident.
        """
        scope = ""
        if context is not None:
            affected = ", ".join(context.affected_services or []) or "unspecified"
            scope = (
                "\n=== SESSION SCOPE ===\n"
                f"This session investigates exactly one incident: {context.incident_id}.\n"
                f"Title: {context.title}\n"
                f"Affected service(s) for this incident: {affected}\n"
                "Every finding you submit must describe THIS incident. Evidence "
                "about a different host, service or alert is not this incident's "
                "cause, even when it looks similar: say so and leave it out. Use "
                "get_related_incidents only to tell the operator whether this "
                "condition keeps recurring — never as the cause of this "
                "occurrence.\n"
                "=== END SESSION SCOPE ===\n\n"
            )
        return scope + (
            "You are an expert infrastructure incident investigator. Your goal is to:\n"
            "1. Understand the symptom and blast radius\n"
            "2. Find the first abnormal signal, not just the loudest alert\n"
            "3. Check recent changes near the time of the incident\n"
            "4. Check dependencies\n"
            "5. Form and test competing hypotheses rather than confirming one\n"
            "6. Use tool calls deliberately - no duplicate queries, no unused fetches\n"
            "7. Prefer UNKNOWN over a fabricated-sounding conclusion when evidence is "
            "insufficient\n"
            "\n"
            "You have access to tools to query Prometheus, Loki, check recent "
            "deployments, and find related incidents.\n"
            "You must use the submit_findings tool to end the investigation with your "
            "final conclusions. A plain text reply is not a submission: if you "
            "answer in prose instead of calling submit_findings, the operator gets "
            "no conclusion from this run. When you have enough evidence, submit.\n"
            "\n"
            "Every finding must be classified as:\n"
            "- FACT: directly observed in metrics/logs\n"
            "- LIKELY_CAUSE: strongly supported by evidence, not certain\n"
            "- HYPOTHESIS: plausible, unverified\n"
            "- UNKNOWN: insufficient evidence; stated explicitly rather than guessed\n"
            "- RECOMMENDATION: a concrete action the operator should take, derived "
            "from the findings. Put these in their own findings — never bury an "
            "action inside a FACT.\n"
            "\n"
            "Every finding must name its source, because the report shows it and a "
            "reader must be able to tell a metric from a log from your inference:\n"
            "- prometheus: a metric you queried\n"
            "- loki: a log line you read\n"
            "- incidents: incident history\n"
            "- llm: your own synthesis across sources (correlation, inference)\n"
            "- system: a harness limit such as a timeout\n"
            "Also set source_detail to the specific query, stream, or metric the "
            'finding rests on, e.g. \'up{job="system_node", instance="koba-labs"}\'. '
            "If a finding combines metrics AND logs, say so in source_detail "
            "(e.g. 'prometheus up + loki docker_logs') and use source llm, since "
            "the combination is your synthesis, not a single observation.\n"
            "\n"
            "Always provide confidence scores (0.0 to 1.0) for your findings.\n"
            "\n"
            "HOW TO WRITE A FINDING:\n"
            "A finding is a claim, in one or two plain sentences, that an on-call "
            "engineer can act on. Write it the way an engineer writes in a "
            "postmortem, not the way a marketing page reads.\n"
            "- Lead with what happened, then the evidence: 'netbird-agent stopped "
            'reporting at 07:12; up{job="system_node"} dropped from 1 to 0 at the '
            "same minute.'\n"
            "- Never narrate your process. No 'Let me check', 'I will now query', "
            "'Looking at the data, I can see'. Those describe your work, not the "
            "incident, and they are not findings.\n"
            "- No em dashes or en dashes. Use a comma, a period, or 'and'.\n"
            "- Avoid filler and inflated wording: delve, leverage, robust, "
            "seamless, crucial, pivotal, comprehensive, holistic, 'it is worth "
            "noting', 'it is important to', 'in conclusion', 'navigate the "
            "complexities', 'unlock', 'showcase', 'underscore'. Say the thing.\n"
            "- Do not use the 'not just X, but Y' construction, and do not pad "
            "lists to three items for rhythm. State the findings you actually "
            "have.\n"
            "- No hedging stacks ('it may potentially possibly'). One qualifier is "
            "enough, and the confidence score already carries it.\n"
            "\n"
            "Always end with at least one RECOMMENDATION: the concrete next step "
            "the on-call engineer should take. If the incident self-resolved, say "
            "so ('No action needed: ...'). An action must be specific enough to "
            "act on — name the component and the check or change, not 'investigate "
            "further'.\n"
            "\n"
            "UPTIME KUMA MONITORS (external availability):\n"
            "Uptime Kuma probes hosts and URLs from outside the cluster and "
            'exports its state to Prometheus under job="uptime_kuma". This is '
            "the second opinion on whether something is actually down: when the "
            "alert says a target is unreachable, the monitor says whether the "
            "probe failed too, since when, and for how long it has been failing.\n"
            "- monitor_status{monitor_name, monitor_hostname, monitor_type} is "
            "the current check result: 1 up, 0 down, 2 pending, 3 maintenance.\n"
            "- monitor_response_time is the last probe latency in milliseconds.\n"
            '- monitor_uptime_ratio{window="1d"|"30d"|"365d"} is the fraction '
            "of successful checks in that window.\n"
            "- monitor_cert_days_remaining and monitor_cert_is_valid cover TLS "
            "expiry for http monitors.\n"
            'monitor_hostname is often the string "null" or a shared front-end '
            "domain for http monitors, so match those on monitor_name or "
            "monitor_url instead. A monitor's name is free text an operator "
            "wrote, so search it loosely, for example "
            'monitor_status{monitor_name=~"(?i)last-labs"}.\n'
            "Use these to answer three questions the alert cannot: did the probe "
            "fail, when exactly did it start failing (changes(monitor_status[1h]) "
            "or the raw series over the incident window), and is this a chronic "
            "flap or a first occurrence (monitor_uptime_ratio).\n"
            "\n"
            "CRITICAL INVESTIGATION DISCIPLINE:\n"
            "- Submit findings as soon as you have enough evidence to answer the "
            "incident question — do not exhaust the tool budget unnecessarily.\n"
            "- If a tool call returns unavailable (external dependency degraded), "
            "note it as a limitation and proceed with what evidence you have; do "
            "not retry the same failing query.\n"
            "- If tool calls keep returning errors or unavailable, submit your best "
            "findings with UNKNOWN confidence rather than looping.\n"
            "- Do not issue duplicate or near-duplicate queries.\n"
            "- Each turn should add new evidence or lead directly to submission.\n"
            "\n"
            "IMPORTANT LOGQL SYNTAX RULES:\n"
            "- Loki supports these line-filter operators: |=, |!=, |~, |!~\n"
            "- Never use |==, |=~, |!=~ — these cause parse errors\n"
            "- Never use 'OR' between stream selectors like {a} OR {b} — Loki "
            "cannot parse. Combine selectors or use pipe-separated values with |~\n"
            "- Never use '|=' with parentheses grouping alternation like |= "
            '("term" OR "other") — Loki only supports pipe-separated values '
            'inside the string: |= "term|other"\n'
            "- Use |!= for a 'does not contain' line filter, e.g. {app=\"x\"} |= "
            '"error" |!= "debug". Note: a bare \'!=\' without the pipe (label '
            'matcher) goes inside curly braces: {service!="value"}\n'
            "- When grouping alternatives, use |~ with pipe-separated values, "
            'e.g. |~ "error|timeout|refused" or combine streams: {service} |~ "x|y"\n'
            "\n"
            "CRITICAL: LABEL NAMES ARE DEPLOYMENT-SPECIFIC.\n"
            "- A selector naming a label or value that does not exist returns "
            "status=success with ZERO streams. That is indistinguishable from "
            "'no logs', so never conclude 'no logs exist' from an empty result "
            "until you have verified the selector.\n"
            "- Call get_loki_labels first to list the real label names, then "
            "get_loki_labels with a specific label to list its values, and build "
            "your selector from what actually exists.\n"
            "- Do not assume a 'service' or 'app' label exists.\n"
        )

    def _build_verification_prompt(self, context: IncidentContext | None = None) -> str:
        """Build the system prompt for a stale-incident verification.

        A verification answers one question — "is this condition actually still
        down?" — that decides whether a silent incident is closed or kept open.
        It is deliberately narrower than a full investigation: the model is told
        to determine the current state of the affected target and stop, not to
        chase a root cause.
        """
        affected = "unspecified"
        if context is not None:
            affected = ", ".join(context.affected_services or []) or "unspecified"
        incident = context.incident_id if context else "unknown"
        title = context.title if context else "unknown"

        return (
            "\n=== SESSION SCOPE (VERIFICATION) ===\n"
            f"This session verifies one incident: {incident}.\n"
            f"Title: {title}\n"
            f"Affected service(s): {affected}\n"
            "Answer ONE question: is this condition actually still down right now?\n"
            "- Query Prometheus and Uptime Kuma first (monitor_status, up, "
            "scrape_duration_seconds) to establish the current state. Call "
            "get_prometheus_labels / get_loki_labels first if the label schema is "
            "unknown.\n"
            "- Then check logs in Loki and incident history (get_related_incidents) "
            "only if the metrics are ambiguous.\n"
            "- Submit a FACT: 'The target is still down' or 'The target has "
            "recovered'. An UNKNOWN must not claim recovery.\n"
            "=== END SESSION SCOPE ===\n"
        )

    def _build_initial_message(self, context: IncidentContext) -> dict:
        """Build the initial user message with incident context."""
        return {
            "role": "user",
            "content": (
                f"Investigate this incident:\n\n"
                f"Incident ID: {context.incident_id}\n"
                f"Title: {context.title}\n"
                f"Status: {context.status}\n"
                f"Started at: {context.started_at}\n"
                f"Severity: {context.severity}\n"
                f"Affected services: {', '.join(context.affected_services or [])}\n\n"
                f"Events in this incident:\n"
                f"{json.dumps(context.events, indent=2)[: self.config.truncation_limit]}\n\n"
                "Begin your investigation by understanding what happened. Use your "
                "tools to gather evidence and form hypotheses."
            ),
        }

    def write_day_summary(self, facts: dict) -> str:
        """Write the daily report's prose summary from the given facts.

        LLM usage stays in this service; the reporter only calls the endpoint.
        Returns "" on any failure so the digest still delivers.
        """
        return _write_day_summary(self.llm, facts)

    def _handle_error_context(self, context: IncidentContext) -> list[Finding]:
        """Handle error context and return error finding."""
        logger.error(f"Failed to get incident context: {context.error}")
        return [
            Finding(
                type=EvidenceType.UNKNOWN,
                content=f"Failed to load incident context: {context.error}",
                source=EvidenceSource.SYSTEM,
                confidence=0.0,
            )
        ]

    async def _execute_tool(self, function_name: str, function_args: dict):  # noqa: C901
        """Execute a tool call and return the result.

        Runs in the asyncio event loop, so it awaits the async adapter
        methods directly (not the sync `self.tools.*` wrappers, which call
        ``loop.run_until_complete`` and would fail inside a running loop).
        Loki/Prometheus parse errors are caught here and returned as
        structured error dicts (status="error") so a single bad query
        does not terminate the whole investigation run.
        """
        try:
            if function_name == "query_prometheus":
                # Guard against a missing start/end/promql argument. A tool call
                # from the LLM may omit required keys; a direct [] lookup would
                # raise KeyError and abort the whole investigation with a useless
                # "Investigation error: 'start'" finding.
                promql = function_args.get("promql")
                start = function_args.get("start")
                end = function_args.get("end")
                if promql is None or start is None or end is None:
                    return (
                        {
                            "status": "error",
                            "error": "query_prometheus requires promql, start, end",
                        },
                        None,
                        False,
                    )
                result = await self.tools._prometheus_adapter.query_range(promql, start, end)
                return result, None, False

            elif function_name == "get_prometheus_labels":
                label = function_args.get("label") or ""
                if label:
                    result = await self.tools._prometheus_adapter.query_label_values(label)
                else:
                    result = await self.tools._prometheus_adapter.query_labels()
                return result, None, False

            elif function_name == "get_loki_labels":
                label = function_args.get("label") or ""
                if label:
                    result = await self.tools._loki_adapter.query_label_values(label)
                else:
                    result = await self.tools._loki_adapter.query_labels()
                return result, None, False

            elif function_name == "query_loki":
                logql = function_args.get("logql")
                start = function_args.get("start")
                end = function_args.get("end")
                if logql is None or start is None or end is None:
                    return (
                        {
                            "status": "error",
                            "error": "query_loki requires logql, start, end",
                        },
                        None,
                        False,
                    )
                result = await self.tools._loki_adapter.query_range(
                    logql,
                    start,
                    end,
                    function_args.get("limit", 100),
                )
                return result, None, False

            elif function_name == "get_related_incidents":
                service = function_args.get("service")
                keywords = function_args.get("symptom_keywords") or []
                if service is None:
                    return (
                        {
                            "status": "error",
                            "error": "get_related_incidents requires service",
                        },
                        None,
                        False,
                    )
                result = await self.tools._related_incidents_adapter.find_related(service, keywords)
                return result, None, False

            elif function_name == "submit_findings":
                raw = function_args.get("findings")
                if not isinstance(raw, list) or not raw:
                    return (
                        {
                            "status": "error",
                            "error": "submit_findings requires a non-empty findings list",
                        },
                        None,
                        False,
                    )
                findings = []
                for f in raw:
                    if not isinstance(f, dict):
                        return (
                            {
                                "status": "error",
                                "error": "submit_findings: each finding must be an object",
                            },
                            None,
                            False,
                        )
                    ftype = f.get("type")
                    content = f.get("content")
                    source = f.get("source")
                    confidence = f.get("confidence")
                    if not ftype or content is None or not source or confidence is None:
                        return (
                            {
                                "status": "error",
                                "error": "submit_findings: finding missing required fields",
                            },
                            None,
                            False,
                        )
                    # A finding's source decides how the report labels it and
                    # whether the claim is checkable, so an unrecognized value
                    # is a tool-call error the model must correct rather than
                    # something to coerce into a default.
                    try:
                        evidence_source = EvidenceSource(source)
                    except ValueError:
                        return (
                            {
                                "status": "error",
                                "error": (
                                    "submit_findings: source must be one of "
                                    "prometheus, loki, incidents, llm, system"
                                ),
                            },
                            None,
                            False,
                        )
                    findings.append(
                        Finding(
                            type=EvidenceType(ftype),
                            content=content,
                            source=evidence_source,
                            confidence=confidence,
                            source_detail=f.get("source_detail") or "",
                        )
                    )
                result = self.tools.submit_findings(findings)
                return result, findings, True  # result, findings, is_final
        except (
            LokiError,
            PrometheusError,
            DeploymentAdapterError,
            RelatedIncidentsError,
        ) as e:
            # A query/tool failure is a structured error, not a run-ending
            # crash. Return it as an error dict so the loop can continue.
            logger.warning("Tool %s returned an error: %s", function_name, e)
            return {"status": "error", "error": str(e)}, None, False
        except Exception as e:
            # Unexpected crash — propagate so the investigation-loop handler
            # (not this function) decides how to persist it.
            logger.error("Tool %s crashed unexpectedly: %s", function_name, e)
            raise
        else:
            return {"error": f"Unknown tool: {function_name}"}, None, False

    def _truncate_result(self, result: any) -> str:
        """Truncate result if too large."""
        result_str = json.dumps(result)
        if len(result_str) > self.config.truncation_limit:
            result_str = self._truncate_context(result_str, self.config.truncation_limit)
            result = {"truncated": result_str}
        return result_str

    @staticmethod
    def _call_signature(function_name: str, function_args: dict) -> str:
        """A stable identity for one tool call.

        Arguments are sorted and serialized so that two spellings of the same
        request — same keys in a different order — compare equal. A model
        re-asking a question rarely re-types it identically.
        """
        try:
            rendered = json.dumps(function_args or {}, sort_keys=True, default=str)
        except (TypeError, ValueError):
            rendered = str(function_args)
        return f"{function_name}:{rendered}"

    def _serve_repeated_call(
        self,
        tool_call,
        function_name: str,
        function_args: dict,
        state: InvestigationState,
        messages: list[dict],
    ) -> bool:
        """Answer a repeated tool call from the first result.

        Returns True when the call was served from the cache, in which case the
        caller must not execute it. The API requires a message for every tool
        call id, so the model still gets a reply — the same result it already
        has, plus a note saying so. `tool_call_count` is deliberately not
        incremented: the point is that a repeat costs no budget.

        `submit_findings` is never cached. It is the terminal action, and its
        arguments carry the findings themselves.
        """
        if function_name == "submit_findings":
            return False

        signature = self._call_signature(function_name, function_args)
        previous = state.seen_calls.get(signature)
        if previous is None:
            return False

        logger.warning(
            "Repeated tool call served from cache: %s (args=%s)",
            function_name,
            function_args,
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    "You already made this exact call earlier in this "
                    "investigation. Its result is unchanged and repeated below. "
                    "Do not call it again. Either query something different or "
                    "submit your findings with submit_findings.\n\n" + previous
                ),
            }
        )
        return True

    def _handle_tool_result(
        self,
        tool_call,
        function_name: str,
        function_args: dict,
        result: any,
        tool_start: float,
        state: InvestigationState,
        messages: list[dict],
    ):
        """Process tool result and update state."""
        result_str = self._truncate_result(result)

        # Add tool result to conversation
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_str})

        state.tool_calls.append(
            ToolResult(
                tool_name=function_name,
                # The resolved arguments, not the envelope: a nested call would
                # otherwise be recorded as {"name": ..., "arguments": {...}}.
                arguments=function_args,
                result=result,
                duration_ms=(time.time() - tool_start) * 1000,
            )
        )

        state.tool_call_count += 1

    def _check_max_tools_reached(
        self, state: InvestigationState, must_submit: bool, messages: list[dict]
    ):
        """Check if max tool calls reached and prompt for submission if needed."""
        if state.tool_call_count >= state.max_tool_calls and not must_submit:
            logger.info("Reached max tool calls, forcing findings submission")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool calls. "
                        "Please submit your findings using the submit_findings tool."
                    ),
                }
            )

    def _handle_text_response(
        self, response, state: InvestigationState, must_submit: bool, messages: list[dict]
    ):
        """Handle text response (no tool calls)."""
        logger.info(f"LLM response (no tool calls): {response.content[:100]}...")
        state.consecutive_text_turns += 1

        if must_submit:
            # Either the turn budget or the patience budget is gone, and the
            # turn was a forced submission — the API was told it may only call
            # submit_findings. A text reply therefore means the endpoint ignored
            # tool_choice. Take what the model said as a hypothesis rather than
            # discarding it: a partial answer beats the null "deadline exceeded"
            # result this path used to produce after burning the full wall-clock
            # budget.
            logger.warning(
                "Text response without tool use (%d consecutive) - recording it as a hypothesis",
                state.consecutive_text_turns,
            )
            content = (response.content or "").strip()
            if not content:
                # Nothing was written, so there is no hypothesis to record.
                # Inventing one ("Investigation completed via text response")
                # put a sentence on the report that the model never said.
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=(
                            "Investigation ended without a stated conclusion; "
                            "the model returned no text and no findings."
                        ),
                        source=EvidenceSource.SYSTEM,
                        confidence=0.0,
                    )
                )
                state.completed = True
                return
            if _is_reasoning_preamble(content):
                # The model narrated its process instead of stating a
                # conclusion ("Let me analyze the data more carefully. I need to
                # convert the Unix timestamps..."). Storing that as a finding
                # put a wall of chain-of-thought under "🎯 Likely cause" on a
                # critical card. There is no conclusion here to record, but the
                # run still needs an honest terminal row so the report says the
                # investigation ended without one rather than showing nothing.
                logger.warning("Discarding reasoning preamble as a finding: %s", content[:120])
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=(
                            "Investigation ended without a stated conclusion; the "
                            "model only described its process."
                        ),
                        source=EvidenceSource.SYSTEM,
                        confidence=0.0,
                    )
                )
                state.completed = True
                return
            state.findings.append(
                Finding(
                    type=EvidenceType.HYPOTHESIS,
                    content=_plain_typography(content),
                    source=EvidenceSource.LLM,
                    confidence=0.3,
                )
            )
            state.completed = True
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue your investigation. Use your tools to gather more "
                        "evidence, or submit your findings when ready."
                    ),
                }
            )

    async def _handle_submit_findings(
        self,
        findings: list[Finding],
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
    ):
        """Handle submit_findings tool call."""
        logger.info("Investigation completed via submit_findings")
        findings = [_sanitize_finding(f) for f in findings]
        state.findings = findings
        state.completed = True

        if self.persistence:
            await self._persist_investigation(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                findings=findings,
                tool_calls=state.tool_calls,
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version="pamawas-investigator-v1",
                tool_contract=1,
                max_tool_calls=self.config.max_tool_calls,
                service=state.service,
            )

    async def investigate_async(
        self, incident_id: str, request_key_hash: str = ""
    ) -> list[Finding]:
        """Run the bounded investigation loop; every terminal path persists
        the run and its evidence."""
        """Run the bounded investigation loop; every terminal path persists
        the run and its evidence.

        Concurrency is bounded outside this method, by the worker pool that
        claims runs from the database. The deadline below starts here, so time
        spent queued is not charged against the investigation's own budget.
        """
        logger.info(f"Starting investigation for incident {incident_id}")
        increment_investigations()

        # Overall investigation deadline (wall-clock). Once exceeded, we stop
        # calling tools and submit whatever partial findings we have.
        deadline = time.monotonic() + self.config.investigation_timeout_seconds

        # Get incident context
        context = self._get_incident_context(incident_id)
        if hasattr(context, "error") and context.error:
            return await self._handle_error_context_path(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                context=context,
            )

        # Initialize investigation state
        state = InvestigationState(
            incident_id=incident_id,
            findings=[],
            tool_calls=[],
            max_tool_calls=self.config.max_tool_calls,
            request_key_hash=request_key_hash,
            service=context.affected_services[0] if context.affected_services else None,
        )

        return await self._run_investigation_loop(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            state=state,
            context=context,
            deadline=deadline,
        )

    async def verify_incident(self, incident_id: str) -> dict[str, Any]:
        """Return a verdict on whether an incident is still actually down.

        This audits a stale-open incident: the Scheduler discovered an incident
        silent past INCIDENT_STALE_AFTER, enqueued it in stale_verifications,
        and the async stale-verification worker executes this bounded check.

        The verdict is evidence-backed. Only a positive claim that the target
        recovered closes the incident; anything else — still down, uncertain, or
        an error — keeps it open. Returns a dict with verdict, a one-line
        reason drawn from the highest-confidence finding, and the findings.
        """
        context = self._get_incident_context(incident_id)
        if hasattr(context, "error") and context.error:
            return {
                "verdict": "still_down",
                "reason": context.error,
                "findings": [f.to_dict() for f in self._handle_error_context(context)],
            }

        state = InvestigationState(
            incident_id=incident_id,
            findings=[],
            tool_calls=[],
            # A verification is one question, not a full investigation. Keep the
            # budget small so it checks the target's availability and stops.
            max_tool_calls=self.config.verify_max_tool_calls,
            request_key_hash="verification",
            service=context.affected_services[0] if context.affected_services else None,
        )

        deadline = time.monotonic() + self.config.verify_timeout_seconds
        findings = await self._run_investigation_loop(
            incident_id=incident_id,
            request_key_hash=f"verification:{incident_id}",
            state=state,
            context=context,
            deadline=deadline,
            verification=True,
        )

        # Highest-confidence evidence drives the reason. A positive recovery
        # claim closes the incident; the default is still_down so an inconclusive
        # verification never closes it.
        verdict = "still_down"
        reason = "The verification did not confirm the condition recovered."
        ranked = sorted(findings, key=lambda f: f.confidence, reverse=True)
        for f in ranked:
            if f.type in (EvidenceType.FACT, EvidenceType.LIKELY_CAUSE) and _claim_recovered(
                f.content
            ):
                verdict = "recovered"
                reason = f.content
                break
        else:
            for f in ranked:
                if (
                    f.type in (
                        EvidenceType.FACT,
                        EvidenceType.LIKELY_CAUSE,
                        EvidenceType.HYPOTHESIS,
                    )
                    and not _is_placeholder_or_dummy(f.content)
                ):
                    reason = f.content
                    break

        return {
            "verdict": verdict,
            "reason": reason,
            "findings": [f.to_dict() for f in findings],
        }

    async def _handle_error_context_path(
        self,
        incident_id: str,
        request_key_hash: str,
        context: IncidentContext,
    ) -> list[Finding]:
        """Persist an error-context finding and return the findings."""
        findings = self._handle_error_context(context)
        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=findings,
            tool_calls=[],
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version="pamawas-investigator-v1",
            tool_contract=1,
            max_tool_calls=self.config.max_tool_calls,
            service=context.affected_services[0] if context.affected_services else None,
        )
        return findings

    def _submission_pressure(
        self,
        incident_id: str,
        remaining: float,
        warned: bool,
        forced_already: bool,
        state: InvestigationState,
        messages: list[dict],
    ) -> tuple[bool, bool]:
        """Decide whether this turn must be a forced submission.

        Two things can force it: the wall-clock deadline entering its reserve,
        and the model spending its no-tool patience on prose instead of a
        submission. Keeping both here means the loop stays one branch.
        """
        force_submit, warned = self._deadline_action(
            incident_id=incident_id,
            remaining=remaining,
            warned=warned,
            forced_already=forced_already,
            messages=messages,
        )
        if force_submit or forced_already:
            return force_submit, warned
        return self._escalate_text_patience(incident_id, state, messages), warned

    def _escalate_text_patience(
        self, incident_id: str, state: InvestigationState, messages: list[dict]
    ) -> bool:
        """Force a structured submission once the model stops calling tools.

        A model that has written several turns without calling a tool and
        without submitting is not going to submit on its own. Giving up here is
        what produced cards that said "UNKNOWN (0%): the model only described
        its process" on a critical incident even though the model had just seen
        real evidence: its prose was discarded while a structured answer was
        still one forced call away.

        Returns True when this turn must be a forced submission.
        """
        if state.consecutive_text_turns < self.MAX_CONSECUTIVE_TEXT_TURNS:
            return False
        logger.warning(
            "Incident %s replied %d times without a tool call; forcing findings "
            "submission instead of recording a null result",
            incident_id,
            state.consecutive_text_turns,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop writing prose. Call submit_findings now and record what you "
                    "have established from the evidence you already gathered, "
                    "including anything you are still unsure about."
                ),
            }
        )
        increment_forced_submission()
        return True

    def _deadline_action(
        self,
        incident_id: str,
        remaining: float,
        warned: bool,
        forced_already: bool,
        messages: list[dict],
    ) -> tuple[bool, bool]:
        """Decide what the remaining wall-clock budget means for this turn.

        Returns (force_submit, warned). The budget is spent in three phases:
        exploring, then a nudge to wrap up, then a hard requirement to submit.
        The reserve exists so evidence already gathered reaches the report
        instead of being thrown away as "deadline exceeded".
        """
        reserve = max(
            self.DEADLINE_RESERVE_MIN_SECONDS,
            self.config.investigation_timeout_seconds * self.DEADLINE_RESERVE_FRACTION,
        )
        if remaining <= reserve:
            if not forced_already:
                logger.warning(
                    "Incident %s is into its deadline reserve (%.0fs left); "
                    "forcing findings submission",
                    incident_id,
                    remaining,
                )
                increment_forced_submission()
            return True, warned

        if not warned and remaining <= reserve * (1 + self.DEADLINE_WARN_FRACTION):
            # A softer warning first: the model can still choose to submit on
            # its own terms, which reads better than a forced summary.
            logger.info(
                "Incident %s has %.0fs left; asking the model to wrap up",
                incident_id,
                remaining,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have about {int(remaining)} seconds left. Stop "
                        "investigating and submit your findings now with "
                        "submit_findings, reporting what you have established "
                        "so far."
                    ),
                }
            )
            return False, True

        return False, warned

    async def _run_investigation_loop(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        context: IncidentContext,
        deadline: float,
        verification: bool = False,
    ) -> list[Finding]:
        """Run the bounded investigation tool-calling loop and persist final findings."""
        # Build system prompt and initial message
        system_prompt = (
            self._build_verification_prompt(context)
            if verification
            else self._build_system_prompt(context)
        )
        initial_message = self._build_initial_message(context)

        messages = [{"role": "system", "content": system_prompt}, initial_message]

        # Investigation loop. Bounded on TURNS (LLM calls), not tool calls: a
        # turn may issue zero tool calls (a prose reply), and if only tool calls
        # advanced the counter such a turn would loop forever until the
        # wall-clock deadline and end with a synthesized UNKNOWN.
        # The reserve exists so the model has time to write up what it found.
        # Exploring must stop when the reserve opens, which is earlier than the
        # hard deadline; tool execution enforces this boundary.
        reserve = max(
            self.DEADLINE_RESERVE_MIN_SECONDS,
            self.config.investigation_timeout_seconds * self.DEADLINE_RESERVE_FRACTION,
        )
        explore_deadline = deadline - reserve

        exhausted_by = ""
        warned_about_deadline = False
        forced_submission_started = False
        while state.turn_count < state.max_tool_calls:
            # Enforce overall deadline: if no time remains, stop and submit.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                exhausted_by = "deadline"
                logger.warning(
                    "Investigation deadline exceeded for incident %s, submitting partial findings",
                    incident_id,
                )
                break

            force_submit, warned_about_deadline = self._submission_pressure(
                incident_id=incident_id,
                remaining=remaining,
                warned=warned_about_deadline,
                forced_already=forced_submission_started,
                state=state,
                messages=messages,
            )
            if force_submit:
                forced_submission_started = True

            state.turn_count += 1
            logger.info(
                "Investigation turn %d/%d (tool calls so far: %d)",
                state.turn_count,
                state.max_tool_calls,
                state.tool_call_count,
            )

            try:
                result = await self._handle_investigation_turn(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    state=state,
                    messages=messages,
                    force_submit=force_submit,
                    deadline=deadline,
                    explore_deadline=explore_deadline,
                )
                if result is not None:
                    return result
            except (
                openai.OpenAIError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
            ) as e:
                logger.error("Error in investigation loop: %s", e)
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=f"Investigation error: {e!s}",
                        source=EvidenceSource.SYSTEM,
                        confidence=0.0,
                    )
                )
                await self._persist_investigation(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    findings=state.findings,
                    tool_calls=state.tool_calls,
                    model_provider=self.config.llm_base_url,
                    model_name=self.config.llm_model,
                    prompt_version="pamawas-investigator-v1",
                    tool_contract=1,
                    max_tool_calls=self.config.max_tool_calls,
                    service=state.service,
                )
                return state.findings

        # If we exit the loop without submitting findings (deadline or turn
        # budget), synthesize a submission from whatever partial findings and
        # tool results we have.
        if not state.findings:
            if exhausted_by == "deadline":
                reason = "Investigation deadline exceeded"
            else:
                reason = "Investigation exhausted its turn budget"
            if state.tool_calls:
                # Count what actually succeeded, rather than reporting the
                # status of the last three calls. "query_prometheus: success;
                # query_loki: success" told a reader nothing about whether the
                # investigation had learned anything before it ran out of time.
                succeeded = sum(
                    1
                    for tc in state.tool_calls
                    if not (isinstance(tc.result, dict) and tc.result.get("status") == "error")
                )
                content = (
                    f"{reason} without submitting findings. "
                    f"{succeeded} of {len(state.tool_calls)} tool calls returned "
                    "data; no conclusion was recorded before the budget ran out."
                )
            else:
                content = f"{reason} without calling any tools"
            state.findings.append(
                Finding(
                    type=EvidenceType.UNKNOWN,
                    content=content,
                    source=EvidenceSource.SYSTEM,
                    confidence=0.0,
                )
            )

        increment_findings_by_type(state.findings)
        if exhausted_by == "deadline":
            increment_deadline_exceeded()
        else:
            increment_max_tool_calls_reached()
        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=state.findings,
            tool_calls=state.tool_calls,
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version="pamawas-investigator-v1",
            tool_contract=1,
            max_tool_calls=self.config.max_tool_calls,
            service=state.service,
        )

        return state.findings

    async def _handle_investigation_turn(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        messages: list[dict],
        force_submit: bool = False,
        deadline: float | None = None,
        explore_deadline: float | None = None,
    ) -> list[Finding] | None:
        """Run a single investigation turn (LLM call + tool handling);
        returns findings if complete."""
        is_final_turn = state.turn_count >= state.max_tool_calls
        # A forced submission is a hard requirement, not a hint: the model is
        # told to call submit_findings and the API refuses anything else, so the
        # turn cannot be spent on another query with no time left to report it.
        must_submit = is_final_turn or force_submit
        tool_choice = (
            {"type": "function", "function": {"name": "submit_findings"}} if must_submit else "auto"
        )
        tools = ToolRegistry.get_tools()

        # Call the LLM off the asyncio event loop: chat_completion is a
        # synchronous SDK call (retries with backoff) that would block health
        # probes and kill the background investigation task.
        #
        # An exploring turn may spend what is left before the reserve opens, not
        # what is left before the hard deadline. Using the deadline here let one
        # slow explore call eat the reserve that exists to write the findings,
        # and the run then ended with nothing to report. The forced-submission
        # turn is the write-up itself, so it may use the whole remainder.
        budget_deadline = deadline if must_submit else (explore_deadline or deadline)
        budget = None if budget_deadline is None else max(0.0, budget_deadline - time.monotonic())
        response = await asyncio.to_thread(
            self.llm.chat_completion,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            session=incident_id,
            budget_seconds=budget,
        )

        # Add the assistant's response to the conversation
        messages.append({"role": "assistant", "content": response.content or ""})

        # Handle tool calls
        tool_calls = getattr(response, "tool_calls", None)

        # A forced turn that comes back with anything other than submit_findings
        # means the endpoint ignored tool_choice. Running that query would spend
        # the reserve that exists to write the findings, and the run would end
        # with nothing to show — the exact failure the reserve was added to
        # prevent. Treat the reply as the final answer instead.
        if (
            force_submit
            and tool_calls
            and not any(_resolved_tool_name(tc) == "submit_findings" for tc in tool_calls)
        ):
            logger.warning(
                "Forced submission turn returned tool calls instead of findings; "
                "completing with the model's text"
            )
            tool_calls = None

        if tool_calls:
            # Real progress: the model is still investigating, so the
            # no-tool patience budget starts over.
            state.consecutive_text_turns = 0
            for tool_call in tool_calls:
                function_name, function_args = _resolve_tool_call(tool_call)

                # The reserve is a wall, not a checkpoint. A single turn may
                # carry several tool calls and each can take up to the tool
                # timeout, so the reserve can be consumed entirely inside one
                # turn: a real run made 13 successful tool calls, blew through
                # its 90-second reserve, hit the hard deadline at the top of the
                # next iteration and submitted nothing. No new tool call starts
                # once the reserve opens — submit_findings always runs, because
                # that is the write-up the reserve exists to protect.
                explore_until = explore_deadline if explore_deadline is not None else deadline
                if (
                    explore_until is not None
                    and function_name != "submit_findings"
                    and time.monotonic() >= explore_until
                ):
                    logger.warning(
                        "Investigation is into its reserve; not starting further tool calls"
                    )
                    return None

                logger.info(f"Calling tool: {function_name} with args: {function_args}")

                # Answer a repeated question from the first answer instead of
                # paying for it again. A real run asked for the same Loki label
                # list eleven times, received the same fifteen names each time,
                # and reached the tool limit having learned nothing after the
                # second call.
                if self._serve_repeated_call(
                    tool_call, function_name, function_args, state, messages
                ):
                    continue

                tool_start = time.time()
                result, findings, is_final = await self._execute_tool(function_name, function_args)

                if is_final:
                    await self._handle_submit_findings(
                        findings, incident_id, request_key_hash, state
                    )
                    increment_findings_by_type(state.findings)
                    return findings

                state.seen_calls[self._call_signature(function_name, function_args)] = (
                    self._truncate_result(result)
                )

                self._handle_tool_result(
                    tool_call, function_name, function_args, result, tool_start, state, messages
                )

                # Incremental persistence after each non-final tool call
                # so that a crash or deadline still leaves a terminal run.
                await self._persist_incremental(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    findings=state.findings,
                    tool_calls=state.tool_calls,
                    model_provider=self.config.llm_base_url,
                    model_name=self.config.llm_model,
                    prompt_version="pamawas-investigator-v1",
                    tool_contract=1,
                    max_tool_calls=self.config.max_tool_calls,
                )

                self._check_max_tools_reached(state, must_submit, messages)
            return None

        self._handle_text_response(response, state, must_submit, messages)
        if state.completed:
            increment_findings_by_type(state.findings)
            await self._persist_investigation(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                findings=state.findings,
                tool_calls=state.tool_calls,
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version="pamawas-investigator-v1",
                tool_contract=1,
                max_tool_calls=self.config.max_tool_calls,
                service=state.service,
            )
            return state.findings
        return None

    async def _persist_investigation(
        self,
        incident_id: str,
        request_key_hash: str,
        findings: list[Finding],
        tool_calls: list[ToolResult],
        model_provider: str,
        model_name: str,
        prompt_version: str,
        tool_contract: int,
        max_tool_calls: int,
        service: str | None = None,
    ):
        """Persist investigation run, tool executions, and evidence in a
        transaction (includes enrichment)."""
        if not self.persistence:
            logger.warning("No persistence layer available, skipping investigation persistence")
            return

        try:
            # Create or get investigation run
            run_id, _ = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=model_provider,
                model_name=model_name,
                prompt_version=prompt_version,
                tool_contract=tool_contract,
                max_tool_calls=max_tool_calls,
            )

            # Update status to running
            self.persistence.update_run_status(run_id, "running")

            # Insert tool executions
            for i, tool_call in enumerate(tool_calls):
                self.persistence.insert_tool_execution(
                    run_id=run_id,
                    sequence_no=i,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    result=tool_call.result,
                    status="completed",
                    duration_ms=int(tool_call.duration_ms),
                )

            # Fetch evidence enrichment (metrics and logs) from Prometheus and Loki
            # Use a default time window for incident context enrichment
            now = datetime.now(UTC)
            five_minutes_ago = datetime.fromtimestamp(now.timestamp() - 300, tz=UTC)
            start_time = five_minutes_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

            enrichment_data = {}
            # Fetch metrics from Prometheus
            if self.metrics_fetcher:
                try:
                    enriched_metrics = await self.metrics_fetcher.fetch_async(
                        incident_id=incident_id,
                        start=start_time,
                        end=end_time,
                        service=service,
                    )
                    enrichment_data["metrics"] = enriched_metrics
                    logger.info(f"Fetched {len(enriched_metrics)} metrics for enrichment")
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(f"Failed to fetch Prometheus metrics for enrichment: {e}")

            # Fetch logs from Loki
            if self.logs_fetcher:
                try:
                    enriched_logs = await self.logs_fetcher.fetch_async(
                        incident_id=incident_id,
                        start=start_time,
                        end=end_time,
                        limit=self.config.loki_max_log_lines,
                        service=service,
                    )
                    enrichment_data["logs"] = enriched_logs
                    logger.info(f"Fetched {len(enriched_logs)} logs for enrichment")
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(f"Failed to fetch Loki logs for enrichment: {e}")

            # Every finding reaches the report and the API through this one
            # call, so normalize it here rather than trusting the prompt.
            findings = [_normalize_finding(f) for f in findings]

            # Insert evidence with enrichment
            self.persistence.insert_evidence(
                incident_id=incident_id,
                run_id=run_id,
                findings=findings,
                metrics=enrichment_data.get("metrics", []),
                logs=enrichment_data.get("logs", []),
            )

            # Mark run as completed
            self.persistence.update_run_status(
                run_id=run_id,
                status="completed",
                completed_at=datetime.now(UTC),
            )

            logger.info(
                f"Persisted investigation run {run_id} with {len(findings)} findings, "
                f"{len(enrichment_data.get('metrics', []))} metrics, "
                f"{len(enrichment_data.get('logs', []))} logs"
            )

        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to persist investigation: %s", e)
            # Try to mark run as failed (terminal state; 'failed_terminal' is
            # the only terminal failure value the DB check constraint permits)
            if "run_id" in locals():
                try:
                    self.persistence.update_run_status(run_id, "failed_terminal")
                except (psycopg2.Error, RuntimeError):
                    pass

    async def _persist_incremental(
        self,
        incident_id: str,
        request_key_hash: str,
        findings: list[Finding],
        tool_calls: list[ToolResult],
        model_provider: str,
        model_name: str,
        prompt_version: str,
        tool_contract: int,
        max_tool_calls: int,
    ):
        """Persist investigation incrementally after each tool call; upserts
        run and appends tool executions/evidence."""
        if not self.persistence:
            logger.warning("No persistence layer available, skipping incremental persistence")
            return

        try:
            run_id, _ = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=model_provider,
                model_name=model_name,
                prompt_version=prompt_version,
                tool_contract=tool_contract,
                max_tool_calls=max_tool_calls,
            )

            # Mark the run as running. Not only when it was just created: the
            # HTTP handler creates the run as `queued` before the work starts,
            # so by the time the first tool result is persisted `created_new`
            # is always False and the run stayed `queued` for its entire life.
            # Anything watching for a terminal status — the scheduler's
            # high-severity follow-up, an operator checking whether a run is
            # alive — saw a run that never started.
            self.persistence.update_run_status(run_id, "running")

            # Insert any new tool executions not yet persisted. We track by
            # sequence_no; insert all tool calls each time is idempotent if the
            # table deduplicates on (run_id, sequence_no), otherwise we only
            # insert beyond the last persisted sequence.
            last_seq = self._last_persisted_sequence(run_id)
            for i, tool_call in enumerate(tool_calls):
                if i <= last_seq:
                    continue
                self.persistence.insert_tool_execution(
                    run_id=run_id,
                    sequence_no=i,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    result=tool_call.result,
                    status="completed",
                    duration_ms=int(tool_call.duration_ms),
                )

            # Insert evidence for new findings (append-only)
            self._insert_incremental_evidence(incident_id, run_id, findings)

        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to persist incrementally: %s", e)

    def _last_persisted_sequence(self, run_id: str) -> int:
        """Return the highest persisted tool-execution sequence number for a run."""
        conn = None
        try:
            conn = self.persistence._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(MAX(sequence_no), -1) FROM tool_executions WHERE run_id = %s",
                    (run_id,),
                )
                row = cursor.fetchone()
            conn.commit()
            return row[0] if row else -1
        except (psycopg2.Error, RuntimeError):
            if conn is not None:
                try:
                    conn.rollback()
                except (psycopg2.Error, AttributeError):
                    pass
            return -1
        finally:
            if conn is not None:
                self.persistence._put_connection(conn)

    def _insert_incremental_evidence(
        self, incident_id: str, run_id: str, findings: list[Finding]
    ) -> None:
        """Insert incremental evidence for new findings (append-only)."""
        if not findings:
            return
        self.persistence.insert_evidence(
            incident_id=incident_id,
            run_id=run_id,
            findings=findings,
            metrics=[],
            logs=[],
        )

    def investigate(self, incident_id: str, request_key_hash: str = "") -> list[Finding]:
        """Synchronous wrapper for investigate_async (backwards compatibility)."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If we're already in an async context, we can't use run_until_complete
            # This shouldn't happen in tests, but just in case
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.investigate_async(incident_id, request_key_hash)
                )
                return future.result()
        else:
            return asyncio.run(self.investigate_async(incident_id, request_key_hash))
