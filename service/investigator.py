"""Main Investigator Service."""

import asyncio
import json
import logging
import time
from datetime import datetime
from datetime import timezone as tz

import openai
import psycopg2
from openai import OpenAI

from config import Config
from metrics import (
    increment_db_errors,
    increment_findings_by_type,
    increment_investigations,
    increment_max_tool_calls_reached,
)
from models import (
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    LogEntry,
    Metric,
    ToolResult,
)
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


class InvestigatorLLM:
    """OpenAI-compatible LLM client for the investigator with endpoint redundancy.

    LLM endpoints are tried in order (primary first, then redundants). On a
    transient failure (timeout/connection/5xx) the next endpoint is attempted,
    so a single unhealthy LLM provider does not stall investigations.
    """

    # Bounded retry policy: transient network/5xx errors are retried with
    # exponential backoff; anything else fails fast.
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 0.5
    MAX_BACKOFF_SECONDS = 4.0
    # SDK-level timeout (seconds) applied to every request. Without this a
    # hung endpoint blocks the investigation loop with no wall-clock cap.
    REQUEST_TIMEOUT = 30.0

    def __init__(self, config: Config):
        # Build ordered endpoint list: explicit redundants first, then the
        # single llm_base_url fallback.
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
        # Single-endpoint fallback (dedup against already-added URLs).
        if config.llm_base_url not in {e[0] for e in endpoints}:
            endpoints.append((config.llm_base_url, config.llm_api_key, config.llm_model))

        self.endpoints = endpoints
        # Each endpoint gets its own OpenAI client.
        self.clients = [
            OpenAI(base_url=url, api_key=key or "", timeout=self.REQUEST_TIMEOUT)
            for url, key, _ in endpoints
        ]
        self.models = [model or config.llm_model for _, _, model in endpoints]

    def _is_retryable(self, error: Exception) -> bool:
        """Return True for transient errors worth retrying / failing over."""
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIConnectionError):
            return True
        if isinstance(error, openai.APIStatusError) and error.status_code >= 500:
            return True
        return False

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ):
        """Call the LLM with optional tool use, failing over across endpoints.

        Each endpoint gets MAX_RETRIES bounded attempts; transient failures move
        to the next endpoint. Non-retryable errors (4xx, auth, bad request)
        propagate immediately. The last error is raised if all endpoints fail.
        """
        import time as _time

        last_error: Exception | None = None
        for idx, (client, model) in enumerate(zip(self.clients, self.models)):
            endpoint_label = f"endpoint {idx + 1}/{len(self.endpoints)}"
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=0.1,  # Low temperature for deterministic output
                        max_tokens=2000,
                    )
                    if idx > 0:
                        logger.warning("LLM failover succeeded on %s", endpoint_label)
                    return response.choices[0].message
                except (
                    openai.OpenAIError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    TimeoutError,
                ) as e:
                    last_error = e
                    if not self._is_retryable(e) or attempt == self.MAX_RETRIES - 1:
                        # Non-retryable, or retries exhausted on this endpoint.
                        if not self._is_retryable(e):
                            logger.error("LLM call failed (non-retryable): %s", e)
                        raise
                    backoff = min(
                        self.INITIAL_BACKOFF_SECONDS * (2**attempt),
                        self.MAX_BACKOFF_SECONDS,
                    )
                    logger.warning(
                        "LLM call failed (attempt %d/%d) on %s, retrying in %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        endpoint_label,
                        backoff,
                        e,
                    )
                    _time.sleep(backoff)

        # All endpoints exhausted.
        if last_error is not None:  # pragma: no cover - defensive
            raise last_error
        raise RuntimeError("LLM call failed after all endpoints exhausted")  # pragma: no cover


class PamawasInvestigator:
    """Main investigator orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = InvestigatorLLM(config)
        self.tools = InvestigatorTools(config)

        # Initialize fetchers for evidence enrichment
        self.metrics_fetcher: MetricsFetcher = PrometheusFetcher(
            base_url=config.prometheus_url,
            query_template=config.prometheus_query_template or '{incident="%s"}',
            timeout_seconds=config.prometheus_timeout_seconds,
        )
        self.logs_fetcher: LogsFetcher = LokiFetcher(
            base_url=config.loki_url,
            query_template=config.loki_query_template or '{incident="%s"}',
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

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the investigator."""
        return (
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
            "final conclusions.\n"
            "\n"
            "Every finding must be classified as:\n"
            "- FACT: directly observed in metrics/logs\n"
            "- LIKELY_CAUSE: strongly supported by evidence, not certain\n"
            "- HYPOTHESIS: plausible, unverified\n"
            "- UNKNOWN: insufficient evidence; stated explicitly rather than guessed\n"
            "\n"
            "Always provide confidence scores (0.0 to 1.0) for your findings.\n"
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
            "- Each turn should add new evidence or lead directly to submission."
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

    def _handle_error_context(self, context: IncidentContext) -> list[Finding]:
        """Handle error context and return error finding."""
        logger.error(f"Failed to get incident context: {context.error}")
        return [
            Finding(
                type=EvidenceType.UNKNOWN,
                content=f"Failed to load incident context: {context.error}",
                source="investigator",
                confidence=0.0,
            )
        ]

    async def _execute_tool(self, function_name: str, function_args: dict):
        """Execute a tool call and return the result."""
        if function_name == "query_prometheus":
            result = await self.tools._prometheus_adapter.query_range(
                function_args["promql"], function_args["start"], function_args["end"]
            )
            return result, None, False
        elif function_name == "query_loki":
            result = await self.tools._loki_adapter.query_range(
                function_args["logql"],
                function_args["start"],
                function_args["end"],
                function_args.get("limit", 100),
            )
            return result, None, False
        elif function_name == "get_recent_deployments":
            result = await self.tools._deployment_adapter.get_recent_deployments(
                function_args["service"], function_args["start"], function_args["end"]
            )
            return result, None, False
        elif function_name == "get_related_incidents":
            result = await self.tools._related_incidents_adapter.find_related(
                function_args["service"], function_args["symptom_keywords"]
            )
            return result, None, False
        elif function_name == "submit_findings":
            findings = [
                Finding(
                    type=EvidenceType(f["type"]),
                    content=f["content"],
                    source=f["source"],
                    confidence=f["confidence"],
                )
                for f in function_args["findings"]
            ]
            result = self.tools.submit_findings(findings)
            return result, findings, True  # result, findings, is_final
        else:
            return {"error": f"Unknown tool: {function_name}"}, None, False

    def _truncate_result(self, result: any) -> str:
        """Truncate result if too large."""
        result_str = json.dumps(result)
        if len(result_str) > self.config.truncation_limit:
            result_str = self._truncate_context(result_str, self.config.truncation_limit)
            result = {"truncated": result_str}
        return result_str

    def _handle_tool_result(
        self,
        tool_call,
        function_name: str,
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
                arguments=json.loads(tool_call.function.arguments),
                result=result,
                duration_ms=(time.time() - tool_start) * 1000,
            )
        )

        state.tool_call_count += 1

    def _check_max_tools_reached(
        self, state: InvestigationState, is_final_turn: bool, messages: list[dict]
    ):
        """Check if max tool calls reached and prompt for submission if needed."""
        if state.tool_call_count >= state.max_tool_calls and not is_final_turn:
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
        self, response, state: InvestigationState, is_final_turn: bool, messages: list[dict]
    ):
        """Handle text response (no tool calls)."""
        logger.info(f"LLM response (no tool calls): {response.content[:100]}...")
        if not is_final_turn:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue your investigation. Use your tools to gather more "
                        "evidence, or submit your findings when ready."
                    ),
                }
            )
        else:
            logger.warning("Final turn reached without tool use - extracting findings from text")
            state.findings.append(
                Finding(
                    type=EvidenceType.HYPOTHESIS,
                    content=response.content or "Investigation completed via text response",
                    source="llm_direct_response",
                    confidence=0.3,
                )
            )
            state.completed = True

    async def _handle_submit_findings(
        self,
        findings: list[Finding],
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
    ):
        """Handle submit_findings tool call."""
        logger.info("Investigation completed via submit_findings")
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
            )

    async def investigate_async(
        self, incident_id: str, request_key_hash: str = ""
    ) -> list[Finding]:
        """Main investigation loop with bounded tool-calling (async version).

        Every terminal return path now persists the run and its evidence so the
        run reaches a terminal state ('completed') and the reporter can load
        findings. Previously only the submit_findings path persisted, leaving
        runs stuck in 'queued' and reports rendering "investigation unavailable".

        Adds:
        - Overall investigation deadline via investigation_timeout_seconds
        - Incremental persistence after each tool call
        - Final-turn submission fallback (synthesizes partial findings if the
          loop exits without submit_findings)
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
        )

        return await self._run_investigation_loop(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            state=state,
            context=context,
            deadline=deadline,
        )

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
        )
        return findings

    async def _run_investigation_loop(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        context: IncidentContext,
        deadline: float,
    ) -> list[Finding]:
        """Run the bounded investigation tool-calling loop.

        Returns the final findings after persisting them to a terminal run.
        """
        # Build system prompt and initial message
        system_prompt = self._build_system_prompt()
        initial_message = self._build_initial_message(context)

        messages = [{"role": "system", "content": system_prompt}, initial_message]

        # Investigation loop
        while state.tool_call_count < state.max_tool_calls:
            # Enforce overall deadline: if no time remains, stop and submit.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Investigation deadline exceeded for incident %s, submitting partial findings",
                    incident_id,
                )
                break

            logger.info(f"Investigation turn {state.tool_call_count + 1}/{state.max_tool_calls}")

            try:
                result = await self._handle_investigation_turn(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    state=state,
                    messages=messages,
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
                        source="investigator_error",
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
                )
                return state.findings

        # If we exit the loop without submitting findings (deadline or max tools),
        # synthesize a submission from whatever partial findings/tool results we have.
        if not state.findings:
            # Build a best-effort UNKNOWN finding from tool results, or a default.
            if state.tool_calls:
                summaries = []
                for tc in state.tool_calls[-3:]:
                    r = tc.result
                    status = r.get("status", "unknown") if isinstance(r, dict) else "unknown"
                    summaries.append(f"{tc.tool_name}: {status}")
                content = (
                    "Investigation reached maximum tool calls without submitting "
                    "findings. Tool results: " + "; ".join(summaries)
                )
            else:
                content = "Investigation reached maximum tool calls without submitting findings"
            state.findings.append(
                Finding(
                    type=EvidenceType.UNKNOWN,
                    content=content,
                    source="investigator_timeout",
                    confidence=0.0,
                )
            )

        increment_findings_by_type(state.findings)
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
        )

        return state.findings

    async def _handle_investigation_turn(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        messages: list[dict],
    ) -> list[Finding] | None:
        """Run a single investigation turn (one LLM call + tool handling).

        Returns the final findings if the investigation is complete, else None.
        """
        is_final_turn = state.tool_call_count == state.max_tool_calls - 1
        tool_choice = (
            {"type": "function", "function": {"name": "submit_findings"}}
            if is_final_turn
            else "auto"
        )
        tools = ToolRegistry.get_tools()

        # Call the LLM off the asyncio event loop — chat_completion is a
        # synchronous OpenAI call (30s timeout × 3 retries) that would block
        # health probes and kill the background investigation task.
        response = await asyncio.to_thread(
            self.llm.chat_completion,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Add the assistant's response to the conversation
        messages.append({"role": "assistant", "content": response.content or ""})

        # Handle tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Calling tool: {function_name} with args: {function_args}")

                tool_start = time.time()
                result, findings, is_final = await self._execute_tool(function_name, function_args)

                if is_final:
                    await self._handle_submit_findings(
                        findings, incident_id, request_key_hash, state
                    )
                    increment_findings_by_type(state.findings)
                    return findings

                self._handle_tool_result(
                    tool_call, function_name, result, tool_start, state, messages
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

                self._check_max_tools_reached(state, is_final_turn, messages)
            return None

        self._handle_text_response(response, state, is_final_turn, messages)
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
    ):
        """Persist investigation run, tool executions, and evidence in a transaction.

        Includes evidence enrichment with metrics and logs from Prometheus and Loki.
        """
        if not self.persistence:
            logger.warning("No persistence layer available, skipping investigation persistence")
            return

        try:
            # Create or get investigation run
            run_id, created_new = self.persistence.create_investigation_run(
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
                    )
                    enrichment_data["logs"] = enriched_logs
                    logger.info(f"Fetched {len(enriched_logs)} logs for enrichment")
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(f"Failed to fetch Loki logs for enrichment: {e}")

            # Convert Metric and LogEntry objects to dicts for persistence
            metrics_for_persistence = [
                m.to_dict() if isinstance(m, Metric) else m
                for m in enrichment_data.get("metrics", [])
            ]
            log_entries = enrichment_data.get("logs", [])
            logs_for_persistence = [
                entry.to_dict() if isinstance(entry, LogEntry) else entry for entry in log_entries
            ]

            # Insert evidence with enrichment
            self.persistence.insert_evidence(
                incident_id=incident_id,
                run_id=run_id,
                findings=findings,
                metrics=metrics_for_persistence,
                logs=logs_for_persistence,
            )

            # Mark run as completed
            self.persistence.update_run_status(
                run_id=run_id,
                status="completed",
                completed_at=datetime.now(UTC),
            )

            logger.info(
                f"Persisted investigation run {run_id} with {len(findings)} findings, "
                f"{len(metrics_for_persistence)} metrics, {len(logs_for_persistence)} logs"
            )

        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to persist investigation: %s", e)
            # Try to mark run as failed
            if "run_id" in locals():
                try:
                    self.persistence.update_run_status(run_id, "failed")
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
        """Persist investigation run incrementally after each tool call.

        Unlike _persist_investigation, this does NOT mark the run completed —
        it upserts the run (queued/running), appends new tool executions and
        evidence, and leaves the run open so subsequent turns can add more.
        This guarantees a terminal state exists even if the investigation is
        interrupted by a deadline or crash.
        """
        if not self.persistence:
            logger.warning("No persistence layer available, skipping incremental persistence")
            return

        try:
            run_id, created_new = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=model_provider,
                model_name=model_name,
                prompt_version=prompt_version,
                tool_contract=tool_contract,
                max_tool_calls=max_tool_calls,
            )

            # Mark run as running if newly created
            if created_new:
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
