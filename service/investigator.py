"""Main Investigator Service."""

import json
import logging
import time
from datetime import UTC, datetime

import psycopg2
from openai import OpenAI

from config import Config
from metrics import increment_db_errors, increment_investigations
from models import (
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    ToolResult,
)
from service.persistence import InvestigationPersistence
from tools import InvestigatorTools, ToolRegistry

logger = logging.getLogger(__name__)


class InvestigatorLLM:
    """OpenAI-compatible LLM client for the investigator."""

    def __init__(self, config: Config):
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        )
        self.model = config.llm_model

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ):
        """Call the LLM with optional tool use."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=2000
            )
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


class PamawasInvestigator:
    """Main investigator orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = InvestigatorLLM(config)
        self.tools = InvestigatorTools(config)
        self.db_conn = None
        self.persistence = None
        self._connect_db()

    def _connect_db(self):
        """Establish database connection."""
        if not self.config.database_url:
            logger.warning("No DATABASE_URL provided, database features disabled")
            return

        try:
            self.db_conn = psycopg2.connect(self.config.database_url)
            self.persistence = InvestigationPersistence(self.config.database_url)
            logger.info("Connected to database")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to connect to database: {e}")
            increment_db_errors()
            self.db_conn = None

    async def close(self):
        """Close investigator and all tools."""
        await self.tools.close()
        if self.db_conn and not self.db_conn.closed:
            self.db_conn.close()

    def _get_incident_context(self, incident_id: str) -> IncidentContext:
        """Retrieve incident context from the database."""
        if not self.db_conn:
            logger.warning("No database connection, returning mock context")
            return IncidentContext(
                incident_id=incident_id,
                title=f"Incident {incident_id}",
                events=[],
                started_at="2026-08-13T02:00:00Z"
            )

        try:
            with self.db_conn.cursor() as cursor:
                # Get incident info
                cursor.execute("""
                    SELECT id, title, status, started_at, resolved_at, severity, affected_services
                    FROM incidents WHERE id = %s
                """, (incident_id,))
                incident_row = cursor.fetchone()

                if not incident_row:
                    return IncidentContext(
                        incident_id=incident_id,
                        title=f"Incident {incident_id} not found",
                        events=[],
                        started_at="",
                        error=f"Incident {incident_id} not found"
                    )

                # Get events for this incident
                cursor.execute("""
                    SELECT e.id, e.source, e.type, e.timestamp, e.service, e.environment,
                           e.severity, e.title, e.status, e.labels, e.raw_payload
                    FROM events e
                    JOIN incident_events ie ON e.id = ie.event_id
                    WHERE ie.incident_id = %s
                    ORDER BY e.timestamp
                """, (incident_id,))
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
                            "raw_payload": e[10]
                        }
                        for e in events
                    ]
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get incident context: {e}")
            return IncidentContext(
                incident_id=incident_id,
                title="Error loading incident",
                events=[],
                started_at="",
                error=str(e)
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
            "Always provide confidence scores (0.0 to 1.0) for your findings."
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
                f"{json.dumps(context.events, indent=2)[:self.config.truncation_limit]}\n\n"
                "Begin your investigation by understanding what happened. Use your "
                "tools to gather evidence and form hypotheses."
            )
        }

    def _handle_error_context(self, context: IncidentContext) -> list[Finding]:
        """Handle error context and return error finding."""
        logger.error(f"Failed to get incident context: {context.error}")
        return [Finding(
            type=EvidenceType.UNKNOWN,
            content=f"Failed to load incident context: {context.error}",
            source="investigator",
            confidence=0.0
        )]

    async def _execute_tool(self, function_name: str, function_args: dict):
        """Execute a tool call and return the result."""
        if function_name == "query_prometheus":
            result = await self.tools._prometheus_adapter.query_range(
                function_args["promql"],
                function_args["start"],
                function_args["end"]
            )
            return result, None, False
        elif function_name == "query_loki":
            result = await self.tools._loki_adapter.query_range(
                function_args["logql"],
                function_args["start"],
                function_args["end"],
                function_args.get("limit", 100)
            )
            return result, None, False
        elif function_name == "get_recent_deployments":
            result = await self.tools._deployment_adapter.get_recent_deployments(
                function_args["service"],
                function_args["start"],
                function_args["end"]
            )
            return result, None, False
        elif function_name == "get_related_incidents":
            result = await self.tools._related_incidents_adapter.find_related(
                function_args["service"],
                function_args["symptom_keywords"]
            )
            return result, None, False
        elif function_name == "submit_findings":
            findings = [
                Finding(
                    type=EvidenceType(f["type"]),
                    content=f["content"],
                    source=f["source"],
                    confidence=f["confidence"]
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
        messages: list[dict]
    ):
        """Process tool result and update state."""
        result_str = self._truncate_result(result)

        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_str
        })

        state.tool_calls.append(ToolResult(
            tool_name=function_name,
            arguments=json.loads(tool_call.function.arguments),
            result=result,
            duration_ms=(time.time() - tool_start) * 1000
        ))

        state.tool_call_count += 1

    def _check_max_tools_reached(
        self, state: InvestigationState, is_final_turn: bool, messages: list[dict]
    ):
        """Check if max tool calls reached and prompt for submission if needed."""
        if state.tool_call_count >= state.max_tool_calls and not is_final_turn:
            logger.info("Reached max tool calls, forcing findings submission")
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Please submit your findings using the submit_findings tool."
                )
            })

    def _handle_text_response(
        self, response, state: InvestigationState, is_final_turn: bool, messages: list[dict]
    ):
        """Handle text response (no tool calls)."""
        logger.info(f"LLM response (no tool calls): {response.content[:100]}...")
        if not is_final_turn:
            messages.append({
                "role": "user",
                "content": (
                    "Continue your investigation. Use your tools to gather more "
                    "evidence, or submit your findings when ready."
                )
            })
        else:
            logger.warning(
                "Final turn reached without tool use - extracting findings "
                "from text"
            )
            state.findings.append(Finding(
                type=EvidenceType.HYPOTHESIS,
                content=response.content or "Investigation completed via text response",
                source="llm_direct_response",
                confidence=0.3
            ))
            state.completed = True

    async def _handle_submit_findings(
        self,
        findings: list[Finding],
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState
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
        """Main investigation loop with bounded tool-calling (async version)."""
        logger.info(f"Starting investigation for incident {incident_id}")
        increment_investigations()

        # Get incident context
        context = self._get_incident_context(incident_id)
        if hasattr(context, 'error') and context.error:
            return self._handle_error_context(context)

        # Initialize investigation state
        state = InvestigationState(
            incident_id=incident_id,
            findings=[],
            tool_calls=[],
            max_tool_calls=self.config.max_tool_calls,
            request_key_hash=request_key_hash,
        )

        # Build system prompt and initial message
        system_prompt = self._build_system_prompt()
        initial_message = self._build_initial_message(context)

        messages = [
            {"role": "system", "content": system_prompt},
            initial_message
        ]

        # Investigation loop
        while state.tool_call_count < state.max_tool_calls:
            logger.info(
                f"Investigation turn {state.tool_call_count + 1}/{state.max_tool_calls}"
            )

            is_final_turn = (state.tool_call_count == state.max_tool_calls - 1)
            tool_choice = "submit_findings" if is_final_turn else "auto"
            tools = ToolRegistry.get_tools()

            try:
                # Call the LLM
                response = self.llm.chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice
                )

                # Add the assistant's response to the conversation
                messages.append({"role": "assistant", "content": response.content or ""})

                # Handle tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        logger.info(
                            f"Calling tool: {function_name} with args: {function_args}"
                        )

                        tool_start = time.time()
                        result, findings, is_final = await self._execute_tool(
                            function_name, function_args
                        )

                        if is_final:
                            await self._handle_submit_findings(
                                findings, incident_id, request_key_hash, state
                            )
                            return findings

                        self._handle_tool_result(
                            tool_call, function_name, result, tool_start, state, messages
                        )

                        self._check_max_tools_reached(state, is_final_turn, messages)

                else:
                    self._handle_text_response(response, state, is_final_turn, messages)
                    if state.completed:
                        return state.findings

            except Exception as e:  # noqa: BLE001
                logger.error(f"Error in investigation loop: {e}")
                state.findings.append(Finding(
                    type=EvidenceType.UNKNOWN,
                    content=f"Investigation error: {e!s}",
                    source="investigator_error",
                    confidence=0.0
                ))
                return state.findings

        # If we exit the loop without submitting findings, create a default finding
        if not state.findings:
            state.findings.append(Finding(
                type=EvidenceType.UNKNOWN,
                content=(
                    "Investigation reached maximum tool calls without submitting findings"
                ),
                source="investigator_timeout",
                confidence=0.0
            ))

        return state.findings

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
        """Persist investigation run, tool executions, and evidence in a transaction."""
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

            # Insert evidence
            self.persistence.insert_evidence(
                incident_id=incident_id,
                run_id=run_id,
                findings=findings,
            )

            # Mark run as completed
            self.persistence.update_run_status(
                run_id=run_id,
                status="completed",
                completed_at=datetime.now(UTC),
            )

            logger.info(f"Persisted investigation run {run_id} with {len(findings)} findings")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to persist investigation: {e}")
            # Try to mark run as failed
            if 'run_id' in locals():
                try:
                    self.persistence.update_run_status(run_id, "failed")
                except Exception:  # noqa: BLE001
                    pass

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
                future = pool.submit(asyncio.run, self.investigate_async(incident_id, request_key_hash))
                return future.result()
        else:
            return asyncio.run(self.investigate_async(incident_id, request_key_hash))
