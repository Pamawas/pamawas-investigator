"""Main Investigator Service."""

import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import psycopg2
from openai import OpenAI

from ..config import Config
from ..models import (
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    ToolResult,
)
from ..tools import InvestigatorTools, ToolRegistry
from ..metrics import (
    increment_investigations,
    increment_findings,
    observe_loop_duration,
    set_running,
    set_uptime,
    increment_db_errors,
)

logger = logging.getLogger(__name__)


class InvestigatorLLM:
    """OpenAI-compatible LLM client for the investigator."""

    def __init__(self, config: Config):
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        )
        self.model = config.llm_model

    def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto"):
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
        self._connect_db()

    def _connect_db(self):
        """Establish database connection."""
        if not self.config.database_url:
            logger.warning("No DATABASE_URL provided, database features disabled")
            return

        try:
            self.db_conn = psycopg2.connect(self.config.database_url)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            increment_db_errors()
            self.db_conn = None

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
        except Exception as e:
            logger.error(f"Failed to get incident context: {e}")
            return IncidentContext(
                incident_id=incident_id,
                title=f"Error loading incident",
                events=[],
                started_at="",
                error=str(e)
            )

    def _truncate_context(self, text: str, limit: int) -> str:
        """Truncate context to prevent excessive growth."""
        if len(text) <= limit:
            return text
        # Keep beginning and end
        half = limit // 2
        return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]

    def investigate(self, incident_id: str) -> List[Finding]:
        """Main investigation loop with bounded tool-calling."""
        logger.info(f"Starting investigation for incident {incident_id}")
        increment_investigations()

        # Get incident context
        context = self._get_incident_context(incident_id)
        if hasattr(context, 'error') and context.error:
            logger.error(f"Failed to get incident context: {context.error}")
            return [Finding(
                type=EvidenceType.UNKNOWN,
                content=f"Failed to load incident context: {context.error}",
                source="investigator",
                confidence=0.0
            )]

        # Initialize investigation state
        state = InvestigationState(
            incident_id=incident_id,
            findings=[],
            tool_calls=[],
            max_tool_calls=self.config.max_tool_calls
        )

        # System prompt that guides the investigator
        system_prompt = """You are an expert infrastructure incident investigator. Your goal is to:
1. Understand the symptom and blast radius
2. Find the first abnormal signal, not just the loudest alert
3. Check recent changes near the time of the incident
4. Check dependencies
5. Form and test competing hypotheses rather than confirming one
6. Use tool calls deliberately - no duplicate queries, no unused fetches
7. Prefer UNKNOWN over a fabricated-sounding conclusion when evidence is insufficient

You have access to tools to query Prometheus, Loki, check recent deployments, and find related incidents.
You must use the submit_findings tool to end the investigation with your final conclusions.

Every finding must be classified as:
- FACT: directly observed in metrics/logs
- LIKELY_CAUSE: strongly supported by evidence, not certain
- HYPOTHESIS: plausible, unverified
- UNKNOWN: insufficient evidence; stated explicitly rather than guessed

Always provide confidence scores (0.0 to 1.0) for your findings."""

        # Initial message with incident context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Investigate this incident:

Incident ID: {context.incident_id}
Title: {context.title}
Status: {context.status}
Started at: {context.started_at}
Severity: {context.severity}
Affected services: {', '.join(context.affected_services)}

Events in this incident:
{json.dumps(context.events, indent=2)[:self.config.truncation_limit]}

Begin your investigation by understanding what happened. Use your tools to gather evidence and form hypotheses."""}
        ]

        # Investigation loop
        while state.tool_call_count < state.max_tool_calls:
            logger.info(f"Investigation turn {state.tool_call_count + 1}/{state.max_tool_calls}")

            # Determine if this is the final turn (force submit_findings)
            is_final_turn = (state.tool_call_count == state.max_tool_calls - 1)
            tool_choice = "submit_findings" if is_final_turn else "auto"

            # Get LLM response with available tools
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

                        logger.info(f"Calling tool: {function_name} with args: {function_args}")

                        # Execute the tool
                        tool_start = time.time()
                        if function_name == "query_prometheus":
                            result = self.tools.query_prometheus(
                                function_args["promql"],
                                function_args["start"],
                                function_args["end"]
                            )
                        elif function_name == "query_loki":
                            result = self.tools.query_loki(
                                function_args["logql"],
                                function_args["start"],
                                function_args["end"],
                                function_args.get("limit", 100)
                            )
                        elif function_name == "get_recent_deployments":
                            result = self.tools.get_recent_deployments(
                                function_args["service"],
                                function_args["start"],
                                function_args["end"]
                            )
                        elif function_name == "get_related_incidents":
                            result = self.tools.get_related_incidents(
                                function_args["service"],
                                function_args["symptom_keywords"]
                            )
                        elif function_name == "submit_findings":
                            # Convert findings dicts back to Finding objects
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
                            # This is the final call - we're done
                            logger.info("Investigation completed via submit_findings")
                            state.findings = findings
                            state.completed = True
                            return findings
                        else:
                            result = {"error": f"Unknown tool: {function_name}"}

                        # Truncate result if too large
                        result_str = json.dumps(result)
                        if len(result_str) > self.config.truncation_limit:
                            result_str = self._truncate_context(result_str, self.config.truncation_limit)
                            result = json.loads(result_str)

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })

                        state.tool_calls.append(ToolResult(
                            tool_name=function_name,
                            arguments=function_args,
                            result=result,
                            duration_ms=(time.time() - tool_start) * 1000
                        ))

                        state.tool_call_count += 1

                        # If we've hit the limit and haven't submitted findings yet,
                        # force a submission on the next turn
                        if state.tool_call_count >= state.max_tool_calls and not is_final_turn:
                            logger.info("Reached max tool calls, forcing findings submission")
                            # Add a message prompting submission
                            messages.append({
                                "role": "user",
                                "content": "You have reached the maximum number of tool calls. Please submit your findings using the submit_findings tool."
                            })

                else:
                    # No tool calls, just text response
                    logger.info(f"LLM response (no tool calls): {response.content[:100]}...")
                    # Continue the conversation - the LLM might want to think more
                    if not is_final_turn:
                        messages.append({
                            "role": "user",
                            "content": "Continue your investigation. Use your tools to gather more evidence, or submit your findings when ready."
                        })
                    else:
                        # Final turn with no tool calls - we need to extract findings from text
                        # This is a fallback - in practice, the LLM should use submit_findings
                        logger.warning("Final turn reached without tool use - extracting findings from text")
                        # Create a simple finding from the response
                        state.findings.append(Finding(
                            type=EvidenceType.HYPOTHESIS,
                            content=response.content or "Investigation completed via text response",
                            source="llm_direct_response",
                            confidence=0.3
                        ))
                        state.completed = True
                        return state.findings

            except Exception as e:
                logger.error(f"Error in investigation loop: {e}")
                state.findings.append(Finding(
                    type=EvidenceType.UNKNOWN,
                    content=f"Investigation error: {str(e)}",
                    source="investigator_error",
                    confidence=0.0
                ))
                return state.findings

        # If we exit the loop without submitting findings, create a default finding
        if not state.findings:
            state.findings.append(Finding(
                type=EvidenceType.UNKNOWN,
                content="Investigation reached maximum tool calls without submitting findings",
                source="investigator_timeout",
                confidence=0.0
            ))

        return state.findings