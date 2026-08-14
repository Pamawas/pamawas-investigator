#!/usr/bin/env python3
"""
Pamawas Investigator - Bounded tool-calling LLM investigation engine
Implements the investigation loop as described in the MVP design.
"""

import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import psycopg2
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evidence types as per MVP design
class EvidenceType(Enum):
    FACT = "fact"
    LIKELY_CAUSE = "likely_cause"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"

@dataclass
class Finding:
    """Represents a piece of evidence or finding from the investigation."""
    type: EvidenceType
    content: str
    source: str
    confidence: float  # 0.0 to 1.0

@dataclass
class InvestigationConfig:
    """Configuration for the investigator."""
    # Database
    database_url: str
    
    # LLM Configuration (OpenAI-compatible)
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    
    # Tool endpoints (these would be internal services in a real deployment)
    prometheus_url: str
    loki_url: str
    deployments_url: str
    related_incidents_url: str
    
    # Investigation limits
    max_tool_calls: int = 6
    context_truncation_limit: int = 8192  # 8KB
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
            loki_url=os.getenv("LOKI_URL", "http://loki:3100"),
            deployments_url=os.getenv("DEPLOYMENTS_URL", "http://deployments:8080"),
            related_incidents_url=os.getenv("RELATED_INCIDENTS_URL", "http://related-incidents:8080"),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
        )

class InvestigatorLLM:
    """OpenAI-compatible LLM client for the investigator."""
    
    def __init__(self, config: InvestigationConfig):
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

class InvestigatorTools:
    """Tool implementations that the LLM can call."""
    
    def __init__(self, config: InvestigationConfig):
        self.config = config
    
    def query_prometheus(self, promql: str, start: str, end: str) -> Dict[str, Any]:
        """Query Prometheus for metrics data."""
        logger.info(f"Querying Prometheus: {promql} [{start} to {end}]")
        # In a real implementation, this would make an HTTP request to Prometheus
        # For now, return mock data
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "http_requests_total", "job": "api-server"},
                        "values": [[start, "100"], [end, "150"]]
                    }
                ]
            }
        }
    
    def query_loki(self, logql: str, start: str, end: str, limit: int = 100) -> Dict[str, Any]:
        """Query Loki for logs data."""
        logger.info(f"Querying Loki: {logql} [{start} to {end}] limit={limit}")
        # In a real implementation, this would make an HTTP request to Loki
        return {
            "status": "success",
            "data": {
                "result": [
                    {
                        "stream": {"job": "api-server", "level": "error"},
                        "values": [[start, "Error: connection refused"], [end, "Error: timeout"]]
                    }
                ]
            }
        }
    
    def get_recent_deployments(self, service: str, start: str, end: str) -> Dict[str, Any]:
        """Get recent deployments for a service."""
        logger.info(f"Getting recent deployments for {service} [{start} to {end}]")
        # Mock data
        return {
            "status": "success",
            "data": [
                {
                    "service": service,
                    "version": "v1.2.3",
                    "deployed_at": start,
                    "image": f"{service}:v1.2.3"
                }
            ]
        }
    
    def get_related_incidents(self, service: str, symptom_keywords: List[str]) -> Dict[str, Any]:
        """Find related incidents from the database."""
        logger.info(f"Finding related incidents for {service} with keywords {symptom_keywords}")
        # Mock data
        return {
            "status": "success",
            "data": [
                {
                    "incident_id": "inc_123",
                    "title": f"High latency in {service}",
                    "started_at": "2026-08-13T02:00:00Z",
                    "resolved_at": "2026-08-13T03:30:00Z",
                    "symptom_keywords": symptom_keywords
                }
            ]
        }
    
    def submit_findings(self, findings: List[Finding]) -> Dict[str, Any]:
        """Submit the final findings - this is the tool that forces structured output."""
        logger.info(f"Submitting {len(findings)} findings")
        # In a real implementation, this would persist findings to the database
        # For now, just return success
        return {
            "status": "success",
            "message": f"Submitted {len(findings)} findings",
            "findings": [asdict(f) for f in findings]
        }

class PamawasInvestigator:
    """Main investigator orchestrator."""
    
    def __init__(self, config: InvestigationConfig):
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
            self.db_conn = None
    
    def _get_incident_context(self, incident_id: str) -> Dict[str, Any]:
        """Retrieve incident context from the database."""
        if not self.db_conn:
            logger.warning("No database connection, returning mock context")
            return {
                "incident_id": incident_id,
                "title": f"Incident {incident_id}",
                "events": [],
                "started_at": "2026-08-13T02:00:00Z"
            }
        
        try:
            with self.db_conn.cursor() as cursor:
                # Get incident info
                cursor.execute("""
                    SELECT id, title, status, started_at, resolved_at, severity, affected_services
                    FROM incidents WHERE id = %s
                """, (incident_id,))
                incident_row = cursor.fetchone()
                
                if not incident_row:
                    return {"error": f"Incident {incident_id} not found"}
                
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
                
                return {
                    "incident_id": incident_row[0],
                    "title": incident_row[1],
                    "status": incident_row[2],
                    "started_at": str(incident_row[3]),
                    "resolved_at": str(incident_row[4]) if incident_row[4] else None,
                    "severity": incident_row[5],
                    "affected_services": incident_row[6],
                    "events": [
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
                }
        except Exception as e:
            logger.error(f"Failed to get incident context: {e}")
            return {"error": str(e)}
    
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
        
        # Get incident context
        context = self._get_incident_context(incident_id)
        if "error" in context:
            logger.error(f"Failed to get incident context: {context['error']}")
            return [Finding(
                type=EvidenceType.UNKNOWN,
                content=f"Failed to load incident context: {context['error']}",
                source="investigator",
                confidence=0.0
            )]
        
        # Initialize investigation state
        findings: List[Finding] = []
        tool_call_count = 0
        
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

Incident ID: {context['incident_id']}
Title: {context['title']}
Status: {context['status']}
Started at: {context['started_at']}
Severity: {context['severity']}
Affected services: {', '.join(context['affected_services'])}

Events in this incident:
{json.dumps(context['events'], indent=2)[:self.config.context_truncation_limit]}

Begin your investigation by understanding what happened. Use your tools to gather evidence and form hypotheses."""}
        ]
        
        # Investigation loop
        while tool_call_count < self.config.max_tool_calls:
            logger.info(f"Investigation turn {tool_call_count + 1}/{self.config.max_tool_calls}")
            
            # Determine if this is the final turn (force submit_findings)
            is_final_turn = (tool_call_count == self.config.max_tool_calls - 1)
            tool_choice = "submit_findings" if is_final_turn else "auto"
            
            # Get LLM response with available tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "query_prometheus",
                        "description": "Query Prometheus for metrics data",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "promql": {"type": "string", "description": "PromQL query expression"},
                                "start": {"type": "string", "description": "Start timestamp (ISO 8601)"},
                                "end": {"type": "string", "description": "End timestamp (ISO 8601)"}
                            },
                            "required": ["promql", "start", "end"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "query_loki",
                        "description": "Query Loki for logs data",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "logql": {"type": "string", "description": "LogQL query expression"},
                                "start": {"type": "string", "description": "Start timestamp (ISO 8601)"},
                                "end": {"type": "string", "description": "End timestamp (ISO 8601)"},
                                "limit": {"type": "integer", "description": "Maximum number of log entries to return"}
                            },
                            "required": ["logql", "start", "end"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_recent_deployments",
                        "description": "Get recent deployments for a service",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "service": {"type": "string", "description": "Service name"},
                                "start": {"type": "string", "description": "Start timestamp (ISO 8601)"},
                                "end": {"type": "string", "description": "End timestamp (ISO 8601)"}
                            },
                            "required": ["service", "start", "end"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_related_incidents",
                        "description": "Find related incidents from the database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "service": {"type": "string", "description": "Service name"},
                                "symptom_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Keywords to match against incident symptoms"
                                }
                            },
                            "required": ["service", "symptom_keywords"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "submit_findings",
                        "description": "Submit final findings and end the investigation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "findings": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["fact", "likely_cause", "hypothesis", "unknown"]
                                            },
                                            "content": {"type": "string"},
                                            "source": {"type": "string"},
                                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                        },
                                        "required": ["type", "content", "source", "confidence"]
                                    }
                                }
                            },
                            "required": ["findings"]
                        }
                    }
                }
            ]
            
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
                            return findings
                        else:
                            result = {"error": f"Unknown tool: {function_name}"}
                        
                        # Truncate result if too large
                        result_str = json.dumps(result)
                        if len(result_str) > self.config.context_truncation_limit:
                            result_str = self._truncate_context(result_str, self.config.context_truncation_limit)
                            result = json.loads(result_str)
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })
                        
                        tool_call_count += 1
                        
                        # If we've hit the limit and haven't submitted findings yet,
                        # force a submission on the next turn
                        if tool_call_count >= self.config.max_tool_calls and not is_final_turn:
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
                        findings.append(Finding(
                            type=EvidenceType.HYPOTHESIS,
                            content=response.content or "Investigation completed via text response",
                            source="llm_direct_response",
                            confidence=0.3
                        ))
                        return findings
                        
            except Exception as e:
                logger.error(f"Error in investigation loop: {e}")
                findings.append(Finding(
                    type=EvidenceType.UNKNOWN,
                    content=f"Investigation error: {str(e)}",
                    source="investigator_error",
                    confidence=0.0
                ))
                return findings
        
        # If we exit the loop without submitting findings, create a default finding
        if not findings:
            findings.append(Finding(
                type=EvidenceType.UNKNOWN,
                content="Investigation reached maximum tool calls without submitting findings",
                source="investigator_timeout",
                confidence=0.0
            ))
        
        return findings

def main():
    """Main entry point for the investigator service."""
    logger.info("Starting Pamawas Investigator")
    
    # Load configuration
    try:
        config = InvestigationConfig.from_env()
        logger.info(f"Loaded config: LLM={config.llm_model} via {config.llm_base_url}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create investigator
    investigator = PamawasInvestigator(config)
    
    # In a real deployment, this would listen for incidents from a queue or API
    # For now, we'll just show that the service is ready
    logger.info("Investigator service is ready to process incidents")
    
    # Example usage (would be replaced by actual incident processing)
    # incident_id = os.getenv("INCIDENT_ID", "test_incident_001")
    # findings = investigator.investigate(incident_id)
    # logger.info(f"Investigation complete: {len(findings)} findings")
    
    return 0

if __name__ == "__main__":
    exit(main())