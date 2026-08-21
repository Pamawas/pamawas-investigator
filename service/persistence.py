"""Database persistence for investigation runs, tool executions, and evidence."""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg2
import psycopg2.extras

from models import Finding

logger = logging.getLogger(__name__)


@dataclass
class InvestigationRun:
    """Represents an investigation run record."""
    id: str
    incident_id: str
    request_key_hash: str
    status: str
    model_provider: str
    model_name: str
    prompt_version: str
    tool_contract: int
    max_tool_calls: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    safe_error_code: str | None = None


@dataclass
class ToolExecution:
    """Represents a tool execution record."""
    id: str
    run_id: str
    sequence_no: int
    tool_name: str
    arguments_redacted: dict[str, Any]
    result_summary: dict[str, Any]
    result_hash: str | None
    status: str
    duration_ms: int


class InvestigationPersistence:
    """Handles persistence of investigation runs, tool executions, and evidence."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._conn: psycopg2.extensions.connection | None = None

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            if not self.database_url:
                raise RuntimeError("Database URL not configured")
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = False  # We manage transactions explicitly
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def create_investigation_run(
        self,
        incident_id: str,
        request_key_hash: str,
        model_provider: str,
        model_name: str,
        prompt_version: str,
        tool_contract: int,
        max_tool_calls: int,
    ) -> tuple[str, bool]:
        """
        Create or get existing investigation run.
        Returns (run_id, created_new).
        If run already exists for this incident+request_key_hash, returns existing run_id and False.
        """
        conn = self._get_connection()
        run_id = f"irun_{uuid.uuid4().hex[:26].upper()}"
        now = datetime.now(UTC)

        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO investigation_runs (
                        id, incident_id, request_key_hash, status,
                        model_provider, model_name, prompt_version,
                        tool_contract, max_tool_calls,
                        started_at, created_at, updated_at
                    ) VALUES (%s, %s, %s, 'queued', %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (incident_id, request_key_hash) DO NOTHING
                    RETURNING id
                    """,
                    (
                        run_id, incident_id, request_key_hash,
                        model_provider, model_name, prompt_version,
                        tool_contract, max_tool_calls,
                        now, now, now,
                    ),
                )
                result = cursor.fetchone()
                if result:
                    conn.commit()
                    logger.info(f"Created investigation run {result[0]} for incident {incident_id}")
                    return result[0], True
                else:
                    # Fetch existing run
                    query = (
                        "SELECT id FROM investigation_runs "
                        "WHERE incident_id = %s AND request_key_hash = %s"
                    )
                    cursor.execute(query, (incident_id, request_key_hash))
                    existing = cursor.fetchone()
                    if existing:
                        logger.info(
                            f"Investigation run {existing[0]} already exists for "
                            f"incident {incident_id}"
                        )
                        return existing[0], False
                    # Should not happen
                    conn.rollback()
                    raise RuntimeError("Failed to create or find investigation run")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to create investigation run: {e}")
                raise

    def update_run_status(
        self,
        run_id: str,
        status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        safe_error_code: str | None = None,
    ):
        """Update investigation run status and timestamps."""
        conn = self._get_connection()
        now = datetime.now(UTC)
        with conn.cursor() as cursor:
            try:
                updates = ["status = %s", "updated_at = %s"]
                params = [status, now]

                if status == "running" and started_at is None:
                    updates.append("started_at = %s")
                    params.append(now)
                elif started_at is not None:
                    updates.append("started_at = %s")
                    params.append(started_at)

                if status in ("completed", "unknown", "failed_retryable", "failed_terminal"):
                    if completed_at is None:
                        updates.append("completed_at = %s")
                        params.append(now)
                    elif completed_at is not None:
                        updates.append("completed_at = %s")
                        params.append(completed_at)

                if safe_error_code:
                    updates.append("safe_error_code = %s")
                    params.append(safe_error_code)

                params.append(run_id)
                query = f"UPDATE investigation_runs SET {', '.join(updates)} WHERE id = %s"
                cursor.execute(query, params)
                conn.commit()
                logger.info(f"Updated investigation run {run_id} status to {status}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update run status: {e}")
                raise

    def insert_tool_execution(
        self,
        run_id: str,
        sequence_no: int,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        status: str,
        duration_ms: int,
    ) -> str:
        """Insert a tool execution record."""
        conn = self._get_connection()
        exec_id = f"texec_{uuid.uuid4().hex[:26].upper()}"
        now = datetime.now(UTC)

        # Redact sensitive arguments
        arguments_redacted = self._redact_arguments(arguments)
        # Create result summary
        result_summary = self._summarize_result(result)
        result_hash = self._hash_result(result)

        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO tool_executions (
                        id, run_id, sequence_no, tool_name,
                        arguments_redacted, result_summary, result_hash,
                        status, duration_ms, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        exec_id, run_id, sequence_no, tool_name,
                        json.dumps(arguments_redacted),
                        json.dumps(result_summary),
                        result_hash,
                        status, duration_ms, now,
                    ),
                )
                conn.commit()
                logger.info(f"Inserted tool execution {exec_id} for run {run_id}")
                return exec_id
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to insert tool execution: {e}")
                raise

    def insert_evidence(
        self,
        incident_id: str,
        run_id: str,
        findings: list[Finding],
    ) -> list[str]:
        """Insert evidence records for all findings in a transaction."""
        conn = self._get_connection()
        evidence_ids = []
        now = datetime.now(UTC)

        with conn.cursor() as cursor:
            try:
                for ordinal, finding in enumerate(findings):
                    evidence_id = f"evd_{uuid.uuid4().hex[:26].upper()}"
                    cursor.execute(
                        """
                        INSERT INTO evidence (
                            id, incident_id, run_id, type, content,
                            source, confidence, supports_evidence,
                            contradicts_evidence, ordinal, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            evidence_id, incident_id, run_id,
                            finding.type.value, finding.content,
                            finding.source, finding.confidence,
                            [], [],  # supports/contradicts empty for MVP
                            ordinal, now,
                        ),
                    )
                    evidence_ids.append(evidence_id)

                conn.commit()
                logger.info(f"Inserted {len(evidence_ids)} evidence records for run {run_id}")
                return evidence_ids
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to insert evidence: {e}")
                raise

    def _redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive information from tool arguments."""
        redacted = {}
        sensitive_keys = {"api_key", "password", "secret", "token", "authorization"}
        for k, v in arguments.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = v
        return redacted

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Create a bounded summary of tool result."""
        summary = {}
        for k, v in result.items():
            if k in ("status", "error", "data", "result"):
                summary[k] = v
            elif isinstance(v, (str, int, float, bool)):
                summary[k] = v
            elif isinstance(v, (list, dict)):
                # Truncate large structures
                summary[k] = f"<{type(v).__name__} len={len(v)}>"
        return summary

    def _hash_result(self, result: dict[str, Any]) -> str:
        """Create a hash of the result for deduplication."""
        import hashlib
        result_str = json.dumps(result, sort_keys=True, default=str)
        return hashlib.sha256(result_str.encode()).hexdigest()[:32]
