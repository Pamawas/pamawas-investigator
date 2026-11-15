"""Database persistence for investigation runs, tool executions, and evidence."""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool

from models import Finding, LogEntry, Metric

# Python 3.9 compatibility for datetime.UTC (added in 3.11)
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

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
        self._pool: psycopg2.pool.ThreadedConnectionPool | None = None

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a database connection from the pool."""
        if not self.database_url:
            raise RuntimeError("Database URL not configured")

        if self._pool is None:
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.database_url,
                    connect_timeout=5,
                )
            except psycopg2.Error as e:
                logger.error("Failed to initialize database pool: %s", e)
                raise

        return self._pool.getconn()

    def _put_connection(self, conn: psycopg2.extensions.connection):
        """Return a database connection to the pool."""
        if self._pool and conn:
            self._pool.putconn(conn)

    def close(self):
        """Close the database pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def commit(self, conn: psycopg2.extensions.connection):
        """Commit a specific connection's transaction."""
        if conn:
            conn.commit()

    def rollback(self, conn: psycopg2.extensions.connection):
        """Roll back a specific connection's transaction."""
        if conn:
            conn.rollback()

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
        """Create or get existing investigation run; returns (run_id, created_new)."""
        conn = self._get_connection()
        run_id = f"irun_{uuid.uuid4().hex[:26].upper()}"
        now = datetime.now(UTC)

        try:
            with conn.cursor() as cursor:
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
                        run_id,
                        incident_id,
                        request_key_hash,
                        model_provider,
                        model_name,
                        prompt_version,
                        tool_contract,
                        max_tool_calls,
                        now,
                        now,
                        now,
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
                        conn.commit()
                        logger.info(
                            f"Investigation run {existing[0]} already exists for "
                            f"incident {incident_id}"
                        )
                        return existing[0], False
                    # Should not happen - the row should exist for this key
                    raise RuntimeError("Failed to create or find investigation run")
        except (psycopg2.Error, RuntimeError) as e:
            conn.rollback()
            logger.error(f"Failed to create investigation run: {e}")
            raise
        finally:
            self._put_connection(conn)

    def mark_run_failed(
        self,
        run_id: str,
        error: str | None = None,
    ) -> bool:
        """Mark a run as failed (idempotent); returns True if status updated."""
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE investigation_runs
                    SET status = 'failed_terminal', updated_at = %s,
                        completed_at = %s, safe_error_code = %s
                    WHERE id = %s
                      AND status NOT IN ('completed', 'failed_terminal')
                    """,
                    (now, now, error, run_id),
                )
                updated = cursor.rowcount > 0
            conn.commit()
            logger.info(
                "Marked investigation run %s as failed (updated=%s)",
                run_id,
                updated,
            )
            return updated
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to mark run %s as failed: %s", run_id, e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return False
        finally:
            if conn is not None:
                self._put_connection(conn)

    def get_run_retry_state(self, run_id: str) -> dict | None:
        """Fetch a run's status, attempts, and max_attempts, or None if absent."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT status, attempts, max_attempts "
                    "FROM investigation_runs WHERE id = %s",
                    (run_id,),
                )
                row = cursor.fetchone()
            conn.commit()
            if row is None:
                return None
            status, attempts, max_attempts = row
            return {"status": status, "attempts": attempts, "max_attempts": max_attempts}
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to get retry state for run %s: %s", run_id, e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None
        finally:
            if conn is not None:
                self._put_connection(conn)

    def schedule_investigation_retry(
        self,
        run_id: str,
        error: str | None,
        next_retry_at: datetime | None,
    ) -> bool:
        """Record a failed attempt and, if the retry budget remains, schedule a retry.

        This is the investigator-side bookkeeping of the auto-retry loop. On every
        failed execution the run's `attempts` is incremented. If the incremented
        count is still below `max_attempts`, the run is moved to
        `failed_retryable` with a `next_retry_at` back-off timestamp, and the
        method returns True (a retry is scheduled). Otherwise the run goes
        `failed_terminal` and the method returns False (budget exhausted).

        The UPDATE is guarded by `status = 'failed_terminal'` NOT being reached:
        it only applies to a run currently in a running/queued/failed_retryable
        state, so a racing completion is a no-op.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE investigation_runs
                    SET attempts = attempts + 1,
                        status = CASE
                            WHEN attempts + 1 < max_attempts THEN 'failed_retryable'
                            ELSE 'failed_terminal'
                        END,
                        next_retry_at = CASE
                            WHEN attempts + 1 < max_attempts THEN %s
                            ELSE NULL
                        END,
                        completed_at = %s,
                        safe_error_code = %s,
                        updated_at = %s
                    WHERE id = %s
                      AND status NOT IN ('completed', 'failed_terminal')
                    """,
                    (next_retry_at, now, error, now, run_id),
                )
                updated = cursor.rowcount > 0
            conn.commit()
            if updated:
                state = self.get_run_retry_state(run_id)
                scheduled = bool(state and state["status"] == "failed_retryable")
                logger.info(
                    "Investigation run %s retry scheduled=%s attempts=%s",
                    run_id,
                    scheduled,
                    (state or {}).get("attempts"),
                )
                return scheduled
            return False
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to schedule retry for run %s: %s", run_id, e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return False
        finally:
            if conn is not None:
                self._put_connection(conn)

    def reclaim_due_investigation_retry(self, run_id: str) -> bool:
        """Atomically move a due `failed_retryable` run back to `running`.

        Guards the auto-retry against double-execution: only one worker can flip
        a run from `failed_retryable` (with `next_retry_at <= now`) to `running`,
        so a racing second reclaim is a no-op.

        `completed_at` and `safe_error_code` are cleared because they describe
        the attempt that just failed. Leaving them set made `started_at = now()`
        land after the previous attempt's `completed_at`, which violates
        `investigation_runs_check1` (completed_at >= started_at) and wedged the
        run in `failed_retryable` forever.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE investigation_runs
                    SET status = 'running',
                        next_retry_at = NULL,
                        completed_at = NULL,
                        safe_error_code = NULL,
                        started_at = %s,
                        updated_at = %s
                    WHERE id = %s
                      AND status = 'failed_retryable'
                      AND (next_retry_at IS NULL OR next_retry_at <= now())
                    """,
                    (now, now, run_id),
                )
                reclaimed = cursor.rowcount > 0
            conn.commit()
            return reclaimed
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to reclaim retry for run %s: %s", run_id, e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return False
        finally:
            if conn is not None:
                self._put_connection(conn)

    def list_due_investigation_retries(self, limit: int = 10) -> list[str]:
        """Return run IDs whose retry back-off has elapsed and are due a re-run.

        Used by the retry scheduler to reclaim `failed_retryable` runs. Only runs
        in the `failed_retryable` state whose `next_retry_at` has passed are
        returned; each is then reclaimed atomically by
        `reclaim_due_investigation_retry` before re-execution.
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM investigation_runs
                    WHERE status = 'failed_retryable'
                      AND (next_retry_at IS NULL OR next_retry_at <= now())
                    ORDER BY next_retry_at NULLS FIRST
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
            conn.commit()
            return [row[0] for row in rows]
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to list due investigation retries: %s", e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return []
        finally:
            if conn is not None:
                self._put_connection(conn)

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
        try:
            with conn.cursor() as cursor:
                updates = ["status = %s", "updated_at = %s"]
                params = [status, now]

                if status == "running" and started_at is None:
                    # COALESCE, not assignment: _persist_investigation re-marks
                    # the run "running" at the END (before completing it), and
                    # a plain assignment reset started_at to that moment. The
                    # run then looked like it took 4 seconds when it had really
                    # spent the full 10-minute deadline, which made every
                    # "why was this slow" investigation start from a lie.
                    updates.append("started_at = COALESCE(started_at, %s)")
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
        except (psycopg2.Error, RuntimeError) as e:
            conn.rollback()
            logger.error(f"Failed to update run status: {e}")
            raise
        finally:
            self._put_connection(conn)

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

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tool_executions (
                        id, run_id, sequence_no, tool_name,
                        arguments_redacted, result_summary, result_hash,
                        status, duration_ms, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, sequence_no) DO NOTHING
                    """,
                    (
                        exec_id,
                        run_id,
                        sequence_no,
                        tool_name,
                        json.dumps(arguments_redacted),
                        json.dumps(result_summary),
                        result_hash,
                        status,
                        duration_ms,
                        now,
                    ),
                )
                conn.commit()
                logger.info(f"Inserted tool execution {exec_id} for run {run_id}")
                return exec_id
        except (psycopg2.Error, RuntimeError) as e:
            conn.rollback()
            logger.error(f"Failed to insert tool execution: {e}")
            raise
        finally:
            self._put_connection(conn)

    def insert_evidence(
        self,
        incident_id: str,
        run_id: str,
        findings: list[Finding],
        metrics: list[Metric] | None = None,
        logs: list[LogEntry] | None = None,
    ) -> list[str]:
        """Insert evidence records for all findings in a transaction; returns evidence IDs."""
        conn = self._get_connection()
        evidence_ids = []
        now = datetime.now(UTC)

        # Prepare enriched data as JSON
        metrics_json = json.dumps([m.to_dict() for m in (metrics or [])])
        logs_json = json.dumps([entry.to_dict() for entry in (logs or [])])

        try:
            with conn.cursor() as cursor:
                for ordinal, finding in enumerate(findings):
                    evidence_id = f"evd_{uuid.uuid4().hex[:26].upper()}"
                    cursor.execute(
                        """
                        INSERT INTO evidence (
                            id, incident_id, run_id, type, content,
                            source, source_detail, confidence,
                            supports_evidence, contradicts_evidence, ordinal,
                            created_at, metrics, logs, version, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            evidence_id,
                            incident_id,
                            run_id,
                            finding.type.value,
                            finding.content,
                            finding.source.value,
                            finding.source_detail,
                            finding.confidence,
                            [],
                            [],  # supports/contradicts empty for MVP
                            ordinal,
                            now,
                            metrics_json,
                            logs_json,
                            1,
                            now,
                        ),
                    )
                    evidence_ids.append(evidence_id)

                conn.commit()
                logger.info(f"Inserted {len(evidence_ids)} evidence records for run {run_id}")
                return evidence_ids
        except (psycopg2.Error, RuntimeError) as e:
            conn.rollback()
            logger.error(f"Failed to insert evidence: {e}")
            raise
        finally:
            self._put_connection(conn)

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
