"""Database persistence for investigation runs, tool executions, and evidence."""

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        # Guarded because pool creation is now reached from several threads at
        # once: the investigation worker pool claims runs through
        # asyncio.to_thread, so a cold start has N workers plus the retry sweep
        # all calling _get_connection simultaneously.
        self._pool_lock = threading.Lock()

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a database connection from the pool."""
        if not self.database_url:
            raise RuntimeError("Database URL not configured")

        # Double-checked locking. Without it two threads can each build a pool;
        # the second assignment discards the first, and a connection handed out
        # by the discarded pool is unknown to the live one — putconn then raises
        # "trying to put unkeyed connection" and that thread's work is lost.
        if self._pool is None:
            with self._pool_lock:
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

    def requeue_stale_running(self, stale_after_seconds: float) -> int:
        """Return runs abandoned by a dead worker to the queue.

        A run is `running` only while a worker is executing it. If the pod
        restarts mid-investigation nothing is left to finish that row, so it
        stays `running` forever — the status endpoint reports progress that
        will never come, and the scheduler's high-severity follow-up waits for a
        terminal state that never arrives.

        `updated_at` advances on every tool execution persisted, so a run whose
        timestamp is older than the investigation deadline plus a generous
        margin belongs to a worker that is gone. The cutoff is measured against
        the deadline, not a fixed number, so a long investigation is never
        mistaken for an abandoned one.

        Returns how many rows were requeued.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE investigation_runs
                    SET status = 'queued',
                        updated_at = %s
                    WHERE status = 'running'
                      AND updated_at < now() - (%s * interval '1 second')
                    """,
                    (now, stale_after_seconds),
                )
                requeued = cursor.rowcount
            conn.commit()
            if requeued:
                logger.warning(
                    "Requeued %d investigation run(s) abandoned by a dead worker", requeued
                )
            return requeued
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to requeue stale running investigations: %s", e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return 0
        finally:
            self._put_connection(conn)

    def claim_next_queued_run(self) -> tuple[str, str, str] | None:
        """Atomically claim one queued run and return (run_id, incident_id, request_key_hash).

        This is the queue read. `investigation_runs` rows with status `queued`
        ARE the queue: the HTTP handler only persists the row and returns 202,
        and a fixed pool of workers claims from this table.

        Claiming in the database rather than in process memory is what makes the
        queue survive a restart. An earlier version spawned one background task
        per request and bounded them with an in-process semaphore: a run waiting
        for a slot held no state anywhere, so when the pod restarted the waiting
        runs were simply gone — production had two runs stuck in `queued` for
        thirteen hours for exactly this reason. A row cannot be lost that way.

        `FOR UPDATE SKIP LOCKED` makes concurrent claims safe: two workers
        polling at the same moment take different rows instead of racing for the
        same one, so no run is executed twice and no worker blocks on a row
        another worker already holds.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE investigation_runs r
                    SET status = 'running',
                        started_at = COALESCE(r.started_at, %s),
                        updated_at = %s
                    WHERE r.id = (
                        SELECT id FROM investigation_runs
                        WHERE status = 'queued'
                        ORDER BY started_at, id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING r.id, r.incident_id, r.request_key_hash
                    """,
                    (now, now),
                )
                row = cursor.fetchone()
            if not row:
                conn.commit()
                return None
            conn.commit()
            logger.info(f"Claimed queued investigation run {row[0]}")
            return row[0], row[1], row[2]
        except (psycopg2.Error, RuntimeError) as e:
            logger.error(f"Failed to claim queued investigation run: {e}")
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None
        finally:
            self._put_connection(conn)

    # -------------------------------------------------------------------------
    # stale_verifications — async verification worker methods
    # -------------------------------------------------------------------------

    def enqueue_stale_verification(self, incident_id: str) -> dict[str, Any] | None:
        """Enqueue an incident into stale_verifications; returns record dict or None.

        If already queued or running, preserves the current status and back-off.
        If previously completed or failed, resets status to 'queued' and next_verify_at to now().
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO stale_verifications (
                        incident_id, status, attempts, next_verify_at, created_at, updated_at
                    ) VALUES (%s, 'queued', 0, %s, %s, %s)
                    ON CONFLICT (incident_id) DO UPDATE SET
                        status = CASE
                            WHEN stale_verifications.status IN ('completed', 'failed') THEN 'queued'
                            ELSE stale_verifications.status
                        END,
                        next_verify_at = CASE
                            WHEN stale_verifications.status IN ('completed', 'failed')
                                THEN EXCLUDED.next_verify_at
                            ELSE stale_verifications.next_verify_at
                        END,
                        updated_at = EXCLUDED.updated_at
                    RETURNING incident_id, status, next_verify_at
                    """,
                    (incident_id, now, now, now),
                )
                row = cursor.fetchone()
            conn.commit()
            if row:
                return {
                    "incident_id": row[0],
                    "status": row[1],
                    "next_verify_at": row[2].isoformat() if row[2] else None,
                }
            return None
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to enqueue stale verification for incident %s: %s", incident_id, e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None
        finally:
            if conn is not None:
                self._put_connection(conn)

    def claim_next_stale_verification(self) -> str | None:
        """Atomically claim one queued stale_verifications row; return incident_id or None.

        Uses FOR UPDATE SKIP LOCKED so multiple async workers can poll
        simultaneously without racing for the same row.  Returns None if no
        eligible row exists (status='queued' AND next_verify_at <= now()).
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            lease_expires = now + timedelta(minutes=10)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE stale_verifications
                    SET status = 'running',
                        lease_expires_at = %s,
                        updated_at = %s
                    WHERE incident_id = (
                        SELECT incident_id
                        FROM stale_verifications
                        WHERE status = 'queued'
                          AND next_verify_at <= %s
                        ORDER BY next_verify_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING incident_id
                    """,
                    (lease_expires, now, now),
                )
                row = cursor.fetchone()
            if not row:
                conn.commit()
                return None
            conn.commit()
            incident_id = row[0]
            logger.info("Claimed stale_verifications row for incident %s", incident_id)
            return incident_id
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to claim stale_verifications row: %s", e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None
        finally:
            if conn is not None:
                self._put_connection(conn)

    def complete_stale_verification(self, incident_id: str, verdict: str, reason: str) -> bool:
        """Mark a stale_verifications row completed with verdict + reason.

        Only transitions from 'running' to prevent overwriting a row that was
        already requeued by the lease-expiry sweeper.  Returns True if a row
        was updated.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE stale_verifications
                    SET status = 'completed',
                        verdict = %s,
                        reason = %s,
                        verified_at = %s,
                        updated_at = %s
                    WHERE incident_id = %s
                      AND status = 'running'
                    """,
                    (verdict, reason, now, now, incident_id),
                )
                updated = cursor.rowcount > 0
            conn.commit()
            logger.info(
                "Completed stale_verifications for incident %s verdict=%s updated=%s",
                incident_id,
                verdict,
                updated,
            )
            return updated
        except (psycopg2.Error, RuntimeError) as e:
            logger.error(
                "Failed to complete stale_verifications for incident %s: %s",
                incident_id,
                e,
            )
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return False
        finally:
            if conn is not None:
                self._put_connection(conn)

    def fail_stale_verification(self, incident_id: str, error: str | None) -> bool:
        """Mark a stale_verifications row as failed.

        Conservative: a failed or inconclusive verification is treated as
        'still_down' by the scheduler.  Only transitions from 'running'.
        Returns True if a row was updated.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE stale_verifications
                    SET status = 'failed',
                        safe_error_code = %s,
                        updated_at = %s
                    WHERE incident_id = %s
                      AND status = 'running'
                    """,
                    (error, now, incident_id),
                )
                updated = cursor.rowcount > 0
            conn.commit()
            logger.info(
                "Failed stale_verifications for incident %s updated=%s",
                incident_id,
                updated,
            )
            return updated
        except (psycopg2.Error, RuntimeError) as e:
            logger.error(
                "Failed to record stale_verifications failure for incident %s: %s",
                incident_id,
                e,
            )
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return False
        finally:
            if conn is not None:
                self._put_connection(conn)

    def requeue_expired_stale_running(self) -> int:
        """Requeue stale_verifications rows stuck in 'running' past lease_expires_at.

        Self-heals rows whose worker died mid-verification.  Resets
        next_verify_at to now() so the worker pool picks them up immediately on
        the next poll rather than waiting for the original back-off window.
        Returns the number of rows requeued.
        """
        conn = None
        try:
            conn = self._get_connection()
            now = datetime.now(UTC)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE stale_verifications
                    SET status = 'queued',
                        next_verify_at = %s,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE status = 'running'
                      AND lease_expires_at <= %s
                    """,
                    (now, now, now),
                )
                count = cursor.rowcount
            conn.commit()
            if count:
                logger.info("Requeued %d expired stale_verifications rows", count)
            return count
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to requeue expired stale_verifications rows: %s", e)
            if conn is not None:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return 0
        finally:
            if conn is not None:
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
                    "SELECT status, attempts, max_attempts FROM investigation_runs WHERE id = %s",
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
        """Atomically move a due `failed_retryable` run back to `queued`.

        It goes back to `queued`, not straight to `running`: the worker pool is
        the only thing that executes runs, and it claims from the queue. Setting
        `running` here would make the run look busy to the pool and to anyone
        polling the status endpoint while nothing was actually running it.

        Guards the auto-retry against double-execution: only one caller can flip
        a run from `failed_retryable` (with `next_retry_at <= now`) to `queued`,
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
                    SET status = 'queued',
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
                cursor.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) FROM evidence WHERE run_id = %s",
                    (run_id,),
                )
                row = cursor.fetchone()
                start_ordinal = (row[0] if row else -1) + 1

                for i, finding in enumerate(findings):
                    ordinal = start_ordinal + i
                    evidence_id = f"evd_{uuid.uuid4().hex[:26].upper()}"
                    cursor.execute(
                        """
                        INSERT INTO evidence (
                            id, incident_id, run_id, type, content,
                            source, source_detail, confidence,
                            supports_evidence, contradicts_evidence, ordinal,
                            created_at, metrics, logs, version, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id, ordinal) DO UPDATE SET
                            type = EXCLUDED.type,
                            content = EXCLUDED.content,
                            source = EXCLUDED.source,
                            source_detail = EXCLUDED.source_detail,
                            confidence = EXCLUDED.confidence,
                            updated_at = EXCLUDED.updated_at
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
