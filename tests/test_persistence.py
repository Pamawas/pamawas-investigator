"""Unit tests for InvestigationPersistence.

These tests cover the persistence layer directly, including edge cases and rollback behavior.
"""

import json
from datetime import UTC
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from service.persistence import InvestigationPersistence

# Named constants for test determinism (hook feedback addressed)
RUN_ID_NEW = "irun_NEW12345678901234567890"
RUN_ID_EXISTING = "irun_EXISTING123456789012345678"
INCIDENT_ID = "inc-1"
REQUEST_KEY_HASH = "abc123hash"


@pytest.fixture
def mock_pool():
    """Create a mock connection pool that simulates the real pool behavior."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_cursor.rowcount = 0  # Track rows affected by UPDATE/DELETE
    mock_cursor.__enter__ = lambda self: self
    mock_cursor.__exit__ = lambda self, *args: None

    mock_conn.cursor.return_value = mock_cursor
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()
    return mock_pool, mock_conn, mock_cursor


class TestCreateInvestigationRun:
    """Tests for InvestigationPersistence.create_investigation_run."""

    def test_returns_run_id_and_created_new_true_on_insert(self, mock_pool):
        """A new run is inserted and (run_id, True) is returned."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = ("irun_ABC123NEWRUN",)  # INSERT RETURNING

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                run_id, created_new = pers.create_investigation_run(
                    incident_id=INCIDENT_ID,
                    request_key_hash=REQUEST_KEY_HASH,
                    model_provider="https://provider.example",
                    model_name="model-v1",
                    prompt_version="v1",
                    tool_contract=1,
                    max_tool_calls=10,
                )

        assert run_id == "irun_ABC123NEWRUN"
        assert created_new is True
        mock_conn.commit.assert_called_once()

    def test_returns_existing_run_and_false_on_idempotent_retry(self, mock_pool):
        """An existing run is found and (existing_id, False) is returned."""
        mock_pool, mock_conn, mock_cursor = mock_pool

        # INSERT returns None (ON CONFLICT DO NOTHING happened)
        # SELECT finds existing row
        mock_cursor.fetchone.side_effect = [
            None,  # INSERT RETURNING - no row (conflict occurred)
            ("irun_EXISTING1234567890123",),  # SELECT finds existing
        ]

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                run_id, created_new = pers.create_investigation_run(
                    incident_id=INCIDENT_ID,
                    request_key_hash=REQUEST_KEY_HASH,
                    model_provider="https://provider.example",
                    model_name="model-v1",
                    prompt_version="v1",
                    tool_contract=1,
                    max_tool_calls=10,
                )

        assert run_id == "irun_EXISTING1234567890123"
        assert created_new is False
        mock_conn.commit.assert_called_once()

    def test_rolls_back_and_reraises_on_insert_failure(self, mock_pool):
        """A failed INSERT rolls back and re-raises the exception."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.side_effect = RuntimeError("db down")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                with pytest.raises(RuntimeError, match="db down"):
                    pers.create_investigation_run(
                        incident_id=INCIDENT_ID,
                        request_key_hash=REQUEST_KEY_HASH,
                        model_provider="https://provider.example",
                        model_name="model-v1",
                        prompt_version="v1",
                        tool_contract=1,
                        max_tool_calls=10,
                    )

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()

    def test_returns_none_from_fetchone_yields_runtime_error(self, mock_pool):
        """Edge case: INSERT returns no row (ON CONFLICT) but SELECT
        returns None; code raises RuntimeError."""
        mock_pool, mock_conn, mock_cursor = mock_pool

        # INSERT returns None (ON CONFLICT DO NOTHING), but SELECT returns None
        # This means the row doesn't exist at all (inconsistent state)
        mock_cursor.fetchone.side_effect = [
            None,  # INSERT RETURNING - no row (but conflict didn't find it either)
            None,  # SELECT - no existing row found
        ]

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                with pytest.raises(
                    RuntimeError, match="Failed to create or find investigation run"
                ):
                    pers.create_investigation_run(
                        incident_id=INCIDENT_ID,
                        request_key_hash=REQUEST_KEY_HASH,
                        model_provider="https://provider.example",
                        model_name="model-v1",
                        prompt_version="v1",
                        tool_contract=1,
                        max_tool_calls=10,
                    )

        mock_conn.rollback.assert_called_once()


class TestTelemetryCatalogRevisions:
    """Tests for immutable catalog pinning on investigation runs."""

    def test_pins_content_addressed_revision_to_unpinned_run(self, mock_pool):
        _, mock_conn, mock_cursor = mock_pool
        catalog = {
            "prometheus": {
                "vendor_queue_depth": {
                    "name": "vendor_queue_depth",
                    "_search_text": "Vendor queue depth",
                }
            },
            "loki": {},
        }
        mock_cursor.fetchall.return_value = [
            (
                "prometheus",
                "vendor_queue_depth",
                "Vendor queue depth",
                {"name": "vendor_queue_depth"},
            )
        ]
        mock_cursor.fetchone.side_effect = [
            (None,),
            None,
            ("tcr_stored",),
            ("tcr_stored", "catalog-hash", catalog, 1),
        ]

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                snapshot = pers.pin_telemetry_catalog_revision("irun_1")

        assert snapshot["revision_id"] == "tcr_stored"
        assert snapshot["catalog"] == catalog
        assert snapshot["entry_count"] == 1
        statements = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("INSERT INTO telemetry_catalog_revisions" in sql for sql in statements)
        assert any("SET catalog_revision_id = %s" in sql for sql in statements)
        mock_conn.commit.assert_called_once()

    def test_existing_pin_is_loaded_without_reading_mutable_catalog(self, mock_pool):
        _, mock_conn, mock_cursor = mock_pool
        catalog = {"prometheus": {}, "loki": {"app": {"name": "app"}}}
        mock_cursor.fetchone.side_effect = [
            ("tcr_existing",),
            ("tcr_existing", "existing-hash", catalog, 1),
        ]

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                snapshot = pers.pin_telemetry_catalog_revision("irun_retry")

        assert snapshot["revision_id"] == "tcr_existing"
        assert snapshot["content_hash"] == "existing-hash"
        mock_cursor.fetchall.assert_not_called()
        statements = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert not any("telemetry_catalog_entries" in sql for sql in statements)
        assert not any("INSERT INTO telemetry_catalog_revisions" in sql for sql in statements)


class TestUpdateRunStatus:
    """Tests for InvestigationPersistence.update_run_status."""

    def test_sets_status_and_updates_timestamps(self, mock_pool):
        """Status update is persisted with timestamps."""
        mock_pool, mock_conn, mock_cursor = mock_pool

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                pers.update_run_status("irun_123", "running")

        mock_conn.commit.assert_called_once()
        # Verify the query was an UPDATE with correct status
        update_call = mock_cursor.execute.call_args[0][0]
        assert "UPDATE investigation_runs" in update_call
        assert "status = %s" in update_call

    def test_rolls_back_on_failure(self, mock_pool):
        """A failed update rolls back the transaction."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = RuntimeError("connection lost")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                with pytest.raises(RuntimeError, match="connection lost"):
                    pers.update_run_status("irun_123", "completed")

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()


class TestInsertToolExecution:
    """Tests for InvestigationPersistence.insert_tool_execution."""

    def test_returns_exec_id_on_success(self, mock_pool):
        """A new tool execution is inserted and its ID is returned."""
        mock_pool, mock_conn, _mock_cursor = mock_pool

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                exec_id = pers.insert_tool_execution(
                    run_id="irun_123",
                    sequence_no=1,
                    tool_name="query_prometheus",
                    arguments={"query": "rate(http_requests[5m])"},
                    result={"status": "ok", "data": [1, 2, 3]},
                    status="success",
                    duration_ms=42,
                )

        assert exec_id.startswith("texec_")
        mock_conn.commit.assert_called_once()

    def test_redacts_sensitive_arguments(self, mock_pool):
        """API keys, passwords, and tokens in arguments are redacted."""
        mock_pool, mock_conn, mock_cursor = mock_pool

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                pers.insert_tool_execution(
                    run_id="irun_123",
                    sequence_no=1,
                    tool_name="get_deployment",
                    arguments={"api_key": "secret123", "password": "pwd", "query": "safe"},
                    result={"status": "ok"},
                    status="success",
                    duration_ms=42,
                )

        # Get the arguments that were passed to INSERT
        insert_call = mock_cursor.execute.call_args
        args = insert_call[0][1]  # Second positional arg to execute
        assert args[3] == "get_deployment"  # tool_name
        # arguments_redacted is JSON in position 4
        import json

        redacted = json.loads(args[4]) if isinstance(args[4], str) else args[4]
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["query"] == "safe"


class TestConnectionPoolManagement:
    """Tests for connection pool lifecycle."""

    def test_pool_not_initialized_without_db_url(self):
        """Pool remains None when DATABASE_URL is empty."""
        pers = InvestigationPersistence(database_url="")

        # Pool should not exist yet and should stay None
        assert pers._pool is None

        # Getting connection should raise
        import pytest

        with pytest.raises(RuntimeError, match="Database URL not configured"):
            pers._get_connection()

    def test_pool_initialized_on_first_use(self):
        """The pool is lazily created when a connection is first requested."""
        with patch("service.persistence.psycopg2.pool.ThreadedConnectionPool") as mock_pool_class:
            with patch("service.persistence.psycopg2.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                pers = InvestigationPersistence(database_url="postgres://test@localhost/test")

                # Pool should not exist yet
                assert pers._pool is None

                # Getting connection creates the pool
                conn = pers._get_connection()
                assert pers._pool is not None
                assert pers._pool == mock_pool

                conn.close()
                pers.close()

    def test_close_returns_all_connections_to_pool(self, mock_pool):
        """Close calls closeall on the pool."""
        mock_pool, mock_conn, _mock_cursor = mock_pool

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            pers = InvestigationPersistence(database_url="test://url")
            pers._pool = mock_pool
            pers.close()

        mock_pool.closeall.assert_called_once()
        assert pers._pool is None


class TestMarkRunFailed:
    """Tests for InvestigationPersistence.mark_run_failed."""

    def test_marks_run_as_failed(self, mock_pool):
        """A pending run is marked as failed with error code."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        # Simulate one row updated by the UPDATE
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                updated = pers.mark_run_failed("irun_123", error="LLM timeout")

        assert updated is True
        mock_conn.commit.assert_called_once()
        # Verify the UPDATE query
        update_call = mock_cursor.execute.call_args[0][0]
        assert "UPDATE investigation_runs" in update_call
        assert "status = 'failed_terminal'" in update_call
        params = mock_cursor.execute.call_args[0][1]
        assert "irun_123" in params

    def test_does_not_update_already_completed_run(self, mock_pool):
        """A run in a terminal state is not updated (idempotent)."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        # Simulate existing rowcount=0 (no matching run - already in terminal state)
        mock_cursor.rowcount = 0

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                updated = pers.mark_run_failed("irun_completed_123", error="already done")

        assert updated is False
        # Transaction is always committed to close it cleanly, even when no rows updated
        mock_conn.commit.assert_called_once()

    def test_rolls_back_on_failure(self, mock_pool):
        """A failed UPDATE rolls back the transaction."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.Error("connection lost")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.mark_run_failed("irun_123", error="db down")

        assert result is False
        mock_conn.rollback.assert_called_once()


class TestReclaimDueInvestigationRetry:
    """Tests for InvestigationPersistence.reclaim_due_investigation_retry."""

    def test_clears_completed_at_from_the_failed_attempt(self, mock_pool):
        """A reclaimed run must not keep the failed attempt's completed_at.

        investigation_runs_check1 requires completed_at >= started_at. The
        reclaim sets started_at to now, so a stale completed_at from the
        previous attempt lands before it and Postgres rejects the update,
        leaving the run failed_retryable forever.
        """
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                reclaimed = pers.reclaim_due_investigation_retry("irun_123")

        assert reclaimed is True
        query = mock_cursor.execute.call_args[0][0]
        assert "completed_at = NULL" in query
        assert "safe_error_code = NULL" in query
        mock_conn.commit.assert_called_once()


class TestInsertEvidenceSerialization:
    """Tests for InvestigationPersistence.insert_evidence JSON encoding."""

    def test_datetime_bearing_samples_serialize(self, mock_pool):
        """Metric and LogEntry samples carry datetimes; they must serialize.

        The enrichment fetchers return typed samples whose `at`/`timestamp`
        are datetime objects. Encoding them with a bare json.dumps raised
        "Object of type datetime is not JSON serializable", which aborted
        persistence and left the run retry-looping with no evidence.
        """
        from datetime import UTC, datetime

        from models import EvidenceSource, EvidenceType, Finding, LogEntry, Metric

        mock_pool, mock_conn, mock_cursor = mock_pool
        now = datetime.now(UTC)

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                pers.insert_evidence(
                    incident_id=INCIDENT_ID,
                    run_id=RUN_ID_NEW,
                    findings=[
                        Finding(
                            type=EvidenceType.FACT,
                            content="target down",
                            source=EvidenceSource.PROMETHEUS,
                            confidence=0.9,
                        )
                    ],
                    metrics=[Metric(name="up", value=1.0, labels={"job": "x"}, at=now)],
                    logs=[LogEntry(timestamp=now, stream="kube_logs", line="hello")],
                )

        # The metrics/logs JSON are the last two params of the evidence insert.
        params = mock_cursor.execute.call_args[0][1]
        assert any('"at":' in str(p) for p in params)
        assert any('"timestamp":' in str(p) for p in params)

    def test_insert_evidence_computes_ordinal_from_existing_max(self, mock_pool):
        from models import EvidenceSource, EvidenceType, Finding

        _mock_pool_obj, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = (4,)

        with (
            patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn),
            patch.object(InvestigationPersistence, "_put_connection", return_value=None),
        ):
            pers = InvestigationPersistence(database_url="test://url")
            pers.insert_evidence(
                incident_id=INCIDENT_ID,
                run_id=RUN_ID_NEW,
                findings=[
                    Finding(
                        type=EvidenceType.FACT,
                        content="new finding",
                        source=EvidenceSource.PROMETHEUS,
                        confidence=0.8,
                    )
                ],
            )

        # The ordinal in the INSERT statement should be max_ordinal + 1 = 5
        params = mock_cursor.execute.call_args[0][1]
        assert params[10] == 5


class TestUpdateRunStatusStartedAt:
    """started_at must record when the run actually began."""

    def test_running_status_preserves_an_existing_started_at(self, mock_pool):
        """Marking a run running twice must not reset its start time.

        _persist_investigation re-marks the run "running" at the END, just
        before completing it. A plain assignment reset started_at to that
        moment, so a run that spent the full 10-minute deadline looked like it
        took 4 seconds — every "why was this slow" investigation then started
        from a false premise.
        """
        mock_pool, mock_conn, mock_cursor = mock_pool

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                pers.update_run_status("irun_123", "running")

        query = mock_cursor.execute.call_args[0][0]
        assert "started_at = COALESCE(started_at, %s)" in query


class TestClaimNextQueuedRun:
    """The queue read: claim one queued run atomically."""

    def test_claims_one_run_and_marks_it_running(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = (
            "irun_CLAIMED",
            "inc-1",
            "hash-1",
        )

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                claimed = pers.claim_next_queued_run()

        assert claimed == ("irun_CLAIMED", "inc-1", "hash-1")
        sql = mock_cursor.execute.call_args[0][0]
        # The claim must be atomic against other workers.
        assert "FOR UPDATE SKIP LOCKED" in sql
        assert "status = 'running'" in sql
        assert "status = 'queued'" in sql
        mock_conn.commit.assert_called_once()

    def test_returns_none_when_queue_is_empty(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = None

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                assert pers.claim_next_queued_run() is None

        mock_conn.commit.assert_called_once()

    def test_returns_none_instead_of_raising_on_db_error(self, mock_pool):
        """A worker poll must not die because one query failed."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("connection lost")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                assert pers.claim_next_queued_run() is None

        mock_conn.rollback.assert_called_once()


class TestRequeueStaleRunning:
    """Runs abandoned by a dead worker must return to the queue."""

    def test_requeues_and_reports_count(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 2

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                assert pers.requeue_stale_running(1200.0) == 2

        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'queued'" in sql
        assert "status = 'running'" in sql
        # The cutoff is derived from the investigation deadline, not hard-coded.
        assert "updated_at < now()" in sql

    def test_returns_zero_on_db_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("gone")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection", return_value=None):
                pers = InvestigationPersistence(database_url="test://url")
                assert pers.requeue_stale_running(1200.0) == 0


class TestPoolInitializationIsThreadSafe:
    """Two threads reaching a cold pool must not build two pools.

    The worker pool claims runs through asyncio.to_thread, so a cold start has
    several threads calling _get_connection at once. If each builds its own
    pool, the second assignment discards the first and every connection handed
    out by the discarded pool fails on return with "trying to put unkeyed
    connection" — which is what production logged before this fix.
    """

    def test_concurrent_first_use_creates_one_pool(self):
        import threading
        import time

        created = []

        class FakePool:
            def __init__(self, **kwargs):
                # Widen the window between the "is the pool built?" check and
                # the assignment. Connecting to postgres is slow, so the window
                # is wide in production; an instantaneous fake would hide the
                # race and the test would pass against the bug.
                time.sleep(0.01)
                created.append(kwargs)
                self.conns = []

            def getconn(self):
                conn = MagicMock()
                self.conns.append(conn)
                return conn

            def putconn(self, conn):
                pass

        pers = InvestigationPersistence(database_url="test://url")
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            pers._get_connection()

        with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=FakePool):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(created) == 1, f"created {len(created)} pools, want exactly 1"


# ---------------------------------------------------------------------------
# stale_verifications persistence methods
# ---------------------------------------------------------------------------

INCIDENT_ID_SV = "inc-sv-001"
RUN_ID_SV = "irun-sv-001"


class TestClaimNextStaleVerification:
    """Tests for InvestigationPersistence.claim_next_stale_verification."""

    def test_returns_none_when_no_eligible_row(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = None  # no queued row

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.claim_next_stale_verification()

        assert result is None

    def test_returns_incident_and_run_id_when_row_claimed(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = (INCIDENT_ID_SV, RUN_ID_SV)
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.claim_next_stale_verification()

        assert result == (INCIDENT_ID_SV, RUN_ID_SV)
        mock_conn.commit.assert_called()

    def test_returns_none_on_db_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("gone")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.claim_next_stale_verification()

        assert result is None


class TestCompleteStaleVerification:
    """Tests for InvestigationPersistence.complete_stale_verification."""

    def test_returns_true_when_row_updated(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.complete_stale_verification(
                    INCIDENT_ID_SV, RUN_ID_SV, "recovered", "service back up"
                )

        assert result is True
        mock_conn.commit.assert_called()

    def test_returns_false_when_no_row_updated(self, mock_pool):
        """Row may already be requeued by lease-expiry sweeper."""
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 0

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.complete_stale_verification(
                    INCIDENT_ID_SV, RUN_ID_SV, "still_down", "metrics still elevated"
                )

        assert result is False

    def test_returns_false_on_db_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("gone")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.complete_stale_verification(
                    INCIDENT_ID_SV, RUN_ID_SV, "still_down", ""
                )

        assert result is False


class TestFailStaleVerification:
    """Tests for InvestigationPersistence.fail_stale_verification."""

    def test_returns_true_when_row_updated(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.fail_stale_verification(
                    INCIDENT_ID_SV, RUN_ID_SV, "LLM timeout"
                )

        assert result is True
        mock_conn.commit.assert_called()

    def test_returns_false_on_db_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("gone")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.fail_stale_verification(
                    INCIDENT_ID_SV, RUN_ID_SV, "timeout"
                )

        assert result is False

    def test_accepts_none_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.fail_stale_verification(INCIDENT_ID_SV, RUN_ID_SV, None)

        assert result is True


def test_result_summary_is_bounded_for_resume_state():
    pers = InvestigationPersistence(database_url="test://url")
    result = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"instance": f"node-{index}"},
                    "values": [[sample, "1"] for sample in range(1000)],
                }
                for index in range(100)
            ]
        },
    }

    summary = pers._summarize_result(result)

    assert len(json.dumps(summary).encode("utf-8")) <= pers.MAX_RESULT_SUMMARY_BYTES
    assert summary["status"] == "success"


def test_result_summary_preserves_bounded_recovery_policy_audits():
    pers = InvestigationPersistence(database_url="test://url")
    result = {
        "status": "success",
        "target_coverage": {
            "validated": True,
            "method": "exact_incident_identity",
            "label": "opaque_scope",
            "value": "alpha-target",
        },
        "_candidate_selection_audit": {
            "policy_version": 1,
            "accepted": [{
                "name": "vendor_domain_metric",
                "domain_overlap": ["vendor"],
                "score": {"domain_overlap": 1},
            }],
            "rejected": [{
                "name": "unrelated_metric",
                "reason": "no_incident_domain_overlap",
            }],
        },
    }

    summary = pers._summarize_result(result)

    assert summary["target_coverage"]["validated"] is True
    assert summary["target_coverage"]["label"] == "opaque_scope"
    audit = summary["_candidate_selection_audit"]
    assert audit["accepted"][0]["score"]["domain_overlap"] == 1
    assert audit["rejected"][0]["reason"] == "no_incident_domain_overlap"


class TestRequeueExpiredStaleRunning:
    """Tests for InvestigationPersistence.requeue_expired_stale_running."""

    def test_returns_count_of_requeued_rows(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 3

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.requeue_expired_stale_running()

        assert result == 3
        mock_conn.commit.assert_called()

    def test_returns_zero_when_nothing_to_requeue(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 0

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.requeue_expired_stale_running()

        assert result == 0


class TestEnqueueStaleVerification:
    """Tests for InvestigationPersistence.enqueue_stale_verification."""

    def test_enqueues_and_returns_dict(self, mock_pool):
        from datetime import datetime

        mock_pool, mock_conn, mock_cursor = mock_pool
        now = datetime.now(UTC)
        mock_cursor.fetchone.return_value = (INCIDENT_ID_SV, "queued", now)

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.enqueue_stale_verification(INCIDENT_ID_SV)

        assert result is not None
        assert result["incident_id"] == INCIDENT_ID_SV
        assert result["status"] == "queued"
        assert result["next_verify_at"] == now.isoformat()
        mock_conn.commit.assert_called()

    def test_returns_none_on_db_error(self, mock_pool):
        mock_pool, mock_conn, mock_cursor = mock_pool
        mock_cursor.execute.side_effect = psycopg2.OperationalError("gone")

        with patch.object(InvestigationPersistence, "_get_connection", return_value=mock_conn):
            with patch.object(InvestigationPersistence, "_put_connection"):
                pers = InvestigationPersistence(database_url="test://url")
                result = pers.enqueue_stale_verification(INCIDENT_ID_SV)

        assert result is None
