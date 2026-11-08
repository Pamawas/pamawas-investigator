"""Unit tests for InvestigationPersistence.

These tests cover the persistence layer directly, including edge cases
and rollback behavior that the HTTP endpoint mocks cannot exercise.
"""

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
        """Edge case: INSERT returns no row (ON CONFLICT), but SELECT returns None.

        This should not happen in production (the ON CONFLICT should find existing),
        but the code guards against it with a RuntimeError in the else branch."""
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
        assert "status = 'failed'" in update_call
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
