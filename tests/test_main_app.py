from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import main


@contextmanager
def client_with(investigator):
    previous = main.investigator
    previous_lifespan = main.app.router.lifespan_context

    @asynccontextmanager
    async def no_lifespan(_app):
        yield

    main.investigator = investigator
    main.app.router.lifespan_context = no_lifespan
    try:
        with TestClient(main.app) as client:
            yield client
    finally:
        main.app.router.lifespan_context = previous_lifespan
        main.investigator = previous


def test_health_and_ready_when_uninitialized():
    with client_with(None) as client:
        assert client.get("/healthz").status_code == 503
        assert client.get("/ready").json()["status"] == "not ready"


def test_health_and_ready_when_database_missing():
    investigator = SimpleNamespace(db_conn=None)
    with client_with(investigator) as client:
        assert client.get("/healthz").json()["error"] == "Database not connected"
        assert client.get("/ready").json()["error"] == "Database not ready"


def test_health_and_ready_query_database(cursor):
    connection = MagicMock()
    connection.cursor.return_value = cursor
    investigator = SimpleNamespace(db_conn=connection)
    with client_with(investigator) as client:
        health = client.get("/healthz")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert ready.json() == {"status": "ready"}
    assert cursor.execute.call_args_list[0].args == ("SELECT 1",)


def test_health_returns_database_error(cursor):
    cursor.execute.side_effect = RuntimeError("offline")
    connection = MagicMock()
    connection.cursor.return_value = cursor
    with client_with(SimpleNamespace(db_conn=connection)) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["error"] == "offline"


def test_investigate_endpoint_is_deprecated_shim():
    """The legacy /api/v1/investigate endpoint is a deprecation shim
    mirroring the async creation contract: persists a queued run and
    returns 202, never inline findings."""
    from unittest.mock import AsyncMock

    service = MagicMock()
    service.db_conn = MagicMock()
    service.persistence.create_investigation_run.return_value = ("irun_legacy1", True)
    service.investigate_async = AsyncMock()
    with client_with(service) as client:
        response = client.post("/api/v1/investigate", params={"incident_id": "inc-1"})
    assert response.status_code == 202
    assert response.headers["Deprecation"] == "true"
    body = response.json()
    assert body["data"]["incident_id"] == "inc-1"
    assert body["data"]["status"] == "queued"
    assert body["data"]["run_id"] == "irun_legacy1"
    # The shim must not invoke the synchronous investigator path.
    service.investigate.assert_not_called()


def test_status_reports_uptime(monkeypatch):
    monkeypatch.setattr(main, "start_time", 100.0)
    monkeypatch.setattr(main.time, "time", lambda: 112.5)
    with client_with(SimpleNamespace()) as client:
        assert client.get("/api/v1/status").json() == {
            "running": True,
            "uptime_seconds": 12.5,
            "version": "1.0.0",
        }


def test_lifespan_initializes_and_shuts_down(config):
    service = MagicMock()
    # Mock the async close method as AsyncMock
    from unittest.mock import AsyncMock

    service.close = AsyncMock()  # Mock the async close method
    shutdown = MagicMock()
    with (
        patch("main.InvestigationConfig.from_env", return_value=config),
        patch("main.PamawasInvestigator", return_value=service),
        patch("main.init_tracer", return_value=shutdown),
        patch("main.set_running") as set_running,
    ):
        with TestClient(main.app):
            assert main.investigator is service
    assert [call.args[0] for call in set_running.call_args_list] == [True, False]
    shutdown.assert_called_once_with()
    service.close.assert_awaited_once()


def test_main_runs_uvicorn_with_config(config):
    with (
        patch("main.InvestigationConfig.from_env", return_value=config),
        patch("uvicorn.run") as run,
    ):
        main.main()
    run.assert_called_once_with("main:app", host="0.0.0.0", port=config.port, log_level="info")


def test_create_investigation_requires_service_token():
    """Test that /api/v1/investigations requires valid service token."""
    import os

    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    service = MagicMock()
    service.persistence = None
    service.db_conn = MagicMock()
    with client_with(service) as client:
        # No token
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid service token"

        # Invalid token
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "wrong-token"},
        )
        assert response.status_code == 401


def test_create_investigation_token_not_configured(monkeypatch):
    """Test that /api/v1/investigations returns 503 when token not configured."""
    monkeypatch.delenv("PAMAWAS_SERVICE_TOKEN", raising=False)
    service = MagicMock()
    service.persistence = None
    service.db_conn = MagicMock()
    with client_with(service) as client:
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "any-token"},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Service token not configured"


def test_create_investigation_idempotent_returns_409():
    """Test that duplicate investigation returns 409 conflict."""
    import os

    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    # Mock persistence that reports an existing run (created_new=False).
    # The idempotency SELECT lives inside create_investigation_run, so the
    # endpoint mock is at the persistence-method level, not the connection level.
    mock_persistence = MagicMock()
    mock_persistence.create_investigation_run.return_value = ("irun_existing123", False)
    # _get_run_status reads the existing run's status on the idempotent-retry path
    status_conn = MagicMock()
    status_cursor = MagicMock()
    status_cursor.fetchone.return_value = ("completed",)
    status_conn.cursor.return_value.__enter__.return_value = status_cursor
    mock_persistence._get_connection.return_value = status_conn

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()

    with client_with(service) as client:
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"},
        )
        assert response.status_code == 409
        assert response.json()["error"] == "Investigation already exists"
        assert response.json()["data"]["run_id"] == "irun_existing123"
        assert response.json()["data"]["status"] == "completed"


def test_create_investigation_success():
    """Test successful investigation creation returns 202 queued per the v1 contract."""
    import os

    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    from unittest.mock import AsyncMock

    mock_persistence = MagicMock()
    mock_persistence.create_investigation_run.return_value = ("irun_new_run", True)

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()
    service.investigate_async = AsyncMock()

    with client_with(service) as client:
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["data"]["incident_id"] == "inc-1"
        assert data["data"]["status"] == "queued"
        assert data["data"]["run_id"] == "irun_new_run"
        # The endpoint must not invoke the synchronous investigator path.
        service.investigate.assert_not_called()


def test_create_investigation_commits_idempotency_check():
    """Idempotency check inside create_investigation_run; connection must commit on success."""
    import os

    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    from unittest.mock import AsyncMock

    # Use the real InvestigationPersistence class with mocked connection pool
    from service.persistence import InvestigationPersistence

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        ("irun_new_run",),
    ]  # INSERT RETURNING returns the row
    mock_cursor.rowcount = 1
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()

    mock_persistence = InvestigationPersistence(database_url="test://url")
    mock_persistence._pool = mock_pool

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()
    service.investigate_async = AsyncMock()

    with client_with(service) as client:
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"},
        )
        assert response.status_code == 202

    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()


def test_create_investigation_rolls_back_on_idempotency_check_failure():
    """A failed idempotency check must roll back, not leak the connection."""
    import os

    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    from service.persistence import InvestigationPersistence

    # Use the real InvestigationPersistence class with mocked connection pool
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = RuntimeError("db down")
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()

    mock_persistence = InvestigationPersistence(database_url="test://url")
    mock_persistence._pool = mock_pool

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()

    with client_with(service) as client:
        response = client.post(
            "/api/v1/investigations",
            json={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"},
        )
        assert response.status_code == 500
        assert response.json()["error"] == "Internal server error"

    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()


def test_get_investigation_status_returns_status():
    """GET /api/v1/investigations/{run_id} returns current status."""
    mock_persistence = MagicMock()
    conn = MagicMock()
    mock_persistence._get_connection.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = ("completed",)

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()

    with client_with(service) as client:
        response = client.get("/api/v1/investigations/irch_abc123")

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "irch_abc123"
    assert data["status"] == "completed"


def test_get_investigation_status_returns_404_for_missing_run():
    """GET /api/v1/investigations/{run_id} returns 404 if run not found."""
    mock_persistence = MagicMock()
    conn = MagicMock()
    mock_persistence._get_connection.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (None,)

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()

    with client_with(service) as client:
        response = client.get("/api/v1/investigations/irch_notfound")

    assert response.status_code == 404
    assert response.json()["error"] == "Investigation not found"


def test_get_investigation_status_returns_503_when_uninitialized():
    """GET /api/v1/investigations/{run_id} returns 503 if investigator not ready."""
    with client_with(None) as client:
        response = client.get("/api/v1/investigations/irch_abc")

    assert response.status_code == 503
    assert "error" in response.json()
