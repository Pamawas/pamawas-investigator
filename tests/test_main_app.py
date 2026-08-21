from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import main
from models import EvidenceType, Finding


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


def test_investigate_endpoint_serializes_findings():
    service = MagicMock()
    service.investigate.return_value = [Finding(EvidenceType.FACT, "observed", "metrics", 0.9)]
    service.db_conn = MagicMock()
    with client_with(service) as client:
        response = client.post("/investigate", params={"incident_id": "inc-1"})
    assert response.status_code == 200
    assert response.json() == {
        "incident_id": "inc-1",
        "findings": [
            {"type": "fact", "content": "observed", "source": "metrics", "confidence": 0.9}
        ],
        "completed": True,
    }
    service.investigate.assert_called_once_with("inc-1")


def test_status_reports_uptime(monkeypatch):
    monkeypatch.setattr(main, "start_time", 100.0)
    monkeypatch.setattr(main.time, "time", lambda: 112.5)
    with client_with(SimpleNamespace()) as client:
        assert client.get("/status").json() == {
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
    """Test that /v1/investigations requires valid service token."""
    import os
    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    service = MagicMock()
    service.persistence = None
    service.db_conn = MagicMock()
    with client_with(service) as client:
        # No token
        response = client.post(
            "/v1/investigations",
            params={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            }
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid service token"

        # Invalid token
        response = client.post(
            "/v1/investigations",
            params={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "wrong-token"}
        )
        assert response.status_code == 401


def test_create_investigation_token_not_configured(monkeypatch):
    """Test that /v1/investigations returns 503 when token not configured."""
    monkeypatch.delenv("PAMAWAS_SERVICE_TOKEN", raising=False)
    service = MagicMock()
    service.persistence = None
    service.db_conn = MagicMock()
    with client_with(service) as client:
        response = client.post(
            "/v1/investigations",
            params={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "any-token"}
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Service token not configured"


def test_create_investigation_idempotent_returns_409():
    """Test that duplicate investigation returns 409 conflict."""
    import os
    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    # Mock persistence that returns existing run
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("irun_existing123", "completed")
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_persistence = MagicMock()
    mock_persistence._get_connection.return_value = mock_conn

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()

    with client_with(service) as client:
        response = client.post(
            "/v1/investigations",
            params={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"}
        )
        assert response.status_code == 409
        assert response.json()["error"] == "Investigation already exists"
        assert response.json()["data"]["run_id"] == "irun_existing123"
        assert response.json()["data"]["status"] == "completed"


def test_create_investigation_success():
    """Test successful investigation creation."""
    import os
    os.environ["PAMAWAS_SERVICE_TOKEN"] = "test-token"

    mock_persistence = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None  # No existing run
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_persistence._get_connection.return_value = mock_conn

    service = MagicMock()
    service.persistence = mock_persistence
    service.db_conn = MagicMock()
    service.investigate.return_value = [
        MagicMock(
            to_dict=lambda: {
                "type": "fact",
                "content": "CPU high",
                "source": "prometheus",
                "confidence": 0.9
            }
        )
    ]

    with client_with(service) as client:
        response = client.post(
            "/v1/investigations",
            params={
                "contract_version": 1,
                "incident_id": "inc-1",
                "reason": "incident_created",
                "correlation_version": 1,
            },
            headers={"x-service-token": "test-token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["incident_id"] == "inc-1"
        assert data["data"]["status"] == "completed"
        assert data["data"]["run_id"].startswith("irun_")
        service.investigate.assert_called_once()
