"""Database-backed related incidents query adapter with circuit breaker."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras

from adapters.circuit_breaker import CircuitBreaker
from metrics import increment_tool_calls, observe_tool_call_duration

logger = logging.getLogger(__name__)


@dataclass
class RelatedIncidentsConfig:
    """Configuration for related incidents adapter."""
    database_url: str
    max_results: int = 10
    max_symptom_keywords: int = 20
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_calls: int = 3


class RelatedIncidentsError(Exception):
    """Base exception for related incidents adapter errors."""
    pass


class RelatedIncidentsAdapter:
    """Parameterized database query for related incidents.

    The model never supplies SQL. All queries are parameterized and
    constructed by the adapter based on bounded input parameters.
    """

    def __init__(self, config: RelatedIncidentsConfig):
        self.config = config
        self._conn: psycopg2.extensions.connection | None = None
        self._circuit_breaker = CircuitBreaker(
            name="related_incidents",
            failure_threshold=config.circuit_breaker_failure_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get or create database connection with timeout."""
        if self._conn is None or self._conn.closed:
            if not self.config.database_url:
                raise RelatedIncidentsError("Database URL not configured")
            self._conn = psycopg2.connect(
                self.config.database_url,
                connect_timeout=5,  # 5-second timeout to prevent startup block
            )
            self._conn.autocommit = True
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def circuit_breaker_state(self) -> dict:
        """Return circuit breaker state for health checks."""
        return self._circuit_breaker.snapshot()

    def _build_symptom_conditions(
        self, symptom_keywords: list[str]
    ) -> tuple[str, list[Any]]:
        """Build parameterized WHERE conditions for symptom keyword matching.

        Returns (where_clause, params) where where_clause includes WHERE keyword.
        """
        if not symptom_keywords:
            return "", []

        # Sanitize and limit keywords
        sanitized = [
            kw.strip()
            for kw in symptom_keywords[: self.config.max_symptom_keywords]
            if kw.strip()
        ]

        if not sanitized:
            return "", []

        # Build ILIKE conditions for title and events
        conditions = []
        params = []

        for kw in sanitized:
            # Use parameterized ILIKE for safety
            pattern = f"%{kw}%"
            conditions.append(
                "(i.title ILIKE %s OR EXISTS "
                "(SELECT 1 FROM incident_events ie "
                "JOIN events e ON e.id = ie.event_id "
                "WHERE ie.incident_id = i.id "
                "AND (e.title ILIKE %s OR e.raw_payload::text ILIKE %s)))"
            )
            params.extend([pattern, pattern, pattern])

        where_clause = "WHERE (" + " OR ".join(conditions) + ")"
        return where_clause, params

    async def find_related(
        self,
        service: str,
        symptom_keywords: list[str],
        environment: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Find related incidents using parameterized query.

        Args:
            service: Service name to filter by
            symptom_keywords: List of symptom keywords to match
            environment: Optional environment filter
            limit: Maximum results (defaults to config max)

        Returns:
            Dict with status and data list of related incidents
        """
        start_time = time.time()

        if not service or not service.strip():
            observe_tool_call_duration("related_incidents", time.time() - start_time)
            increment_tool_calls("related_incidents", "unavailable")
            return self._unavailable_response("invalid_service_name")

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            observe_tool_call_duration("related_incidents", time.time() - start_time)
            increment_tool_calls("related_incidents", "circuit_open")
            logger.warning("Related incidents circuit breaker open, rejecting query")
            return self._unavailable_response("circuit_breaker_open")

        # Validate and limit results
        max_results = min(limit or self.config.max_results, self.config.max_results)

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Build parameterized query
                # Match by service (from affected_services array) and symptoms
                where_conditions = ["%s = ANY(i.affected_services)"]
                params: list[Any] = [service.strip()]

                if environment:
                    where_conditions.append("i.environment = %s")
                    params.append(environment.strip())

                # Add symptom keyword conditions
                symptom_clause, symptom_params = self._build_symptom_conditions(
                    symptom_keywords
                )
                if symptom_clause:
                    where_conditions.append(symptom_clause.lstrip("WHERE "))
                    params.extend(symptom_params)

                where_clause = "WHERE " + " AND ".join(where_conditions)

                query = f"""
                    SELECT
                        i.id as incident_id,
                        i.title,
                        i.status,
                        i.started_at,
                        i.resolved_at,
                        i.severity,
                        i.environment,
                        i.affected_services,
                        i.correlation_policy,
                        i.correlation_version
                    FROM incidents i
                    {where_clause}
                    ORDER BY i.started_at DESC
                    LIMIT %s
                """
                params.append(max_results)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to dict format
                incidents = []
                for row in rows:
                    incidents.append({
                        "incident_id": row["incident_id"],
                        "title": row["title"],
                        "status": row["status"],
                        "started_at": row["started_at"].isoformat()
                        if row["started_at"]
                        else None,
                        "resolved_at": row["resolved_at"].isoformat()
                        if row["resolved_at"]
                        else None,
                        "severity": row["severity"],
                        "environment": row["environment"],
                        "affected_services": row["affected_services"],
                        "correlation_policy": row["correlation_policy"],
                        "correlation_version": row["correlation_version"],
                    })

                self._circuit_breaker.record_success()
                observe_tool_call_duration("related_incidents", time.time() - start_time)
                increment_tool_calls("related_incidents", "success")

                return {
                    "status": "success",
                    "data": incidents,
                }

        except psycopg2.Error as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("related_incidents", time.time() - start_time)
            increment_tool_calls("related_incidents", "error")
            logger.error("Related incidents query failed: %s", e)
            raise RelatedIncidentsError(f"Database query failed: {e}")
        except (KeyError, ValueError, TypeError) as e:
            observe_tool_call_duration("related_incidents", time.time() - start_time)
            increment_tool_calls("related_incidents", "error")
            logger.error("Related incidents query failed: %s", e)
            raise RelatedIncidentsError(f"Related incidents query failed: {e}")

    def _unavailable_response(
        self, reason: str = "related_incidents_not_available"
    ) -> dict[str, Any]:
        """Return standardized unavailable response."""
        return {
            "status": "unavailable",
            "reason_code": reason,
            "data": [],
        }

    async def find_by_service_and_time(
        self,
        service: str,
        start_time: str,
        end_time: str,
        environment: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Find incidents for a service within a time window.

        Args:
            service: Service name
            start_time: Start timestamp (RFC3339)
            end_time: End timestamp (RFC3339)
            environment: Optional environment filter
            limit: Maximum results

        Returns:
            Dict with status and data list of incidents
        """
        query_start = time.time()

        if not service or not service.strip():
            observe_tool_call_duration(
                "related_incidents", time.time() - query_start
            )
            increment_tool_calls("related_incidents", "unavailable")
            return self._unavailable_response("invalid_service_name")

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            observe_tool_call_duration(
                "related_incidents", time.time() - query_start
            )
            increment_tool_calls("related_incidents", "circuit_open")
            logger.warning("Related incidents circuit breaker open, rejecting time query")
            return self._unavailable_response("circuit_breaker_open")

        max_results = min(limit or self.config.max_results, self.config.max_results)

        conn = self._get_connection()

        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                where_conditions = [
                    "%s = ANY(i.affected_services)",
                    "i.started_at >= %s",
                    "i.started_at <= %s",
                ]
                params = [service.strip(), start_dt, end_dt]

                if environment:
                    where_conditions.append("i.environment = %s")
                    params.append(environment.strip())

                where_clause = "WHERE " + " AND ".join(where_conditions)

                query = f"""
                    SELECT
                        i.id as incident_id,
                        i.title,
                        i.status,
                        i.started_at,
                        i.resolved_at,
                        i.severity,
                        i.environment,
                        i.affected_services,
                        i.correlation_policy,
                        i.correlation_version
                    FROM incidents i
                    {where_clause}
                    ORDER BY i.started_at DESC
                    LIMIT %s
                """
                params.append(max_results)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                incidents = []
                for row in rows:
                    incidents.append({
                        "incident_id": row["incident_id"],
                        "title": row["title"],
                        "status": row["status"],
                        "started_at": row["started_at"].isoformat()
                        if row["started_at"]
                        else None,
                        "resolved_at": row["resolved_at"].isoformat()
                        if row["resolved_at"]
                        else None,
                        "severity": row["severity"],
                        "environment": row["environment"],
                        "affected_services": row["affected_services"],
                        "correlation_policy": row["correlation_policy"],
                        "correlation_version": row["correlation_version"],
                    })

                self._circuit_breaker.record_success()
                observe_tool_call_duration(
                    "related_incidents", time.time() - query_start
                )
                increment_tool_calls("related_incidents", "success")

                return {
                    "status": "success",
                    "data": incidents,
                }

        except psycopg2.Error as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration(
                "related_incidents", time.time() - query_start
            )
            increment_tool_calls("related_incidents", "error")
            logger.error("Related incidents time-range query failed: %s", e)
            raise RelatedIncidentsError(f"Database query failed: {e}")
        except (KeyError, ValueError, TypeError) as e:
            observe_tool_call_duration(
                "related_incidents", time.time() - query_start
            )
            increment_tool_calls("related_incidents", "error")
            logger.error("Related incidents time-range query failed: %s", e)
            raise RelatedIncidentsError(f"Related incidents query failed: {e}")
