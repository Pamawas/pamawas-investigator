"""Grafana managed-alerting condition-state adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from adapters.circuit_breaker import CircuitBreaker
from adapters.condition import ConditionEvaluation, ConditionState
from models import ConditionReference


class GrafanaAlertingError(Exception):
    """Grafana alert-state request or response was not safely evaluable."""


@dataclass
class GrafanaAlertingConfig:
    base_url: str
    api_token: str
    org_id: str = "1"
    timeout_seconds: float = 10.0
    max_response_bytes: int = 1_000_000
    max_alerts: int = 1000
    max_rules: int = 1000
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_calls: int = 3

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("Grafana base_url is required")
        if not self.api_token:
            raise ValueError("Grafana API token is required")
        if not self.org_id:
            raise ValueError("Grafana org_id is required")
        if self.timeout_seconds <= 0:
            raise ValueError("Grafana timeout must be positive")
        if self.max_response_bytes <= 0 or self.max_alerts <= 0 or self.max_rules <= 0:
            raise ValueError("Grafana response bounds must be positive")
        self.base_url = self.base_url.rstrip("/")


class GrafanaAlertingAdapter:
    """Read exact current rule and alert-instance state from Grafana."""

    _RULES_PATH = "/api/prometheus/grafana/api/v1/rules"
    _ALERTS_PATH = "/api/alertmanager/grafana/api/v2/alerts"

    def __init__(
        self,
        config: GrafanaAlertingConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._circuit_breaker = CircuitBreaker(
            name="grafana_alerting",
            failure_threshold=config.circuit_breaker_failure_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Accept": "application/json",
                    "X-Grafana-Org-Id": self.config.org_id,
                },
                timeout=httpx.Timeout(self.config.timeout_seconds),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def circuit_breaker_state(self) -> dict[str, Any]:
        return self._circuit_breaker.snapshot()

    async def _get_json(self, path: str, params: dict[str, str]) -> Any:
        client = await self._get_client()
        response = await client.get(path, params=params)
        response.raise_for_status()
        if len(response.content) > self.config.max_response_bytes:
            raise GrafanaAlertingError("Grafana response exceeded the configured byte limit")
        try:
            return response.json()
        except ValueError as exc:
            raise GrafanaAlertingError("Grafana returned invalid JSON") from exc

    @staticmethod
    def _rule_uid(rule: dict[str, Any]) -> str:
        for key in ("uid", "ruleUID", "ruleUid"):
            value = rule.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        labels = rule.get("labels")
        if isinstance(labels, dict):
            value = labels.get("__alert_rule_uid__")
            if isinstance(value, str):
                return value.strip()
        return ""

    def _matching_rules(self, payload: Any, resource_id: str) -> list[dict[str, Any]]:
        data = payload.get("data") if isinstance(payload, dict) else None
        groups = data.get("groups") if isinstance(data, dict) else None
        if not isinstance(groups, list):
            raise GrafanaAlertingError("Grafana rules response has no groups array")
        rules: list[dict[str, Any]] = []
        total = 0
        for group in groups:
            if not isinstance(group, dict):
                raise GrafanaAlertingError("Grafana rules response contains an invalid group")
            raw_rules = group.get("rules")
            if not isinstance(raw_rules, list):
                raise GrafanaAlertingError("Grafana rule group has no rules array")
            total += len(raw_rules)
            if total > self.config.max_rules:
                raise GrafanaAlertingError("Grafana rules response exceeded the rule limit")
            if any(not isinstance(rule, dict) for rule in raw_rules):
                raise GrafanaAlertingError("Grafana rules response contains an invalid rule")
            rules.extend(rule for rule in raw_rules if self._rule_uid(rule) == resource_id)
        return rules

    @staticmethod
    def _alert_rule_uid(alert: dict[str, Any]) -> str:
        labels = alert.get("labels")
        if isinstance(labels, dict):
            value = labels.get("__alert_rule_uid__")
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("ruleUID", "ruleUid", "rule_uid"):
            value = alert.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _active_alerts(self, payload: Any) -> list[dict[str, Any]]:
        raw = payload
        if isinstance(payload, dict):
            raw = payload.get("data")
        if not isinstance(raw, list):
            raise GrafanaAlertingError("Grafana alerts response is not an array")
        if len(raw) > self.config.max_alerts:
            raise GrafanaAlertingError("Grafana alerts response exceeded the alert limit")
        if any(not isinstance(alert, dict) for alert in raw):
            raise GrafanaAlertingError("Grafana alerts response contains an invalid alert")
        return raw

    @staticmethod
    def _normalized_rule_state(rule: dict[str, Any]) -> ConditionState:
        health = str(rule.get("health") or "").strip().lower().replace("-", "_")
        state = str(rule.get("state") or "").strip().lower().replace("-", "_")
        if health in {"error", "err"} or state in {"error", "err"}:
            return ConditionState.ERROR
        if health in {"nodata", "no_data", "no data"} or state in {
            "nodata",
            "no_data",
            "no data",
        }:
            return ConditionState.NO_DATA
        if health and health not in {"ok", "healthy"}:
            return ConditionState.UNKNOWN
        if state in {"firing", "alerting", "active"}:
            return ConditionState.FIRING
        if state in {"pending"}:
            return ConditionState.PENDING
        if state in {"normal", "inactive", "ok", "resolved"}:
            # Absence from the active-alert set is recovery evidence only when
            # the rule evaluation itself explicitly reports healthy.
            return (
                ConditionState.NORMAL
                if health in {"ok", "healthy"}
                else ConditionState.UNKNOWN
            )
        return ConditionState.UNKNOWN

    @staticmethod
    def _evaluation(
        reference: ConditionReference,
        state: ConditionState,
        reason_code: str,
        detail: str,
        *,
        authoritative: bool = True,
    ) -> ConditionEvaluation:
        return ConditionEvaluation(
            state=state,
            source=reference.source,
            evaluator=reference.evaluator,
            reason_code=reason_code,
            detail=detail,
            authoritative=authoritative,
            observed_at=datetime.now(UTC),
        )

    @staticmethod
    def _reference_error(
        reference: ConditionReference, expected_org_id: str
    ) -> ConditionEvaluation | None:
        if reference.resource_type != "alert_rule":
            return ConditionEvaluation.unknown(
                reference,
                "unsupported_resource_type",
                "Grafana managed alerting requires an alert_rule resource.",
            )
        if reference.tenant_id != expected_org_id:
            return ConditionEvaluation.unknown(
                reference,
                "tenant_mismatch",
                "The condition tenant does not match the configured Grafana organization.",
            )
        return None

    def _match_alert_instances(
        self,
        reference: ConditionReference,
        alerts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ConditionEvaluation | None]:
        unexpected_alert = next(
            (
                alert
                for alert in alerts
                if self._alert_rule_uid(alert) != reference.resource_id
                or not str(alert.get("fingerprint") or "").strip()
            ),
            None,
        )
        if unexpected_alert is not None:
            error = ConditionEvaluation.unknown(
                reference,
                "invalid_alert_instance",
                "Grafana returned an alert that could not be matched exactly "
                "to the rule and instance.",
            )
            return [], [], error
        same_rule = [
            alert
            for alert in alerts
            if self._alert_rule_uid(alert) == reference.resource_id
        ]
        exact = [
            alert
            for alert in same_rule
            if str(alert.get("fingerprint") or "").strip() == reference.instance_id
        ]
        if len(exact) > 1:
            error = ConditionEvaluation.unknown(
                reference,
                "ambiguous_alert_instance",
                "Grafana returned more than one alert with the same instance identity.",
            )
            return same_rule, exact, error
        return same_rule, exact, None

    def _evaluate_matched_state(
        self,
        reference: ConditionReference,
        rule_state: ConditionState,
        same_rule: list[dict[str, Any]],
        exact: list[dict[str, Any]],
    ) -> ConditionEvaluation:
        if rule_state in {ConditionState.ERROR, ConditionState.NO_DATA}:
            return self._evaluation(
                reference,
                rule_state,
                f"rule_{rule_state.value}",
                f"Grafana reports the rule state as {rule_state.value}.",
            )

        if exact:
            if rule_state == ConditionState.NORMAL:
                return ConditionEvaluation.unknown(
                    reference,
                    "conflicting_grafana_state",
                    "Grafana returned the instance as active while the rule state was normal.",
                )
            state = (
                ConditionState.PENDING
                if rule_state == ConditionState.PENDING
                else ConditionState.FIRING
            )
            return self._evaluation(
                reference,
                state,
                f"instance_{state.value}",
                f"Grafana reports the exact alert instance as {state.value}.",
            )

        if rule_state == ConditionState.NORMAL:
            return self._evaluation(
                reference,
                ConditionState.NORMAL,
                "instance_normal",
                "Grafana reports the rule as normal and the exact instance is not active.",
            )
        if rule_state == ConditionState.PENDING:
            return self._evaluation(
                reference,
                ConditionState.PENDING,
                "rule_pending",
                "Grafana reports the rule as pending, so recovery is not established.",
            )
        if rule_state == ConditionState.FIRING and same_rule:
            return self._evaluation(
                reference,
                ConditionState.NORMAL,
                "instance_resolved_other_instances_firing",
                "The exact instance is absent from Grafana's active set while other "
                "instances of the rule remain firing.",
            )
        if rule_state == ConditionState.FIRING:
            return ConditionEvaluation.unknown(
                reference,
                "inconsistent_active_alert_set",
                "Grafana reports the rule as firing but returned no active instance for it.",
            )
        return ConditionEvaluation.unknown(
            reference,
            "unknown_rule_state",
            "Grafana returned an unsupported rule state.",
        )

    async def evaluate(self, reference: ConditionReference) -> ConditionEvaluation:
        if error := self._reference_error(reference, self.config.org_id):
            return error
        if not self._circuit_breaker.allow_call():
            return ConditionEvaluation.unknown(
                reference,
                "circuit_breaker_open",
                "The Grafana alert-state adapter is temporarily unavailable.",
            )

        escaped_uid = reference.resource_id.replace("\\", "\\\\").replace('"', '\\"')
        try:
            rules_payload, alerts_payload = await asyncio.gather(
                self._get_json(self._RULES_PATH, {"rule_uid": reference.resource_id}),
                self._get_json(
                    self._ALERTS_PATH,
                    {
                        "active": "true",
                        "silenced": "true",
                        "inhibited": "true",
                        "filter": f'__alert_rule_uid__="{escaped_uid}"',
                    },
                ),
            )
            rules = self._matching_rules(rules_payload, reference.resource_id)
            alerts = self._active_alerts(alerts_payload)
        except (
            httpx.HTTPError,
            GrafanaAlertingError,
            TimeoutError,
            OSError,
        ) as exc:
            self._circuit_breaker.record_failure()
            return ConditionEvaluation.unknown(
                reference,
                "grafana_api_unavailable",
                f"Grafana alert state could not be read ({type(exc).__name__}).",
            )

        self._circuit_breaker.record_success()
        if len(rules) != 1:
            reason = "rule_not_found" if not rules else "ambiguous_rule"
            return ConditionEvaluation.unknown(
                reference,
                reason,
                "Grafana did not return exactly one matching alert rule.",
            )

        rule_state = self._normalized_rule_state(rules[0])
        same_rule, exact, error = self._match_alert_instances(reference, alerts)
        if error is not None:
            return error
        return self._evaluate_matched_state(reference, rule_state, same_rule, exact)
