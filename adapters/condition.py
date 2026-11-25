"""Provider-neutral condition evaluation registry."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from models import ConditionReference

logger = logging.getLogger(__name__)


class ConditionState(StrEnum):
    """Normalized current states returned by condition-source adapters."""

    FIRING = "firing"
    PENDING = "pending"
    NORMAL = "normal"
    NO_DATA = "no_data"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConditionEvaluation:
    """One bounded, auditable evaluation of a condition reference."""

    state: ConditionState
    source: str
    evaluator: str
    reason_code: str
    detail: str
    authoritative: bool
    observed_at: datetime

    @classmethod
    def unknown(
        cls,
        reference: ConditionReference | None,
        reason_code: str,
        detail: str,
    ) -> ConditionEvaluation:
        return cls(
            state=ConditionState.UNKNOWN,
            source=reference.source if reference is not None else "system",
            evaluator=reference.evaluator if reference is not None else "none",
            reason_code=reason_code,
            detail=detail,
            authoritative=False,
            observed_at=datetime.now(timezone.utc),
        )


class ConditionEvaluatorRegistry:
    """Dispatch condition references without source-specific core branches."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], Any] = {}

    def register(self, source: str, evaluator: str, adapter: Any) -> None:
        key = (source.strip(), evaluator.strip())
        if not all(key):
            raise ValueError("condition evaluator source and name are required")
        self._adapters[key] = adapter

    async def evaluate(
        self, reference: ConditionReference | None
    ) -> ConditionEvaluation:
        if reference is None:
            return ConditionEvaluation.unknown(
                None,
                "missing_condition_reference",
                "No typed condition reference was persisted for this incident.",
            )
        adapter = self._adapters.get((reference.source, reference.evaluator))
        if adapter is None:
            return ConditionEvaluation.unknown(
                reference,
                "unsupported_condition_evaluator",
                "No configured adapter supports this condition source and evaluator.",
            )
        try:
            evaluation = await adapter.evaluate(reference)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Condition evaluator %s/%s failed",
                reference.source,
                reference.evaluator,
            )
            return ConditionEvaluation.unknown(
                reference,
                "condition_evaluator_failure",
                f"The condition evaluator failed with {type(exc).__name__}.",
            )
        if not isinstance(evaluation, ConditionEvaluation):
            return ConditionEvaluation.unknown(
                reference,
                "invalid_condition_evaluation",
                "The configured adapter returned an invalid evaluation.",
            )
        return evaluation

    async def close(self) -> None:
        adapters = list(dict.fromkeys(self._adapters.values()))
        await asyncio.gather(
            *(adapter.close() for adapter in adapters if hasattr(adapter, "close")),
            return_exceptions=True,
        )

    def circuit_breaker_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for (source, evaluator), adapter in self._adapters.items():
            snapshot = getattr(adapter, "circuit_breaker_state", None)
            if snapshot is not None:
                states[f"{source}/{evaluator}"] = snapshot()
        return states
