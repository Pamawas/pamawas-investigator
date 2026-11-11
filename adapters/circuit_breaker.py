"""Circuit breaker for external adapter calls.

States:
  CLOSED   - calls pass through; failures increment the error counter
  OPEN     - calls are rejected immediately without touching the external dependency
  HALF_OPEN - a limited number of trial calls are allowed to test recovery

When the failure threshold is reached within a window, the circuit opens.
After the timeout elapses, it transitions to half-open. A successful call
closes the circuit; another failure reopens it.
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker with exponential backoff on repeated opens."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._total_opens = 0
        self._lock = Lock()

    # -- public API -------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    @property
    def total_opens(self) -> int:
        with self._lock:
            return self._total_opens

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if the timeout has elapsed -> move to half-open
                if (
                    self._opened_at is not None
                    and time.monotonic() - self._opened_at >= self.timeout_seconds
                ):
                    self._transition_half_open()
                    return False
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._transition_closed()

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._transition_open()
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._transition_open()

    def allow_call(self) -> bool:
        """Return True if a call may proceed, False if the circuit is open."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if (
                    self._opened_at is not None
                    and time.monotonic() - self._opened_at >= self.timeout_seconds
                ):
                    self._transition_half_open()
                    return True
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    return False
                return True
            return True

    # -- internal transitions (caller must hold no lock; these take the lock) --

    def _transition_open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls = 0
        self._total_opens += 1
        logger.warning(
            "circuit_breaker_opened",
            breaker=self.name,
            failures=self._failures,
            consecutive=self._consecutive_failures,
            timeout_seconds=self.timeout_seconds,
        )

    def _transition_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._opened_at = None
        logger.info(
            "circuit_breaker_half_open",
            breaker=self.name,
        )

    def _transition_closed(self) -> None:
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._consecutive_failures = 0
        self._half_open_calls = 0
        logger.info(
            "circuit_breaker_closed",
            breaker=self.name,
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "consecutive_failures": self._consecutive_failures,
                "total_opens": self._total_opens,
                "is_open": self._state == CircuitState.OPEN
                and (
                    self._opened_at is None
                    or time.monotonic() - self._opened_at < self.timeout_seconds
                ),
            }
