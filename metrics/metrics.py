"""Metrics for the Investigator service."""

from prometheus_client import Counter, Gauge, Histogram

# Names follow the pamawas_<subsystem>_<metric> convention (design review
# P3): namespace `pamawas`, subsystem `investigator`.
# Counters
investigations_total = Counter(
    "pamawas_investigator_investigations_total",
    "Total number of investigations started",
)

verify_total = Counter(
    "pamawas_investigator_verify_total",
    "Total number of stale-incident verifications run",
    ["verdict"],
)

verify_duration = Histogram(
    "pamawas_investigator_verify_duration_seconds",
    "Stale-incident verification wall-clock duration in seconds",
)

max_tool_calls_reached = Counter(
    "pamawas_investigator_max_tool_calls_reached_total",
    "Total number of investigations that hit max tool calls without submitting findings",
)

deadline_exceeded = Counter(
    "pamawas_investigator_deadline_exceeded_total",
    "Total number of investigations that ran out of wall-clock budget without submitting findings",
)

forced_submissions = Counter(
    "pamawas_investigator_forced_submissions_total",
    "Total number of investigations whose findings submission was forced by the deadline reserve",
)

tool_calls_total = Counter(
    "pamawas_investigator_tool_calls_total",
    "Total number of tool calls made",
    ["tool", "status"],
)

findings_total = Counter(
    "pamawas_investigator_findings_total",
    "Total number of findings generated",
    ["type"],
)

db_connection_errors = Counter(
    "pamawas_investigator_db_connection_errors_total",
    "Total number of database connection errors",
)

investigation_retry_scheduled_total = Counter(
    "pamawas_investigator_retry_scheduled_total",
    "Total number of investigation retries scheduled after a transient failure",
)

investigation_retry_reclaimed_total = Counter(
    "pamawas_investigator_retry_reclaimed_total",
    "Total number of due failed_retryable runs reclaimed by the retry scheduler",
)

investigation_retry_exhausted_total = Counter(
    "pamawas_investigator_retry_exhausted_total",
    "Total number of investigations whose retry budget was exhausted",
)

investigation_retry_errors_total = Counter(
    "pamawas_investigator_retry_errors_total",
    "Total number of retry scheduler loop errors",
)

# Histograms
loop_duration = Histogram(
    "pamawas_investigator_loop_duration_seconds",
    "Investigation loop duration in seconds",
)

tool_call_duration = Histogram(
    "pamawas_investigator_tool_call_duration_seconds",
    "Tool call duration in seconds",
    ["tool"],
)

context_truncations = Histogram(
    "pamawas_investigator_context_truncations_total",
    "Number of context truncations",
)

# Gauges
investigator_running = Gauge(
    "pamawas_investigator_running",
    "Whether investigator is currently running (1) or not (0)",
)

investigator_uptime = Gauge(
    "pamawas_investigator_uptime_seconds",
    "Uptime of the investigator in seconds",
)


def increment_max_tool_calls_reached():
    max_tool_calls_reached.inc()


def increment_deadline_exceeded():
    deadline_exceeded.inc()


def increment_forced_submission():
    forced_submissions.inc()


def increment_investigations():
    investigations_total.inc()


def increment_verify(verdict: str, duration_seconds: float):
    """Record a stale-incident verification and its wall-clock duration."""
    verify_total.labels(verdict=verdict).inc()
    verify_duration.observe(duration_seconds)


def increment_tool_calls(tool: str, status: str):
    tool_calls_total.labels(tool=tool, status=status).inc()


def increment_findings(finding_type: str):
    findings_total.labels(type=finding_type).inc()


def increment_findings_by_type(findings):
    """Increment findings counter per finding type."""
    for f in findings:
        increment_findings(f.type.value if hasattr(f, "type") else f["type"])


def increment_db_errors():
    db_connection_errors.inc()


def observe_loop_duration(seconds: float):
    loop_duration.observe(seconds)


def observe_tool_call_duration(tool: str, seconds: float):
    tool_call_duration.labels(tool=tool).observe(seconds)


def observe_truncations(count: int):
    context_truncations.observe(count)


def set_running(running: bool):
    investigator_running.set(1 if running else 0)


def set_uptime(seconds: float):
    investigator_uptime.set(seconds)
