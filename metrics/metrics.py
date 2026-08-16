"""Metrics for the Investigator service."""

from prometheus_client import Counter, Gauge, Histogram

# Counters
investigations_total = Counter(
    "investigator_investigations_total",
    "Total number of investigations started",
)

tool_calls_total = Counter(
    "investigator_tool_calls_total",
    "Total number of tool calls made",
    ["tool", "status"],
)

findings_total = Counter(
    "investigator_findings_total",
    "Total number of findings generated",
    ["type"],
)

db_connection_errors = Counter(
    "investigator_db_connection_errors_total",
    "Total number of database connection errors",
)

# Histograms
loop_duration = Histogram(
    "investigator_loop_duration_seconds",
    "Investigation loop duration in seconds",
)

tool_call_duration = Histogram(
    "investigator_tool_call_duration_seconds",
    "Tool call duration in seconds",
    ["tool"],
)

context_truncations = Histogram(
    "investigator_context_truncations_total",
    "Number of context truncations",
)

# Gauges
investigator_running = Gauge(
    "investigator_running",
    "Whether investigator is currently running (1) or not (0)",
)

investigator_uptime = Gauge(
    "investigator_uptime_seconds",
    "Uptime of the investigator in seconds",
)


def increment_investigations():
    investigations_total.inc()


def increment_tool_calls(tool: str, status: str):
    tool_calls_total.labels(tool=tool, status=status).inc()


def increment_findings(finding_type: str):
    findings_total.labels(type=finding_type).inc()


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
