from unittest.mock import MagicMock, patch

import metrics.metrics as metrics


def test_metric_helpers_delegate_to_collectors():
    with (
        patch.object(metrics.investigations_total, "inc") as investigations,
        patch.object(metrics.db_connection_errors, "inc") as db_errors,
        patch.object(metrics.loop_duration, "observe") as loop,
        patch.object(metrics.context_truncations, "observe") as truncations,
        patch.object(metrics.investigator_running, "set") as running,
        patch.object(metrics.investigator_uptime, "set") as uptime,
    ):
        metrics.increment_investigations()
        metrics.increment_db_errors()
        metrics.observe_loop_duration(1.5)
        metrics.observe_truncations(2)
        metrics.set_running(True)
        metrics.set_running(False)
        metrics.set_uptime(10)
    investigations.assert_called_once_with()
    db_errors.assert_called_once_with()
    loop.assert_called_once_with(1.5)
    truncations.assert_called_once_with(2)
    assert [call.args[0] for call in running.call_args_list] == [1, 0]
    uptime.assert_called_once_with(10)


def test_labeled_metric_helpers_use_labels():
    tool_child = MagicMock()
    finding_child = MagicMock()
    duration_child = MagicMock()
    with (
        patch.object(metrics.tool_calls_total, "labels", return_value=tool_child) as tool_labels,
        patch.object(
            metrics.findings_total, "labels", return_value=finding_child
        ) as finding_labels,
        patch.object(
            metrics.tool_call_duration, "labels", return_value=duration_child
        ) as duration_labels,
    ):
        metrics.increment_tool_calls("loki", "success")
        metrics.increment_findings("fact")
        metrics.observe_tool_call_duration("loki", 0.25)
    tool_labels.assert_called_once_with(tool="loki", status="success")
    tool_child.inc.assert_called_once_with()
    finding_labels.assert_called_once_with(type="fact")
    finding_child.inc.assert_called_once_with()
    duration_labels.assert_called_once_with(tool="loki")
    duration_child.observe.assert_called_once_with(0.25)
