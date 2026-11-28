"""Versioned operator actions selected from validated observations."""

from dataclasses import dataclass

from models import IncidentContext, InvestigationPolicy


@dataclass(frozen=True)
class PolicyObservation:
    provider: str
    expression: str
    metric_type: str = ""
    metric_unit: str = ""


def _matches_incident_labels(policy: InvestigationPolicy, context: IncidentContext) -> bool:
    if not policy.labels:
        return True
    labels = [event.get("labels") for event in context.events or []]
    if context.condition_ref is not None:
        labels.append(context.condition_ref.labels)
    return any(
        isinstance(values, dict)
        and all(isinstance(value, str) and values.get(key) == value
                for key, value in policy.labels.items())
        for values in labels
    )


def _matches_observation(policy: InvestigationPolicy, observation: PolicyObservation) -> bool:
    return (
        policy.provider == observation.provider
        and (policy.metric_type is None or policy.metric_type == observation.metric_type)
        and (policy.metric_unit is None or policy.metric_unit == observation.metric_unit)
    )


def select_recommendation_policy(
    context: IncidentContext, observations: list[PolicyObservation]
) -> InvestigationPolicy | None:
    ordered = sorted(context.investigation_policies, key=lambda p: (-p.priority, p.id, -p.version))
    for policy in ordered:
        if not _matches_incident_labels(policy, context):
            continue
        expressions = {
            observation.expression for observation in observations
            if _matches_observation(policy, observation)
        }
        if len(expressions) >= policy.minimum_observations:
            return policy
    return None
