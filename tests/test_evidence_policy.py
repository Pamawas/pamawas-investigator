from models import ConditionReference, EvidenceSource, EvidenceType, Finding, IncidentContext
from service.evidence_policy import EVIDENCE_POLICY


def fact(content: str, source: EvidenceSource, detail: str) -> Finding:
    return Finding(EvidenceType.FACT, content, source, 1.0, detail)


def test_repeated_windows_of_one_signal_are_not_independent_support():
    facts = [
        fact(
            "Vendor request latency increased.",
            EvidenceSource.PROMETHEUS,
            "query_prometheus: vendor_request_latency [10:00 to 10:10]",
        ),
        fact(
            "Vendor request latency remained elevated.",
            EvidenceSource.PROMETHEUS,
            "query_prometheus: vendor_request_latency [10:05 to 10:15]",
        ),
    ]
    candidate = Finding(
        EvidenceType.LIKELY_CAUSE,
        "Vendor request latency caused the condition.",
        EvidenceSource.LLM,
        0.9,
    )

    validated = EVIDENCE_POLICY.validate(candidate, facts)

    assert validated is not None
    assert validated.type is EvidenceType.HYPOTHESIS
    assert validated.supports_evidence == [0]


def test_target_identity_alone_is_scope_not_mechanism_support():
    context = IncidentContext(
        "inc_generic",
        "Vendor request latency",
        affected_services=["alpha-target"],
    )
    facts = [
        fact(
            "Alpha target nightly backup completed successfully.",
            EvidenceSource.LOKI,
            'query_loki: {scope="alpha-target"}',
        )
    ]
    candidate = Finding(
        EvidenceType.HYPOTHESIS,
        "Alpha target worker saturation may have caused request latency.",
        EvidenceSource.LLM,
        0.6,
    )

    assert EVIDENCE_POLICY.validate(candidate, facts, context) is None


def test_incident_title_terms_alone_are_not_mechanism_support():
    context = IncidentContext(
        "inc_generic",
        "Vendor service request latency",
        affected_services=["alpha-target"],
    )
    facts = [
        fact(
            "Vendor service request latency reached five seconds.",
            EvidenceSource.PROMETHEUS,
            "query: vendor_request_latency",
        ),
        fact(
            "Vendor service request bytes changed during the incident.",
            EvidenceSource.PROMETHEUS,
            "query: vendor_request_bytes",
        ),
    ]
    candidate = Finding(
        EvidenceType.LIKELY_CAUSE,
        "A traffic spike overwhelmed the vendor service and caused request latency.",
        EvidenceSource.LLM,
        0.9,
    )

    assert EVIDENCE_POLICY.validate(candidate, facts, context) is None


def test_descriptive_condition_labels_do_not_erase_mechanism_terms():
    context = IncidentContext(
        "inc_generic",
        "Vendor request latency",
        affected_services=["alpha-target"],
        condition_ref=ConditionReference(
            version=1,
            source="generic-alerting",
            evaluator="managed-rules",
            resource_type="rule",
            resource_id="rule-1",
            instance_id="instance-1",
            labels={
                "scope": "alpha-target",
                "rule_name": "Vendor Request Latency",
            },
        ),
    )
    facts = [
        fact(
            "Vendor request bytes increased before request latency.",
            EvidenceSource.PROMETHEUS,
            "query: vendor_request_bytes",
        ),
        fact(
            "Vendor request volume increased during the same interval.",
            EvidenceSource.PROMETHEUS,
            "query: vendor_request_volume",
        ),
    ]
    candidate = Finding(
        EvidenceType.LIKELY_CAUSE,
        "Increased vendor request volume and bytes caused request latency.",
        EvidenceSource.LLM,
        0.9,
    )

    validated = EVIDENCE_POLICY.validate(candidate, facts, context)

    assert validated is not None
    assert validated.type is EvidenceType.LIKELY_CAUSE
    assert validated.supports_evidence == [0, 1]


def test_prompt_contract_is_rendered_from_enforced_policy_values():
    contract = EVIDENCE_POLICY.prompt_contract()

    assert EVIDENCE_POLICY.version in contract
    assert f"at least {EVIDENCE_POLICY.likely_cause_min_facts}" in contract
    assert "same provider or different providers" in contract
    assert "incident title and target identity" in contract
