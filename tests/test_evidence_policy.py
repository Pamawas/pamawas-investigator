from models import EvidenceSource, EvidenceType, Finding, IncidentContext
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


def test_prompt_contract_is_rendered_from_enforced_policy_values():
    contract = EVIDENCE_POLICY.prompt_contract()

    assert EVIDENCE_POLICY.version in contract
    assert f"at least {EVIDENCE_POLICY.likely_cause_min_facts}" in contract
    assert "same provider or different providers" in contract
    assert "target identity alone is scope" in contract
