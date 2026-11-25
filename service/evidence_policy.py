"""Deterministic, provider-neutral causal evidence policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from models import EvidenceSource, EvidenceType, Finding, IncidentContext

_TERM = re.compile(r"[a-z0-9]+")
_QUERY_WINDOW = re.compile(r"\s+\[[^\[\]]+\s+to\s+[^\[\]]+\]\s*$")
_GENERIC_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "being",
        "by",
        "cause",
        "caused",
        "causing",
        "component",
        "condition",
        "could",
        "data",
        "during",
        "event",
        "events",
        "fact",
        "facts",
        "for",
        "from",
        "had",
        "has",
        "have",
        "hour",
        "hours",
        "hypothesis",
        "in",
        "incident",
        "is",
        "it",
        "its",
        "last",
        "latest",
        "likely",
        "line",
        "lines",
        "log",
        "logs",
        "may",
        "mechanism",
        "metric",
        "metrics",
        "might",
        "minute",
        "minutes",
        "observation",
        "observations",
        "of",
        "on",
        "or",
        "peaked",
        "plausible",
        "possibly",
        "query",
        "reported",
        "result",
        "results",
        "second",
        "seconds",
        "service",
        "signal",
        "signals",
        "system",
        "target",
        "than",
        "that",
        "the",
        "their",
        "this",
        "time",
        "timestamp",
        "to",
        "unit",
        "units",
        "value",
        "values",
        "was",
        "were",
        "window",
        "with",
        "without",
        "would",
    }
)
_NON_DIRECT_OBSERVATION_SOURCES = frozenset(
    {
        EvidenceSource.ALERT,
        EvidenceSource.INCIDENTS,
        EvidenceSource.LLM,
        EvidenceSource.SYSTEM,
    }
)


def _term_key(term: str) -> str:
    """Collapse common inflections without a deployment vocabulary."""
    if term.endswith("ies") and len(term) > 5:
        term = f"{term[:-3]}y"
    elif term.endswith("s") and not term.endswith("ss") and len(term) > 5:
        term = term[:-1]
    # Prefix keys align forms such as saturation/saturated while retaining
    # short, meaningful terms such as cpu, oom, queue, and error unchanged.
    return term[:6] if len(term) > 7 else term


def _meaningful_terms(text: str) -> set[str]:
    return {
        _term_key(term)
        for term in _TERM.findall((text or "").lower())
        if len(term) >= 3 and not term.isdigit() and term not in _GENERIC_TERMS
    }


def _premise_terms(context: IncidentContext | None) -> set[str]:
    if context is None:
        return set()
    terms = _meaningful_terms(context.title)
    terms.update(
        {
        term
        for target in context.affected_services or []
        for term in _meaningful_terms(str(target))
        }
    )
    return terms


def _observation_identity(finding: Finding) -> tuple[str, str]:
    detail = _QUERY_WINDOW.sub("", " ".join((finding.source_detail or "").split()))
    return finding.source.value, (detail or " ".join(finding.content.split())).lower()


@dataclass(frozen=True)
class CausalSupport:
    """Relevant, materially distinct observations supporting one mechanism."""

    fact_indexes: tuple[int, ...]
    dimensions: frozenset[str]


@dataclass(frozen=True)
class EvidencePolicy:
    """One causal contract shared by prompt generation and final validation."""

    version: str = "causal-evidence-v1"
    likely_cause_min_facts: int = 2
    likely_cause_min_dimensions: int = 2
    hypothesis_min_facts: int = 1
    likely_cause_min_confidence: float = 0.7
    likely_cause_confidence_cap: float = 0.75
    hypothesis_confidence_cap: float = 0.6
    downgraded_confidence_cap: float = 0.55

    def prompt_contract(self) -> str:
        """Render the exact causal rules enforced by :meth:`validate`."""
        return (
            f"- CAUSAL POLICY {self.version}: the deterministic validator is authoritative.\n"
            f"- LIKELY_CAUSE requires at least {self.likely_cause_min_facts} materially "
            "distinct relevant facts, evidence for multiple parts of the named mechanism, "
            f"and confidence of at least {self.likely_cause_min_confidence:.1f}. The facts "
            "may come from the same provider or different providers; provider count is not "
            "evidence independence.\n"
            f"- HYPOTHESIS requires at least {self.hypothesis_min_facts} relevant fact that "
            "supports a named plausible mechanism but lacks confirmation for LIKELY_CAUSE.\n"
            "- UNKNOWN is required when no collected fact supports a named mechanism. "
            "Recovery, temporal overlap, an empty query, or provider availability alone is "
            "not a mechanism.\n"
            "- A cause can be supported by one provider. Cite the observed signal terms that "
            "connect each supporting fact to the mechanism; the incident title and target "
            "identity are premise/scope, not causal support.\n"
        )

    def support(
        self,
        candidate: Finding,
        facts: list[Finding],
        context: IncidentContext | None = None,
    ) -> CausalSupport:
        """Return distinct facts semantically connected to the candidate mechanism."""
        ignored = _premise_terms(context)
        mechanism_terms = _meaningful_terms(candidate.content) - ignored
        if not mechanism_terms:
            return CausalSupport((), frozenset())

        indexes: list[int] = []
        dimensions: set[str] = set()
        identities: set[tuple[str, str]] = set()
        for index, fact in enumerate(facts):
            if (
                fact.type is not EvidenceType.FACT
                or fact.source in _NON_DIRECT_OBSERVATION_SOURCES
            ):
                continue
            identity = _observation_identity(fact)
            if identity in identities:
                continue
            shared = mechanism_terms.intersection(_meaningful_terms(fact.content) - ignored)
            if not shared:
                continue
            identities.add(identity)
            indexes.append(index)
            dimensions.update(shared)
        return CausalSupport(tuple(indexes), frozenset(dimensions))

    def validate(
        self,
        candidate: Finding,
        facts: list[Finding],
        context: IncidentContext | None = None,
    ) -> Finding | None:
        """Classify one model mechanism using the same rules shown in the prompt."""
        support = self.support(candidate, facts, context)
        supported = len(support.fact_indexes)
        links = list(support.fact_indexes)

        if candidate.type is EvidenceType.LIKELY_CAUSE:
            confirmed = (
                supported >= self.likely_cause_min_facts
                and len(support.dimensions) >= self.likely_cause_min_dimensions
                and float(candidate.confidence) >= self.likely_cause_min_confidence
            )
            if confirmed:
                return replace(
                    candidate,
                    confidence=min(
                        float(candidate.confidence), self.likely_cause_confidence_cap
                    ),
                    supports_evidence=links,
                )
            if supported >= self.hypothesis_min_facts:
                return replace(
                    candidate,
                    type=EvidenceType.HYPOTHESIS,
                    confidence=min(
                        float(candidate.confidence), self.downgraded_confidence_cap
                    ),
                    supports_evidence=links,
                )
            return None

        if candidate.type is EvidenceType.HYPOTHESIS:
            if supported < self.hypothesis_min_facts:
                return None
            return replace(
                candidate,
                confidence=min(float(candidate.confidence), self.hypothesis_confidence_cap),
                supports_evidence=links,
            )
        return None


EVIDENCE_POLICY = EvidencePolicy()
