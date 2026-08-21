# pamawas-investigator

**Bounded Tool-Calling LLM Investigation Engine** — Investigates incidents using an LLM with read-only access to Prometheus, Loki, deployment history, and past incidents. Produces evidence-typed findings.

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com/)

---

## Purpose

Orchestrates AI-powered root cause analysis for correlated incidents. The investigator runs a **bounded tool-calling loop** (max 6 tool calls) against curated incident context, using real observability tools to gather evidence and produce structured findings.

## Investigation Tools

| Tool | Parameters | Purpose |
|------|------------|---------|
| `query_prometheus` | `promql`, `start`, `end` | Query metrics from Prometheus |
| `query_loki` | `logql`, `start`, `end`, `limit` | Query logs from Loki |
| `get_recent_deployments` | `service`, `start`, `end` | Check recent deployments (placeholder) |
| `get_related_incidents` | `service`, `symptom_keywords` | Find historical incidents from DB |
| `submit_findings` | `findings[]` | **Required** — forces structured output on final turn |

## Evidence Classification

Every finding is typed with a confidence score:

| Type | Description | Confidence Range |
|------|-------------|------------------|
| `FACT` | Directly observed in metrics/logs | 0.8–1.0 |
| `LIKELY_CAUSE` | Strongly supported, not certain | 0.6–0.9 |
| `HYPOTHESIS` | Plausible, unverified | 0.3–0.7 |
| `UNKNOWN` | Insufficient evidence | 0.0–0.3 |

## Key Principles

- **Read-only** — Zero autonomous remediation
- **Honest uncertainty** — Prefers `UNKNOWN` over fabricated conclusions
- **Bounded** — Hard cap of 6 tool calls per investigation
- **Auditable** — Full tool call logging (tool + args + result + duration)
- **Context-aware** — 8KB context truncation cap to control growth

## Quick Start

```bash
# Docker
docker run -e DATABASE_URL="postgres://user:pass@host:5432/db" \
  -e LLM_API_KEY="sk-..." \
  -e LLM_MODEL="gpt-4o-mini" \
  -e PROMETHEUS_URL="http://prometheus:9090" \
  -e LOKI_URL="http://loki:3100" \
  -e OTEL_EXPORTER_OTLP_ENDPOINT="tempo:4317" \
  ghcr.io/yoganovvaindra/pamawas-investigator:latest

# Local development
pip install -r requirements.txt
python main.py
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | **Required** |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key | **Required** |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `PROMETHEUS_URL` | Prometheus HTTP endpoint | `http://prometheus:9090` |
| `LOKI_URL` | Loki HTTP endpoint | `http://loki:3100` |
| `DEPLOYMENTS_URL` | Deployments service URL | `http://deployments:8080` |
| `MAX_TOOL_CALLS` | Hard cap on tool rounds | `6` |
| `LOG_LEVEL` | debug, info, warn, error | `info` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Tempo OTLP gRPC endpoint | `tempo:4317` |

## LLM Provider Flexibility

Works with any OpenAI-compatible API — no code changes needed:

- **OpenAI** — `base_url: https://api.openai.com/v1`, `model: gpt-4o-mini`
- **Ollama** — `base_url: http://localhost:11434/v1`, `api_key: ollama`, `model: llama3.1:70b`
- **vLLM** — `base_url: http://localhost:8000/v1`, `model: meta-llama/Llama-3.1-70B-Instruct`
- **OpenRouter** — `base_url: https://openrouter.ai/api/v1`, `model: anthropic/claude-sonnet-4.5`

## Database Schema

```sql
-- Evidence (written by investigator)
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('fact','likely_cause','hypothesis','unknown')),
    content TEXT NOT NULL,
    source TEXT,
    confidence DOUBLE PRECISION CHECK (confidence >= 0.0 AND confidence <= 1.0)
);
```

## Observability

| Feature | Endpoint |
|---------|----------|
| Prometheus Metrics | `/metrics` — `investigator_investigations_total`, `investigator_tool_calls_total`, `investigator_findings_total`, `investigator_loop_duration_seconds` |
| JSON Logging | stdout — trace_id, span_id, service, method, path, status_code, duration_ms |
| OpenTelemetry | OTLP gRPC → Tempo:4317 |

## Building

```bash
docker build -t pamawas-investigator .
# Or install locally
pip install -r requirements.txt
```

## Related

- **Root README**: [../README.md](../README.md)
- **Correlator**: [../pamawas-correlator/README.md](../pamawas-correlator/README.md)
- **Reporter**: [../pamawas-reporter/README.md](../pamawas-reporter/README.md)
- **Database Schema**: [../pamawas-schema/README.md](../pamawas-schema/README.md)