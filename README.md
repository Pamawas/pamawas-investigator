# pamawas-investigator

**Bounded Tool-Calling LLM Investigation Engine** — OpenAI-compatible, evidence-based RCA

Language: Python 3.11+

## Purpose

Orchestrates the investigation of incidents using an LLM with bounded tool access to observability data (Prometheus metrics, Loki logs, deployment history, related incidents). Produces structured findings with evidence classification (FACT, LIKELY_CAUSE, HYPOTHESIS, UNKNOWN) and confidence scores. This is the AI investigation engine from MVP §6.

## MVP Reference

- **MVP §10 Build Order #3**: Investigator — bounded tool-calling loop against curated incident context
- **MVP §6 Investigation Engine — Bounded Tool-Calling (v1.1)**: Tools, loop mechanics, system prompt intent, evidence classification
- **MVP §7 LLM Provider — OpenAI-Compatible**: Single thin client, config-driven vendor swapping
- **MVP §8 Architecture Overview**: Investigator (Python, bounded tool-calling LLM) component

## Responsibilities

- Load incident context from PostgreSQL (incident + events)
- Run bounded tool-calling investigation loop (MAX_TOOL_CALLS=6)
- Provide tools: `query_prometheus`, `query_loki`, `get_recent_deployments`, `get_related_incidents`, `submit_findings`
- Force `submit_findings` on final turn for structured output
- Persist evidence to `evidence` table with classification and confidence
- Full audit logging of all tool calls (tool + args + result + duration)
- Context truncation (8KB cap) to control growth across turns

## Investigation Tools (MVP §6)

| Tool | Parameters | Purpose |
|------|------------|---------|
| `query_prometheus` | `promql`, `start`, `end` | Query metrics from Prometheus |
| `query_loki` | `logql`, `start`, `end`, `limit` | Query logs from Loki |
| `get_recent_deployments` | `service`, `start`, `end` | Check recent deployments (placeholder for MVP) |
| `get_related_incidents` | `service`, `symptom_keywords` | Find historical incidents from own DB |
| `submit_findings` | `findings[]` | **Required** — forces structured output |

## Evidence Classification (MVP §6)

| Type | Description | Confidence Range |
|------|-------------|------------------|
| `FACT` | Directly observed in metrics/logs | 0.8-1.0 |
| `LIKELY_CAUSE` | Strongly supported by evidence, not certain | 0.6-0.9 |
| `HYPOTHESIS` | Plausible, unverified | 0.3-0.7 |
| `UNKNOWN` | Insufficient evidence; stated explicitly | 0.0-0.3 |

## System Prompt Intent (MVP §6)

The investigator is instructed to:
1. Understand the symptom and blast radius
2. Find the first abnormal signal, not just the loudest alert
3. Check recent changes near that time
4. Check dependencies
5. Form and test competing hypotheses rather than confirming one
6. Use tool calls deliberately — no duplicate queries, no unused fetches
7. Prefer `UNKNOWN` over a fabricated-sounding conclusion when evidence is insufficient

## Configuration (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key | Required |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `PROMETHEUS_URL` | Prometheus HTTP endpoint | `http://prometheus:9090` |
| `LOKI_URL` | Loki HTTP endpoint | `http://loki:3100` |
| `DEPLOYMENTS_URL` | Deployments service URL | `http://deployments:8080` |
| `RELATED_INCIDENTS_URL` | Related incidents service URL | `http://related-incidents:8080` |
| `MAX_TOOL_CALLS` | Hard cap on tool rounds | `6` |
| `LOG_LEVEL` | Log level | `info` |

## Database Schema (from pamawas-schema)

```sql
-- Evidence table (written by investigator)
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('fact', 'likely_cause', 'hypothesis', 'unknown')),
    content TEXT NOT NULL,
    source TEXT,
    confidence DOUBLE PRECISION CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Incidents & events (read by investigator)
-- Defined in pamawas-schema migrations
```

## LLM Provider Flexibility (MVP §7)

The investigator uses a thin OpenAI-compatible client that works with:
- **OpenAI** — `base_url: https://api.openai.com/v1`, `model: gpt-4o-mini`
- **Ollama** — `base_url: http://localhost:11434/v1`, `api_key: ollama`, `model: llama3.1:70b`
- **vLLM** — `base_url: http://localhost:8000/v1`, `model: meta-llama/Llama-3.1-70B-Instruct`
- **OpenRouter** — `base_url: https://openrouter.ai/api/v1`, `model: anthropic/claude-sonnet-4.5`

No code changes needed — only config.

## Current Implementation Status

- ✅ OpenAI-compatible client (`InvestigatorLLM`)
- ✅ All 5 tool implementations (Prometheus, Loki, Deployments, Related Incidents, Submit Findings)
- ✅ Investigation loop with MAX_TOOL_CALLS=6 cap
- ✅ Forced `submit_findings` on final turn
- ✅ Context truncation (8KB)
- ✅ Full tool call audit logging
- ✅ Evidence classification (FACT, LIKELY_CAUSE, HYPOTHESIS, UNKNOWN)
- ✅ PostgreSQL persistence for evidence
- ✅ Incident context loading from DB
- ✅ Requirements.txt with dependencies
- ✅ Dockerfile (Python base)
- ✅ GitHub Actions workflow (main + dev branches, GHCR publishing)
- ⬜ Real Prometheus/Loki HTTP clients (currently mocked)
- ⬜ Structured JSON logging
- ⬜ Configuration management (YAML + ENV)
- ⬜ Unit tests for tool functions (target 80%+ coverage)

## Kanban Tasks

- `t_d089def7` — Design bounded tool-calling LLM agent architecture (architect)
- `t_8ab53d61` — Implement core tool-calling loop with OpenAI-compatible client (python-dev)
- `t_1e52c0c7` — Implement Prometheus and Loki query tools (python-dev)
- `t_dbe23b41` — Implement deployment and related incidents tools (python-dev)
- `t_c8a4e4ee` — Write unit tests for tool functions (qa-dev)

## Dependencies

- **PostgreSQL** — incidents, events, evidence tables (via pamawas-schema)
- **pamawas-schema** — Shared types and migrations (parent: `t_d1cdd7a9`)
- **pamawas-correlator** — Produces incidents to investigate
- **Prometheus** — Metrics queries (tool dependency)
- **Loki** — Log queries (tool dependency)
- **Deployments service** — Deployment history (placeholder for MVP)

## Build & Run

```bash
# Local development
pip install -r requirements.txt
python main.py

# Docker
docker build -t pamawas-investigator .
docker run -e DATABASE_URL="postgres://..." \
  -e LLM_API_KEY="..." \
  -e LLM_MODEL="gpt-4o-mini" \
  -e PROMETHEUS_URL="http://prometheus:9090" \
  -e LOKI_URL="http://loki:3100" \
  pamawas-investigator
```

## Running an Investigation

```bash
# The investigator runs as a service with HTTP endpoints
# Trigger investigation via API or it can be called programmatically

# Example: investigate incident via API (when endpoint added)
curl -X POST http://localhost:8080/investigate \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "inc_123"}'
```

## Example Investigation Flow

```
Incident: "High API latency in payment-api" (183 alerts → 1 incident)

Turn 1: LLM queries Prometheus for payment-api latency metrics
Turn 2: LLM queries Loki for payment-api error logs
Turn 3: LLM checks recent deployments for payment-api
Turn 4: LLM queries related incidents for "latency" + "payment-api"
Turn 5: LLM forms hypothesis, queries more specific metrics
Turn 6 (forced): LLM calls submit_findings with structured output:
  - FACT: payment-api p99 latency spiked from 200ms to 5s at 01:47
  - FACT: Database connection pool exhausted (97% utilization)
  - LIKELY_CAUSE: Database connection exhaustion following 01:47 deployment (confidence: 0.87)
  - HYPOTHESIS: Deployment increased connection demand (confidence: 0.65)
  - UNKNOWN: Whether connection leak was introduced in deployment
  - RECOMMENDATION: Review connection pool config, compare deployment DB behavior
```