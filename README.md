# pamawas-investigator

Bounded tool-calling LLM investigation engine, OpenAI-compatible client

Language: Python

## Purpose
Orchestrates the investigation of incidents using an LLM with bounded tool access to observability data (Prometheus, Loki, etc.).

## Responsibilities
- Receive incident context from the correlator
- Plan investigation steps
- Use tools to query Prometheus, Loki, check recent deployments, and find related incidents
- Generate structured findings (facts, likely cause, hypothesis, unknowns, confidence, recommendations)
- Persist investigation audit log

## TODO
- Set up Python environment
- Implement OpenAI-compatible LLM client
- Define tool functions for Prometheus, Loki, deployments, related incidents
- Implement the investigation loop with tool call limits
- Connect to PostgreSQL to read incident context and write evidence/findings
- Add logging and metrics

