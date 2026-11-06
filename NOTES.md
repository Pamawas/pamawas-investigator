# Notes - Investigating LLM Feature

## Investigation Status

### Current State
- Investigating whether LLM feature is working properly
- Need to check: investigator deployment env vars, configmap updates, API connectivity

### Key Config Changes Made
1. **configmap.yml** - Added Prometheus/Loki timeout settings:
   - `PROMETHEUS_TIMEOUT_SECONDS: "30"` (was `PROMETHEUS_TIMEOUT`)
   - `LOKI_TIMEOUT_SECONDS: "30"` (was `LOKI_TIMEOUT`)  
   - `LOKI_MAX_LOG_LINES: "100"` (was `LOKI_MAX_LINES`)

2. **investigator.yml** - Updated env var names to match Python config expectations

### Next Steps
- [ ] Verify configmap applied to cluster
- [ ] Check investigator pod logs for LLM connectivity errors
- [ ] Test with kubectl port-forward if needed