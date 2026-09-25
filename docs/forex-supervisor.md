# Forex shadow supervisor

The supervisor is the operational entry point for the read-only MT5 shadow
collector:

```powershell
python -m cli.forex_supervisor run --db-path data_cache/live-market-clean-20260923.db
```

It uses a dedicated local Ollama endpoint at `127.0.0.1:11435`, with
`qwen3.5:2b` for quick prose nodes, `qwen3.5:4b` for deep/structured nodes,
temperature `0`, a 16,384-token server context, and bounded output settings.
The supervisor verifies model availability and loaded context, prewarms the
models with bounded health requests, then delegates scheduling and MT5 lease
fencing to `forex-watch`. It never kills the normal Ollama tray process.

Status is scalar-only and does not construct MT5 or TradingAgents resources:

```powershell
python -m cli.forex_supervisor status --db-path data_cache/live-market-clean-20260923.db
```

The health levels are `HEALTHY`, `DEGRADED`, and
`OPERATOR_REVIEW_REQUIRED`. Recovery is bounded; stale watcher takeover still
uses the existing lease/process-identity rules. No supervisor path sends an
order, changes prompts/risk settings, rewrites historical decisions, or
regenerates frozen Phase 7/11A artifacts.

Revision-scoped read-only evidence is available with:

```powershell
python -m cli.forex_validate_revision --db-path data_cache/live-market-clean-20260923.db --commit <commit-sha>
```

New watcher rows carry commit, application, prompt/config, collector-contract,
safe-config, and fingerprint provenance. Existing rows are not backfilled.
