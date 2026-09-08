"""Standalone MT5 forex shadow-mode command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from cli.stats_handler import StatsCallbackHandler
from tradingagents.forex.runner import ForexShadowRunner

_FOREX_ANALYSTS = frozenset({"market", "news"})


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_analysts(raw: str) -> tuple[str, ...]:
    if not isinstance(raw, str):
        raise ValueError("analysts must be a comma-separated list")
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("at least one forex analyst is required")
    invalid = tuple(name for name in names if name not in _FOREX_ANALYSTS)
    if invalid:
        raise ValueError(
            "forex shadow mode only supports market/news analysts; "
            f"unsupported: {', '.join(invalid)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("analysts must not contain duplicates")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-shadow",
        description="Read-only MT5 forex analysis in explicit shadow mode.",
    )
    parser.add_argument("--symbol", default="EURUSD", help="requested forex pair")
    parser.add_argument(
        "--count",
        type=_positive_int,
        default=100,
        help="number of bars per timeframe (must be positive)",
    )
    parser.add_argument(
        "--analysis-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="analysis date in ISO format (defaults to UTC today)",
    )
    parser.add_argument(
        "--terminal-path",
        default=None,
        help="optional local MetaTrader 5 terminal executable path",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite path for persisted shadow decisions",
    )
    parser.add_argument(
        "--analysts",
        default="market,news",
        help="comma-separated forex-safe analysts (default: market,news)",
    )
    parser.add_argument(
        "--analysis-profile",
        default="INTRADAY",
        help="bounded forex analysis profile (default: INTRADAY)",
    )

    # Non-secret runtime overrides. Credentials remain in the provider/host
    # configuration and are never accepted as command-line arguments.
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument(
        "--quick-model", "--quick-think-llm", dest="quick_model", default=None
    )
    parser.add_argument(
        "--deep-model", "--deep-think-llm", dest="deep_model", default=None
    )
    parser.add_argument(
        "--backend-url", "--llm-backend-url", dest="backend_url", default=None
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=_positive_int, default=None)
    return parser


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "llm_provider": args.llm_provider,
        "quick_think_llm": args.quick_model,
        "deep_think_llm": args.deep_model,
        "backend_url": args.backend_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    return {key: value for key, value in values.items() if value is not None}


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value) if value is not None else "unknown"


def _format_raw_result(decision: Any) -> str:
    raw_json = getattr(decision, "raw_portfolio_manager_result_json", None)
    if isinstance(raw_json, str) and raw_json:
        return raw_json
    raw_result = getattr(decision, "raw_portfolio_manager_result", None)
    if raw_result is None:
        return "unavailable"
    try:
        return json.dumps(raw_result, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(raw_result)


def _metric(metrics: Mapping[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return value if value is not None else "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parsed_date = date.fromisoformat(args.analysis_date)
        analysts = _parse_analysts(args.analysts)
    except (TypeError, ValueError) as exc:
        print(f"FOREX SHADOW ERROR: {exc}", file=sys.stderr)
        return 2

    # Keep the safety boundary visible before provider/graph construction.
    print("MT5 FOREX — SHADOW MODE")
    print("NO ORDER WILL BE SENT")

    stats_handler = StatsCallbackHandler()
    try:
        runner = ForexShadowRunner(config=_runtime_config(args) or None)
        result = runner.run(
            symbol=args.symbol,
            analysis_date=parsed_date,
            count=args.count,
            terminal_path=args.terminal_path,
            db_path=args.db_path,
            analysts=analysts,
            callbacks=[stats_handler],
            analysis_profile=args.analysis_profile,
        )
    except Exception as exc:  # CLI boundary: preserve a concise non-zero error
        print(f"FOREX SHADOW ERROR: {exc}", file=sys.stderr)
        return 1

    decision = result.decision
    if getattr(decision, "executed", False) is not False:
        print("FOREX SHADOW ERROR: executed invariant was violated", file=sys.stderr)
        return 1
    metrics = result.metrics if isinstance(getattr(result, "metrics", None), Mapping) else {}
    database_path = args.db_path or getattr(getattr(runner, "store", None), "path", "configured default")
    print("SHADOW DECISION RECORDED")
    print(f"DECISION ID: {decision.decision_id}")
    print(f"LLM PROVIDER: {getattr(decision, 'llm_provider', None) or 'unknown'}")
    print(f"QUICK MODEL: {getattr(decision, 'quick_model', None) or 'unknown'}")
    print(f"DEEP MODEL: {getattr(decision, 'deep_model', None) or 'unknown'}")
    print(f"REQUESTED SYMBOL: {getattr(decision, 'requested_symbol', None) or 'unknown'}")
    print(f"RESOLVED SYMBOL: {getattr(decision, 'resolved_symbol', None) or 'unknown'}")
    print(f"SNAPSHOT TIMESTAMP: {_format_timestamp(getattr(decision, 'snapshot_timestamp', None))}")
    print(f"BID: {getattr(decision, 'reference_bid', 'unknown')}")
    print(f"ASK: {getattr(decision, 'reference_ask', 'unknown')}")
    print(f"SPREAD: {getattr(decision, 'spread', 'unknown')}")
    print(f"SPREAD POINTS: {getattr(decision, 'spread_points', 'unknown')}")
    print(f"ANALYSIS PROFILE: {getattr(decision, 'analysis_profile', args.analysis_profile)}")
    print(f"VALID FOR SECONDS: {getattr(decision, 'valid_for_seconds', None) or 'unknown'}")
    print(f"VALID UNTIL: {_format_timestamp(getattr(decision, 'valid_until', None))}")
    print(f"MACRO/EVENT STATUS: {_metric(metrics, 'macro_event_status')}")
    bars_used = metrics.get("bars_used")
    if isinstance(bars_used, Mapping):
        print(f"BARS USED: {json.dumps(dict(bars_used), sort_keys=True)}")
    print(f"RUNTIME SECONDS: {getattr(result, 'elapsed_seconds', _metric(metrics, 'elapsed_seconds'))}")
    print(f"LLM CALLS: {_metric(metrics, 'llm_calls')}")
    print(f"TOOL CALLS: {_metric(metrics, 'tool_calls')}")
    print(f"TOKENS IN: {_metric(metrics, 'tokens_in')}")
    print(f"TOKENS OUT: {_metric(metrics, 'tokens_out')}")
    print(f"REASONING TOKENS: {_metric(metrics, 'reasoning_tokens')}")
    agents = metrics.get("agents")
    if isinstance(agents, Mapping):
        for agent_name, agent_metrics in agents.items():
            if not isinstance(agent_metrics, Mapping):
                continue
            print(
                "AGENT METRICS: "
                f"{agent_name} "
                f"model={agent_metrics.get('model', 'unknown')} "
                f"calls={agent_metrics.get('calls', 'unknown')} "
                f"tokens_in={agent_metrics.get('tokens_in', 'unknown')} "
                f"tokens_out={agent_metrics.get('tokens_out', 'unknown')} "
                f"reasoning_tokens={agent_metrics.get('reasoning_tokens', 'unknown')} "
                f"elapsed_seconds={agent_metrics.get('elapsed_seconds', 'unknown')}"
            )
    print(f"RAW PORTFOLIO MANAGER RESULT: {_format_raw_result(decision)}")
    print(f"NORMALIZED ACTION: {getattr(decision, 'action', None) or 'UNRESOLVED'}")
    print(f"NORMALIZATION STATUS: {decision.normalization_status}")
    print(
        "DECISION CONTEXT STATUS: "
        f"{getattr(decision, 'decision_context_status', 'INCOMPLETE')}"
    )
    context_integrity = metrics.get("context_integrity")
    if isinstance(context_integrity, Mapping):
        missing_context = context_integrity.get("missing")
        if (
            isinstance(missing_context, Sequence)
            and not isinstance(missing_context, (str, bytes))
            and missing_context
        ):
            print(
                "CONTEXT INTEGRITY MISSING: "
                + ", ".join(str(item) for item in missing_context)
            )
    normalization_error = getattr(decision, "normalization_error", None)
    if normalization_error:
        print(f"NORMALIZATION ERROR: {normalization_error}")
    print(f"SHADOW DATABASE: {database_path}")
    print("EXECUTED: FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
