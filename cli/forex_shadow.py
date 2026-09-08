"""Standalone MT5 forex shadow-mode command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any

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

    try:
        runner = ForexShadowRunner(config=_runtime_config(args) or None)
        result = runner.run(
            symbol=args.symbol,
            analysis_date=parsed_date,
            count=args.count,
            terminal_path=args.terminal_path,
            db_path=args.db_path,
            analysts=analysts,
        )
    except Exception as exc:  # CLI boundary: preserve a concise non-zero error
        print(f"FOREX SHADOW ERROR: {exc}", file=sys.stderr)
        return 1

    decision = result.decision
    print("SHADOW DECISION RECORDED")
    print(f"DECISION ID: {decision.decision_id}")
    print(f"ACTION: {decision.action or 'UNRESOLVED'}")
    print(f"NORMALIZATION STATUS: {decision.normalization_status}")
    normalization_error = getattr(decision, "normalization_error", None)
    if normalization_error:
        print(f"NORMALIZATION ERROR: {normalization_error}")
    print("EXECUTED: FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
