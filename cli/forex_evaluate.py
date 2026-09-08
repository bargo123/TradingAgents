"""Standalone read-only evaluator for persisted forex shadow decisions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tradingagents.forex.evaluation import (
    EvaluationConfig,
    ShadowDecisionEvaluationResult,
    ShadowEvaluationBatchResult,
    ShadowEvaluationStore,
    ShadowOutcomeEvaluator,
)
from tradingagents.forex.shadow import ShadowDecisionStore


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-evaluate",
        description="Read-only MT5 outcome evaluation for persisted forex shadow decisions.",
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--decision-id", help="evaluate one persisted decision")
    selector.add_argument(
        "--pending",
        action="store_true",
        help="evaluate all decisions whose legacy shadow status is pending",
    )
    parser.add_argument(
        "--db-path",
        default="data_cache/shadow_decisions.db",
        help="SQLite path containing shadow decisions and evaluations",
    )
    parser.add_argument(
        "--terminal-path",
        default=None,
        help="optional local MetaTrader 5 terminal executable path",
    )
    parser.add_argument(
        "--observation-tolerance-seconds",
        type=_non_negative_int,
        default=30,
        help="terminal observation tolerance in seconds (default: 30)",
    )
    return parser


def _metric(metrics: Mapping[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return value if value is not None else "unknown"


def _print_result(result: ShadowDecisionEvaluationResult | ShadowEvaluationBatchResult) -> None:
    decision = getattr(result, "decision", None)
    if decision is not None:
        print(f"DECISION ID: {decision.decision_id}")
        print(f"RESOLVED SYMBOL: {decision.resolved_symbol}")
    else:
        print(f"DECISIONS SCANNED: {_metric(result.metrics, 'decisions_scanned')}")
    statuses = getattr(result, "status_by_basis", {})
    for basis in ("ANALYSIS_SNAPSHOT", "DECISION_REFERENCE"):
        print(f"{basis} STATUS: {statuses.get(basis, 'PENDING')}")
    metrics = result.metrics if isinstance(result.metrics, Mapping) else {}
    for key, label in (
        ("decisions_scanned", "DECISIONS SCANNED"),
        ("horizons_evaluated", "HORIZONS EVALUATED"),
        ("historical_ticks_processed", "HISTORICAL TICKS PROCESSED"),
        ("database_seconds", "DATABASE SECONDS"),
        ("mt5_read_seconds", "MT5 READ SECONDS"),
        ("total_runtime_seconds", "TOTAL RUNTIME SECONDS"),
        ("llm_calls", "LLM CALLS"),
    ):
        print(f"{label}: {_metric(metrics, key)}")
    errors = getattr(result, "errors", ())
    for error in errors:
        print(f"EVALUATION ERROR: {error}", file=sys.stderr)


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_factory: Callable[..., Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    print("MT5 FOREX — OUTCOME EVALUATION (READ ONLY)")
    print("NO ORDER WILL BE SENT")
    try:
        db_path = Path(args.db_path)
        decision_store = ShadowDecisionStore(db_path)
        evaluation_store = ShadowEvaluationStore(db_path)
        config = EvaluationConfig(
            observation_tolerance_seconds=args.observation_tolerance_seconds
        )
        factory = evaluator_factory or ShadowOutcomeEvaluator
        evaluator = factory(
            decision_store=decision_store,
            evaluation_store=evaluation_store,
            config=config,
        )
        if args.decision_id is not None:
            result = evaluator.evaluate_decision(
                args.decision_id,
                terminal_path=args.terminal_path,
            )
        else:
            result = evaluator.evaluate_pending(terminal_path=args.terminal_path)
        _print_result(result)
        result_metrics = getattr(result, "metrics", {})
        if (
            isinstance(result_metrics, Mapping)
            and result_metrics.get("llm_calls", 0) not in (0, None)
        ):
            print("FOREX EVALUATION ERROR: evaluator reported non-zero LLM calls", file=sys.stderr)
            return 1
        return 1 if getattr(result, "errors", ()) else 0
    except Exception as exc:  # CLI boundary: preserve a concise non-zero error
        print(f"FOREX EVALUATION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
