"""Bounded offline HFT entry-margin tuning; never constructs a broker gateway.

Select on development data once, then evaluate the frozen choice on later
periods. All results use the current deterministic HFT engines and a neutral
research regime, not a reconstruction of historical live regime decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from .causal import CausalSegment, load_causal_tick_dataset
from .evaluation import _aggregate_replay_metrics
from .models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from .replay import ReplayEvaluator

STAGES = ("DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT")


def partition_segments(segments, *, guard_seconds: float = 60.0):
    """Partition globally by broker time, retaining independent runtime state."""
    segments = tuple(segments)
    items = sorted((item for segment in segments for item in segment.ticks), key=lambda item: item.timestamp)
    if len(items) < 8 or not math.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("insufficient ticks or invalid chronological guard")
    if len({item.timestamp for item in items}) != len(items):
        raise ValueError("overlapping runtime captures must be resolved before tuning")
    first = items[len(items) // 2].timestamp
    second = items[3 * len(items) // 4].timestamp
    gap = timedelta(seconds=guard_seconds)
    bounds = {
        "DEVELOPMENT": (None, first - gap),
        "VALIDATION": (first + gap, second - gap),
        "UNSEEN_HOLDOUT": (second + gap, None),
    }
    result = {}
    for stage, (lower, upper) in bounds.items():
        parts = []
        for segment in sorted(segments, key=lambda value: value.start):
            ticks = tuple(item for item in segment.ticks
                          if (lower is None or item.timestamp >= lower)
                          and (upper is None or item.timestamp < upper))
            if ticks:
                parts.append(replace(segment, segment_id=f"{segment.segment_id}:{stage}", ticks=ticks))
        if not any(len(part.ticks) >= 2 for part in parts):
            raise ValueError(f"chronological guard leaves {stage} empty")
        result[stage] = tuple(parts)
    return result


def _marked_expectancy(report):
    trades = int(report.get("trades", 0))
    open_count = int(report.get("open_positions", 0))
    pnl = float(report.get("expectancy") or 0) * trades + float(report.get("unrealized_pnl") or 0)
    return pnl / max(1, trades + open_count)


def select_margin(development_reports, *, minimum_trades=20):
    """Selection accepts development evidence only, including boundary marks."""
    if minimum_trades <= 0:
        raise ValueError("minimum_trades must be positive")
    eligible = [(margin, report) for margin, report in development_reports.items()
                if int(report.get("trades", 0)) >= minimum_trades]
    return max(eligible, key=lambda item: (_marked_expectancy(item[1]), -item[0]))[0] if eligible else None


def _evaluate(segments, candidate, *, margin, slippage, point):
    evaluator = ReplayEvaluator(slippage_points=slippage, cost_safety_margin_points=margin)
    reports = tuple(evaluator.evaluate((item.tick for item in segment.ticks), candidate)
                    for segment in segments if len(segment.ticks) >= 2)
    metrics = _aggregate_replay_metrics(reports).to_dict()
    metrics["marked_expectancy_points"] = _marked_expectancy(metrics) / point
    metrics["closed_pnl_points"] = float(metrics["expectancy"] or 0) * metrics["trades"] / point
    metrics["unrealized_pnl_points"] = metrics["unrealized_pnl"] / point
    metrics["slippage_points_per_fill"] = slippage
    return metrics


def run_tuning(source: str | Path, *, source_commit="unknown", minimum_trades=20):
    """Test a fixed four-margin grid for both built-in HFT strategy families."""
    source = Path(source).resolve()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset = load_causal_tick_dataset(source)
    if dataset.invalid_rows:
        raise ValueError("invalid source ticks prevent this experiment")
    # Preserve observed quotes for diagnostics, never invent a replacement spread.
    # Unverified zero-spread captures cannot produce a validated candidate.
    quality_reasons = ["ZERO_SPREAD_PROVENANCE_UNVERIFIED"] if dataset.zero_spread_ticks else []
    stages = partition_segments(dataset.segments)
    points = {tick.point for tick in dataset.ticks}
    if len(points) != 1:
        raise ValueError("a tuning dataset must use one broker point size")
    point = points.pop()
    family_results = {}
    for family in ("range_rejection", "momentum_continuation"):
        parent = StrategyVersion(family, "on-disk-incumbent", "native-exit-profiles",
                                 ExitPolicyConfig().to_dict(), source_commit)
        candidate = CandidateSpec(f"tune-{family}", parent, family, ExitPolicyConfig(),
                                  "Require a larger expected move above observed spread.")
        development = {}
        for margin in (1.0, 3.0, 5.0, 8.0):
            print(f"TUNING {family} development margin={margin:g}", flush=True)
            development[margin] = _evaluate(stages["DEVELOPMENT"], candidate, margin=margin,
                                            slippage=1.0, point=point)
        selected = select_margin(development, minimum_trades=minimum_trades)
        if selected is None:
            family_results[family] = {"status": "INSUFFICIENT_EVIDENCE", "development": development}
            continue
        print(f"FROZEN {family} margin={selected:g}; evaluating later data", flush=True)
        checks = {}
        reasons = list(quality_reasons)
        if _marked_expectancy(development[selected]) <= 0:
            reasons.append("DEVELOPMENT_EXPECTANCY_NONPOSITIVE")
        for stage in ("VALIDATION", "UNSEEN_HOLDOUT"):
            checks[stage] = {}
            for slippage in (1.0, 2.0):
                baseline = _evaluate(stages[stage], candidate, margin=1.0, slippage=slippage, point=point)
                chosen = baseline if selected == 1.0 else _evaluate(
                    stages[stage], candidate, margin=selected, slippage=slippage, point=point)
                checks[stage][slippage] = {"baseline": baseline, "selected": chosen}
                if chosen["trades"] < minimum_trades:
                    reasons.append(f"INSUFFICIENT_TRADES:{stage}:{slippage:g}")
                if _marked_expectancy(chosen) <= 0:
                    reasons.append(f"NONPOSITIVE_EXPECTANCY:{stage}:{slippage:g}")
                if _marked_expectancy(chosen) <= _marked_expectancy(baseline):
                    reasons.append(f"NO_IMPROVEMENT:{stage}:{slippage:g}")
                if chosen["max_drawdown"] > baseline["max_drawdown"]:
                    reasons.append(f"DRAWDOWN_REGRESSION:{stage}:{slippage:g}")
        family_results[family] = {
            "status": "RESEARCH_VALIDATED" if not reasons else "REJECTED",
            "selected_margin_points": selected, "development": development,
            "later_period_checks": checks, "reasons": reasons,
        }
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError("source changed during tuning; repeat on a frozen snapshot")
    return {
        "execution_mode": "RESEARCH", "real_money": False, "source": str(source),
        "source_sha256": before, "dataset": dataset.to_dict(), "source_commit": source_commit,
        "data_quality_eligible": not quality_reasons, "quality_reasons": quality_reasons,
        "selection": "development only; choice frozen before later-period evaluation",
        "guard_gap_seconds": 60, "families": family_results,
        "assumptions": {"regime": "NEUTRAL_BOTH_RESEARCH", "point": point,
                        "commission_known": False, "latency_price_effect_modeled": False,
                        "account_metrics": "unleveraged unit-size simulation",
                        "live_configuration_applied": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hft-path", required=True, help="Frozen HFT SQLite snapshot")
    parser.add_argument("--output", required=True, help="New report file; existing files are preserved")
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output == Path(args.hft_path).resolve():
        raise FileExistsError("output must be a new file distinct from the source")
    report = run_tuning(args.hft_path, source_commit=args.source_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"REPORT {output}", flush=True)
    for family, result in report["families"].items():
        print(f"{family}: {result['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
