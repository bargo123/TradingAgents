"""Bounded offline qualification, never a broker connection or promotion API.

Extends the existing replay/tuning pipeline. Historical last partitions are
diagnostic, NOT sealed. Unknown fees and research history remain hard blockers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

from tradingagents.forex.hft.models import FastAction
from .causal import load_causal_tick_dataset
from .evaluation import _aggregate_replay_metrics
from .models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from .replay import ReplayEvaluator
from .tuning import partition_segments, _marked_expectancy


def trend_filter(name):
    if name not in ('none', 'aligned', 'countertrend_cap'):
        raise ValueError('unknown causal trend filter')

    def accepts(action, features):
        sign = 1 if action is FastAction.ENTER_LONG else -1
        opposing = sign * features.momentum < 0
        if name == 'aligned':
            return not opposing
        if name == 'countertrend_cap':
            return not (opposing and abs(features.momentum) / features.point > 2
                        and features.direction_persistence >= .75)
        return True
    return accepts


def _time(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('evidence timestamps must be timezone aware')
    return stamp.astimezone(timezone.utc)


def _quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def clustered_interval(outcomes, *, trial_count, replicates=4000, minimum_clusters=20, seed=14):
    """Resample whole UTC-hour clusters, with local Bonferroni tail allocation.

    Approximate empirical interval, not a guarantee. Empty hours are omitted;
    dependence beyond one hour and historical trials are not corrected here.
    """
    if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 1:
        raise ValueError('trial_count must be a positive integer')
    if replicates < 1000 or minimum_clusters < 2:
        raise ValueError('insufficient bootstrap resolution or cluster minimum')
    grouped = defaultdict(list)
    for record in outcomes:
        value = float(record['net_points'])
        if not math.isfinite(value):
            raise ValueError('nonfinite outcome')
        grouped[_time(record['timestamp']).replace(minute=0, second=0, microsecond=0)].append(value)
    blocks = [(sum(values), len(values)) for _, values in sorted(grouped.items())]
    result = dict(clusters=len(blocks), familywise_alpha=.05, per_trial_alpha=.05/trial_count,
                  trial_count=trial_count, lower_points=None, upper_points=None,
                  status='INSUFFICIENT_CLUSTERS', method='UTC_HOUR_CLUSTER_BOOTSTRAP_LOCAL_BONFERRONI',
                  replicates=replicates, seed=seed)
    if len(blocks) < minimum_clusters:
        return result
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        draw = rng.choices(blocks, k=len(blocks))
        means.append(sum(pnl for pnl, count in draw) / sum(count for pnl, count in draw))
    tail = .05 / (2 * trial_count)
    result.update(status='ESTIMATED', lower_points=_quantile(means, tail),
                  upper_points=_quantile(means, 1-tail))
    return result


def calibrate_costs(rows, *, point, development_end):
    if not math.isfinite(point) or point <= 0 or development_end.tzinfo is None:
        raise ValueError('invalid point or development boundary')
    adverse, latencies = [], []
    for row in rows:
        if _time(row['timestamp']) > development_end:
            continue
        requested, filled = float(row['requested_price']), float(row['fill_price'])
        if row['direction'] not in ('LONG', 'SHORT') or not all(math.isfinite(x) and x > 0 for x in (requested, filled)):
            raise ValueError('invalid broker fill evidence')
        sign = 1 if row['direction'] == 'LONG' else -1
        adverse.append(max(0, sign * (filled-requested) / point))
        latency = row.get('latency_ms')
        if latency is not None and math.isfinite(float(latency)) and float(latency) >= 0:
            latencies.append(float(latency))
    return dict(fills=len(adverse), adverse_slippage_p95_points=_quantile(adverse, .95) if adverse else None,
                latency_median_ms=statistics.median(latencies) if latencies else None,
                latency_p99_ms=_quantile(latencies, .99) if latencies else None,
                commission_known=False, calibration_end=development_end.isoformat(),
                latency_price_effect_modeled=False)


def _broker_rows(path):
    # Read frozen evidence directly: constructing a DemoStore could migrate it.
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('''SELECT r.observed_at AS timestamp, i.direction,
            i.requested_price, r.fill_price, r.request_payload
            FROM demo_order_results r JOIN demo_order_intents i USING(intent_id)
            WHERE r.classification='FILLED' AND r.broker_order_sent=1
            AND r.real_money=0 AND i.real_money=0 AND i.account_trade_mode=0
            AND i.symbol='EURUSD' AND r.fill_price>0''').fetchall()
    return [dict(timestamp=row['timestamp'], direction=row['direction'], requested_price=row['requested_price'],
                 fill_price=row['fill_price'], latency_ms=json.loads(row['request_payload']).get('order_submission_latency_ms'))
            for row in rows]


def qualification_reasons(*, zero_spread_ticks, cost_fills):
    # There is deliberately no switch that declares already-inspected data sealed.
    reasons = ['COMMISSION_UNKNOWN', 'HOLDOUT_NOT_SEALED', 'HISTORICAL_TRIAL_COUNT_UNKNOWN',
               'LATENCY_PRICE_EFFECT_NOT_MODELED', 'DRAWDOWN_LIMIT_NOT_APPROVED']
    if zero_spread_ticks:
        reasons.append('ZERO_SPREAD_PROVENANCE_UNVERIFIED')
    if cost_fills < 20:
        reasons.append('DEVELOPMENT_COST_SAMPLE_INSUFFICIENT')
    return reasons


def _evaluate(segments, candidate, *, margin, filter_name, slippage, commission, point, trial_count):
    evaluator = ReplayEvaluator(slippage_points=slippage, cost_safety_margin_points=margin,
                                commission_round_trip_points=commission)
    outcomes, reports = [], []
    for segment in segments:
        if len(segment.ticks) >= 2:
            reports.append(evaluator.evaluate((item.tick for item in segment.ticks), candidate,
                           entry_filter=trend_filter(filter_name), outcome_observer=outcomes.append))
    metrics = _aggregate_replay_metrics(tuple(reports)).to_dict()
    metrics['marked_expectancy_points'] = _marked_expectancy(metrics) / point
    metrics['clustered_interval'] = clustered_interval(outcomes, trial_count=trial_count)
    metrics['outcomes'] = outcomes
    return metrics


def run_qualification(source, *, demo_source, source_commit='unknown', minimum_trades=20):
    if minimum_trades < 1:
        raise ValueError('minimum_trades must be positive')
    paths = [Path(source).resolve(), Path(demo_source).resolve()]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    dataset = load_causal_tick_dataset(paths[0])
    if dataset.invalid_rows:
        raise ValueError('invalid source ticks')
    points = {tick.point for tick in dataset.ticks}
    if len(points) != 1:
        raise ValueError('one point size required')
    point = points.pop()
    stages = partition_segments(dataset.segments)
    development_end = max(segment.end for segment in stages['DEVELOPMENT'])
    costs = calibrate_costs(_broker_rows(paths[1]), point=point, development_end=development_end)
    # Fixed, recorded grid; no repeated search or live settings changes.
    grid = [(family, margin, filter_name)
            for family in ('range_rejection', 'momentum_continuation')
            for margin in (1., 5.) for filter_name in ('none', 'aligned', 'countertrend_cap')]
    slippage = max(1., costs['adverse_slippage_p95_points'] or 0.)
    scenarios = [(slippage, 0.), (max(2., slippage * 2), 2.)]
    trials, families = [], {}
    blockers = qualification_reasons(zero_spread_ticks=dataset.zero_spread_ticks, cost_fills=costs['fills'])
    for family in ('range_rejection', 'momentum_continuation'):
        parent = StrategyVersion(family, 'on-disk-incumbent', 'native-exit-profiles', ExitPolicyConfig().to_dict(), source_commit)
        candidate = CandidateSpec('qualify-'+family, parent, family, ExitPolicyConfig(), 'Offline causal trend qualification')
        development = []
        for _, margin, filter_name in (row for row in grid if row[0] == family):
            print(f'QUALIFY {family} DEVELOPMENT margin={margin:g} filter={filter_name}', flush=True)
            metrics = _evaluate(stages['DEVELOPMENT'], candidate, margin=margin, filter_name=filter_name,
                                slippage=slippage, commission=0., point=point, trial_count=len(grid))
            trial = dict(strategy=family, margin_points=margin, trend_filter=filter_name,
                         slippage_per_fill_points=slippage, assumed_round_trip_commission_points=0., development=metrics)
            trials.append(trial)
            development.append(trial)
        eligible = [trial for trial in development if trial['development']['trades'] >= minimum_trades]
        selected = max(eligible, key=lambda trial: trial['development']['marked_expectancy_points']) if eligible else None
        reasons, checks = list(blockers), []
        if selected is None:
            reasons.append('INSUFFICIENT_DEVELOPMENT_TRADES')
        else:
            if selected['development']['marked_expectancy_points'] <= 0:
                reasons.append('DEVELOPMENT_EXPECTANCY_NONPOSITIVE')
            for stage in ('VALIDATION', 'UNSEEN_HOLDOUT'):
                for slip, fee in scenarios:
                    print(f'FROZEN {family} {stage} slippage={slip:g} commission={fee:g}', flush=True)
                    metrics = _evaluate(stages[stage], candidate, margin=selected['margin_points'],
                                        filter_name=selected['trend_filter'], slippage=slip, commission=fee,
                                        point=point, trial_count=len(grid))
                    interval = metrics['clustered_interval']
                    if metrics['trades'] < minimum_trades:
                        reasons.append('INSUFFICIENT_TRADES:'+stage)
                    if metrics['marked_expectancy_points'] <= 0:
                        reasons.append('NONPOSITIVE_EXPECTANCY:'+stage)
                    if interval['lower_points'] is None or interval['lower_points'] <= 0:
                        reasons.append('CLUSTERED_LOWER_BOUND_NOT_POSITIVE:'+stage)
                    checks.append(dict(stage=stage, sealed=False, slippage_per_fill_points=slip,
                                       assumed_round_trip_commission_points=fee, metrics=metrics))
        families[family] = dict(status='NOT_QUALIFIED', selected=selected, checks=checks, reasons=sorted(set(reasons)))
    if hashes != [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]:
        raise ValueError('frozen source changed during qualification')
    return dict(execution_mode='RESEARCH', real_money=False, live_configuration_applied=False,
                sources=[dict(path=str(path), sha256=digest) for path, digest in zip(paths, hashes)],
                dataset=dataset.to_dict(), source_commit=source_commit, measured_costs=costs,
                trial_count=len(grid), historical_trial_count=None, trials=trials, families=families,
                assumptions=dict(regime='NEUTRAL_BOTH_RESEARCH', sealed_holdout=False,
                                 commission_known=False, latency_price_effect_modeled=False,
                                 unit='broker points, unit-size replay; not account-dollar profit',
                                 calibration='development broker fills only; fallback stress is assumed, not measured'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hft-path', required=True)
    parser.add_argument('--demo-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', default='unknown')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output in (Path(args.hft_path).resolve(), Path(args.demo_path).resolve()):
        raise FileExistsError('output must be new and distinct from both snapshots')
    report = run_qualification(args.hft_path, demo_source=args.demo_path, source_commit=args.source_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(f'REPORT {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
