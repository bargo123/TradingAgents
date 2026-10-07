from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from tradingagents.forex.hft.features import TickFeatureEngine
from tradingagents.forex.hft.models import FastAction, Tick
from tradingagents.self_enhancement.models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.replay import ReplayEvaluator


def sample():
    start = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    ticks = tuple(Tick('EURUSD', start + timedelta(seconds=i), m-.00001, m+.00001, .00001, i)
                  for i, m in enumerate((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005, 1.1003)))
    parent = StrategyVersion('range_rejection', 'v1', 'cfg', ExitPolicyConfig().to_dict(), 'abc')
    return ticks, CandidateSpec('test', parent, 'range_rejection', ExitPolicyConfig(), 'test')


def test_research_filter_blocks_entries_not_exits_and_collects_costed_outcomes():
    ticks, candidate = sample()
    outcomes = []
    def entry_only(action, features):
        assert action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT)
        return True

    metrics = ReplayEvaluator(commission_round_trip_points=2).evaluate(
        ticks, candidate, entry_filter=entry_only, outcome_observer=outcomes.append)
    assert metrics.trades == 1
    assert metrics.expectancy == pytest.approx(.00016)
    assert outcomes[0]['net_points'] == pytest.approx(16)
    assert outcomes[0]['timestamp'] == ticks[-1].timestamp.isoformat()
    blocked = ReplayEvaluator().evaluate(ticks, candidate, entry_filter=lambda action, features: False)
    assert blocked.trades == 0 and blocked.open_positions == 0


def test_boundary_mark_includes_modeled_round_trip_commission():
    ticks, candidate = sample()
    outcomes = []
    metrics = ReplayEvaluator(commission_round_trip_points=2).evaluate(
        ticks[:-1], candidate, outcome_observer=outcomes.append)
    assert metrics.open_positions == 1
    assert metrics.unrealized_pnl == pytest.approx(.00036)
    assert outcomes[-1]['boundary_mark'] is True
    assert outcomes[-1]['net_points'] == pytest.approx(36)


@pytest.mark.parametrize('cost', [-1, float('nan'), float('inf')])
def test_invalid_commission_cannot_create_artificial_replay_profit(cost):
    with pytest.raises(ValueError):
        ReplayEvaluator(commission_round_trip_points=cost)


def test_causal_trend_filters_reject_only_the_declared_opposing_entries():
    from tradingagents.self_enhancement.qualification import trend_filter
    ticks, _ = sample()
    features = replace(TickFeatureEngine().update(ticks[0]), momentum=.00003, direction_persistence=.9)
    assert trend_filter('aligned')(FastAction.ENTER_LONG, features)
    assert not trend_filter('aligned')(FastAction.ENTER_SHORT, features)
    assert not trend_filter('countertrend_cap')(FastAction.ENTER_SHORT, features)
    quiet = replace(features, momentum=.00001)
    assert trend_filter('countertrend_cap')(FastAction.ENTER_SHORT, quiet)
    with pytest.raises(ValueError):
        trend_filter('unknown')


def test_clustered_interval_counts_hours_not_individual_trades_and_corrects_trials():
    from tradingagents.self_enhancement.qualification import clustered_interval
    records = [{'timestamp': '2026-01-01T10:00:00+00:00', 'net_points': 1} for _ in range(100)]
    report = clustered_interval(records, trial_count=12)
    assert report['clusters'] == 1
    assert report['status'] == 'INSUFFICIENT_CLUSTERS'
    assert report['lower_points'] is None
    assert report['familywise_alpha'] == pytest.approx(.05)
    assert report['per_trial_alpha'] == pytest.approx(.05 / 12)
    records = [{'timestamp': f'2026-01-{day:02d}T10:00:00+00:00', 'net_points': 2} for day in range(1, 22)]
    report = clustered_interval(records, trial_count=12)
    assert report['lower_points'] == 2 and report['upper_points'] == 2


def test_cost_calibration_cannot_use_later_fills_or_favorable_slippage_as_negative_cost():
    from tradingagents.self_enhancement.qualification import calibrate_costs
    rows = [dict(timestamp='2026-01-01T10:00:00+00:00', direction='LONG', requested_price=1.1,
                 fill_price=1.09999, latency_ms=200),
            dict(timestamp='2026-01-02T10:00:00+00:00', direction='SHORT', requested_price=1.1,
                 fill_price=1.09, latency_ms=300)]
    result = calibrate_costs(rows, point=.00001, development_end=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
    assert result['fills'] == 1
    assert result['adverse_slippage_p95_points'] == 0
    assert result['latency_median_ms'] == 200
    assert result['commission_known'] is False


def test_missing_qualification_evidence_blocks_even_positive_research_returns():
    from tradingagents.self_enhancement.qualification import qualification_reasons
    reasons = qualification_reasons(zero_spread_ticks=1, cost_fills=0)
    assert {'ZERO_SPREAD_PROVENANCE_UNVERIFIED', 'DEVELOPMENT_COST_SAMPLE_INSUFFICIENT',
            'COMMISSION_UNKNOWN', 'HOLDOUT_NOT_SEALED', 'HISTORICAL_TRIAL_COUNT_UNKNOWN'} <= set(reasons)


def test_cost_sensitivity_preserves_round_trip_commission():
    from tradingagents.self_enhancement.evaluation import cost_sensitivity
    ticks, candidate = sample()
    result = cost_sensitivity(ReplayEvaluator(commission_round_trip_points=2), ticks, candidate, (0,))
    assert result.metrics_by_scenario[0].expectancy == pytest.approx(.00016)


def test_offline_qualification_records_every_trial_without_modifying_sources(tmp_path):
    import hashlib
    import json
    import sqlite3
    from tradingagents.self_enhancement.qualification import run_qualification

    source, demo = tmp_path / 'ticks.sqlite3', tmp_path / 'demo.sqlite3'
    ticks, _ = sample()
    start = ticks[0].timestamp
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE hft_ticks(run_id,tick_key,symbol,timestamp,bid,ask,features_json)')
        db.executemany('INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?)', [
            ('run', str(i), 'EURUSD', (start + timedelta(seconds=10*i)).isoformat(),
             ticks[i % 7].bid, ticks[i % 7].ask, json.dumps({'point': .00001})) for i in range(280)])
    with sqlite3.connect(demo) as db:
        db.execute('CREATE TABLE demo_order_results(intent_id,observed_at,fill_price,request_payload,classification,broker_order_sent,real_money)')
        db.execute('CREATE TABLE demo_order_intents(intent_id,direction,requested_price,account_trade_mode,symbol,real_money)')
        # A future fill cannot calibrate earlier development costs.
        db.execute("INSERT INTO demo_order_results VALUES('1','2026-01-02T10:00:00Z',1.101,'{}','FILLED',1,0)")
        db.execute("INSERT INTO demo_order_intents VALUES('1','LONG',1.1,0,'EURUSD',0)")
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, demo)]
    report = run_qualification(source, demo_source=demo, minimum_trades=1)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, demo)] == before
    assert report['measured_costs']['fills'] == 0
    assert report['trial_count'] == len(report['trials']) == 12
    assert report['live_configuration_applied'] is False and report['real_money'] is False
    assert all(family['status'] == 'NOT_QUALIFIED' for family in report['families'].values())
    assert all(check['sealed'] is False for family in report['families'].values() for check in family['checks'])
    json.dumps(report, allow_nan=False)
