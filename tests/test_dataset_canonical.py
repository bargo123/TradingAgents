import json
from datetime import datetime, timezone

import pytest

from tradingagents.datasets.canonical import canonicalize
from tradingagents.datasets.eligibility import EligibilityResult, join_observations
from tradingagents.datasets.models import (
 DatasetExclusionReason,
 EvaluationObservation,
 EvidenceObservation,
 JoinedObservation,
 SourceFingerprint,
 SourceObservation,
)
from tradingagents.datasets.sources import SourceReadResult

UTC = timezone.utc

def obs():
 d=SourceObservation('d1',datetime(2025,1,1,tzinfo=UTC),datetime(2025,1,1,0,1,tzinfo=UTC),source_run_id='r1',requested_symbol='EURUSD',resolved_symbol='EURUSD',analysis_profile='p',analysis_timeframe='M5',action='HOLD',fields={'source_decision_fingerprint':'dfp','decision_context_status':'COMPLETE','normalization_status':'NORMALIZED','snapshot_json':{'quote':{'bid':1.1,'ask':1.2,'spread_points':10},'point':0.0001,'digits':5,'features':{'M5':{'return_over_bars':1.0}}}})
 ev=EvaluationObservation('d1','ANALYSIS_SNAPSHOT',300,'COMPLETE',True,{'source_evaluation_fingerprint':'efp','source_run_id':'r1','source_decision_fingerprint':'dfp','entry_bid':1.1,'entry_ask':1.2,'future_bid':1.0,'future_ask':1.1,'buy_net_points':-100.0,'sell_net_points':100.0,'selected_action':'HOLD','selected_action_net_points':0.0,'best_counterfactual_action':'SELL','best_counterfactual_net_points':100.0,'hold_opportunity_cost_points':100.0,'buy_mfe_points':20.0,'buy_mae_points':-120.0,'sell_mfe_points':130.0,'sell_mae_points':-10.0,'provenance':{'evaluation_fingerprint':'efp','source_decision_id':'d1'}})
 e=EvidenceObservation('COMPLETE','USED',('K1','E1','S1'),(),{'context_hash':'ctx','available_knowledge_ids':['K1'],'available_experience_ids':['E1'],'available_statistics_ids':['S1'],'knowledge_generation_id':'kg','experience_generation_id':'eg'})
 return JoinedObservation(d,ev,e,{'experience':{'experience_id':'x1','trust':'TIER_A_HIGH_TRUST','trust_policy_version':'tp1','experience_schema_version':'e1','feature_schema_version':'f1','feature_extractor_version':'x1','source_decision_fingerprint':'dfp','provenance':{'source_decision_id':'d1','source_decision_fingerprint':'dfp'},'market_state':{'quote':{'bid':1.1}},'source_evaluation_fingerprints':{'ANALYSIS_SNAPSHOT:300':'efp'}},'audit':{'decision_id':'d1','context_hash':'ctx','available_knowledge_ids':['K1'],'available_experience_ids':['E1'],'available_statistics_ids':['S1'],'evidence_refs_used':['K1','E1','S1'],'evidence_refs_rejected':[]},'source_fingerprint':{'source_id':'s1','snapshot_fingerprint':'sfp'},'duplicate':False})

def test_canonical_stable_json_and_fields():
 o = obs()
 ok = EligibilityResult(True, details={'decision_id': 'd1'})
 a = canonicalize(o, ok)
 b = canonicalize(o, ok)
 assert a.example_id==b.example_id and a.to_json()==b.to_json()
 assert a.outcome['buy_net_points']==-100.0 and a.outcome['sell_mfe_points']==130.0
 assert a.outcome['hold_opportunity_cost_points']==100.0
 assert a.provenance['knowledge']['ids']==('K1',) and a.provenance['phase8']['ids']==('E1','S1')
 assert 'context_hash' in a.provenance
 json.loads(a.to_json())

def test_canonical_rejects_ineligible_and_forbidden_recursive():
 o=obs()
 with pytest.raises(ValueError):
  canonicalize(o, EligibilityResult(False, (DatasetExclusionReason.OUTCOME_INELIGIBLE,), {'decision_id': 'd1'}))
 with pytest.raises(ValueError):
  canonicalize({'nested': {'prompt': 'hello'}}, EligibilityResult(True, details={'decision_id': 'd1'}))

def test_duplicate_rejected():
 o = obs()
 o = o.__class__(o.decision, o.evaluation, o.evidence, {**o.fields, 'duplicate': True})
 with pytest.raises(ValueError):
  canonicalize(o, EligibilityResult(True, details={'decision_id': 'd1'}))


def test_id_uses_only_approved_identity_fields():
    o = obs()
    ok = EligibilityResult(True, details={'decision_id': 'd1'})
    first = canonicalize(o, ok)
    changed = o.__class__(o.decision, o.evaluation, o.evidence,
                          {**o.fields, 'irrelevant': 'changed'})
    assert canonicalize(changed, ok).example_id == first.example_id
    assert 'canonicalization_version' not in first.provenance.get('identity', {})


def test_snapshot_is_bounded_and_fingerprints_are_phase_separated():
    o = obs()
    source = {'source_id': 's1', 'canonical_path': 'x', 'schema_fingerprint': 'sf',
              'file_sha256': 'fh', 'snapshot_fingerprint': 'ss'}
    fields = {**o.fields, 'source_fingerprint': source,
              'audit': {**o.fields.get('audit', {}), 'phase9_fingerprint': 'p9'}}
    d = o.decision.__class__(o.decision.decision_id, o.decision.analysis_snapshot_timestamp,
        o.decision.decision_completed_timestamp, source_run_id=o.decision.source_run_id,
        requested_symbol=o.decision.requested_symbol, resolved_symbol=o.decision.resolved_symbol,
        analysis_profile=o.decision.analysis_profile, analysis_timeframe=o.decision.analysis_timeframe,
        action=o.decision.action, fields={**o.decision.fields, 'snapshot_json': {
            'quote': {'bid': 1, 'ask': 2, 'spread_points': 3}, 'point': .1, 'digits': 2,
            'features': {'M5': {'return_over_bars': 1, 'candle_dump': ['x'] * 100}},
            'account': {'balance': 999}, 'rendered_context': 'drop me'}})
    joined = o.__class__(d, o.evaluation, o.evidence, fields)
    result = canonicalize(joined, EligibilityResult(True, details={'decision_id': 'd1'}))
    snapshot = result.market['snapshot']
    assert snapshot['quote']['bid'] == 1
    assert 'account' not in snapshot and 'rendered_context' not in snapshot
    assert result.provenance['phase56']['snapshot_fingerprint'] == 'ss'
    assert result.provenance['phase9']['phase9_fingerprint'] == 'p9'


def test_evidence_duplicates_and_overlap_fail_closed():
    o = obs()
    evidence = o.evidence.__class__('COMPLETE', 'USED', ('K1', 'K1'), (),
                                    {'available_knowledge_ids': ['K1'], 'context_hash': 'ctx'})
    bad = o.__class__(o.decision, o.evaluation, evidence, o.fields)
    with pytest.raises(ValueError):
        canonicalize(bad, EligibilityResult(True, details={'decision_id': 'd1'}))


def test_evidence_partition_must_account_for_every_available_reference():
    o = obs()
    evidence = o.evidence.__class__(
        'COMPLETE', 'USED', ('K1',), (),
        {'available_knowledge_ids': ['K1', 'K2'], 'context_hash': 'ctx'},
    )
    bad = o.__class__(
        o.decision,
        o.evaluation,
        evidence,
        {**o.fields, "audit": {**o.fields["audit"], "available_knowledge_ids": ["K1", "K2"]}},
    )
    with pytest.raises(ValueError, match="evidence partition"):
        canonicalize(bad, EligibilityResult(True, details={'decision_id': 'd1'}))


def test_conflicting_authoritative_and_fallback_available_ids_fail_closed():
    o = obs()
    evidence = o.evidence.__class__(
        "COMPLETE", "USED", ("K1", "E1", "S1"), (),
        {"available_knowledge_ids": ["K2"], "context_hash": "ctx"},
    )
    with pytest.raises(ValueError, match="available.*IDs"):
        canonicalize(
            o.__class__(o.decision, o.evaluation, evidence, o.fields),
            EligibilityResult(True, details={'decision_id': 'd1'}),
        )


def test_rejection_reason_is_preserved_in_canonical_provenance():
    o = obs()
    evidence = o.evidence.__class__(
        'COMPLETE', 'NONE_RELEVANT', (),
        ({'ref': 'K1', 'reason': 'LOW_RELEVANCE'},),
        {'available_knowledge_ids': ['K1'], 'context_hash': 'ctx'},
    )
    fields = {**o.fields, 'audit': {**o.fields['audit'],
        'available_knowledge_ids': ['K1'], 'evidence_refs_used': [],
        'available_experience_ids': [], 'available_statistics_ids': [],
        'evidence_refs_rejected': [{'ref': 'K1', 'reason': 'LOW_RELEVANCE'}]}}
    result = canonicalize(
        o.__class__(o.decision, o.evaluation, evidence, fields),
        EligibilityResult(True, details={'decision_id': 'd1'}),
    )
    assert result.provenance['phase9']['rejected'] == (
        {'ref': 'K1', 'reason': 'LOW_RELEVANCE'},
    )


def test_design_fields_and_phase_labeled_provenance():
    o = obs()
    o = o.__class__(o.decision, o.evaluation, o.evidence,
                    {**o.fields, 'source_fingerprint': {'source_id': 'p56', 'snapshot_fingerprint': 'f56'},
                     'audit': {'phase9_fingerprint': 'f9', 'context_integrity': 'COMPLETE',
                               'evidence_use_status': 'USED', 'evidence_refs_used': ['K1', 'E1', 'S1'],
                               'evidence_refs_rejected': []}})
    result = canonicalize(o, EligibilityResult(True, details={'decision_id': 'd1'}))
    assert result.decision['normalization_status'] == 'NORMALIZED'
    assert result.decision['decision_context_status'] == 'COMPLETE'
    assert result.decision['raw_result_fingerprint'] == result.provenance['raw_result_fingerprint']
    assert result.market['snapshot_fingerprint']
    assert result.provenance['phase56']['source_id'] == 'p56'
    assert result.provenance['phase8']['source_decision_fingerprint'] == 'dfp'
    assert result.provenance['phase9']['phase9_fingerprint'] == 'f9'
    assert result.research['context_integrity'] == 'COMPLETE'
    assert result.research['evidence_use_status'] == 'USED'


def test_oversized_snapshot_and_evidence_fail_closed():
    o = obs()
    huge = {f'feature_{i}': float(i) for i in range(100)}
    snapshot = {'quote': {'bid': 1, 'ask': 2, 'spread_points': 3}, 'point': .1, 'digits': 2,
                'features': {'M5': huge}}
    d = o.decision.__class__(o.decision.decision_id, o.decision.analysis_snapshot_timestamp,
        o.decision.decision_completed_timestamp, source_run_id=o.decision.source_run_id,
        requested_symbol=o.decision.requested_symbol, resolved_symbol=o.decision.resolved_symbol,
        analysis_profile=o.decision.analysis_profile, analysis_timeframe=o.decision.analysis_timeframe,
        action=o.decision.action, fields={**o.decision.fields, 'snapshot_json': snapshot})
    with pytest.raises(ValueError):
        canonicalize(o.__class__(d, o.evaluation, o.evidence, o.fields), EligibilityResult(True, details={'decision_id': 'd1'}))
    evidence = o.evidence.__class__('COMPLETE', 'USED', tuple(f'K{i}' for i in range(100)), (),
                                    {'available_knowledge_ids': tuple(f'K{i}' for i in range(100)), 'context_hash': 'ctx'})
    with pytest.raises(ValueError):
        canonicalize(o.__class__(o.decision, o.evaluation, evidence, o.fields), EligibilityResult(True, details={'decision_id': 'd1'}))


def test_exact_decision_contract_training_eligibility_and_research_metadata():
    o = obs()
    ev = o.evaluation.__class__(o.evaluation.decision_id, o.evaluation.evaluation_basis,
        o.evaluation.horizon_seconds, o.evaluation.evaluation_status,
        o.evaluation.source_context_eligible,
        {**o.evaluation.fields, 'training_eligible': True, 'training_eligibility_reason': 'APPROVED'})
    audit = {'context_integrity': 'COMPLETE', 'evidence_use_status': 'USED',
             'evidence_refs_used': ['K1', 'E1', 'S1'], 'evidence_refs_rejected': [],
             'bundle_status': 'COMPLETE', 'integration_status': 'INJECTED',
             'selected_counts': {'knowledge': 1, 'experience': 1, 'statistics': 1},
             'dropped_counts': {'knowledge': 0, 'experience': 0, 'statistics': 0},
             'knowledge_query_fingerprint': 'qf', 'query_policy_version': 'qp',
             'knowledge_generation_id': 'kg', 'experience_generation_id': 'eg'}
    joined = o.__class__(o.decision, ev, o.evidence, {**o.fields, 'audit': audit})
    result = canonicalize(joined, EligibilityResult(True, details={'decision_id': 'd1'}))
    assert set(result.decision) == {'decision_id', 'source_run_id', 'requested_symbol', 'resolved_symbol',
        'analysis_profile', 'analysis_timeframe', 'analysis_snapshot_timestamp',
        'decision_completed_timestamp', 'action', 'normalization_status', 'decision_context_status',
        'raw_result_fingerprint'}
    assert result.outcome['training_eligible'] is True
    assert result.outcome['training_eligibility_reason'] == 'APPROVED'
    assert result.research['bundle_status'] == 'COMPLETE'
    assert result.research['selected_counts']['knowledge'] == 1
    assert result.research['knowledge_query_fingerprint'] == 'qf'


def test_phase8_provenance_values_and_evaluation_fingerprints_are_bounded():
    o = obs()
    bad_record = {**o.fields['experience'],
                  'source_evaluation_fingerprints': {'ANALYSIS_SNAPSHOT:300': ['bad']}}
    with pytest.raises(ValueError, match="evaluation fingerprint"):
        canonicalize(
            o.__class__(o.decision, o.evaluation, o.evidence,
                        {**o.fields, 'experience': bad_record}),
            EligibilityResult(True, details={'decision_id': 'd1'}),
        )

    bad_record = {**o.fields['experience'],
                  'provenance': {'source_decision_id': {'arbitrary': 'object'}}}
    with pytest.raises(ValueError, match="Phase 8 provenance"):
        canonicalize(
            o.__class__(o.decision, o.evaluation, o.evidence,
                        {**o.fields, 'experience': bad_record}),
            EligibilityResult(True, details={'decision_id': 'd1'}),
        )


def test_arbitrary_phase8_provenance_is_rejected():
    o = obs()
    bad = {**o.fields, 'experience': {**o.fields['experience'], 'provenance': {'notes': 'arbitrary prose'}}}
    with pytest.raises(ValueError):
        canonicalize(o.__class__(o.decision, o.evaluation, o.evidence, bad),
                      EligibilityResult(True, details={'decision_id': 'd1'}))


def test_phase_source_fingerprints_survive_join_and_canonical_projection():
    o = obs()
    f56 = SourceFingerprint("p56", "phase56", "schema56", "file56", "snap56")
    f8 = SourceFingerprint("p8", "phase8", "schema8", "file8", "snap8")
    f9 = SourceFingerprint("p9", "phase9", "schema9", "file9", "snap9")
    phase56 = SourceReadResult(decisions=(o.decision,), evaluations=(o.evaluation,), fingerprint=f56)
    phase8 = SourceReadResult(records=({**o.fields["experience"],
                                        "source_decision_id": "d1",
                                        "source_run_id": "r1",
                                        "source_decision_fingerprint": "dfp"},), fingerprint=f8)
    phase9 = SourceReadResult(audits=(o.fields["audit"],), fingerprint=f9)
    joined = join_observations(phase56, phase8, phase9)[0]
    assert joined.fields["source_fingerprints"]["phase8"]["source_id"] == "p8"
    result = canonicalize(joined, EligibilityResult(True, details={"decision_id": "d1"}))
    assert result.provenance["phase56"]["source_id"] == "p56"
    assert result.provenance["phase8"]["source_fingerprint"]["source_id"] == "p8"
    assert result.provenance["phase9"]["source_fingerprint"]["source_id"] == "p9"


@pytest.mark.parametrize("phase", ["phase56", "phase8", "phase9"])
def test_phase_labeled_source_fingerprint_requires_complete_contract(phase):
    o = obs()
    fields = {**o.fields, "source_fingerprints": {phase: {"source_id": phase}}}
    with pytest.raises(ValueError, match=f"invalid {phase} source fingerprint"):
        canonicalize(
            o.__class__(o.decision, o.evaluation, o.evidence, fields),
            EligibilityResult(True, details={'decision_id': 'd1'}),
        )


def test_non_mapping_snapshot_feature_section_fails_closed():
    o = obs()
    d = o.decision.__class__(
        o.decision.decision_id,
        o.decision.analysis_snapshot_timestamp,
        o.decision.decision_completed_timestamp,
        source_run_id=o.decision.source_run_id,
        requested_symbol=o.decision.requested_symbol,
        resolved_symbol=o.decision.resolved_symbol,
        analysis_profile=o.decision.analysis_profile,
        analysis_timeframe=o.decision.analysis_timeframe,
        action=o.decision.action,
        fields={**o.decision.fields, "snapshot_json": {"features": {"M5": ["bad"]}}},
    )
    with pytest.raises(ValueError, match="feature section"):
        canonicalize(o.__class__(d, o.evaluation, o.evidence, o.fields),
                     EligibilityResult(True, details={"decision_id": "d1"}))


def test_snapshot_rejects_oversized_timeframe_key_before_bounding():
    o = obs()
    snapshot = {**o.decision.fields["snapshot_json"],
                "features": {"M" * 33: {"return_over_bars": 1.0}}}
    d = o.decision.__class__(
        o.decision.decision_id,
        o.decision.analysis_snapshot_timestamp,
        o.decision.decision_completed_timestamp,
        source_run_id=o.decision.source_run_id,
        requested_symbol=o.decision.requested_symbol,
        resolved_symbol=o.decision.resolved_symbol,
        analysis_profile=o.decision.analysis_profile,
        analysis_timeframe=o.decision.analysis_timeframe,
        action=o.decision.action,
        fields={**o.decision.fields, "snapshot_json": snapshot},
    )
    with pytest.raises(ValueError, match="timeframe"):
        canonicalize(o.__class__(d, o.evaluation, o.evidence, o.fields),
                     EligibilityResult(True, details={"decision_id": "d1"}))


def test_phase9_metadata_types_and_bounds_fail_closed():
    o = obs()
    audit = {**o.fields["audit"], "selected_counts": [1, 2]}
    bad = o.__class__(o.decision, o.evaluation, o.evidence, {**o.fields, "audit": audit})
    with pytest.raises(ValueError, match="selected_counts"):
        canonicalize(bad, EligibilityResult(True, details={"decision_id": "d1"}))


@pytest.mark.parametrize(
    "field",
    (
        "integration_status",
        "integration",
        "bundle_status",
        "evidence_audit_status",
        "context_integrity",
        "evidence_use_status",
    ),
)
def test_phase9_metadata_rejects_unknown_status_values(field):
    o = obs()
    audit = {**o.fields["audit"], field: "UNKNOWN_STATUS"}
    bad = o.__class__(o.decision, o.evaluation, o.evidence, {**o.fields, "audit": audit})
    with pytest.raises(ValueError, match=field):
        canonicalize(bad, EligibilityResult(True, details={"decision_id": "d1"}))

    audit = {**o.fields["audit"], "integration": []}
    bad = o.__class__(o.decision, o.evaluation, o.evidence, {**o.fields, "audit": audit})
    with pytest.raises(ValueError, match="integration"):
        canonicalize(bad, EligibilityResult(True, details={"decision_id": "d1"}))

    audit = {**o.fields["audit"], "unapproved_runtime_detail": "must not leak"}
    result = canonicalize(
        o.__class__(o.decision, o.evaluation, o.evidence,
                    {**o.fields, "audit": audit}),
        EligibilityResult(True, details={"decision_id": "d1"}),
    )
    assert "unapproved_runtime_detail" not in result.provenance["phase9"]

    audit = {**o.fields["audit"], "selected_counts": {"knowledge": 1, "unknown": 2}}
    bad = o.__class__(o.decision, o.evaluation, o.evidence, {**o.fields, "audit": audit})
    with pytest.raises(ValueError, match="selected_counts"):
        canonicalize(bad, EligibilityResult(True, details={"decision_id": "d1"}))
