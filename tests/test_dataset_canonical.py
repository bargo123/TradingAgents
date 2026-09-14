import json
from datetime import datetime, timezone

import pytest

from tradingagents.datasets.canonical import canonicalize
from tradingagents.datasets.eligibility import EligibilityResult
from tradingagents.datasets.models import (
 DatasetExclusionReason,
 EvaluationObservation,
 EvidenceObservation,
 JoinedObservation,
 SourceObservation,
)

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
