"""Research-only binding gate; normalized distances are not feature thresholds."""
from .book_v2_pipeline import validate_envelope
from .book_normalization_models import CandidateValidationV2, NormalizationStatus
from .phase14c_mapping import capture_current_strategy_contracts, map_current_strategies, StrategyContractDriftError


def create_research_strategy(candidate: CandidateValidationV2, *, source_lookup):
    if not isinstance(candidate, CandidateValidationV2):
        raise TypeError("V2 validation record required")
    validated = validate_envelope(candidate.envelope, source_lookup=source_lookup)
    if validated.status is not NormalizationStatus.EXECUTABLE_ELIGIBLE:
        raise ValueError("V2 candidate is not replay eligible: " + ",".join(validated.reason_codes))
    # The enabled V2 grammar currently describes price distances only. None
    # binds to a reviewed threshold primitive; never convert it to momentum.
    raise ValueError("V2 candidate has no reviewed deterministic primitive binding")


def map_v2(candidate: CandidateValidationV2, *, snapshot, source_lookup):
    if not isinstance(candidate, CandidateValidationV2):
        raise TypeError("V2 validation record required")
    checked = validate_envelope(candidate.envelope, source_lookup=source_lookup)
    if checked.status is NormalizationStatus.REJECTED:
        raise ValueError("invalid V2 normalization proof")
    if capture_current_strategy_contracts().fingerprint != snapshot.fingerprint:
        raise StrategyContractDriftError("current strategy implementation changed after snapshot capture")
    # No price-distance rule matches the incumbent feature-based signatures.
    # Reuse the exact mapping report contract, without feeding V2 quotes to V1.
    return map_current_strategies((), snapshot=snapshot)
