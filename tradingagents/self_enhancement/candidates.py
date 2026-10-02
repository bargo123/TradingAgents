"""Deterministic bounded candidate generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .models import CandidateSpec, ExitPolicyConfig, StrategyVersion


class CandidateGenerator:
    """Generate only the small V1 exit-policy experiment set."""

    _variants = (
        ("baseline", {}, "Measure the incumbent exit policy as the control."),
        ("expected_move", {"profit_target_fraction": 1.5}, "Take profit at a bounded expected-move multiple."),
        ("mfe_giveback", {"protection_arm_fraction": 0.4, "giveback_fraction": 0.2}, "Protect favorable excursion earlier."),
        ("micro_reversal", {"micro_reversal_fraction": 0.35}, "Exit earlier after a causal micro reversal."),
        ("trailing", {"protection_arm_fraction": 0.4, "trailing_distance_fraction": 0.35}, "Use a tighter causal trailing distance."),
        ("no_progress", {"no_progress_cap_seconds": 20.0, "max_duration_seconds": 60.0}, "Reduce time spent without causal progress."),
    )

    def generate(self, incumbent: StrategyVersion) -> tuple[CandidateSpec, ...]:
        base_keys = set(ExitPolicyConfig().to_dict())
        base_values = {key: incumbent.parameters[key] for key in base_keys if key in incumbent.parameters}
        base = ExitPolicyConfig.from_mapping(base_values)
        result: list[CandidateSpec] = []
        for name, overrides, hypothesis in self._variants:
            policy = replace(base, **overrides)
            canonical = json.dumps(
                {"parent": incumbent.config_hash, "variant": name, "policy": policy.to_dict()},
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate_id = "cand-" + hashlib.sha256(canonical.encode()).hexdigest()[:24]
            result.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    parent=incumbent,
                    strategy_id=incumbent.strategy_id,
                    exit_policy=policy,
                    hypothesis=hypothesis,
                )
            )
        return tuple(result)


__all__ = ["CandidateGenerator"]
