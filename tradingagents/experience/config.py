"""Versioned configuration for the independent Phase 8 artifact store."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from .models import TrustTier

EXPERIENCE_SCHEMA_VERSION = "phase8.experience.v1"
FEATURE_SCHEMA_VERSION = "experience-features.v1"
FEATURE_EXTRACTOR_VERSION = "phase8-feature-extractor.v1"
SIMILARITY_PROFILE_VERSION = "similarity-profile.v1"
TRUST_POLICY_VERSION = "trust-policy.v1"
STATISTICS_POLICY_VERSION = "statistics-policy.v1"
PROJECTION_VERSION = "experience-projection.v1"

@dataclass(frozen=True, slots=True)
class ExperienceConfig:
    artifact_root: Path = Path("data_cache/experience")
    source_databases: tuple[Path, ...] = ()
    experience_schema_version: str = EXPERIENCE_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_extractor_version: str = FEATURE_EXTRACTOR_VERSION
    similarity_profile_version: str = SIMILARITY_PROFILE_VERSION
    trust_policy_version: str = TRUST_POLICY_VERSION
    statistics_policy_version: str = STATISTICS_POLICY_VERSION
    projection_version: str = PROJECTION_VERSION
    trust_tiers: tuple[TrustTier, ...] = (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)
    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root)); object.__setattr__(self, "source_databases", tuple(Path(p) for p in self.source_databases)); object.__setattr__(self, "trust_tiers", tuple(TrustTier(v) for v in self.trust_tiers))
    def to_dict(self):
        return {"artifact_root": str(self.artifact_root), "source_databases": [str(p) for p in self.source_databases], "experience_schema_version": self.experience_schema_version, "feature_schema_version": self.feature_schema_version, "feature_extractor_version": self.feature_extractor_version, "similarity_profile_version": self.similarity_profile_version, "trust_policy_version": self.trust_policy_version, "statistics_policy_version": self.statistics_policy_version, "projection_version": self.projection_version, "trust_tiers": [v.value for v in self.trust_tiers]}
    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
