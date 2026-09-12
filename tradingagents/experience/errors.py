"""Typed boundary errors for Experience Memory services."""


class ExperienceError(Exception):
    """Base class for typed Phase 8 errors."""


_ERRORS = [
    "SourceDatabaseUnavailableError",
    "SourceSchemaIncompatibleError",
    "SourceSnapshotChangedError",
    "SourceDecisionConflictError",
    "DecisionEvidenceMalformedError",
    "FeatureExtractionIncompleteError",
    "InsufficientComparableFeaturesError",
    "SimilarityProjectionIncompatibleError",
    "SimilarityProfileMismatchError",
    "TrustPolicyMismatchError",
    "EvaluationEvidenceUnavailableError",
    "Phase7KnowledgeUnavailableError",
    "ExperienceMemoryUnavailableError",
    "ExperienceArtifactNotEmptyError",
    "PartialEvidenceError",
    "ProvenanceViolationError",
    "ExperienceImportLockedError",
    "ExperienceImportInterrupted",
]

for _name in _ERRORS:
    globals()[_name] = type(_name, (ExperienceError,), {})

__all__ = ["ExperienceError", *_ERRORS]
