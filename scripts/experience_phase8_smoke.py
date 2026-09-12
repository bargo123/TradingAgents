"""Bounded, read-only Phase 8 acceptance smoke (local and offline only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.errors import (
    ExperienceArtifactNotEmptyError,
    Phase7KnowledgeUnavailableError,
)
from tradingagents.experience.features import extract_market_state
from tradingagents.experience.importer import ExperienceImporter, ExperienceRebuilder
from tradingagents.experience.models import EvidenceRequest, OutcomeStatsRequest, TrustTier
from tradingagents.experience.normalization import NormalizationCohortV1, build_profile
from tradingagents.experience.orchestrator import EvidenceOrchestrator
from tradingagents.experience.outcomes import OutcomeStatsCalculator
from tradingagents.experience.query import ExperienceQueryService
from tradingagents.experience.source_reader import ReadonlySourceReader
from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.embeddings import FastEmbedProvider
from tradingagents.knowledge.index_generation import IncompatibleIndexGeneration
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.vector_index import VectorIndexReader


class NetworkAttempt(RuntimeError):
    """An unexpected network connection was attempted during the smoke."""


class OfflineNetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._connect = socket.socket.connect
        self._create = socket.create_connection
        self._urlopen = urllib.request.urlopen

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def __enter__(self):
        def blocked(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
            self.attempts.append(str(address))
            raise NetworkAttempt(f"offline smoke blocked network attempt: {address}")

        def blocked_create(address: Any, *args: Any, **kwargs: Any) -> None:
            self.attempts.append(str(address))
            raise NetworkAttempt(f"offline smoke blocked network attempt: {address}")

        def blocked_urlopen(url: Any, *args: Any, **kwargs: Any) -> None:
            self.attempts.append(str(url))
            raise NetworkAttempt(f"offline smoke blocked network attempt: {url}")

        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.create_connection = blocked_create  # type: ignore[assignment]
        urllib.request.urlopen = blocked_urlopen  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        socket.socket.connect = self._connect  # type: ignore[method-assign]
        socket.create_connection = self._create  # type: ignore[assignment]
        urllib.request.urlopen = self._urlopen  # type: ignore[assignment]


REPORT_KEYS = frozenset({"experience_artifact_root"})


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"sha256": None, "size": 0, "mtime_ns": None}
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_integrity(path: Path) -> dict[str, Any]:
    return {"db": _fingerprint(path), "wal": _fingerprint(Path(str(path) + "-wal"))}


def _json(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalization_fingerprint(profiles: dict[Any, Any]) -> str | None:
    """Return the canonical fingerprint for the first built profile."""

    for profile in profiles.values():
        if profile is not None:
            return profile.to_fingerprint()
    return None


def _tier_counts(records: tuple[Any, ...]) -> dict[str, int]:
    """Count catalog trust tiers using their stable serialized values."""

    def value(record: Any) -> str:
        trust = getattr(record, "trust", "")
        return trust.value if isinstance(trust, TrustTier) else str(trust)

    return {
        tier.value: sum(value(record) == tier.value for record in records) for tier in TrustTier
    }


def _ensure_fresh(root: Path) -> None:
    if root.exists():
        raise ExperienceArtifactNotEmptyError(f"Phase 8 artifact root already exists: {root}")
    elif not root.parent.is_dir():
        raise ExperienceArtifactNotEmptyError(
            f"Phase 8 artifact parent does not exist: {root.parent}"
        )
    root.mkdir(parents=False, exist_ok=False)


def run_smoke(
    source_db: str | Path,
    *,
    experience_artifact_root: str | Path,
    knowledge_artifact_root: str | Path | None,
    knowledge_embedding_model_path: str | Path | None,
    offline: bool = False,
) -> dict[str, Any]:
    if not offline:
        raise ValueError("--offline is required")
    if knowledge_artifact_root is None or knowledge_embedding_model_path is None:
        raise ValueError("Phase 7 artifact and local embedding model paths are required")
    source = Path(source_db).expanduser().resolve()
    root = Path(experience_artifact_root).expanduser().resolve()
    knowledge_root = Path(knowledge_artifact_root).expanduser().resolve()
    model_path = Path(knowledge_embedding_model_path).expanduser().resolve()
    _ensure_fresh(root)
    if not knowledge_root.is_dir() or not (knowledge_root / "catalog.sqlite3").is_file():
        raise Phase7KnowledgeUnavailableError(
            f"existing Phase 7 artifact root/catalog.sqlite3 is required: {knowledge_root}"
        )
    before = _source_integrity(source)
    started = time.monotonic()
    os.environ.update(
        {"KNOWLEDGE_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    )
    guard = OfflineNetworkGuard()
    with guard:
        snapshot = ReadonlySourceReader(source).read_snapshot()
        catalog = ExperienceCatalog(root)
        imported = ExperienceImporter(catalog).import_sources((source,))
        generation = ExperienceRebuilder(catalog).rebuild()
        records = tuple(catalog.active_records()) + tuple(catalog.historical_records())
        vectors = {}
        for record in records:
            try:
                vector = extract_market_state(
                    next(
                        r
                        for r in snapshot.decisions
                        if str(r["decision_id"]) == record.source_decision_id
                    )
                )
                vectors[record.experience_id] = vector
            except Exception:
                continue
        feature_rows = {
            record.experience_id: {
                "experience_id": record.experience_id,
                "values": vector.values,
                "mask": vector.mask,
                "feature_names": vector.feature_names,
                "cohort": vector.cohort,
                "feature_fingerprint": vector.fingerprint,
                "trust_tier": record.trust,
                "trust": record.trust,
            }
            for record in records
            if (vector := vectors.get(record.experience_id))
        }
        profiles = {}
        for cohort_values in {tuple(v.cohort) for v in vectors.values()}:
            cohort = NormalizationCohortV1(*cohort_values)
            rows = [r for r in feature_rows.values() if tuple(r["cohort"]) == cohort_values]
            profiles[cohort] = (
                build_profile(rows, cohort, (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED))
                if rows
                else None
            )
        experience_service = ExperienceQueryService(
            records,
            feature_vectors=feature_rows,
            profiles={k: v for k, v in profiles.items() if v},
            generation_id=generation.generation_id,
        )
        stats = OutcomeStatsCalculator(catalog)
        config_source_root = source.parent
        try:
            knowledge_root.relative_to(config_source_root)
            config_source_root = Path.cwd().resolve()
        except ValueError:
            pass
        config = KnowledgeConfig(
            source_root=config_source_root,
            artifact_root=knowledge_root,
            embedding_model_path=model_path,
            offline=True,
        )
        knowledge_catalog = KnowledgeCatalog(knowledge_root / "catalog.sqlite3")
        knowledge_generation = knowledge_catalog.active_generation()
        if knowledge_generation is None:
            raise IncompatibleIndexGeneration("no active Phase 7 generation")
        vector_reader = VectorIndexReader(knowledge_generation.vector_location)
        lexical_reader = LexicalIndexReader(knowledge_generation.lexical_location)
        embedder = FastEmbedProvider.from_config(config)
        knowledge_service = KnowledgeQueryService(
            vector_reader, lexical_reader, knowledge_catalog, embedder
        )
        orchestrator = EvidenceOrchestrator(knowledge_service, experience_service, stats)
        market_state = next(iter(vectors.values()), None)
        state = (
            {
                "values": list(market_state.values),
                "mask": list(market_state.mask),
                "feature_names": list(market_state.feature_names),
                "cohort": list(market_state.cohort),
            }
            if market_state
            else {"values": [], "mask": []}
        )
        bundle = orchestrator.query(
            EvidenceRequest(
                research_question="market risk and execution",
                market_state=state,
                knowledge_top_k=3,
                experience_top_k=3,
            )
        )
    after = _source_integrity(source)
    report = {
        "source_path": str(source),
        "source_hash_before": before["db"]["sha256"],
        "source_hash_after": after["db"]["sha256"],
        "source_size_before": before["db"]["size"],
        "source_size_after": after["db"]["size"],
        "source_mtime_before": before["db"]["mtime_ns"],
        "source_mtime_after": after["db"]["mtime_ns"],
        "source_wal_before": before["wal"],
        "source_wal_after": after["wal"],
        "source_unchanged": before == after,
        "decision_count": len(snapshot.decisions),
        "evaluation_count": len(snapshot.evaluations),
        "import_count": imported.indexed_count,
        "experience_count": len(records),
        "tier_counts": _tier_counts(records),
        "quarantine_count": catalog.quarantine_count(),
        "alias_count": sum(catalog.current_alias_count(r.experience_id) for r in records),
        "feature_population_count": len(vectors),
        "active_generation_id": generation.generation_id,
        "normalization_cohort_count": len(profiles),
        "normalization_population": len(vectors),
        "normalization_fingerprint": _normalization_fingerprint(profiles),
        "similarity_examples": [_json(hit) for hit in bundle.experience[:3]],
        "statistics": _json(
            stats.calculate(
                OutcomeStatsRequest(
                    experience_ids=tuple(r.experience_id for r in records),
                    evaluation_basis="DECISION_REFERENCE",
                    horizon_seconds=0,
                )
            )
        ),
        "phase7_generation_id": knowledge_generation.generation_id,
        "knowledge_embedding_spec": _json(knowledge_generation.embedding_spec),
        "knowledge_artifact_identity": knowledge_generation.population_hash,
        "knowledge_hit_count": len(bundle.knowledge),
        "experience_hit_count": len(bundle.experience),
        "evidence_status": bundle.status,
        "evidence_bundle": _json(bundle),
        "analysis_invocations": 0,
        "network_attempts": guard.attempt_count,
        "bounded_latency_ms": round((time.monotonic() - started) * 1000, 2),
        "fresh_experience_artifact_root": str(root),
        "diagnostics": [e.message for e in bundle.errors],
    }
    report["experience_artifact_root"] = str(root)
    if not report["source_unchanged"] or guard.attempt_count:
        raise RuntimeError("Phase 8 smoke integrity/network gate failed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--experience-artifact-root", required=True)
    parser.add_argument("--knowledge-artifact-root", required=True)
    parser.add_argument("--knowledge-embedding-model-path", required=True)
    parser.add_argument("--offline", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_smoke(
                    args.source_db,
                    experience_artifact_root=args.experience_artifact_root,
                    knowledge_artifact_root=args.knowledge_artifact_root,
                    knowledge_embedding_model_path=args.knowledge_embedding_model_path,
                    offline=args.offline,
                ),
                sort_keys=True,
                default=str,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"PHASE 8 NOT COMPLETE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
