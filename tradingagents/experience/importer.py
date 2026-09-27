"""Deterministic Phase 8 source importer and projection rebuilder."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import ExperienceCatalog
from .config import ExperienceConfig
from .errors import (
    ExperienceImportLockedError,
    FeatureExtractionIncompleteError,
    ProvenanceViolationError,
    SourceDecisionConflictError,
)
from .features import extract_market_state
from .identity import source_decision_fingerprint, source_evaluation_fingerprint
from .models import TrustTier
from .provenance import build_evaluation_provenance
from .source_reader import ReadonlySourceReader, ReadonlySourceSnapshot
from .trust import classify_trust

_MISSING = object()


def _primary_source_value(row: dict[str, Any], primary: str, fallback: str) -> Any:
    value = row.get(primary, _MISSING)
    if value is _MISSING or value is None or (isinstance(value, str) and value == ""):
        return row.get(fallback)
    return value


def _source_symbol(row: dict[str, Any]) -> Any:
    value = _primary_source_value(row, "resolved_symbol", "symbol")
    return "UNKNOWN" if value is None else value


def _evaluation_fingerprint(value: dict[str, Any]) -> str:
    """Select an evaluation identity without bypassing malformed fields."""
    for name in ("source_evaluation_fingerprint", "fingerprint"):
        if name not in value:
            continue
        candidate = value[name]
        if candidate is None or candidate == "":
            continue
        if not isinstance(candidate, str) or not candidate.strip():
            raise ProvenanceViolationError(
                f"{name} must be a non-empty evaluation fingerprint"
            )
        return candidate
    return source_evaluation_fingerprint(value)


@dataclass(frozen=True)
class ImportReport:
    indexed_count: int = 0
    unchanged_count: int = 0
    experience_count: int = 0
    alias_count: int = 0
    quarantined_count: int = 0
    failed_scan_count: int = 0
    evaluation_snapshot_count: int = 0
    fabricated_evaluation_snapshot_count: int = 0
    run_id: str = ""


@dataclass(frozen=True)
class GenerationManifest:
    generation_id: str
    population_fingerprint: str
    feature_count: int = 0
    experience_count: int = 0
    metadata: dict[str, Any] | None = None


def _stable_source_id(snapshot: ReadonlySourceSnapshot) -> str:
    schema = json.dumps(snapshot.source_schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(
        ("phase8.source.v1\0" + snapshot.canonical_path + "\0" + schema).encode()
    ).hexdigest()


def _generation_metadata(row_count: int) -> dict[str, Any]:
    metadata = {"rows": row_count, "retains_historical_features": True}
    metadata.update(
        {
            key: value
            for key, value in ExperienceConfig().to_dict().items()
            if key.endswith("_version")
        }
    )
    return metadata


def _feature_projection_metadata(
    market_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    config = ExperienceConfig()
    return (
        {**market_state, "version": config.feature_extractor_version},
        config.feature_schema_version,
    )


class ExperienceImporter:
    def __init__(self, catalog: ExperienceCatalog, reader_factory=ReadonlySourceReader) -> None:
        self.catalog = catalog
        self.reader_factory = reader_factory
        self.lock_path = catalog.artifact_root / "locks" / "import.lock"

    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self.lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ExperienceImportLockedError("Phase 8 import already running") from exc

    def _append_evaluations(
        self, record, snapshot: ReadonlySourceSnapshot, decision_id: str, source_id: str
    ) -> int:
        """Append only evaluation rows observed in this immutable source read."""
        added = 0
        for evaluation in snapshot.evaluations:
            evaluation_decision_id = evaluation.get("decision_id")
            if not isinstance(evaluation_decision_id, str) or evaluation_decision_id != decision_id:
                continue
            efp = _evaluation_fingerprint(evaluation)
            ep = build_evaluation_provenance(
                efp,
                source_database_id=source_id,
                source_snapshot_fingerprint=snapshot.source_snapshot_fingerprint,
            )
            if self.catalog.append_evaluation_snapshot(record.experience_id, evaluation, efp, ep):
                added += 1
        return added

    def import_sources(self, source_paths: Iterable[str | os.PathLike[str]]) -> ImportReport:
        run_id = uuid.uuid4().hex
        lock = self._lock()
        staging = self.catalog.artifact_root / ".staging" / run_id
        staging.mkdir(parents=True, exist_ok=False)
        indexed = unchanged = experiences = aliases = quarantined = failed = eval_count = 0
        try:
            self.catalog.reconcile_interrupted_imports()
            self.catalog.record_import_event("RUNNING", detail={"run_id": run_id})
            for path in sorted((Path(p) for p in source_paths), key=lambda p: str(p.resolve())):
                committed_counts = (indexed, unchanged, aliases, quarantined, eval_count)
                try:
                    reader = self.reader_factory(path)
                    try:
                        snapshot = reader.read_snapshot()
                    finally:
                        close = getattr(reader, "close", None)
                        if callable(close):
                            close()
                    source_id = _stable_source_id(snapshot)
                    with self.catalog.transaction():
                        schema_fp = hashlib.sha256(
                            json.dumps(snapshot.source_schema, sort_keys=True, default=str).encode()
                        ).hexdigest()
                        self.catalog.register_source(
                            source_id,
                            canonical_path=snapshot.canonical_path,
                            source_fingerprint=snapshot.source_snapshot_fingerprint,
                            source_schema_fingerprint=schema_fp,
                            metadata={
                                "file": snapshot.file_fingerprint,
                                "wal": snapshot.wal_fingerprint,
                            },
                        )
                        seen: set[str] = set()
                        for row in snapshot.decisions:
                            raw_decision_id = row.get("decision_id")
                            if not isinstance(raw_decision_id, str) or not raw_decision_id.strip():
                                raise ValueError("decision_id must be a non-empty string")
                            decision_id = raw_decision_id
                            seen.add(decision_id)
                            fingerprint = source_decision_fingerprint(row)
                            existing = next(
                                (
                                    r
                                    for r in self.catalog.active_records()
                                    + self.catalog.historical_records()
                                    if r.source_decision_id == decision_id
                                ),
                                None,
                            )
                            alias_key = f"{source_id}:{decision_id}"
                            if (
                                existing
                                and existing.source_decision_fingerprint == fingerprint
                                and existing.source_aliases.get(alias_key) == "CURRENT"
                            ):
                                unchanged += 1
                                eval_count += self._append_evaluations(
                                    existing, snapshot, decision_id, source_id
                                )
                                continue
                            try:
                                vector = extract_market_state(row)
                                trust = classify_trust(
                                    {
                                        **row,
                                        "source_decision_fingerprint": fingerprint,
                                        "provenance": {
                                            "source_decision_fingerprint": fingerprint,
                                            "feature_fingerprint": vector.fingerprint,
                                        },
                                    },
                                    vector,
                                )
                                market_state = {
                                    "values": vector.values,
                                    "mask": vector.mask,
                                    "feature_names": vector.feature_names,
                                    "cohort": vector.cohort,
                                }
                            except FeatureExtractionIncompleteError:
                                vector = None
                                trust = type("T", (), {"tier": TrustTier.TIER_C_DIAGNOSTIC_ONLY})()
                                market_state = {
                                    "symbol": _source_symbol(row),
                                    "action": row.get("action"),
                                }
                            provenance = {
                                "source_database_id": source_id,
                                "source_snapshot_fingerprint": snapshot.source_snapshot_fingerprint,
                                "source_decision_fingerprint": fingerprint,
                            }
                            try:
                                record = self.catalog.upsert_source_alias(
                                    source_id,
                                    decision_id,
                                    fingerprint,
                                    source_run_id=row.get("source_run_id"),
                                    symbol=_source_symbol(row),
                                    requested_symbol=row.get("requested_symbol"),
                                    analysis_profile=row.get("analysis_profile"),
                                    analysis_timeframe=row.get("analysis_timeframe"),
                                    analysis_snapshot_timestamp=_primary_source_value(
                                        row,
                                        "analysis_snapshot_timestamp",
                                        "snapshot_timestamp",
                                    ),
                                    decision_completed_timestamp=row.get(
                                        "decision_completed_timestamp"
                                    ),
                                    decision_reference_timestamp=row.get(
                                        "decision_reference_timestamp"
                                    ),
                                    market_state=market_state,
                                    decision_evidence={"action": row.get("action")},
                                    provenance=provenance,
                                    trust=trust.tier.value,
                                )
                            except SourceDecisionConflictError:
                                quarantined += 1
                                continue
                            indexed += 1
                            experiences += 1
                            if existing is not None:
                                aliases += 1
                            if vector is not None:
                                projection, schema_version = _feature_projection_metadata(market_state)
                                self.catalog.store_feature_projection(
                                    record.experience_id, projection, schema_version=schema_version
                                )
                            eval_count += self._append_evaluations(
                                record, snapshot, decision_id, source_id
                            )
                        for alias in self.catalog.aliases_for_source(source_id):
                            if (
                                alias["source_decision_id"] not in seen
                                and alias["state"] == "CURRENT"
                            ):
                                self.catalog.mark_alias_removed(
                                    source_id,
                                    alias["source_decision_id"],
                                    alias["accepted_fingerprint"],
                                )
                except Exception as exc:
                    # The source transaction was rolled back; discard all
                    # counters that were incremented while it was tentative.
                    indexed, unchanged, aliases, quarantined, eval_count = committed_counts
                    failed += 1
                    self.catalog.record_failed_scan(str(path), type(exc).__name__)
            self.catalog.record_import_event("COMPLETED", detail={"run_id": run_id})
            return ImportReport(
                indexed,
                unchanged,
                len(self.catalog.active_records()),
                aliases,
                quarantined,
                failed,
                eval_count,
                0,
                run_id,
            )
        except Exception:
            self.catalog.record_import_event("INTERRUPTED", detail={"run_id": run_id})
            raise
        finally:
            # Import currently stores no files in this staging directory.  Do
            # not accumulate empty run folders; if a future stage writes
            # diagnostics, rmdir will fail closed and preserve them for review.
            with suppress(OSError):
                staging.rmdir()
            lock.close()
            with suppress(FileNotFoundError):
                self.lock_path.unlink()

    def reconcile_removed_aliases(self, successful_source_scan: dict[str, Iterable[str]]) -> int:
        removed = 0
        for source_id, present in successful_source_scan.items():
            present = set(present)
            for alias in self.catalog.aliases_for_source(source_id):
                if alias["state"] == "CURRENT" and alias["source_decision_id"] not in present:
                    self.catalog.mark_alias_removed(
                        source_id, alias["source_decision_id"], alias["accepted_fingerprint"]
                    )
                    removed += 1
        return removed


class ExperienceRebuilder:
    def __init__(self, catalog: ExperienceCatalog) -> None:
        self.catalog = catalog

    def rebuild(self, fail_after_stage: str | None = None) -> GenerationManifest:
        run_id = uuid.uuid4().hex
        staging = self.catalog.artifact_root / ".staging" / run_id
        staging.mkdir(parents=True, exist_ok=False)
        try:
            records = self.catalog.active_records() + self.catalog.historical_records()
            if fail_after_stage == "features":
                raise RuntimeError("requested rebuild failure")
            ids = [r.experience_id for r in records]
            population = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
            generation_id = "gen-" + population[:24]
            metadata = _generation_metadata(len(records))
            self.catalog.publish_generation(generation_id, population, metadata)
            return GenerationManifest(
                generation_id, population, len(records), len(records), metadata
            )
        finally:
            with suppress(OSError):
                staging.rmdir()


__all__ = ["ExperienceImporter", "ExperienceRebuilder", "ImportReport", "GenerationManifest"]
