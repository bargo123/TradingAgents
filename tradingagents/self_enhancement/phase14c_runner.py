"""Read-only preflight and bounded offline replay adapter for Phase 14C."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .book_factory import BookStrategyFactory
from .book_strategies import BookStrategyRegistry
from .causal import CausalDatasetError, load_causal_tick_dataset
from .models import CandidateState, ExitPolicyConfig, StrategyVersion
from .orchestrator import (
    BookCandidateExperiment,
    _read_verified_phase14a,
    _source_file_fingerprint,
    _validate_demo_source_readonly,
)
from .phase14c_inventory import (
    ArtifactRootValidationError,
    read_phase7_inventory,
    validate_fresh_artifact_root,
)
from .phase14c_models import (
    CandidateEvaluationRecord,
    Phase14CCandidateReason,
    Phase14CCandidateStatus,
    Phase14CPreflightReason,
    Phase14CReplayPreflight,
    Phase14CRunIdentity,
    Phase14CSourcePaths,
)
from .phase14c_queries import (
    build_discovery_query_bank,
    retrieve_discovery_evidence,
)
from .phase14c_selection import select_diverse_evidence
from .replay import ReplayError
from .strategy_specs import StrategySpec

_SHADOW_STATES = frozenset(
    {
        CandidateState.INSUFFICIENT_EVIDENCE.value,
        CandidateState.REJECTED.value,
        CandidateState.SHADOW_CHALLENGER.value,
    }
)
RUN_MANIFEST_NAME = "run-manifest.json"
RUN_REPORT_NAME = "phase14c-report.json"
RESUME_RESULTS_NAME = "resume-results.jsonl"
RUN_MANIFEST_SCHEMA_VERSION = "phase14c-run.v1"
MAX_LOCAL_EXTRACTION_CALLS_PER_SELECTED_GROUP = 15
_RESUME_RESULT_FIELDS = frozenset(
    {"status", "result_id", "reason_code", "stage", "accepted", "counts"}
)


class Phase14CResumeError(ValueError):
    """A Phase 14C artifact cannot be safely resumed or interpreted."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase14CResumeError("artifact JSON contains a duplicate field")
        result[key] = value
    return result


def _read_json_object(path: Path, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise Phase14CResumeError("artifact file exceeds its size limit")
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except Phase14CResumeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise Phase14CResumeError("artifact JSON is missing or invalid") from exc
    if not isinstance(value, dict):
        raise Phase14CResumeError("artifact JSON must contain an object")
    return value


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise Phase14CResumeError("run manifest already exists; refusing overwrite") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_run_manifest(
    artifact_root: Path,
    identity: Phase14CRunIdentity,
    *,
    status: str = "IN_PROGRESS",
) -> dict[str, Any]:
    """Create a safe manifest without replacing any existing run state."""

    root = Path(artifact_root).expanduser().resolve(strict=True)
    if not root.is_dir() or not isinstance(identity, Phase14CRunIdentity):
        raise Phase14CResumeError("fresh artifact directory and run identity are required")
    if status not in {"IN_PROGRESS", "INTERRUPTED"}:
        raise Phase14CResumeError("new run status must be IN_PROGRESS or INTERRUPTED")
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "identity_fingerprint": identity.fingerprint,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    path = root / RUN_MANIFEST_NAME
    _exclusive_write(path, (_canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def validate_resume_identity(
    artifact_root: Path,
    expected: Phase14CRunIdentity,
) -> dict[str, Any]:
    """Require an exactly matching, still-incomplete run identity."""

    if not isinstance(expected, Phase14CRunIdentity):
        raise TypeError("expected must be Phase14CRunIdentity")
    manifest = _read_json_object(Path(artifact_root) / RUN_MANIFEST_NAME)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise Phase14CResumeError("run manifest schema is incompatible")
    if manifest.get("identity") != expected.to_dict() or manifest.get("identity_fingerprint") != expected.fingerprint:
        raise Phase14CResumeError("run identity mismatch; refusing resume")
    if manifest.get("status") not in {"IN_PROGRESS", "INTERRUPTED"}:
        raise Phase14CResumeError("run is not resumable")
    return manifest


def update_run_manifest_status(
    artifact_root: Path,
    expected: Phase14CRunIdentity,
    status: str,
) -> dict[str, Any]:
    """Atomically advance a manifest after confirming its exact run identity."""

    if status not in {"IN_PROGRESS", "INTERRUPTED", "COMPLETE", "FAILED"}:
        raise Phase14CResumeError("unsupported run status")
    root = Path(artifact_root).expanduser().resolve(strict=True)
    manifest = _read_json_object(root / RUN_MANIFEST_NAME)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise Phase14CResumeError("run manifest schema is incompatible")
    if manifest.get("identity") != expected.to_dict() or manifest.get("identity_fingerprint") != expected.fingerprint:
        raise Phase14CResumeError("run identity mismatch; refusing manifest update")
    updated = dict(manifest)
    updated["status"] = status
    updated["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary = root / f".{RUN_MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(updated) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / RUN_MANIFEST_NAME)
    finally:
        if temporary.exists():
            temporary.unlink()
    return updated


def _safe_resume_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or not result:
        raise Phase14CResumeError("resume result must be a non-empty structured object")
    forbidden = {"prompt", "completion", "reasoning", "chainofthought", "apikey", "authorization", "password", "credential", "secret", "rawtext", "rawcompletion", "prose"}
    if any(str(key).lower().replace("_", "") in forbidden for key in result):
        raise Phase14CResumeError("resume result contains a sensitive field")
    if set(result) - _RESUME_RESULT_FIELDS or "status" not in result:
        raise Phase14CResumeError("resume result has unsupported or missing fields")
    safe: dict[str, Any] = {}
    for key, value in result.items():
        if key == "counts":
            if not isinstance(value, dict):
                raise Phase14CResumeError("resume counts must be a structured mapping")
            counts: dict[str, int] = {}
            for count_name, count_value in value.items():
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", str(count_name)):
                    raise Phase14CResumeError("resume count key is invalid")
                if str(count_name).lower().replace("_", "") in forbidden:
                    raise Phase14CResumeError("resume result contains a sensitive field")
                if type(count_value) is not int or count_value < 0:
                    raise Phase14CResumeError("resume counts must be non-negative integers")
                counts[str(count_name)] = count_value
            safe[key] = dict(sorted(counts.items()))
        elif key == "accepted":
            if type(value) is not bool:
                raise Phase14CResumeError("resume accepted value must be boolean")
            safe[key] = value
        else:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", value):
                raise Phase14CResumeError("resume status and identifiers must be safe codes")
            safe[key] = value
    return safe


def load_resume_results(path: Path) -> tuple[dict[str, Any], ...]:
    """Read complete results; repair only a malformed unterminated final record."""

    target = Path(path)
    if not target.exists():
        return ()
    try:
        raw = target.read_bytes()
        if len(raw) > 128_000_000:
            raise Phase14CResumeError("resume log exceeds its size limit")
        text = raw.decode("utf-8")
    except Phase14CResumeError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise Phase14CResumeError("resume log is unreadable") from exc
    if not text:
        return ()
    lines = text.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        complete_line = line.endswith(("\n", "\r"))
        body = line.rstrip("\r\n")
        final = index == len(lines) - 1
        try:
            value = json.loads(body, object_pairs_hook=_unique_json_object)
            if not isinstance(value, dict):
                raise Phase14CResumeError("resume record must be an object")
            cache_key = value.pop("cache_key", None)
            if not isinstance(cache_key, str) or not re.fullmatch(r"[0-9a-f]{64}", cache_key):
                raise Phase14CResumeError("resume cache key is invalid")
            result = _safe_resume_result(value)
            if cache_key in seen:
                raise Phase14CResumeError("resume log repeats a completed cache key")
            seen.add(cache_key)
            records.append({"cache_key": cache_key, **result})
            if final and not complete_line:
                try:
                    with target.open("ab") as stream:
                        stream.write(b"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as exc:
                    raise Phase14CResumeError("completed trailing resume record could not be terminated") from exc
        except (json.JSONDecodeError, Phase14CResumeError):
            if not final or complete_line:
                raise Phase14CResumeError("resume log contains a corrupt completed record") from None
            last_complete_bytes = sum(len(item.encode("utf-8")) for item in lines[:-1])
            try:
                with target.open("r+b") as stream:
                    stream.truncate(last_complete_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise Phase14CResumeError("interrupted trailing record could not be recovered") from exc
            break
    return tuple(records)


def append_resume_result(path: Path, cache_key: str, result: Any) -> bool:
    """Append one safe completed key; return False when it is already present."""

    if not isinstance(cache_key, str) or not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise Phase14CResumeError("resume cache key must be a lowercase SHA-256 digest")
    normalized = _safe_resume_result(result)
    records = load_resume_results(Path(path))
    if any(item["cache_key"] == cache_key for item in records):
        return False
    record = {"cache_key": cache_key, **normalized}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("ab") as stream:
            stream.write((_canonical_json(record) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise Phase14CResumeError("resume result could not be appended") from exc
    return True


class _ReadOnlyKnowledgeCatalog:
    """Minimal Phase 7 catalog adapter that can only open SQLite read-only."""

    def __init__(self, catalog_path: Path) -> None:
        self.path = Path(catalog_path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise FileNotFoundError("Phase 7 catalog does not exist")

    @contextmanager
    def _read(self):
        connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            yield connection
        finally:
            connection.close()

    def active_generation(self):
        from tradingagents.knowledge.catalog import KnowledgeCatalog

        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_index_generations WHERE is_active = 1"
            ).fetchone()
        return None if row is None else KnowledgeCatalog._row_generation(row)

    def get_document(self, document_id: str):
        from tradingagents.knowledge.catalog import KnowledgeCatalog

        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return None if row is None else KnowledgeCatalog._row_document(row)

    def document_is_retrieval_ready(self, document_id: str) -> bool:
        with self._read() as connection:
            row = connection.execute(
                """SELECT 1 FROM knowledge_documents AS document
                   JOIN knowledge_index_generations AS generation
                     ON generation.generation_id = document.projection_generation
                    AND generation.is_active = 1
                   WHERE document.document_id = ?
                     AND document.active = 1
                     AND document.vector_ready = 1
                     AND document.lexical_ready = 1
                     AND EXISTS (
                         SELECT 1 FROM knowledge_aliases AS alias
                         WHERE alias.document_id = document.document_id
                           AND alias.relation_status = 'CURRENT'
                     )""",
                (document_id,),
            ).fetchone()
        return row is not None


class _ReadOnlyLexicalIndexReader:
    """FTS reader using mode=ro so planning cannot create a SQLite journal."""

    def __init__(self, location: Path) -> None:
        self.location = Path(location).expanduser().resolve(strict=True)
        if not self.location.is_file():
            raise FileNotFoundError("Phase 7 lexical projection does not exist")

    @contextmanager
    def _read(self):
        connection = sqlite3.connect(
            f"{self.location.as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            yield connection
        finally:
            connection.close()

    def metadata(self) -> dict[str, Any]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM knowledge_index_metadata ORDER BY key"
            ).fetchall()
        try:
            return {row["key"]: json.loads(row["value_json"]) for row in rows}
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 7 lexical metadata is invalid") from exc

    def search(self, query: str, *, limit: int = 50, **_filters: Any) -> tuple[dict[str, Any], ...]:
        if not str(query).strip():
            return ()
        from tradingagents.knowledge.lexical_index import _normalize_text

        with self._read() as connection:
            rows = connection.execute(
                "SELECT provenance.provenance_json, bm25(knowledge_fts) AS lexical_score "
                "FROM knowledge_fts JOIN knowledge_chunk_provenance AS provenance "
                "ON provenance.chunk_id = knowledge_fts.chunk_id "
                "WHERE knowledge_fts MATCH ? ORDER BY lexical_score, provenance.chunk_id LIMIT ?",
                (_normalize_text(query), int(limit)),
            ).fetchall()
        result = []
        for row in rows:
            try:
                item = json.loads(row["provenance_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Phase 7 lexical provenance is invalid") from exc
            item["lexical_score"] = float(row["lexical_score"])
            result.append(item)
        return tuple(result)


def _disjoint_source_root(knowledge_root: Path, embedding_path: Path) -> Path:
    candidates = (Path.cwd(), Path(os.getenv("TEMP", "")), Path.home())
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if not resolved.is_dir():
            continue
        if _overlap(resolved, knowledge_root) or _overlap(resolved, embedding_path):
            continue
        return resolved
    raise ValueError("no existing local directory is disjoint from the Phase 7 and embedding roots")


def _overlap(left: Path, right: Path) -> bool:
    try:
        common = Path(os.path.commonpath((str(left.resolve()), str(right.resolve()))))
    except (OSError, ValueError):
        return False
    return common in {left.resolve(), right.resolve()}


def _knowledge_config_for_generation(
    generation: Any,
    *,
    knowledge_root: Path,
    embedding_model_path: Path,
):
    from tradingagents.knowledge.config import KnowledgeConfig

    spec = generation.embedding_spec
    return KnowledgeConfig(
        source_root=_disjoint_source_root(knowledge_root, embedding_model_path),
        artifact_root=knowledge_root,
        embedding_model_id=spec.model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimensions=spec.dimensions,
        embedding_runtime=spec.runtime,
        embedding_normalization=spec.normalization_policy,
        embedding_model_version=spec.resolved_model_version,
        embedding_artifact_hash=spec.artifact_hash,
        embedding_tokenizer_fingerprint=spec.tokenizer_fingerprint,
        embedding_max_input_tokens=spec.model_max_input_tokens,
        embedding_special_token_budget=spec.special_token_budget,
        embedding_effective_content_token_limit=spec.effective_corpus_content_token_limit,
        embedding_truncation=spec.truncation,
        embedding_corpus_instruction_policy=spec.corpus_instruction_policy,
        embedding_corpus_instruction_version=spec.corpus_instruction_version,
        embedding_query_instruction_policy=spec.query_instruction_policy,
        embedding_query_instruction_version=spec.query_instruction_version,
        offline=True,
        worker_count=1,
    )


def plan_phase14c(
    *,
    knowledge_root: Path,
    expected_generation_id: str,
    expected_generation_fingerprint: str,
    expected_population_hash: str,
    embedding_model_path: Path,
    timeout_seconds: float = 300.0,
    top_k_per_formulation: int = 10,
) -> dict[str, Any]:
    """Perform deterministic, read-only retrieval planning without a teacher."""

    root = Path(knowledge_root).expanduser().resolve(strict=True)
    model_path = Path(embedding_model_path).expanduser().resolve(strict=True)
    if not root.is_dir() or not model_path.is_dir() or not any(model_path.iterdir()):
        raise ValueError("knowledge root and pre-provisioned embedding model directory are required")
    if isinstance(timeout_seconds, bool) or not math.isfinite(float(timeout_seconds)) or not 0 < timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be bounded between 0 and 600")
    if type(top_k_per_formulation) is not int or not 1 <= top_k_per_formulation <= 50:
        raise ValueError("top_k_per_formulation must be between 1 and 50")

    pin, inventory = read_phase7_inventory(
        root,
        expected_generation_id=expected_generation_id,
        expected_generation_fingerprint=expected_generation_fingerprint,
        expected_population_hash=expected_population_hash,
    )
    catalog = _ReadOnlyKnowledgeCatalog(root / "catalog.sqlite3")
    generation = catalog.active_generation()
    if generation is None or generation.generation_id != pin.generation_id:
        raise ValueError("active Phase 7 generation changed after inventory validation")
    config = _knowledge_config_for_generation(
        generation,
        knowledge_root=root,
        embedding_model_path=model_path,
    )

    from tradingagents.knowledge.embeddings import FastEmbedProvider
    from tradingagents.knowledge.query import KnowledgeQueryService
    from tradingagents.knowledge.vector_index import VectorIndexReader

    embedder = FastEmbedProvider.from_config(config)
    if embedder.spec.to_dict() != generation.embedding_spec.to_dict():
        raise ValueError("local embedding model specification differs from the pinned Phase 7 index")
    vector_location = Path(generation.vector_location)
    lexical_location = Path(generation.lexical_location)
    if not vector_location.is_absolute():
        vector_location = root / vector_location
    if not lexical_location.is_absolute():
        lexical_location = root / lexical_location
    query_service = KnowledgeQueryService(
        VectorIndexReader(vector_location),
        _ReadOnlyLexicalIndexReader(lexical_location),
        catalog,
        embedder,
    )
    bank = build_discovery_query_bank()
    retrieval = retrieve_discovery_evidence(
        query_service,
        bank,
        pinned_generation_id=pin.generation_id,
        pinned_population_hash=pin.population_hash,
        top_k_per_formulation=top_k_per_formulation,
    )
    if retrieval.status != "COMPLETE":
        raise ValueError("Phase 7 retrieval planning failed closed")
    selection = select_diverse_evidence(retrieval)
    after_pin, after_inventory = read_phase7_inventory(
        root,
        expected_generation_id=expected_generation_id,
        expected_generation_fingerprint=expected_generation_fingerprint,
        expected_population_hash=expected_population_hash,
    )
    if pin != after_pin or inventory != after_inventory:
        raise ValueError("frozen Phase 7 identity or inventory changed during plan-only retrieval")
    return {
        "status": "PLANNED",
        "generation_id": pin.generation_id,
        "generation_fingerprint": pin.generation_fingerprint,
        "population_hash": pin.population_hash,
        "source_manifest_fingerprint": inventory.source_manifest_fingerprint,
        "inventory": asdict(inventory),
        "query_bank_version": retrieval.query_bank_version,
        "query_bank_fingerprint": retrieval.query_bank_fingerprint,
        "query_count": retrieval.completed_query_count,
        "raw_hit_count": retrieval.raw_hit_count,
        "unique_hit_count": retrieval.unique_hit_count,
        "selected_group_count": selection.selected_group_count,
        "deferred_group_count": selection.deferred_group_count,
        "unclassified_chunk_count": selection.unclassified_chunk_count,
        "selected_groups_by_family": dict(selection.selected_groups_by_family),
        "deferred_groups_by_family": dict(selection.deferred_groups_by_family),
        "llm_calls": 0,
        "conservative_estimated_teacher_calls": (
            selection.selected_group_count * MAX_LOCAL_EXTRACTION_CALLS_PER_SELECTED_GROUP
        ),
        "conservative_estimated_runtime_seconds": (
            selection.selected_group_count
            * MAX_LOCAL_EXTRACTION_CALLS_PER_SELECTED_GROUP
            * float(timeout_seconds)
        ),
        "runtime_estimate_basis": (
            "selected groups x 15 maximum local extraction attempts per group x timeout"
        ),
    }


def validate_discovery_paths(
    *,
    knowledge_root: Path,
    embedding_model_path: Path,
    artifact_root: Path,
    atomic_cache_path: Path,
    source_paths: Phase14CSourcePaths,
    allow_existing_artifact_root: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Validate all Phase 14C output/cache boundaries before any model setup."""

    knowledge = Path(knowledge_root).expanduser().resolve(strict=True)
    embedding = Path(embedding_model_path).expanduser().resolve(strict=True)
    artifact = Path(artifact_root)
    cache = Path(atomic_cache_path)
    if not knowledge.is_dir():
        raise ValueError("knowledge root must be an existing directory")
    if not embedding.is_dir() or not any(embedding.iterdir()):
        raise ValueError("embedding model path must be a non-empty pre-provisioned local directory")
    if _overlap(knowledge, embedding):
        raise ArtifactRootValidationError("embedding model path overlaps the frozen Phase 7 root")
    if not artifact.is_absolute() or not cache.is_absolute():
        raise ValueError("Phase 14C artifact and atomic cache paths must be explicit absolute paths")
    artifact = artifact.expanduser().resolve(strict=False)
    cache = cache.expanduser().resolve(strict=False)
    paths = (source_paths.phase14a_path, source_paths.hft_path, source_paths.demo_path)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("all three explicit read-only Phase 14 sources must exist")
    if len(set(paths)) != 3:
        raise ValueError("Phase 14A, HFT, and DEMO source files must be distinct")
    source_parent_roots = tuple(path.parent for path in paths)
    protected = (knowledge, embedding, *paths, *source_parent_roots)
    if any(_overlap(embedding, source_root) for source_root in source_parent_roots):
        raise ArtifactRootValidationError("embedding model path overlaps a read-only source directory")
    if allow_existing_artifact_root:
        if not artifact.is_dir():
            raise FileNotFoundError("resume artifact root must already exist")
        for protected_path in protected:
            if _overlap(artifact, protected_path):
                raise ArtifactRootValidationError("resume artifact root overlaps protected data")
    else:
        validate_fresh_artifact_root(artifact, protected_roots=protected)
    cache_parent = cache.parent
    if not cache_parent.is_dir():
        raise ValueError("atomic cache parent directory must already exist")
    for protected_path in (*protected, artifact):
        if _overlap(cache, protected_path):
            raise ArtifactRootValidationError("atomic cache overlaps a protected source or artifact root")
    if cache.exists() and not cache.is_file():
        raise ValueError("atomic cache path must be a file when it already exists")
    return knowledge, embedding, artifact, cache


def load_complete_discovery_artifacts(artifact_root: Path) -> tuple[tuple[StrategySpec, ...], dict[str, Any]]:
    """Load only integrity-checked, complete, strictly parsed StrategySpecs."""

    root = Path(artifact_root).expanduser().resolve(strict=True)
    manifest = _read_json_object(root / RUN_MANIFEST_NAME)
    report = _read_json_object(root / RUN_REPORT_NAME)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION or manifest.get("status") != "COMPLETE":
        raise Phase14CResumeError("discovery run is not complete or has an unsupported manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("identity_fingerprint") != hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest():
        raise Phase14CResumeError("completed run identity integrity is invalid")
    if report.get("status") != "COMPLETE" or report.get("artifact_integrity") != "VALID":
        raise Phase14CResumeError("discovery report is incomplete or failed integrity validation")
    filename = report.get("validated_specs_file")
    expected_digest = report.get("validated_specs_sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not re.fullmatch(r"[0-9a-f]{64}", str(expected_digest))
    ):
        raise Phase14CResumeError("validated-spec artifact reference is invalid")
    specs_path = root / filename
    try:
        payload_bytes = specs_path.read_bytes()
    except OSError as exc:
        raise Phase14CResumeError("validated-spec artifact is missing") from exc
    if hashlib.sha256(payload_bytes).hexdigest() != expected_digest:
        raise Phase14CResumeError("validated-spec artifact fingerprint mismatch")
    try:
        specs_payload = json.loads(payload_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase14CResumeError("validated-spec artifact is malformed") from exc
    if (
        not isinstance(specs_payload, dict)
        or specs_payload.get("schema_version") != "phase14c-validated-specs.v1"
        or not isinstance(specs_payload.get("specs"), list)
    ):
        raise Phase14CResumeError("validated-spec artifact schema is unsupported")
    try:
        specs = tuple(StrategySpec.from_dict(item) for item in specs_payload["specs"])
    except (TypeError, ValueError, KeyError) as exc:
        raise Phase14CResumeError("a validated StrategySpec failed strict parsing") from exc
    for spec in specs:
        if (
            spec.generation_id != identity.get("generation_id")
            or spec.knowledge_fingerprint != identity.get("generation_fingerprint")
        ):
            raise Phase14CResumeError("a validated StrategySpec does not match the pinned Phase 7 generation")
        supported = {
            item.rule_fingerprint: item.status.value
            for item in spec.validation_results
        }
        if any(supported.get(claim.fingerprint) != "SUPPORTED" for claim in spec.source_supported_rules):
            raise Phase14CResumeError("artifact contains a StrategySpec without complete evidence validation")
    return specs, report


def _source_path_map(paths: Phase14CSourcePaths) -> dict[str, Path]:
    return {
        "phase14a": paths.phase14a_path,
        "hft": paths.hft_path,
        "demo": paths.demo_path,
    }


def _fingerprint_sources(paths: Phase14CSourcePaths) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name, path in _source_path_map(paths).items():
        try:
            if path.is_file():
                fingerprints[name] = _source_file_fingerprint(path)
        except OSError:
            continue
    return fingerprints


def _demo_failure_reason(exc: Exception) -> Phase14CPreflightReason:
    message = str(exc).lower()
    if "reconciliation" in message or "chronology" in message:
        return Phase14CPreflightReason.DEMO_RECONCILIATION_UNCERTAIN
    return Phase14CPreflightReason.INVALID_DEMO_SOURCE


def preflight_phase14_sources(
    phase14a_path: Path,
    hft_path: Path,
    demo_path: Path,
) -> Phase14CReplayPreflight:
    """Inspect three explicit source databases without modifying any of them."""

    paths = Phase14CSourcePaths(phase14a_path, hft_path, demo_path)
    before = _fingerprint_sources(paths)
    reason: Phase14CPreflightReason | None = None
    if len(before) != 3:
        reason = Phase14CPreflightReason.SOURCE_UNAVAILABLE

    verified_count = quarantined_count = tick_count = segment_count = 0
    path_map = _source_path_map(paths)

    if "phase14a" in before:
        try:
            experiences, quarantined_count = _read_verified_phase14a(path_map["phase14a"])
            verified_count = len(experiences)
            if not experiences and reason is None:
                reason = Phase14CPreflightReason.NO_VERIFIED_PHASE14A_EXPERIENCE
        except (OSError, ValueError, TypeError, sqlite3.Error):
            if reason is None:
                reason = Phase14CPreflightReason.INVALID_PHASE14A

    if "demo" in before:
        try:
            _validate_demo_source_readonly(path_map["demo"])
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            if reason is None:
                reason = _demo_failure_reason(exc)

    if "hft" in before:
        try:
            dataset = load_causal_tick_dataset(path_map["hft"], symbol="EURUSD")
            tick_count = dataset.valid_rows
            segment_count = len(dataset.segments)
            if (dataset.valid_rows == 0 or dataset.invalid_rows > 0 or not dataset.segments) and reason is None:
                reason = Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA
        except (OSError, ValueError, TypeError, sqlite3.Error, ReplayError, CausalDatasetError):
            if reason is None:
                reason = Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA

    after = _fingerprint_sources(paths)
    if before != after:
        reason = Phase14CPreflightReason.SOURCE_MUTATED_DURING_PREFLIGHT
    if reason is None:
        reason = Phase14CPreflightReason.READY

    return Phase14CReplayPreflight(
        source_paths=paths,
        source_fingerprints=before,
        verified_experience_count=verified_count,
        quarantined_experience_count=quarantined_count,
        causal_tick_count=tick_count,
        causal_segment_count=segment_count,
        commission_status="UNKNOWN",
        ready=reason is Phase14CPreflightReason.READY,
        reason_code=reason,
    )


def _record_not_run(
    spec: StrategySpec,
    reason: Phase14CCandidateReason | Phase14CPreflightReason,
    source_fingerprints: dict[str, str],
    *,
    candidate_id: str | None = None,
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        spec_id=spec.spec_id,
        candidate_id=candidate_id,
        status=Phase14CCandidateStatus.NOT_RUN,
        reason_code=reason,
        artifact_path=None,
        source_fingerprints=source_fingerprints,
    )


def _record_failed(
    spec: StrategySpec,
    candidate_id: str,
    reason: Phase14CCandidateReason,
    artifact_path: Path,
    source_fingerprints: dict[str, str],
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        spec_id=spec.spec_id,
        candidate_id=candidate_id,
        status=Phase14CCandidateStatus.FAILED,
        reason_code=reason,
        artifact_path=artifact_path,
        source_fingerprints=source_fingerprints,
    )


def evaluate_phase14c_candidates(
    specs: Sequence[StrategySpec],
    *,
    artifact_root: Path,
    source_paths: Phase14CSourcePaths,
    source_commit: str,
    max_candidate_runs: int = 5,
) -> tuple[CandidateEvaluationRecord, ...]:
    """Run a bounded batch through the existing read-only causal experiment."""

    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of StrategySpec values")
    requested = tuple(specs)
    if any(not isinstance(spec, StrategySpec) for spec in requested):
        raise TypeError("specs must contain only StrategySpec values")
    if not isinstance(source_paths, Phase14CSourcePaths):
        raise TypeError("source_paths must be Phase14CSourcePaths")
    if type(max_candidate_runs) is not int or not 1 <= max_candidate_runs <= 10:
        raise ValueError("max_candidate_runs must be an integer from 1 through 10")
    commit = str(source_commit).strip()
    if not commit:
        raise ValueError("source_commit is required")
    root = Path(artifact_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError("Phase 14C candidate artifact root must be fresh")
    if not requested:
        return ()

    preflight = preflight_phase14_sources(
        source_paths.phase14a_path,
        source_paths.hft_path,
        source_paths.demo_path,
    )
    fingerprints = dict(preflight.source_fingerprints)
    if not preflight.ready:
        return tuple(
            _record_not_run(spec, preflight.reason_code, fingerprints)
            for spec in requested
        )

    records: list[CandidateEvaluationRecord] = []
    runs = 0
    seen_spec_hashes: set[str] = set()
    stop_for_source_mutation = False
    registry = BookStrategyRegistry()
    factory = BookStrategyFactory()

    for spec in requested:
        if stop_for_source_mutation:
            records.append(
                _record_not_run(
                    spec,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    fingerprints,
                )
            )
            continue
        if not spec.is_executable:
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.SPEC_NOT_EXECUTABLE, fingerprints)
            )
            continue
        if spec.content_hash in seen_spec_hashes:
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.DUPLICATE_SPEC, fingerprints)
            )
            continue
        seen_spec_hashes.add(spec.content_hash)

        try:
            implementation = registry.create(spec)
        except (TypeError, ValueError):
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.UNIMPLEMENTED_SPEC, fingerprints)
            )
            continue

        parent = StrategyVersion(
            strategy_id=implementation.strategy_id,
            strategy_version="phase14-reviewed-control-v1",
            config_version="phase14-default-exits-v1",
            parameters=ExitPolicyConfig().to_dict(),
            source_commit=commit,
        )
        candidate = factory.from_validated_spec(spec, parent=parent)
        if runs >= max_candidate_runs:
            records.append(
                _record_not_run(
                    spec,
                    Phase14CCandidateReason.MAX_CANDIDATE_RUNS_REACHED,
                    fingerprints,
                    candidate_id=candidate.candidate_id,
                )
            )
            continue

        if _fingerprint_sources(source_paths) != fingerprints:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    root / candidate.candidate_id,
                    _fingerprint_sources(source_paths),
                )
            )
            stop_for_source_mutation = True
            continue

        run_root = root / candidate.candidate_id
        runs += 1
        try:
            report = BookCandidateExperiment(run_root).run(
                phase14a_path=source_paths.phase14a_path,
                hft_path=source_paths.hft_path,
                demo_path=source_paths.demo_path,
                candidate=candidate,
                source_commit=commit,
            )
        except (OSError, ValueError, TypeError, ReplayError):
            current = _fingerprint_sources(source_paths)
            changed = current != fingerprints
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION
                    if changed
                    else Phase14CCandidateReason.REPLAY_FAILED,
                    run_root,
                    current,
                )
            )
            stop_for_source_mutation = changed
            continue

        current = _fingerprint_sources(source_paths)
        report_fingerprints = dict(getattr(report, "source_fingerprints", {}))
        if current != fingerprints or report_fingerprints != fingerprints or not report.source_unchanged:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    run_root,
                    current,
                )
            )
            stop_for_source_mutation = True
            continue

        candidate_state = str(report.candidate_state)
        if report.status != "COMPLETED":
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.REPLAY_FAILED,
                    run_root,
                    current,
                )
            )
            continue
        if candidate_state not in _SHADOW_STATES:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.STATE_EXCEEDS_SHADOW_CEILING,
                    run_root,
                    current,
                )
            )
            continue
        records.append(
            CandidateEvaluationRecord(
                spec_id=spec.spec_id,
                candidate_id=candidate.candidate_id,
                status=Phase14CCandidateStatus.COMPLETED,
                reason_code=Phase14CCandidateReason.REPLAY_COMPLETED,
                artifact_path=run_root,
                gate_decision=report.gate_decision,
                gate_reasons=tuple(report.gate_reasons),
                source_fingerprints=current,
                candidate_state=candidate_state,
            )
        )

    return tuple(records)


__all__ = [
    "RUN_MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_REPORT_NAME",
    "RESUME_RESULTS_NAME",
    "MAX_LOCAL_EXTRACTION_CALLS_PER_SELECTED_GROUP",
    "Phase14CResumeError",
    "append_resume_result",
    "evaluate_phase14c_candidates",
    "load_complete_discovery_artifacts",
    "load_resume_results",
    "plan_phase14c",
    "preflight_phase14_sources",
    "update_run_manifest_status",
    "validate_discovery_paths",
    "validate_resume_identity",
    "write_run_manifest",
]
