"""Read-only preflight and bounded offline replay adapter for Phase 14C."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from tradingagents.forex.hft.features import TickFeatures

from .book_atomic_extraction import (
    _PRESENCE_PROMPT_CACHE_VERSION,
    _PRESENCE_SCHEMA_CACHE_VERSION,
    ATOMIC_PROMPT_VERSION,
    EVIDENCE_SEGMENTATION_VERSION,
    AtomicStrategyExtractor,
    PresenceStatus,
)
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
from .phase14c_assembly import (
    PHASE14C_ASSEMBLY_SCHEMA_VERSION,
    PHASE14C_ASSEMBLY_VERSION,
    assemble_atomic_concepts,
)
from .phase14c_inventory import (
    ArtifactRootValidationError,
    read_phase7_inventory,
    validate_fresh_artifact_root,
)
from .phase14c_mapping import (
    PHASE14C_MAPPING_SCHEMA_VERSION,
    PHASE14C_MAPPING_VERSION,
    capture_current_strategy_contracts,
    map_current_strategies,
)
from .phase14c_models import (
    CandidateEvaluationRecord,
    ChunkIdentity,
    CorpusInventory,
    Phase14CCandidateReason,
    Phase14CCandidateStatus,
    Phase14CGenerationPin,
    Phase14CPreflightReason,
    Phase14CReplayPreflight,
    Phase14CRunIdentity,
    Phase14CSourcePaths,
)
from .phase14c_queries import (
    DISCOVERY_QUERY_BANK_VERSION,
    build_discovery_query_bank,
    query_bank_fingerprint,
    retrieve_discovery_evidence,
)
from .phase14c_selection import (
    DISCOVERY_SELECTION_VERSION,
    select_diverse_evidence,
)
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
RESUME_SCHEMA_VERSION = "phase14c-resume.v2"
ATOMIC_CACHE_SCHEMA_VERSION = "phase14b-atomic-results.v1"
VALIDATED_SPEC_SCHEMA_VERSION = "phase14c-validated-specs.v1"
REPORT_SCHEMA_VERSION = "phase14c-report.v1"
MAX_LOCAL_EXTRACTION_CALLS_PER_SELECTED_GROUP = 15
_DISCOVERY_ARTIFACT_FILES = (
    "corpus-inventory.json",
    "query-bank.json",
    "retrieval/evidence.jsonl",
    "selection/coverage.json",
    "classification/validated-results.jsonl",
    "concepts/concepts.jsonl",
    "concepts/conflicts.jsonl",
    "specs/strategy-specs.jsonl",
    "specs/assembly-records.jsonl",
    "mappings/current-strategy-mapping.json",
    "validated-specs.json",
    RESUME_RESULTS_NAME,
)
_RESUME_RESULT_FIELDS = frozenset(
    {"status", "result_id", "reason_code", "stage", "accepted", "counts", "telemetry"}
)
_TELEMETRY_FIELDS = frozenset(
    {
        "stage",
        "evidence_count",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
        "finish_reason",
        "outcome",
        "retry_depth",
        "cache_hit",
        "error_type",
    }
)
_TELEMETRY_CODE = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{0,127}")


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
        elif key == "telemetry":
            if not isinstance(value, list) or len(value) > 256:
                raise Phase14CResumeError("resume telemetry must be a bounded list")
            safe[key] = [_safe_telemetry_record(item) for item in value]
        elif key == "accepted":
            if type(value) is not bool:
                raise Phase14CResumeError("resume accepted value must be boolean")
            safe[key] = value
        else:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", value):
                raise Phase14CResumeError("resume status and identifiers must be safe codes")
            safe[key] = value
    return safe


def _safe_telemetry_record(value: Any) -> dict[str, Any]:
    """Validate provider metadata without allowing arbitrary response text."""

    if not isinstance(value, Mapping) or set(value) != _TELEMETRY_FIELDS:
        raise Phase14CResumeError("resume telemetry has unsupported fields")
    safe: dict[str, Any] = {}
    for field_name in ("stage", "outcome"):
        field_value = value[field_name]
        if not isinstance(field_value, str) or not _TELEMETRY_CODE.fullmatch(field_value):
            raise Phase14CResumeError("resume telemetry contains an invalid code")
        safe[field_name] = field_value
    for field_name in ("finish_reason", "error_type"):
        field_value = value[field_name]
        if field_value is not None and (
            not isinstance(field_value, str) or not _TELEMETRY_CODE.fullmatch(field_value)
        ):
            raise Phase14CResumeError("resume telemetry contains unsafe text")
        safe[field_name] = field_value
    for field_name in (
        "evidence_count",
        "retry_depth",
        "input_tokens",
        "output_tokens",
    ):
        field_value = value[field_name]
        if field_value is None and field_name in {"input_tokens", "output_tokens"}:
            safe[field_name] = None
            continue
        if type(field_value) is not int or field_value < 0 or field_value > 10_000_000:
            raise Phase14CResumeError("resume telemetry count is outside its allowed range")
        safe[field_name] = field_value
    elapsed = value["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or not 0 <= float(elapsed) <= 86_400
    ):
        raise Phase14CResumeError("resume telemetry elapsed time is outside its allowed range")
    safe["elapsed_seconds"] = round(float(elapsed), 6)
    if type(value["cache_hit"]) is not bool:
        raise Phase14CResumeError("resume telemetry cache_hit must be boolean")
    safe["cache_hit"] = value["cache_hit"]
    return safe


def _validate_resume_log_path(path: Path, artifact_root: Path | None = None) -> None:
    """Reject resume-log aliases and paths outside the owning run directory."""

    target = Path(path)
    if artifact_root is not None:
        try:
            root = Path(artifact_root).resolve(strict=True)
            parent = target.parent.resolve(strict=True)
        except OSError as exc:
            raise Phase14CResumeError("unsafe resume log path") from exc
        if not root.is_dir() or parent != root:
            raise Phase14CResumeError("unsafe resume log path")

    try:
        info = target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise Phase14CResumeError("unsafe resume log path") from exc

    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    if (
        target.is_symlink()
        or attributes & reparse_point
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise Phase14CResumeError("unsafe resume log path")


def load_resume_results(
    path: Path, *, artifact_root: Path | None = None
) -> tuple[dict[str, Any], ...]:
    """Read complete results; repair only a malformed unterminated final record."""

    target = Path(path)
    _validate_resume_log_path(target, artifact_root)
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
                    _validate_resume_log_path(target, artifact_root)
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
                _validate_resume_log_path(target, artifact_root)
                with target.open("r+b") as stream:
                    stream.truncate(last_complete_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise Phase14CResumeError("interrupted trailing record could not be recovered") from exc
            break
    return tuple(records)


def append_resume_result(
    path: Path,
    cache_key: str,
    result: Any,
    *,
    artifact_root: Path | None = None,
) -> bool:
    """Append one safe completed key; return False when it is already present."""

    if not isinstance(cache_key, str) or not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise Phase14CResumeError("resume cache key must be a lowercase SHA-256 digest")
    normalized = _safe_resume_result(result)
    target = Path(path)
    _validate_resume_log_path(target, artifact_root)
    records = load_resume_results(target, artifact_root=artifact_root)
    if any(item["cache_key"] == cache_key for item in records):
        return False
    record = {"cache_key": cache_key, **normalized}
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _validate_resume_log_path(target, artifact_root)
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
        # Resolve identity from the local artifact bytes. Passing the pinned
        # index values here would make a different local model appear to match
        # merely because its config was populated from the generation.
        embedding_artifact_hash="",
        embedding_tokenizer_fingerprint="",
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


def _build_local_query_service(
    knowledge_root: Path,
    embedding_model_path: Path,
    generation_pin: Phase14CGenerationPin,
) -> tuple[Any, Any]:
    """Construct an offline read-only query stack matching the pinned generation."""

    from tradingagents.knowledge.embeddings import FastEmbedProvider
    from tradingagents.knowledge.query import KnowledgeQueryService
    from tradingagents.knowledge.vector_index import VectorIndexReader

    root = Path(knowledge_root).expanduser().resolve(strict=True)
    embedding_path = Path(embedding_model_path).expanduser().resolve(strict=True)
    catalog = _ReadOnlyKnowledgeCatalog(root / "catalog.sqlite3")
    generation = catalog.active_generation()
    if (
        generation is None
        or generation.generation_id != generation_pin.generation_id
        or generation.population_hash != generation_pin.population_hash
        or generation.status != "VALIDATED"
        or not generation.vector_ready
        or not generation.lexical_ready
    ):
        raise ValueError("active Phase 7 generation changed after inventory validation")
    config = _knowledge_config_for_generation(
        generation,
        knowledge_root=root,
        embedding_model_path=embedding_path,
    )
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
    return query_service, embedder


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
    query_service, _embedder = _build_local_query_service(root, model_path, pin)
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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported Phase 14C artifact value: {type(value).__name__}")


def _write_artifact(root: Path, relative_path: str, payload: bytes) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase14CResumeError("artifact path must remain inside the Phase 14C root")
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise Phase14CResumeError("Phase 14C artifact already exists; refusing overwrite") from exc
    except OSError as exc:
        raise Phase14CResumeError("Phase 14C artifact could not be written") from exc
    return target


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(_jsonable(value)) + "\n").encode("utf-8")


def _jsonl_bytes(values: Sequence[Any]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)


def _phase7_identity(pin: Phase14CGenerationPin, inventory: CorpusInventory) -> dict[str, Any]:
    return {"generation": _jsonable(pin), "inventory": _jsonable(inventory)}


def _build_run_identity(
    *,
    pin: Phase14CGenerationPin,
    inventory: CorpusInventory,
    embedding_spec: Any,
    query_bank: Sequence[Any],
    top_k_per_formulation: int,
    selection_limits: Any,
    endpoint: str,
    model: str,
    model_version: str,
    timeout_seconds: float,
    max_output_tokens: int,
    context_tokens: int,
    atomic_cache_path: Path,
    source_paths: Phase14CSourcePaths,
    source_fingerprints: Mapping[str, str],
) -> Phase14CRunIdentity:

    artifact_schema = {
        "manifest": RUN_MANIFEST_SCHEMA_VERSION,
        "report": REPORT_SCHEMA_VERSION,
        "validated_specs": VALIDATED_SPEC_SCHEMA_VERSION,
        "resume": RESUME_SCHEMA_VERSION,
    }
    stage_budgets = dict(AtomicStrategyExtractor._STAGE_BUDGETS)
    source_path_map = _source_path_map(source_paths)
    return Phase14CRunIdentity(
        generation_id=pin.generation_id,
        generation_fingerprint=pin.generation_fingerprint,
        population_hash=pin.population_hash,
        embedding_spec_fingerprint=_sha256_json(embedding_spec.to_dict()),
        query_bank_version=DISCOVERY_QUERY_BANK_VERSION,
        query_bank_fingerprint=query_bank_fingerprint(query_bank),
        top_k_per_formulation=top_k_per_formulation,
        selection_version=DISCOVERY_SELECTION_VERSION,
        selection_config_fingerprint=_sha256_json(selection_limits),
        assembly_version=PHASE14C_ASSEMBLY_VERSION,
        assembly_schema_version=PHASE14C_ASSEMBLY_SCHEMA_VERSION,
        artifact_schema_fingerprint=_sha256_json(artifact_schema),
        provider="ollama-local",
        ollama_endpoint=endpoint,
        model_id=model,
        model_version=model_version,
        temperature=0.0,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        context_tokens=context_tokens,
        atomic_cache_path=atomic_cache_path,
        atomic_cache_schema_version=ATOMIC_CACHE_SCHEMA_VERSION,
        presence_prompt_version=_PRESENCE_PROMPT_CACHE_VERSION,
        presence_schema_version=_PRESENCE_SCHEMA_CACHE_VERSION,
        extraction_prompt_version=ATOMIC_PROMPT_VERSION,
        segmentation_version=EVIDENCE_SEGMENTATION_VERSION,
        stage_budgets_fingerprint=_sha256_json(stage_budgets),
        resume_schema_version=RESUME_SCHEMA_VERSION,
        source_paths={name: str(path) for name, path in source_path_map.items()},
        source_fingerprints=source_fingerprints,
    )


def _presence_cache_key(identity: Phase14CRunIdentity, group: Any) -> str:
    return _sha256_json(
        {
            "identity_fingerprint": identity.fingerprint,
            "group_id": group.group_id,
            "evidence_ids": [sentence.evidence_id for sentence in group.sentences],
        }
    )


def _validate_teacher(teacher: Any, *, endpoint: str, model: str, model_version: str,
                      timeout_seconds: float, max_output_tokens: int, context_tokens: int) -> None:
    expected = {
        "endpoint": endpoint.rstrip("/"),
        "model": model,
        "model_version": model_version,
        "timeout_seconds": float(timeout_seconds),
        "max_output_tokens": max_output_tokens,
        "context_tokens": context_tokens,
    }
    for name, value in expected.items():
        if getattr(teacher, name, None) != value:
            raise ValueError(f"local teacher {name} differs from the explicit run identity")
    if not callable(getattr(teacher, "classify_presence", None)) or not callable(
        getattr(teacher, "extract_actionable", None)
    ):
        raise TypeError("local teacher does not expose the Phase 14B bounded extraction API")


def classify_discovery_outcome(
    *,
    complete_spec_count: int,
    hft_suitable_spec_count: int,
    deferred_group_count: int,
    classification_failure_count: int,
) -> tuple[str, str]:
    """Return a closed discovery state without treating partial coverage as complete."""

    counts = (complete_spec_count, hft_suitable_spec_count, deferred_group_count, classification_failure_count)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("discovery outcome counts must be non-negative integers")
    if deferred_group_count:
        return "INCOMPLETE", "DISCOVERY_INCOMPLETE_BUDGET_LIMITED"
    if classification_failure_count:
        return "INCOMPLETE", "DISCOVERY_INCOMPLETE_CLASSIFICATION"
    if complete_spec_count == 0:
        return "COMPLETE", "DISCOVERY_COMPLETE_NO_COMPLETE_SPEC"
    if hft_suitable_spec_count == 0:
        return "COMPLETE", "DISCOVERY_COMPLETE_NO_HFT_SUITABLE_SPEC"
    return "COMPLETE", "CANDIDATES_READY_FOR_EVALUATION"


def _telemetry_summary(calls: Sequence[Any]) -> dict[str, Any]:
    outcomes = Counter(str(call.outcome) for call in calls)
    stages: dict[str, dict[str, Any]] = {}
    for stage in sorted({str(call.stage) for call in calls}):
        stage_calls = tuple(call for call in calls if str(call.stage) == stage)
        stages[stage] = {
            "call_records": len(stage_calls),
            "llm_calls": sum(not call.cache_hit for call in stage_calls),
            "cache_hits": sum(bool(call.cache_hit) for call in stage_calls),
            "elapsed_seconds": round(sum(float(call.elapsed_seconds) for call in stage_calls), 6),
            "input_tokens": sum(call.input_tokens or 0 for call in stage_calls),
            "output_tokens": sum(call.output_tokens or 0 for call in stage_calls),
            "outcomes": dict(sorted(Counter(str(call.outcome) for call in stage_calls).items())),
        }
    return {
        "call_records": len(calls),
        "llm_calls": sum(not call.cache_hit for call in calls),
        "cache_hits": sum(bool(call.cache_hit) for call in calls),
        "retry_count": sum(int(call.retry_depth > 0) for call in calls),
        "elapsed_seconds": round(sum(float(call.elapsed_seconds) for call in calls), 6),
        "input_tokens": sum(call.input_tokens or 0 for call in calls),
        "output_tokens": sum(call.output_tokens or 0 for call in calls),
        "outcomes": dict(sorted(outcomes.items())),
        "stages": stages,
        "calls": [call.to_dict() for call in calls],
    }


def _evidence_reference(sentence: Any) -> dict[str, Any]:
    hit = sentence.hit
    return {
        "evidence_id": sentence.evidence_id,
        "document_id": hit.document_id,
        "source_hash": hit.source_hash,
        "source_filename": hit.source_filename,
        "chunk_id": hit.chunk_id,
        "page": hit.page,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "chapter": hit.chapter,
        "section_path": list(hit.section_path),
        "start_offset": sentence.start_offset,
        "end_offset": sentence.end_offset,
        "quote": sentence.text,
    }


def _retrieval_record(record: Any) -> dict[str, Any]:
    hit = record.hit
    return {
        "document_id": hit.document_id,
        "source_hash": hit.source_hash,
        "source_filename": hit.source_filename,
        "source_relative_path": hit.source_relative_path,
        "title": hit.title,
        "authors": list(hit.authors),
        "publication_year": hit.publication_year,
        "chunk_id": hit.chunk_id,
        "content_type": hit.content_type.value,
        "page": hit.page,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "chapter": hit.chapter,
        "section_path": list(hit.section_path),
        "text": hit.text,
        "references": [_jsonable(reference) for reference in record.references],
    }


def _concept_record(concept: Any) -> dict[str, Any]:
    return {
        "family": concept.family.value,
        "rules": [
            {
                "stage": item.stage.value,
                "parsed_rule": _jsonable(item.parsed),
                "evidence": _evidence_reference(item.evidence),
            }
            for item in concept.rules
        ],
        "unresolved_rules": [
            {
                "stage": item.stage.value,
                "reason_code": item.reason_code,
                "evidence": _evidence_reference(item.evidence),
            }
            for item in concept.unresolved_rules
        ],
    }


def _manifest_artifact_hashes(root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def _validate_artifact_hashes(root: Path, report: Mapping[str, Any]) -> None:
    hashes = report.get("artifact_files")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_DISCOVERY_ARTIFACT_FILES):
        raise Phase14CResumeError("discovery artifact manifest is incomplete")
    root_resolved = Path(root).resolve(strict=True)
    for relative, expected_digest in hashes.items():
        if not isinstance(relative, str):
            raise Phase14CResumeError("discovery artifact manifest contains an invalid entry")
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or not isinstance(expected_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        ):
            raise Phase14CResumeError("discovery artifact manifest contains an invalid entry")
        try:
            target = (root_resolved / path).resolve(strict=True)
            target.relative_to(root_resolved)
            if not target.is_file():
                raise OSError("artifact is not a file")
            actual_digest = _manifest_artifact_hashes(root_resolved, (relative,))[relative]
        except (OSError, ValueError) as exc:
            raise Phase14CResumeError("discovery artifact is missing or outside its root") from exc
        if actual_digest != expected_digest:
            raise Phase14CResumeError("discovery artifact fingerprint mismatch")


def run_phase14c_discovery(
    *,
    artifact_root: Path,
    knowledge_root: Path,
    embedding_model_path: Path,
    atomic_cache_path: Path,
    source_paths: Phase14CSourcePaths,
    source_fingerprints: Mapping[str, str],
    generation_pin: Phase14CGenerationPin,
    inventory: CorpusInventory,
    expected_generation_id: str,
    expected_generation_fingerprint: str,
    expected_population_hash: str,
    endpoint: str,
    model: str,
    model_version: str,
    timeout_seconds: float,
    max_output_tokens: int,
    context_tokens: int,
    resume: bool,
    teacher_factory: Any,
    top_k_per_formulation: int = 10,
) -> dict[str, Any]:
    """Run pinned local discovery, writing only structured evidence and safe telemetry."""

    if not isinstance(source_paths, Phase14CSourcePaths):
        raise TypeError("source_paths must be Phase14CSourcePaths")
    if not isinstance(generation_pin, Phase14CGenerationPin) or not isinstance(inventory, CorpusInventory):
        raise TypeError("discovery requires validated Phase 7 generation and inventory records")
    if not callable(teacher_factory):
        raise TypeError("teacher_factory must be an explicit local-model factory")
    if type(resume) is not bool:
        raise TypeError("resume must be an explicit boolean")
    if type(top_k_per_formulation) is not int or not 1 <= top_k_per_formulation <= 50:
        raise ValueError("top_k_per_formulation must be between 1 and 50")

    knowledge, embedding_path, root, cache_path = validate_discovery_paths(
        knowledge_root=knowledge_root,
        embedding_model_path=embedding_model_path,
        artifact_root=artifact_root,
        atomic_cache_path=atomic_cache_path,
        source_paths=source_paths,
        allow_existing_artifact_root=resume,
    )
    expected_pin, expected_inventory = read_phase7_inventory(
        knowledge,
        expected_generation_id=expected_generation_id,
        expected_generation_fingerprint=expected_generation_fingerprint,
        expected_population_hash=expected_population_hash,
    )
    if expected_pin != generation_pin or expected_inventory != inventory:
        raise ValueError("Phase 7 pin or inventory changed after command preflight")

    supplied_source_fingerprints = dict(source_fingerprints)
    preflight = preflight_phase14_sources(
        source_paths.phase14a_path,
        source_paths.hft_path,
        source_paths.demo_path,
    )
    if not preflight.ready:
        raise ValueError(f"Phase 14 replay preflight is not READY: {preflight.reason_code.value}")
    if supplied_source_fingerprints != dict(preflight.source_fingerprints):
        raise ValueError("source database fingerprints changed after command preflight")
    source_fingerprints = dict(preflight.source_fingerprints)

    query_service, embedder = _build_local_query_service(knowledge, embedding_path, generation_pin)
    if not hasattr(embedder, "spec") or not callable(getattr(embedder.spec, "to_dict", None)):
        raise TypeError("local query embedder did not expose a complete embedding specification")
    query_bank = build_discovery_query_bank()
    retrieval = retrieve_discovery_evidence(
        query_service,
        query_bank,
        pinned_generation_id=generation_pin.generation_id,
        pinned_population_hash=generation_pin.population_hash,
        top_k_per_formulation=top_k_per_formulation,
    )
    if retrieval.status != "COMPLETE":
        raise Phase14CResumeError("Phase 7 retrieval failed closed; no teacher was constructed")
    selection = select_diverse_evidence(retrieval)
    pin_after_plan, inventory_after_plan = read_phase7_inventory(
        knowledge,
        expected_generation_id=expected_generation_id,
        expected_generation_fingerprint=expected_generation_fingerprint,
        expected_population_hash=expected_population_hash,
    )
    if pin_after_plan != generation_pin or inventory_after_plan != inventory:
        raise ValueError("frozen Phase 7 identity or inventory changed during deterministic planning")
    fingerprints_after_plan = _fingerprint_sources(source_paths)
    if fingerprints_after_plan != dict(source_fingerprints):
        raise ValueError("read-only Phase 14 source fingerprint changed during deterministic planning")

    identity = _build_run_identity(
        pin=generation_pin,
        inventory=inventory,
        embedding_spec=embedder.spec,
        query_bank=query_bank,
        top_k_per_formulation=top_k_per_formulation,
        selection_limits=selection.limits,
        endpoint=endpoint,
        model=model,
        model_version=model_version,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        context_tokens=context_tokens,
        atomic_cache_path=cache_path,
        source_paths=source_paths,
        source_fingerprints=source_fingerprints,
    )

    if resume:
        validate_resume_identity(root, identity)
        if any((root / item).exists() for item in (RUN_REPORT_NAME, "corpus-inventory.json", "retrieval")):
            raise Phase14CResumeError("partially published result artifacts cannot be overwritten or resumed")
        update_run_manifest_status(root, identity, "IN_PROGRESS")
    else:
        if not root.parent.is_dir():
            raise FileNotFoundError("Phase 14C artifact parent must already exist")
        root.mkdir()
        write_run_manifest(root, identity, status="IN_PROGRESS")

    resume_path = root / RESUME_RESULTS_NAME
    try:
        prior_results = load_resume_results(resume_path, artifact_root=root)
        prior_by_key = {str(item["cache_key"]): item for item in prior_results}
        presence_results: list[Any] = []
        presence_telemetry: list[Any] = []
        teacher: Any | None = None

        def get_teacher() -> Any:
            nonlocal teacher
            if teacher is None:
                teacher = teacher_factory()
                _validate_teacher(
                    teacher,
                    endpoint=endpoint,
                    model=model,
                    model_version=model_version,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    context_tokens=context_tokens,
                )
            return teacher

        from tradingagents.self_enhancement.book_atomic_extraction import PresenceGroupResult

        for group in selection.groups:
            key = _presence_cache_key(identity, group)
            prior = prior_by_key.get(key)
            if prior is not None:
                status_code = prior.get("status")
                reason_code = prior.get("reason_code")
                if status_code in {item.value for item in PresenceStatus}:
                    result = PresenceGroupResult(group, PresenceStatus(status_code))
                elif status_code == "FAILED" and reason_code:
                    result = PresenceGroupResult(group, None, str(reason_code))
                else:
                    raise Phase14CResumeError("resume presence result is not a valid closed status")
                raw_calls = prior.get("telemetry", [])
                for item in raw_calls:
                    presence_telemetry.append(_telemetry_from_dict(item))
            else:
                classified = get_teacher().classify_presence((group,))
                if len(classified.results) != 1 or classified.results[0].group.group_id != group.group_id:
                    raise Phase14CResumeError("presence stage returned a mismatched evidence group")
                result = classified.results[0]
                presence_telemetry.extend(classified.call_telemetry)
                record = {
                    "status": result.status.value if result.status is not None else "FAILED",
                    "result_id": group.group_id,
                    "stage": "PRESENCE",
                    "telemetry": [call.to_dict() for call in classified.call_telemetry],
                }
                if result.failure_code:
                    record["reason_code"] = result.failure_code
                append_resume_result(resume_path, key, record, artifact_root=root)
                prior_by_key[key] = {"cache_key": key, **record}
            presence_results.append(result)

        actionable_sentences = tuple(
            sentence
            for result in presence_results
            if result.status is PresenceStatus.ACTIONABLE
            for sentence in result.group.sentences
        )
        unique_actionable: dict[str, Any] = {}
        for sentence in actionable_sentences:
            unique_actionable.setdefault(sentence.evidence_id, sentence)

        if unique_actionable:
            extraction = get_teacher().extract_actionable(tuple(unique_actionable.values()))
        else:
            from tradingagents.self_enhancement.book_atomic_extraction import AtomicExtractionReport

            extraction = AtomicExtractionReport((), (), 0, 0, 0, 0, 0, 0, 0, 0)
        extraction_telemetry = list(extraction.call_telemetry)
        concepts = tuple(extraction.concepts)

        assembly = assemble_atomic_concepts(
            concepts,
            generation_id=generation_pin.generation_id,
            generation_fingerprint=generation_pin.generation_fingerprint,
            model_id=model,
            available_features=tuple(TickFeatures.__dataclass_fields__),
            duplicate_clusters=selection.duplicate_index,
        )
        complete_records = tuple(
            record
            for record in assembly.records
            if record.status.value == "COMPLETE" and record.spec is not None
        )
        complete_specs = tuple(record.spec for record in complete_records if record.spec is not None)
        mapping_snapshot = capture_current_strategy_contracts()
        mappings = map_current_strategies(complete_specs, snapshot=mapping_snapshot)

        registry = BookStrategyRegistry()
        eligible_specs: list[StrategySpec] = []
        ineligible_candidates: Counter[str] = Counter()
        for record in complete_records:
            spec = record.spec
            if not record.executable_eligible or spec is None:
                ineligible_candidates["NOT_EXECUTABLE"] += 1
            elif spec.suitability.value != "HFT_SUITABLE":
                ineligible_candidates[spec.suitability.value] += 1
            else:
                try:
                    registry.create(spec)
                except (TypeError, ValueError):
                    ineligible_candidates["UNIMPLEMENTED_SPEC"] += 1
                else:
                    eligible_specs.append(spec)

        status_counts = Counter(
            result.status.value if result.status is not None else f"FAILED_{result.failure_code}"
            for result in presence_results
        )
        classification_failures = sum(result.status is None for result in presence_results)
        classification_failures += sum(
            call.outcome in {"SCHEMA_INVALID", "TRUNCATED", "PROVIDER_ERROR", "PROVENANCE_INVALID"}
            for call in presence_telemetry
        )
        classification_failures += sum(
            call.outcome in {"SCHEMA_INVALID", "TRUNCATED", "PROVIDER_ERROR", "PROVENANCE_INVALID"}
            for call in extraction_telemetry
        )
        classification_failures += extraction.provenance_failure_count + extraction.cache_write_failures
        status, outcome = classify_discovery_outcome(
            complete_spec_count=len(complete_specs),
            hft_suitable_spec_count=len(eligible_specs),
            deferred_group_count=selection.deferred_group_count,
            classification_failure_count=classification_failures,
        )

        pin_after, inventory_after = read_phase7_inventory(
            knowledge,
            expected_generation_id=expected_generation_id,
            expected_generation_fingerprint=expected_generation_fingerprint,
            expected_population_hash=expected_population_hash,
        )
        source_fingerprints_after = _fingerprint_sources(source_paths)
        if pin_after != generation_pin or inventory_after != inventory:
            raise ValueError("frozen Phase 7 identity or inventory changed during discovery")
        if source_fingerprints_after != dict(source_fingerprints):
            raise ValueError("read-only Phase 14 source fingerprints changed during discovery")

        all_calls = (*presence_telemetry, *extraction_telemetry)
        retrieved_document_ids = {record.hit.document_id for record in retrieval.records}
        actionable_identities = {
            (sentence.hit.document_id, sentence.hit.chunk_id, sentence.hit.source_hash)
            for sentence in unique_actionable.values()
        }
        actionable_document_ids = set(
            selection.duplicate_index.independent_document_ids(
                tuple(
                    ChunkIdentity(document_id, chunk_id, source_hash)
                    for document_id, chunk_id, source_hash in actionable_identities
                )
            )
        )
        sections = {
            tuple(group.sentences[0].hit.section_path)
            or ((group.sentences[0].hit.section or "UNSPECIFIED"),)
            for group in selection.groups
        }
        assembly_modes = Counter(
            record.assembly_mode.value
            for record in assembly.records
            if record.assembly_mode is not None
        )
        suitability_counts = Counter(spec.suitability.value for spec in complete_specs)
        evaluation_summary = {
            "status": "NOT_RUN",
            "outcome": "CANDIDATES_READY_FOR_EVALUATION" if eligible_specs else "NO_ELIGIBLE_CANDIDATES",
            "candidate_count": 0,
            "eligible_candidate_count": len(eligible_specs),
        }
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": status,
            "outcome": outcome,
            "artifact_integrity": "PENDING",
            "generation_id": generation_pin.generation_id,
            "generation_fingerprint": generation_pin.generation_fingerprint,
            "population_hash": generation_pin.population_hash,
            "source_manifest_fingerprint": inventory.source_manifest_fingerprint,
            "inventory": _jsonable(inventory),
            "coverage": {
                "raw_query_hits": retrieval.raw_hit_count,
                "unique_retrieved_chunks": retrieval.unique_hit_count,
                "retrieved_book_count": len(retrieved_document_ids),
                "queryable_books_never_retrieved": max(
                    0, inventory.queryable_document_count - len(retrieved_document_ids)
                ),
                "books_with_actionable_evidence": len(actionable_document_ids),
                "selected_groups_by_family": dict(selection.selected_groups_by_family),
                "deferred_groups_by_family": dict(selection.deferred_groups_by_family),
                "selected_section_count": len(sections),
                "eligible_group_count": selection.eligible_group_count,
                "selected_group_count": selection.selected_group_count,
                "deferred_group_count": selection.deferred_group_count,
                "unclassified_chunk_count": selection.unclassified_chunk_count,
                "coverage_complete": selection.coverage_complete,
            },
            "retrieval": {
                "query_bank_version": retrieval.query_bank_version,
                "query_bank_fingerprint": retrieval.query_bank_fingerprint,
                "query_count": retrieval.completed_query_count,
                "raw_hit_count": retrieval.raw_hit_count,
                "unique_hit_count": retrieval.unique_hit_count,
                "duplicate_cluster_count": selection.duplicate_cluster_count,
                "duplicate_chunk_count": selection.duplicate_chunk_count,
                "duplicate_group_suppressed_count": selection.duplicate_group_suppressed_count,
                "query_failures": [_jsonable(item) for item in retrieval.query_failures],
            },
            "classification": {
                "status_counts": dict(sorted(status_counts.items())),
                "failure_count": classification_failures,
                "unique_actionable_evidence_count": len(unique_actionable),
                "concept_count": len(concepts),
            },
            "extraction": {
                "atomic_rule_count": extraction.atomic_rule_count,
                "unsupported_rule_count": extraction.unsupported_rule_count,
                "provenance_failure_count": extraction.provenance_failure_count,
                "cache_write_failures": extraction.cache_write_failures,
                "concept_count": extraction.concept_count,
                "independent_multi_source_concept_count": sum(
                    len(record.independent_document_ids) > 1 for record in assembly.records
                ),
            },
            "assembly": {
                "complete_count": assembly.complete_count,
                "partial_count": assembly.partial_count,
                "rejected_count": assembly.rejected_count,
                "conflict_count": len(assembly.conflicts),
                "parameter_request_count": len(assembly.parameter_requests),
                "rejection_count": len(assembly.rejections),
                "modes": dict(sorted(assembly_modes.items())),
                "suitability": dict(sorted(suitability_counts.items())),
            },
            "mapping": {
                "schema_version": PHASE14C_MAPPING_SCHEMA_VERSION,
                "contract_version": PHASE14C_MAPPING_VERSION,
                "contract_fingerprint": mapping_snapshot.fingerprint,
                "strategies": [_jsonable(item) for item in mappings],
            },
            "candidate_eligibility": {
                "eligible_count": len(eligible_specs),
                "ineligible_reason_counts": dict(sorted(ineligible_candidates.items())),
                "evaluation_invoked": False,
            },
            "evaluation": evaluation_summary,
            "telemetry": _telemetry_summary(all_calls),
            "runtime_seconds": round(sum(float(call.elapsed_seconds) for call in all_calls), 6),
            "phase7_before": _phase7_identity(generation_pin, inventory),
            "phase7_after": _phase7_identity(pin_after, inventory_after),
            "phase7_unchanged": pin_after == generation_pin and inventory_after == inventory,
            "source_fingerprints_before": dict(source_fingerprints),
            "source_fingerprints_after": dict(source_fingerprints_after),
            "source_unchanged": source_fingerprints_after == dict(source_fingerprints),
            "artifact_identity_fingerprint": identity.fingerprint,
        }

        # Persist only source-bearing evidence and deterministic/strictly validated records.
        for path, values in (
            ("retrieval/evidence.jsonl", tuple(_retrieval_record(item) for item in retrieval.records)),
            (
                "classification/validated-results.jsonl",
                tuple(
                    {
                        "group_id": result.group.group_id,
                        "family_id": result.group.family_id,
                        "status": result.status.value if result.status is not None else "FAILED",
                        "failure_code": result.failure_code,
                        "evidence_ids": [sentence.evidence_id for sentence in result.group.sentences],
                        "source_refs": [_evidence_reference(sentence) for sentence in result.group.sentences],
                    }
                    for result in presence_results
                ),
            ),
            ("concepts/concepts.jsonl", tuple(_concept_record(item) for item in concepts)),
            ("concepts/conflicts.jsonl", tuple(_jsonable(item) for item in assembly.conflicts)),
            ("specs/strategy-specs.jsonl", tuple(_jsonable(record.spec) for record in assembly.records if record.spec)),
            ("specs/assembly-records.jsonl", tuple(_jsonable(item) for item in assembly.records)),
        ):
            _write_artifact(root, path, _jsonl_bytes(values))
        _write_artifact(root, "corpus-inventory.json", _json_bytes(inventory))
        _write_artifact(
            root,
            "query-bank.json",
            _json_bytes(
                {
                    "schema_version": retrieval.query_bank_version,
                    "fingerprint": retrieval.query_bank_fingerprint,
                    "queries": [_jsonable(item) for item in query_bank],
                }
            ),
        )
        _write_artifact(root, "selection/coverage.json", _json_bytes(selection))
        _write_artifact(root, "mappings/current-strategy-mapping.json", _json_bytes(report["mapping"]))
        validated_specs = tuple(record.spec for record in complete_records if record.executable_eligible and record.spec)
        validated_payload = {
            "schema_version": VALIDATED_SPEC_SCHEMA_VERSION,
            "specs": [_jsonable(spec) for spec in validated_specs],
        }
        validated_bytes = _json_bytes(validated_payload)
        _write_artifact(root, "validated-specs.json", validated_bytes)
        # The resumable status log contains only IDs, closed statuses, and safe scalar telemetry.
        if not resume_path.exists():
            _write_artifact(root, RESUME_RESULTS_NAME, b"")
        report["artifact_files"] = _manifest_artifact_hashes(root, _DISCOVERY_ARTIFACT_FILES)
        report["validated_specs_file"] = "validated-specs.json"
        report["validated_specs_sha256"] = hashlib.sha256(validated_bytes).hexdigest()
        report["artifact_integrity"] = "VALID"
        _write_artifact(root, RUN_REPORT_NAME, _json_bytes(report))
        update_run_manifest_status(root, identity, "COMPLETE" if status == "COMPLETE" else "FAILED")
        return report
    except Exception:
        try:
            if (root / RUN_MANIFEST_NAME).exists():
                manifest = _read_json_object(root / RUN_MANIFEST_NAME)
                next_status = "FAILED" if any(
                    (root / name).exists() for name in (RUN_REPORT_NAME, "corpus-inventory.json", "retrieval")
                ) else "INTERRUPTED"
                if manifest.get("status") in {"IN_PROGRESS", "INTERRUPTED"}:
                    update_run_manifest_status(root, identity, next_status)
        except Exception:
            pass
        raise
    finally:
        transport = getattr(teacher, "transport", None) if teacher is not None else None
        close = getattr(transport, "close", None)
        if callable(close):
            close()


def _telemetry_from_dict(value: Mapping[str, Any]) -> Any:
    from tradingagents.self_enhancement.book_atomic_extraction import ModelCallTelemetry
    return ModelCallTelemetry(**_safe_telemetry_record(value))


def summarize_phase14c_evaluation(records: Sequence[CandidateEvaluationRecord]) -> dict[str, Any]:
    values = tuple(records)
    if any(not isinstance(item, CandidateEvaluationRecord) for item in values):
        raise TypeError("evaluation records must be CandidateEvaluationRecord values")
    if not values:
        return {"status": "NOT_RUN", "outcome": "NO_ELIGIBLE_CANDIDATES", "evaluated_candidate_count": 0}
    preflight_blockers = {
        Phase14CPreflightReason.SOURCE_UNAVAILABLE.value,
        Phase14CPreflightReason.INVALID_PHASE14A.value,
        Phase14CPreflightReason.NO_VERIFIED_PHASE14A_EXPERIENCE.value,
        Phase14CPreflightReason.INVALID_DEMO_SOURCE.value,
        Phase14CPreflightReason.DEMO_RECONCILIATION_UNCERTAIN.value,
        Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA.value,
        Phase14CPreflightReason.SOURCE_MUTATED_DURING_PREFLIGHT.value,
    }
    if all(item.status is Phase14CCandidateStatus.NOT_RUN for item in values) and any(
        item.reason_code in preflight_blockers for item in values
    ):
        return {
            "status": "BLOCKED",
            "outcome": "REPLAY_PREFLIGHT_BLOCKED",
            "evaluated_candidate_count": 0,
            "reason_counts": dict(sorted(Counter(item.reason_code for item in values).items())),
        }
    challenger_count = sum(item.candidate_state == CandidateState.SHADOW_CHALLENGER.value for item in values)
    completed = sum(item.status is Phase14CCandidateStatus.COMPLETED for item in values)
    failed = sum(item.status is Phase14CCandidateStatus.FAILED for item in values)
    not_run = sum(item.status is Phase14CCandidateStatus.NOT_RUN for item in values)
    if completed == 0 and failed > 0:
        status, outcome = "FAILED", "CANDIDATE_REPLAY_FAILED"
    elif challenger_count:
        status, outcome = "EVALUATED", "SHADOW_CHALLENGER_CREATED"
    elif completed:
        status, outcome = "EVALUATED", "CANDIDATES_EVALUATED_NONE_PASSED"
    else:
        status, outcome = "NOT_RUN", "NO_ELIGIBLE_CANDIDATES"
    return {
        "status": status,
        "outcome": outcome,
        "evaluated_candidate_count": completed,
        "challenger_count": challenger_count,
        "failed_candidate_count": failed,
        "not_run_candidate_count": not_run,
        "reason_counts": dict(sorted(Counter(item.reason_code for item in values).items())),
        "max_candidate_state": (
            CandidateState.SHADOW_CHALLENGER.value if challenger_count else None
        ),
    }


def run_phase14c_evaluation(
    *,
    artifact_root: Path,
    source_paths: Phase14CSourcePaths,
    source_commit: str,
    max_candidate_runs: int = 5,
) -> dict[str, Any]:
    """Evaluate only integrity-checked HFT specs under the existing replay gate."""

    root = Path(artifact_root).expanduser().resolve(strict=True)
    specs, discovery_report = load_complete_discovery_artifacts(root)
    evaluation_root = root / "evaluations"
    if evaluation_root.exists():
        raise FileExistsError("Phase 14C evaluations already exist; refusing overwrite")
    registry = BookStrategyRegistry()
    candidates: list[StrategySpec] = []
    for spec in specs:
        if not spec.is_executable or spec.suitability.value != "HFT_SUITABLE":
            continue
        try:
            registry.create(spec)
        except (TypeError, ValueError):
            continue
        candidates.append(spec)
    records = evaluate_phase14c_candidates(
        tuple(candidates),
        artifact_root=evaluation_root,
        source_paths=source_paths,
        source_commit=source_commit,
        max_candidate_runs=max_candidate_runs,
    ) if candidates else ()
    summary = summarize_phase14c_evaluation(records)
    report = {
        "schema_version": "phase14c-evaluation-report.v1",
        **summary,
        "discovery_status": discovery_report["status"],
        "candidate_count": len(candidates),
        "records": [
            {
                "spec_id": item.spec_id,
                "candidate_id": item.candidate_id,
                "status": item.status.value,
                "reason_code": item.reason_code,
                "gate_decision": item.gate_decision,
                "gate_reasons": list(item.gate_reasons),
                "candidate_state": item.candidate_state,
                "artifact_path": (
                    str(item.artifact_path.relative_to(root)) if item.artifact_path is not None else None
                ),
                "source_fingerprints": dict(item.source_fingerprints),
            }
            for item in records
        ],
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    _write_artifact(evaluation_root, "phase14c-evaluation-report.json", _json_bytes(report))
    return report


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
    _validate_artifact_hashes(root, report)
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
