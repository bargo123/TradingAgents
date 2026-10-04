"""Explicit offline Phase 14B book-strategy research command.

No command in this module starts the forex supervisor or constructs MT5.  The
local Ollama adapter is created only by the explicit ``draft-and-evaluate``
subcommand and only after all pinned-source and fresh-output checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlsplit

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.embeddings import FastEmbedProvider
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.models import IndexGeneration
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.vector_index import VectorIndexReader
from tradingagents.self_enhancement.book_factory import BookStrategyFactory
from tradingagents.self_enhancement.models import ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.orchestrator import (
    BookCandidateExperiment,
    _read_verified_phase14a,
    _source_file_fingerprint,
    _validate_demo_source_readonly,
)
from tradingagents.self_enhancement.strategy_specs import StrategySuitability

_EXIT_OK = 0
_EXIT_FAILURE = 1
_EXIT_USAGE = 2
_REPORT_NAME = "phase14b-run.json"
_DEFAULT_QUERIES = (
    "forex range rejection failed breakout reversion entry invalidation exit holding horizon",
    "short horizon momentum continuation breakout confirmation filters stop exit holding horizon",
)


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _fingerprint_generation(generation: IndexGeneration) -> str:
    canonical = json.dumps(
        generation.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _emit(value: Any, *, as_json: bool) -> None:
    payload = _json_value(value)
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    elif isinstance(payload, Mapping):
        for key, item in payload.items():
            rendered = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
            print(f"{str(key).upper()}: {rendered}")
    else:
        print(payload)


def _open_knowledge(knowledge_root: Path) -> tuple[KnowledgeCatalog, IndexGeneration | None]:
    catalog_path = knowledge_root / "catalog.sqlite3"
    if not catalog_path.is_file():
        raise FileNotFoundError("Phase 7 catalog.sqlite3 is missing from knowledge root")
    catalog = KnowledgeCatalog(catalog_path)
    return catalog, catalog.active_generation()


def _safe_query_source_root(artifact_root: Path, embedding_path: Path | None) -> Path:
    candidates = (Path.home(), Path.cwd(), Path(os.getenv("TEMP", "")))
    for candidate in candidates:
        try:
            candidate = candidate.resolve(strict=True)
            if candidate == artifact_root or candidate in artifact_root.parents or artifact_root in candidate.parents:
                continue
            if embedding_path is not None and (
                candidate == embedding_path
                or candidate in embedding_path.parents
                or embedding_path in candidate.parents
            ):
                continue
            return candidate
        except (OSError, RuntimeError):
            continue
    raise ValueError("no existing local source path is disjoint from Phase 7 and the embedding model")


def _embedding_model_path(args: argparse.Namespace) -> Path | None:
    configured = args.embedding_model_path or os.getenv("KNOWLEDGE_EMBEDDING_MODEL_PATH")
    if configured:
        return _path(configured)
    candidate = _path(args.knowledge_root) / "models" / "embedding"
    return candidate if candidate.is_dir() else None


def _build_query_service(
    args: argparse.Namespace,
    catalog: KnowledgeCatalog,
    generation: IndexGeneration,
) -> KnowledgeQueryService:
    spec = generation.embedding_spec
    model_path = _embedding_model_path(args)
    knowledge_root = _path(args.knowledge_root)
    config = KnowledgeConfig(
        source_root=_safe_query_source_root(knowledge_root, model_path),
        artifact_root=knowledge_root,
        embedding_model_id=spec.model_id,
        embedding_model_path=model_path,
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
    )
    embedder = FastEmbedProvider.from_config(config)
    vector_reader = VectorIndexReader(generation.vector_location)
    lexical_reader = LexicalIndexReader(generation.lexical_location)
    return KnowledgeQueryService(vector_reader, lexical_reader, catalog, embedder)


def _create_drafter(args: argparse.Namespace) -> Any:
    # Import at the opt-in command boundary only.  Status never imports or
    # constructs a model client.
    from tradingagents.self_enhancement.book_drafter import OllamaStrategyDrafter

    return OllamaStrategyDrafter(
        args.ollama_endpoint,
        args.model,
        model_version=getattr(args, "model_version", None),
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        context_tokens=args.context_tokens,
        cache_path=getattr(args, "_resolved_atomic_cache_path", None),
    )


class _RecordingDrafter:
    """Retain only the drafter's allowlisted scalar diagnostics."""

    _FIELDS = (
        "provider",
        "model",
        "model_version",
        "prompt_version",
        "schema_version",
        "timeout_seconds",
        "max_output_tokens",
        "context_tokens",
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "finish_reason",
        "draft_digest",
        "error_code",
        "error_type",
        "ok",
        "llm_calls",
        "cache_hits",
        "call_telemetry",
        "evidence_sentence_count",
        "evidence_groups_processed",
        "actionable_evidence_groups",
        "actionable_concept_count",
        "atomic_rule_count",
        "unsupported_rule_count",
        "provenance_failure_count",
        "schema_failure_count",
        "truncation_count",
        "provider_failure_count",
        "retry_count",
        "insufficient_specification_count",
        "cache_write_failures",
    )

    def __init__(self, drafter: Any) -> None:
        self._drafter = drafter
        self.result: Any = None
        self.calls = 0

    def draft(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.result = self._drafter.draft(*args, **kwargs)
        return self.result

    def telemetry(self) -> dict[str, Any] | None:
        if self.result is None:
            return None
        values = {name: getattr(self.result, name, None) for name in self._FIELDS}
        calls = values.get("call_telemetry")
        if calls is not None:
            values["call_telemetry"] = [
                call.to_dict() if hasattr(call, "to_dict") else call for call in calls
            ]
        return values

    @property
    def llm_calls(self) -> int:
        if self.result is not None:
            value = getattr(self.result, "llm_calls", None)
            if type(value) is int and value >= 0:
                return value
        return self.calls

    @property
    def cache_hits(self) -> int:
        if self.result is not None:
            value = getattr(self.result, "cache_hits", None)
            if type(value) is int and value >= 0:
                return value
        return 0


def _nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _draft_performance(telemetry: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(telemetry, Mapping):
        return {}
    raw_calls = telemetry.get("call_telemetry")
    calls = [
        call for call in raw_calls
        if isinstance(call, Mapping) and call.get("cache_hit") is not True
    ] if isinstance(raw_calls, (tuple, list)) else []
    runtimes = [float(call["elapsed_seconds"]) for call in calls if isinstance(call.get("elapsed_seconds"), (int, float))]
    input_tokens = [int(call["input_tokens"]) for call in calls if type(call.get("input_tokens")) is int]
    output_tokens = [int(call["output_tokens"]) for call in calls if type(call.get("output_tokens")) is int]
    stage_counts = Counter(str(call.get("stage")) for call in calls)
    return {
        "model_calls": len(calls),
        "cache_hits": int(telemetry.get("cache_hits", 0) or 0),
        "input_tokens_total": sum(input_tokens),
        "input_tokens_average": round(sum(input_tokens) / len(input_tokens), 2) if input_tokens else None,
        "input_tokens_p50": round(median(input_tokens), 3) if input_tokens else None,
        "input_tokens_p95": _nearest_rank(input_tokens, 0.95),
        "output_tokens_total": sum(output_tokens),
        "output_tokens_average": round(sum(output_tokens) / len(output_tokens), 2) if output_tokens else None,
        "output_tokens_p50": round(median(output_tokens), 3) if output_tokens else None,
        "output_tokens_p95": _nearest_rank(output_tokens, 0.95),
        "runtime_seconds_total": round(sum(runtimes), 3),
        "runtime_seconds_p50": round(median(runtimes), 3) if runtimes else None,
        "runtime_seconds_p95": _nearest_rank(runtimes, 0.95),
        "truncations": sum(call.get("outcome") == "TRUNCATED" for call in calls),
        "schema_failures": sum(call.get("outcome") == "SCHEMA_INVALID" for call in calls),
        "provenance_failures": sum(call.get("outcome") == "PROVENANCE_INVALID" for call in calls),
        "provider_failures": sum(call.get("outcome") == "PROVIDER_ERROR" for call in calls),
        "retry_calls": sum(int(call.get("retry_depth", 0) or 0) > 0 for call in calls),
        "cache_write_failures": int(telemetry.get("cache_write_failures", 0) or 0),
        "stage_call_counts": dict(sorted(stage_counts.items())),
    }


class _RecordingQueryService:
    def __init__(self, service: KnowledgeQueryService) -> None:
        self._service = service
        self.catalog = service.catalog
        self.hits: list[Any] = []

    def search(self, request: Any) -> tuple[Any, ...]:
        hits = tuple(self._service.search(request))
        self.hits.extend(hits)
        return hits


def _create_pipeline(query_service: Any, drafter: Any, *, generation: str, fingerprint: str) -> Any:
    # Like the Ollama adapter, the extraction pipeline is loaded only by the
    # explicit draft command, not by metadata-only status inspection.
    from tradingagents.self_enhancement.book_pipeline import BookStrategyPipeline

    return BookStrategyPipeline(
        query_service,
        drafter,
        pinned_generation=generation,
        pinned_fingerprint=fingerprint,
    )


def _source_fingerprints(paths: Mapping[str, Path]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name, path in paths.items():
        fingerprints[name] = _source_file_fingerprint(path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.is_file():
                fingerprints[f"{name}{suffix}"] = _source_file_fingerprint(sidecar)
    return fingerprints


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise ValueError("could not resolve a valid source commit")
    return commit


def _status(args: argparse.Namespace) -> int:
    root = _path(args.artifact_root)
    report_path = root / _REPORT_NAME
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except (OSError, json.JSONDecodeError):
            report = {}
    _emit(
        {
            "artifact_root": str(root),
            "artifact_root_initialized": root.is_dir(),
            "run_report_exists": report_path.is_file(),
            "status": report.get("status"),
            "generation_id": report.get("generation_id"),
            "generation_fingerprint": report.get("generation_fingerprint"),
            "provider": report.get("provider"),
            "model": report.get("model"),
            "validated_spec_count": report.get("validated_spec_count"),
            "hft_suitable_count": report.get("hft_suitable_count"),
            "candidate_experiment_count": len(report.get("candidate_experiments", ()))
            if isinstance(report.get("candidate_experiments"), list)
            else 0,
        },
        as_json=args.json,
    )
    return _EXIT_OK


def _validate_source_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "phase14a": _path(args.phase14a_db),
        "hft": _path(args.hft_db),
        "demo": _path(args.demo_db),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required read-only source file missing: {','.join(missing)}")
    if len(set(paths.values())) != len(paths):
        raise ValueError("Phase 14A, HFT and DEMO inputs must be distinct files")
    return paths


def _validate_output_root(root: Path, knowledge_root: Path, source_paths: Mapping[str, Path]) -> None:
    if root == knowledge_root or root in knowledge_root.parents or knowledge_root in root.parents:
        raise ValueError("Phase 14B output root must be disjoint from the frozen Phase 7 root")
    if any(path == root or root in path.parents for path in source_paths.values()):
        raise ValueError("Phase 14B output root must not contain a read-only source database")


def _validate_local_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama endpoint must be a loopback URL") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Ollama endpoint must be a loopback URL")


def _preflight_replay_sources(source_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Validate replay inputs read-only; blockers suppress replay, not drafting."""

    result: dict[str, Any] = {
        "candidate_replay_ready": False,
        "reason_code": None,
        "verified_experience_count": 0,
        "quarantined_experience_excluded": 0,
        "tick_count": 0,
        "segment_count": 0,
        "dataset_fingerprint": None,
        "demo_reconciliation_status": "UNKNOWN",
    }
    phase14a_ready = False
    try:
        verified, quarantined = _read_verified_phase14a(source_paths["phase14a"])
        result["verified_experience_count"] = len(verified)
        result["quarantined_experience_excluded"] = quarantined
        phase14a_ready = bool(verified)
        if not verified:
            result["reason_code"] = "NO_VERIFIED_PHASE14A_EXPERIENCE"
    except Exception:
        result["reason_code"] = "PHASE14A_SOURCE_INVALID"

    demo_ready = False
    try:
        _validate_demo_source_readonly(source_paths["demo"])
        demo_ready = True
        result["demo_reconciliation_status"] = "CLEAN"
    except ValueError as exc:
        message = str(exc)
        if "unresolved reconciliation" in message.lower():
            result["demo_reconciliation_status"] = "RECONCILIATION_REQUIRED"
            if result["reason_code"] is None:
                result["reason_code"] = "DEMO_RECONCILIATION_REQUIRED"
        else:
            result["demo_reconciliation_status"] = "INVALID"
            if result["reason_code"] is None:
                result["reason_code"] = "DEMO_SOURCE_INVALID"
    except Exception:
        result["demo_reconciliation_status"] = "INVALID"
        if result["reason_code"] is None:
            result["reason_code"] = "DEMO_SOURCE_INVALID"

    from tradingagents.self_enhancement.causal import CausalDatasetError, load_causal_tick_dataset
    from tradingagents.self_enhancement.replay import ReplayError

    hft_ready = False
    try:
        dataset = load_causal_tick_dataset(source_paths["hft"], symbol="EURUSD")
    except (CausalDatasetError, ReplayError):
        if result["reason_code"] is None:
            result["reason_code"] = "HFT_SOURCE_INVALID"
    except Exception:
        if result["reason_code"] is None:
            result["reason_code"] = "HFT_SOURCE_INVALID"
    else:
        result["tick_count"] = dataset.valid_rows
        result["segment_count"] = len(dataset.segments)
        result["dataset_fingerprint"] = dataset.source_fingerprint
        hft_ready = dataset.valid_rows > 0 and dataset.invalid_rows == 0
        if not hft_ready and result["reason_code"] is None:
            result["reason_code"] = "HFT_DATA_QUALITY_INVALID"
    result["candidate_replay_ready"] = phase14a_ready and demo_ready and hft_ready
    if result["candidate_replay_ready"]:
        result["reason_code"] = None
    elif result["reason_code"] is None:
        result["reason_code"] = "REPLAY_PREFLIGHT_BLOCKED"
    return result


def _candidate_experiment(
    *,
    spec: Any,
    source_paths: Mapping[str, Path],
    artifact_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    from tradingagents.self_enhancement.book_strategies import BookStrategyRegistry

    # Registry construction is the deterministic implementation boundary; a
    # StrategySpec is not executable merely because its provenance is valid.
    BookStrategyRegistry().create(spec)
    parent = StrategyVersion(
        "range_rejection",
        "incumbent-v1",
        "config-v1",
        ExitPolicyConfig().to_dict(),
        source_commit,
    )
    candidate = BookStrategyFactory().from_validated_spec(spec, parent=parent)
    experiment_root = artifact_root / "candidate-experiments" / candidate.candidate_id
    result = BookCandidateExperiment(experiment_root).run(
        phase14a_path=source_paths["phase14a"],
        hft_path=source_paths["hft"],
        demo_path=source_paths["demo"],
        candidate=candidate,
        source_commit=source_commit,
        symbol="EURUSD",
    )
    return result.to_dict()


def _run_draft_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = _path(args.phase14_artifact_root)
    if root.exists():
        raise FileExistsError("Phase 14B artifact root must be new; existing data is preserved")
    _validate_local_endpoint(args.ollama_endpoint)
    knowledge_root = _path(args.knowledge_root)
    if root == knowledge_root or root in knowledge_root.parents or knowledge_root in root.parents:
        raise ValueError("Phase 14B output root must be disjoint from the frozen Phase 7 root")
    catalog, generation = _open_knowledge(knowledge_root)
    if generation is None:
        raise ValueError("PINNED_GENERATION_MISMATCH: active Phase 7 generation is missing")
    generation_fingerprint = _fingerprint_generation(generation)
    if (
        generation.generation_id != args.expected_generation
        or generation_fingerprint != args.expected_fingerprint
        or generation.status != "VALIDATED"
        or not generation.vector_ready
        or not generation.lexical_ready
    ):
        raise ValueError("PINNED_GENERATION_MISMATCH: Phase 7 generation or fingerprint is not approved")

    source_paths = _validate_source_paths(args)
    _validate_output_root(root, knowledge_root, source_paths)
    requested_cache_path = getattr(args, "atomic_cache_path", None)
    cache_path = _path(requested_cache_path) if requested_cache_path else root / "atomic-results.sqlite3"
    if cache_path == knowledge_root or cache_path in knowledge_root.parents or knowledge_root in cache_path.parents:
        raise ValueError("atomic cache path must be outside the frozen Phase 7 root")
    if cache_path in source_paths.values():
        raise ValueError("atomic cache path must not overlap a Phase 14/12 source database")
    if cache_path.exists() and cache_path.is_dir():
        raise ValueError("atomic cache path must identify a file")
    args._resolved_atomic_cache_path = str(cache_path)
    source_before = _source_fingerprints(source_paths)
    replay_preflight = _preflight_replay_sources(source_paths)
    service = _build_query_service(args, catalog, generation)
    recorded_service = _RecordingQueryService(service)
    recorder = _RecordingDrafter(_create_drafter(args))
    pipeline = _create_pipeline(
        recorded_service,
        recorder,
        generation=generation.generation_id,
        fingerprint=generation.population_hash,
    )
    queries = tuple(args.query or _DEFAULT_QUERIES)
    _source_commit_value = _source_commit()

    # Create the report root only after all preflight gates have passed.  From
    # here onward, even a model/schema failure is preserved as a safe report.
    root.mkdir(parents=True, exist_ok=False)
    from tradingagents.self_enhancement.book_pipeline import (
        deduplicate_specs,
        map_existing_strategy,
    )

    batch = pipeline.extract(queries, max_specs=args.max_specs)
    specs = deduplicate_specs(batch.specs)
    mappings = tuple(
        map_existing_strategy(strategy_id, recorded_service.hits)
        for strategy_id in ("range_rejection", "momentum_continuation")
    )
    suitability_counts = Counter(spec.suitability.value for spec in specs)
    validation_counts = Counter(
        result.status.value
        for spec in specs
        for result in spec.validation_results
    )
    validation_counts.update(result.status.value for result in batch.rejected_rules)
    candidate_experiments: list[dict[str, Any]] = []
    candidate_errors: list[str] = []
    runnable_specs = [
        spec for spec in specs
        if spec.suitability is StrategySuitability.HFT_SUITABLE and spec.is_executable
    ]
    if runnable_specs and not replay_preflight["candidate_replay_ready"]:
        candidate_experiments.extend(
            {
                "spec_id": spec.spec_id,
                "status": "NOT_RUN",
                "reason_code": replay_preflight["reason_code"],
            }
            for spec in runnable_specs[: args.max_candidate_runs]
        )
    elif replay_preflight["candidate_replay_ready"]:
        for spec in runnable_specs[: args.max_candidate_runs]:
            try:
                candidate_experiments.append(
                    _candidate_experiment(
                        spec=spec,
                        source_paths=source_paths,
                        artifact_root=root,
                        source_commit=_source_commit_value,
                    )
                )
            except Exception as exc:
                # Error messages can include untrusted provider or document text;
                # persist only a fixed code and exception class.
                candidate_errors.append(type(exc).__name__)
    source_after = _source_fingerprints(source_paths)
    active_after = catalog.active_generation()
    final_generation_fingerprint = (
        _fingerprint_generation(active_after) if active_after is not None else None
    )
    source_unchanged = source_before == source_after
    generation_unchanged = (
        active_after is not None
        and active_after.generation_id == generation.generation_id
        and final_generation_fingerprint == generation_fingerprint
    )
    error_code = batch.error_code
    if not source_unchanged:
        error_code = "READ_ONLY_SOURCE_CHANGED"
    elif not generation_unchanged:
        error_code = "PHASE7_GENERATION_CHANGED"
    elif candidate_errors:
        error_code = "CANDIDATE_EXPERIMENT_FAILED"
    elif batch.error_code:
        error_code = batch.error_code
    elif runnable_specs and not replay_preflight["candidate_replay_ready"]:
        error_code = "REPLAY_PREFLIGHT_BLOCKED"

    telemetry = recorder.telemetry()
    draft_performance = _draft_performance(telemetry)
    report: dict[str, Any] = {
        "schema_version": "phase14b-run-report-v1",
        "run_id": "phase14b-" + uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "FAILED" if error_code else ("COMPLETED" if specs else "NO_VALIDATED_SPECS"),
        "error_code": error_code,
        "provider": telemetry.get("provider") if telemetry else None,
        "model": telemetry.get("model") if telemetry else None,
        "model_telemetry": telemetry,
        "generation_id": generation.generation_id,
        "generation_fingerprint": generation_fingerprint,
        "generation_population_hash": generation.population_hash,
        "generation_status": generation.status,
        "knowledge_root": str(knowledge_root),
        "query_count": batch.query_count,
        "retrieved_count": batch.retrieved_count,
        "draft_count": batch.draft_count,
        "evidence_sentence_count": int((telemetry or {}).get("evidence_sentence_count", 0) or 0),
        "evidence_groups_processed": int((telemetry or {}).get("evidence_groups_processed", 0) or 0),
        "actionable_evidence_groups": int((telemetry or {}).get("actionable_evidence_groups", 0) or 0),
        "actionable_concepts_found": int((telemetry or {}).get("actionable_concept_count", 0) or 0),
        "atomic_rules_extracted": int((telemetry or {}).get("atomic_rule_count", 0) or 0),
        "unsupported_rules_rejected": int((telemetry or {}).get("unsupported_rule_count", 0) or 0),
        "insufficient_specifications": int((telemetry or {}).get("insufficient_specification_count", 0) or 0),
        "atomic_cache_path": str(cache_path),
        "draft_performance": draft_performance,
        "replay_source_preflight": replay_preflight,
        "validated_spec_count": len(specs),
        "validated_specs": [spec.to_dict() for spec in specs],
        "rule_validation_counts": dict(sorted(validation_counts.items())),
        "suitability_counts": dict(sorted(suitability_counts.items())),
        "hft_suitable_count": int(suitability_counts.get(StrategySuitability.HFT_SUITABLE.value, 0)),
        "existing_strategy_mappings": _json_value(mappings),
        "candidate_experiments": candidate_experiments,
        "candidate_experiment_error_types": candidate_errors,
        "candidate_runs_requested": args.max_candidate_runs,
        "source_fingerprints_before": source_before,
        "source_fingerprints_after": source_after,
        "source_unchanged": source_unchanged,
        "generation_unchanged": generation_unchanged,
        "llm_calls": recorder.llm_calls,
        "cache_hits": recorder.cache_hits,
        "mt5_calls": 0,
        "prompts_persisted": False,
        "completions_persisted": False,
        "reasoning_persisted": False,
        "execution_enabled": False,
    }
    _write_report_contents(root / _REPORT_NAME, report)
    return report


def _write_report_contents(target: Path, report: Mapping[str, Any]) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase14b",
        description="Explicit local-only Phase 14B StrategySpec research; never starts trading.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="inspect a Phase 14B artifact root without model setup")
    status.add_argument("--artifact-root", required=True, metavar="PATH")
    status.add_argument("--json", action="store_true")

    run = commands.add_parser(
        "draft-and-evaluate",
        help="retrieve frozen Phase 7 evidence, explicitly draft locally, and replay eligible candidates",
    )
    run.add_argument("--knowledge-root", required=True, metavar="PATH")
    run.add_argument("--expected-generation", required=True)
    run.add_argument("--expected-fingerprint", required=True)
    run.add_argument("--embedding-model-path", metavar="PATH", help="pre-provisioned local Phase 7 embedder")
    run.add_argument("--phase14a-db", required=True, metavar="PATH")
    run.add_argument("--hft-db", required=True, metavar="PATH")
    run.add_argument("--demo-db", required=True, metavar="PATH")
    run.add_argument("--phase14-artifact-root", required=True, metavar="NEW_PATH")
    run.add_argument(
        "--atomic-cache-path",
        metavar="PATH",
        help="optional resumable cache of validated atomic outputs; never stores prompts or completions",
    )
    run.add_argument("--model", required=True, help="installed local Ollama model; no implicit default")
    run.add_argument(
        "--model-version",
        help="optional immutable local model digest/revision used to invalidate atomic cache entries",
    )
    run.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    run.add_argument("--timeout-seconds", type=float, default=300.0)
    run.add_argument("--max-output-tokens", type=_positive_int, default=1024)
    run.add_argument("--context-tokens", type=_positive_int, default=8192)
    run.add_argument("--max-specs", type=_positive_int, default=2)
    run.add_argument("--max-candidate-runs", type=_positive_int, default=1)
    run.add_argument("--query", action="append", help="bounded targeted query; may be repeated")
    run.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "status":
            return _status(args)
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.timeout_seconds > 600:
            raise ValueError("timeout_seconds must be between 0 and 600")
        if args.max_output_tokens > 8192:
            raise ValueError("max_output_tokens must not exceed 8192")
        if args.context_tokens < 512 or args.context_tokens > 32768:
            raise ValueError("context_tokens must be between 512 and 32768")
        if args.max_specs > 10:
            raise ValueError("max_specs must be between 1 and 10")
        if args.max_candidate_runs > 3:
            raise ValueError("max_candidate_runs must be between 1 and 3")
        if not args.model or args.model.strip() != args.model:
            raise ValueError("model must be an explicit non-empty local model identifier")
        report = _run_draft_and_evaluate(args)
        _emit(report, as_json=args.json)
        return _EXIT_FAILURE if report["status"] == "FAILED" else _EXIT_OK
    except SystemExit:
        raise
    except Exception as exc:
        # Keep the CLI boundary concise; never print prompts, completions or
        # untrusted provider messages.
        print(f"PHASE14B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE if isinstance(exc, (ValueError, FileExistsError)) else _EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
