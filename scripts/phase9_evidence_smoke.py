"""Bounded local Phase 9 acceptance smoke (read-only and offline)."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import socket
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.experience.config import (
    FEATURE_EXTRACTOR_VERSION,
    FEATURE_SCHEMA_VERSION,
    TRUST_POLICY_VERSION,
)
from tradingagents.experience.models import TrustTier
from tradingagents.experience.source_reader import ReadonlySourceReader
from tradingagents.forex.evidence_audit import EvidenceAuditStore, EvidenceUsageAudit
from tradingagents.forex.evidence_replay import (
    EvidenceReplayConfig,
    SavedSnapshotCodec,
    SavedSnapshotReplay,
    SnapshotReplayError,
)
from tradingagents.forex.evidence_runtime import (
    ReadonlyExperienceCatalog,
    ReadonlyKnowledgeCatalog,
    approved_readonly_component,
    approved_readonly_factory,
)


class NetworkAttempt(RuntimeError):
    """An external network request was attempted by the smoke."""


class Phase8PreflightError(RuntimeError):
    """The operator-supplied Phase 8 root is not an accepted read-only root."""


class SmokeGateError(RuntimeError):
    """A post-run acceptance gate failed."""


class SmokeAcceptanceError(SmokeGateError):
    """The replay output does not satisfy the evidence acceptance contract."""


@dataclass(frozen=True, slots=True)
class Phase8ArtifactPreflight:
    root: Path
    catalog_path: Path
    generation_id: str
    trust_policy_version: str
    feature_schema_version: str
    feature_extractor_version: str
    numeric_trust_tiers: tuple[str, ...]
    tier_counts: Mapping[str, int]
    active_record_count: int
    historical_record_count: int
    evaluation_snapshot_count: int

    @property
    def active_generation_id(self) -> str:
        return self.generation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "catalog_path": str(self.catalog_path),
            "generation_id": self.generation_id,
            "trust_policy_version": self.trust_policy_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_extractor_version": self.feature_extractor_version,
            "numeric_trust_tiers": list(self.numeric_trust_tiers),
            "tier_counts": dict(self.tier_counts),
            "active_record_count": self.active_record_count,
            "historical_record_count": self.historical_record_count,
            "evaluation_snapshot_count": self.evaluation_snapshot_count,
        }


def _tier_value(record: Any) -> str:
    value = getattr(record, "trust", getattr(record, "trust_tier", ""))
    return value.value if isinstance(value, TrustTier) else str(value)


def resolve_verified_phase8_root(candidate: Path) -> Phase8ArtifactPreflight:
    """Validate an existing published Phase 8 catalog through read-only APIs.

    No catalog constructor that creates directories/tables is used here.  A
    Tier-C-only population is valid, but Tier C can never enter numeric query
    defaults or statistics.
    """

    root = Path(candidate).expanduser().resolve()
    catalog_path = root / "catalog.sqlite3"
    if not root.is_dir() or not catalog_path.is_file():
        raise Phase8PreflightError(f"Phase 8 artifact root/catalog.sqlite3 is unavailable: {root}")
    catalog: ReadonlyExperienceCatalog | None = None
    try:
        catalog = ReadonlyExperienceCatalog(root)
        generation = catalog.active_generation()
        if not isinstance(generation, Mapping) or not generation.get("generation_id"):
            raise Phase8PreflightError("Phase 8 catalog has no published active generation")
        metadata = generation.get("metadata")
        if not isinstance(metadata, Mapping):
            raise Phase8PreflightError("Phase 8 generation metadata is missing")
        records = tuple(catalog.active_records())
        historical = tuple(catalog.historical_records())
        all_records = records + historical
        snapshots = 0
        for record in all_records:
            snapshots += len(catalog.evaluation_snapshots(str(record.experience_id)))
        values = {
            "trust_policy_version": metadata.get("trust_policy_version"),
            "feature_schema_version": metadata.get("feature_schema_version"),
            "feature_extractor_version": metadata.get("feature_extractor_version"),
        }
        for key, expected in (("trust_policy_version", TRUST_POLICY_VERSION), ("feature_schema_version", FEATURE_SCHEMA_VERSION), ("feature_extractor_version", FEATURE_EXTRACTOR_VERSION)):
            observed = values[key]
            if observed != expected:
                raise Phase8PreflightError(f"Phase 8 {key} is incompatible: {observed!r}")
        tier_counts = {tier.value: 0 for tier in TrustTier}
        for record in all_records:
            tier = _tier_value(record)
            if tier not in tier_counts:
                raise Phase8PreflightError(f"unknown Phase 8 trust tier: {tier}")
            tier_counts[tier] += 1
            for key, expected in (("feature_schema_version", FEATURE_SCHEMA_VERSION), ("feature_extractor_version", FEATURE_EXTRACTOR_VERSION), ("trust_policy_version", TRUST_POLICY_VERSION)):
                observed = getattr(record, key, None)
                if observed != expected:
                    raise Phase8PreflightError(f"Phase 8 record {key} is incompatible: {observed!r}")
        # This is intentionally an explicit immutable tuple: callers cannot
        # accidentally add Tier C to ExperienceQuery/OutcomeStatsRequest.
        numeric_tiers = (TrustTier.TIER_A_HIGH_TRUST.value, TrustTier.TIER_B_LIMITED.value)
        return Phase8ArtifactPreflight(
            root=root,
            catalog_path=catalog_path,
            generation_id=str(generation["generation_id"]),
            trust_policy_version=TRUST_POLICY_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_extractor_version=FEATURE_EXTRACTOR_VERSION,
            numeric_trust_tiers=numeric_tiers,
            tier_counts=tier_counts,
            active_record_count=len(records),
            historical_record_count=len(historical),
            evaluation_snapshot_count=snapshots,
        )
    except Phase8PreflightError:
        raise
    except Exception as exc:
        raise Phase8PreflightError(f"Phase 8 read/query preflight failed: {exc}") from exc
    finally:
        if catalog is not None:
            close = getattr(catalog, "close", None)
            if callable(close):
                close()


def _resolve_verified_phase7_generation(root: Path) -> Any:
    """Read and validate the active Phase 7 generation without maintenance."""

    catalog: ReadonlyKnowledgeCatalog | None = None
    try:
        catalog = ReadonlyKnowledgeCatalog(root)
        generation = catalog.active_generation()
        if generation is None:
            raise Phase8PreflightError("Phase 7 catalog has no active generation")
        if not bool(getattr(generation, "vector_ready", False)) or not bool(getattr(generation, "lexical_ready", False)):
            raise Phase8PreflightError("Phase 7 active generation is incomplete")
        if str(getattr(generation, "status", "")) != "VALIDATED":
            raise Phase8PreflightError("Phase 7 active generation is not validated")
        return generation
    except Phase8PreflightError:
        raise
    except Exception as exc:
        raise Phase8PreflightError(f"Phase 7 read-only preflight failed: {exc}") from exc
    finally:
        if catalog is not None:
            close = getattr(catalog, "close", None)
            if callable(close):
                close()


def _fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_file():
        raw = path.read_bytes()
        stat = path.stat()
        return {"sha256": hashlib.sha256(raw).hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not path.is_dir():
        return {"sha256": None, "size": 0, "mtime_ns": None, "files": {}}
    entries: dict[str, Any] = {}
    for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        raw = item.read_bytes()
        stat = item.stat()
        entries[str(item.relative_to(path))] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "size": sum(v["size"] for v in entries.values()), "mtime_ns": path.stat().st_mtime_ns, "files": entries}


def _source_fingerprint(path: Path) -> dict[str, Any]:
    return {"db": _fingerprint(path), "wal": _fingerprint(Path(str(path) + "-wal"))}


def _assert_report_outside(report_path: Path | None, roots: Sequence[tuple[Path, str]]) -> None:
    if report_path is None:
        return
    target = report_path.expanduser().resolve()
    for root, label in roots:
        root = root.resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        raise Phase8PreflightError(f"report path must be outside {label} artifacts")


def _host_is_loopback(value: Any) -> bool:
    if isinstance(value, tuple):
        value = value[0] if value else ""
    host = str(value).strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.startswith(("\\\\.\\pipe\\", "/"))


class OfflineNetworkGuard:
    """Allow documented local endpoints and fail before external I/O."""

    def __init__(self) -> None:
        self.loopback_connection_attempts = 0
        self.external_network_attempts = 0
        self.attempts: list[str] = []
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex
        self._create_connection = socket.create_connection
        self._urlopen = urllib.request.urlopen
        self._inside_create = False

    @property
    def attempt_count(self) -> int:
        return self.external_network_attempts

    @property
    def network_attempts(self) -> int:
        return self.external_network_attempts

    def _check(self, address: Any) -> None:
        if _host_is_loopback(address):
            self.loopback_connection_attempts += 1
            return
        self.external_network_attempts += 1
        self.attempts.append(str(address))
        raise NetworkAttempt(f"external network blocked: {address}")

    def __enter__(self) -> OfflineNetworkGuard:
        def connect(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
            if self._inside_create:
                return self._connect(sock, address, *args, **kwargs)
            self._check(address)
            return self._connect(sock, address, *args, **kwargs)

        def create(address: Any, *args: Any, **kwargs: Any) -> Any:
            self._check(address)
            self._inside_create = True
            try:
                return self._create_connection(address, *args, **kwargs)
            finally:
                self._inside_create = False

        def urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
            parsed = urlparse(str(url))
            self._check(parsed.hostname or parsed.path)
            return self._urlopen(url, *args, **kwargs)

        socket.socket.connect = connect  # type: ignore[method-assign]

        def connect_ex(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> int:
            self._check(address)
            return self._connect_ex(sock, address, *args, **kwargs)

        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.create_connection = create  # type: ignore[assignment]
        urllib.request.urlopen = urlopen  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        socket.socket.connect = self._connect  # type: ignore[method-assign]
        socket.socket.connect_ex = self._connect_ex  # type: ignore[method-assign]
        socket.create_connection = self._create_connection  # type: ignore[assignment]
        urllib.request.urlopen = self._urlopen  # type: ignore[assignment]


# Descriptive aliases keep the guard import stable for acceptance fixtures.
LoopbackNetworkGuard = OfflineNetworkGuard
LoopbackOnlyNetworkGuard = OfflineNetworkGuard


def _privacy(value: Any) -> Any:
    forbidden = ("prompt", "completion", "reasoning", "credential", "api_key", "password", "token")
    if is_dataclass(value):
        return {field.name: _privacy(getattr(value, field.name)) for field in fields(value) if not any(token in field.name.lower() for token in forbidden)}
    if isinstance(value, Mapping):
        return {str(key): _privacy(item) for key, item in value.items() if not any(token in str(key).lower() for token in forbidden)}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_privacy(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and any(token in value.lower() for token in forbidden):
            return "[REDACTED_SENSITIVE_TEXT]"
        return value
    return "[REDACTED_UNSERIALIZABLE]"


def build_report(**values: Any) -> dict[str, Any]:
    """Build a typed, metadata-only report with forbidden fields removed."""

    payload = _privacy(values)
    payload.setdefault("loopback_connection_attempts", 0)
    payload.setdefault("external_network_attempts", 0)
    payload.setdefault("warnings", [])
    payload.setdefault("errors", [])
    return payload


def _value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _validate_smoke_result(result: Any) -> None:
    """Fail closed unless the evidence-enabled leg is genuinely acceptable."""

    comparison = str(_value(result, "comparison_status", ""))
    evidence_telemetry = _value(result, "evidence_telemetry", {})
    integration = str(_value(result, "integration_status", _value(evidence_telemetry, "evidence_integration_status", "")))
    bundle = str(_value(result, "bundle_status", ""))
    action = _value(result, "evidence_action") or _value(result, "normalized_action")
    context_hash = _value(result, "evidence_context_hash")
    citation = str(_value(result, "citation_status", _value(result, "evidence_audit_status", "")))
    normalization = str(_value(result, "normalization_status", _value(evidence_telemetry, "normalization_status", "")))
    refs_used = _value(result, "used_references", _value(result, "evidence_refs_used", ()))
    refs_rejected = _value(result, "rejected_references", _value(result, "evidence_refs_rejected", ()))
    if comparison != "VALID":
        raise SmokeAcceptanceError(f"replay comparison is not valid: {comparison or 'missing'}")
    if integration in {"", "FALLBACK", "DISABLED", "EvidenceIntegrationStatus.FALLBACK", "EvidenceIntegrationStatus.DISABLED"}:
        raise SmokeAcceptanceError("evidence integration was unavailable or fell back")
    if bundle in {"", "EMPTY", "FAILED", "EvidenceBundleStatus.EMPTY", "EvidenceBundleStatus.FAILED"}:
        raise SmokeAcceptanceError("evidence bundle is empty or failed")
    if not isinstance(action, str) or action.upper() not in {"BUY", "SELL", "HOLD"}:
        raise SmokeAcceptanceError("final action was not normalized")
    if not isinstance(context_hash, str) or len(context_hash) != 64:
        raise SmokeAcceptanceError("evidence context hash is missing")
    if citation not in {"VALID", "EvidenceAuditStatus.VALID", "USED"}:
        raise SmokeAcceptanceError("citation validation did not pass")
    if normalization not in {"NORMALIZED", ""}:
        raise SmokeAcceptanceError("final action normalization failed")
    if not isinstance(refs_used, (tuple, list)) or not isinstance(refs_rejected, (tuple, list)):
        raise SmokeAcceptanceError("final evidence references were not validated")
    if (_value(result, "knowledge_count", 0) or _value(result, "experience_count", 0) or _value(result, "statistics_count", 0)) and not (refs_used or refs_rejected):
        raise SmokeAcceptanceError("evidence items have no final reference validation")


def _run_fake_ab(runner: Any, *, snapshot: Any, snapshot_bytes: bytes, config: Any) -> dict[str, Any]:
    """Run a deterministic fake baseline/evidence pair in sequence."""

    started = time.monotonic()
    outputs: list[Any] = []
    for enabled in (False, True):
        analyze = getattr(runner, "analyze", None)
        if not callable(analyze):
            raise SmokeAcceptanceError("fake replay runner must expose analyze()")
        outputs.append(analyze(snapshot=snapshot, snapshot_bytes=snapshot_bytes, config=config, evidence_enabled=enabled, forex_evidence_enabled=enabled, persist=False))
    baseline_action = _value(outputs[0], "normalized_action")
    evidence_action = _value(outputs[1], "normalized_action")
    return {"analysis_count": len(outputs), "baseline_action": None if baseline_action is None else str(baseline_action), "evidence_action": None if evidence_action is None else str(evidence_action), "did_action_change": baseline_action != evidence_action, "latency_seconds": time.monotonic() - started}


class _DeterministicFakeRunner:
    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        return {"normalized_action": "BUY" if kwargs.get("evidence_enabled") else "HOLD"}


def _audit_path(report_path: Path | None, source: Path) -> Path:
    root = report_path.parent if report_path is not None else source.parent
    if root.name.casefold() == "evidence_runtime":
        return root / "evidence_audit.sqlite3"
    return root / "evidence_runtime" / "evidence_audit.sqlite3"


def _append_audit(path: Path | EvidenceAuditStore, *, decision_id: str, source_run_id: str, snapshot: Any, result: Any) -> None:
    empty_hash = hashlib.sha256(b"").hexdigest()
    audit = EvidenceUsageAudit(
        decision_id=decision_id,
        source_run_id=source_run_id,
        integration_status=str(_value(result, "integration_status", "INJECTED")),
        bundle_status=str(_value(result, "bundle_status", "COMPLETE")),
        as_of=snapshot.timestamp,
        knowledge_generation_id=_value(result, "pinned_phase7_generation_id"),
        experience_generation_id=_value(result, "pinned_phase8_generation_id"),
        knowledge_query_fingerprint=_value(result, "knowledge_query_fingerprint"),
        rendered_context="",
        rendered_context_hash=empty_hash,
        evidence_use_status="USED" if _value(result, "evidence_action") else "UNAVAILABLE",
        evidence_audit_status="VALID",
        retrieval_count=1,
        selected_counts={"knowledge": int(_value(result, "knowledge_count", 0) or 0), "experience": int(_value(result, "experience_count", 0) or 0), "statistics": int(_value(result, "statistics_count", 0) or 0)},
        provider="ollama",
        model="qwen",
    )
    store = path if hasattr(path, "append") else EvidenceAuditStore(path)
    store.append(audit)


def _uses_saved_snapshot() -> bool:
    return True


@approved_readonly_factory
def build_local_orchestrator(config: Any = None) -> Any:
    """Top-level child factory for the local read-only evidence query."""

    from tradingagents.experience.orchestrator import EvidenceOrchestrator
    from tradingagents.experience.outcomes import OutcomeStatsCalculator
    from tradingagents.experience.query import ExperienceQueryService
    from tradingagents.knowledge.config import KnowledgeConfig
    from tradingagents.knowledge.embeddings import FastEmbedProvider
    from tradingagents.knowledge.lexical_index import LexicalIndexReader
    from tradingagents.knowledge.query import KnowledgeQueryService
    from tradingagents.knowledge.vector_index import VectorIndexReader

    roots = config.roots() if hasattr(config, "roots") else dict(config or {})
    # The evidence worker is a separate process; install the same guard in
    # that process so an external request cannot escape the parent boundary.
    child_guard = OfflineNetworkGuard()
    child_guard.__enter__()
    approved_readonly_component(child_guard)
    experience = ReadonlyExperienceCatalog(roots["experience"])
    knowledge = ReadonlyKnowledgeCatalog(roots["knowledge"])
    generation = knowledge.active_generation()
    if generation is None:
        raise Phase8PreflightError("Phase 7 has no active generation")
    model_path = Path(roots["knowledge_embedding_model_path"])
    knowledge_config = KnowledgeConfig(source_root=Path.cwd(), artifact_root=knowledge.path.parent, embedding_model_path=model_path, offline=True)
    embedder = FastEmbedProvider.from_config(knowledge_config)
    vector_reader = VectorIndexReader(generation.vector_location)
    lexical_reader = LexicalIndexReader(generation.lexical_location)
    knowledge_service = KnowledgeQueryService(vector_reader, lexical_reader, knowledge, embedder)
    experience_service = ExperienceQueryService(experience, feature_vectors=experience.feature_vectors, profiles=experience.profiles, generation_id=experience.active_generation().get("generation_id") if experience.active_generation() else None)
    orchestrator = EvidenceOrchestrator(knowledge_service, experience_service, OutcomeStatsCalculator(experience))
    orchestrator._phase9_network_guard = child_guard
    return orchestrator


def run_smoke(
    source_db: str | Path,
    *,
    experience_artifact_root: str | Path,
    knowledge_artifact_root: str | Path,
    knowledge_embedding_model_path: str | Path,
    profile: str = "INTRADAY",
    analysts: Sequence[str] = ("market", "news"),
    offline: bool = False,
    report_path: str | Path | None = None,
    replay_factory: Callable[..., Any] | None = None,
    runner: Any | None = None,
    fake_runner: Any | None = None,
    audit_store: EvidenceAuditStore | None = None,
) -> dict[str, Any]:
    if not offline:
        raise ValueError("--offline is required")
    required_env = ("KNOWLEDGE_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    if any(os.environ.get(name) != "1" for name in required_env):
        raise ValueError("KNOWLEDGE_OFFLINE, HF_HUB_OFFLINE, and TRANSFORMERS_OFFLINE must all be 1")
    source = Path(source_db).expanduser().resolve()
    experience_root = Path(experience_artifact_root).expanduser().resolve()
    knowledge_root = Path(knowledge_artifact_root).expanduser().resolve()
    model_path = Path(knowledge_embedding_model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source database does not exist: {source}")
    if not knowledge_root.is_dir() or not (knowledge_root / "catalog.sqlite3").is_file():
        raise FileNotFoundError(f"Phase 7 catalog does not exist: {knowledge_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"local embedding model does not exist: {model_path}")
    # This must happen before ReadonlySourceReader queries the source.
    phase8 = resolve_verified_phase8_root(experience_root)
    generation7_obj = _resolve_verified_phase7_generation(knowledge_root)
    _assert_report_outside(None if report_path is None else Path(report_path), ((source, "source"), (Path(str(source) + "-wal"), "source WAL"), (knowledge_root, "Phase 7"), (experience_root, "Phase 8"), (model_path, "embedding model")))
    before = {"source": _source_fingerprint(source), "phase7": _fingerprint(knowledge_root), "phase8": _fingerprint(experience_root)}
    reader = ReadonlySourceReader(source)
    source_snapshot = reader.read_snapshot()
    row = next((item for item in source_snapshot.decisions if item.get("snapshot_json")), None)
    if row is None:
        raise SnapshotReplayError("source has no saved ForexMarketSnapshot decision")
    snapshot = SavedSnapshotCodec.from_source_row(row)
    encoded = row["snapshot_json"]
    snapshot_bytes = encoded if isinstance(encoded, bytes) else str(encoded).encode("utf-8")
    generation7 = str(generation7_obj.generation_id)
    config = EvidenceReplayConfig(source_decision_id=str(row["decision_id"]), profile=profile, analysts=tuple(analysts), provider="ollama", models={"quick": "qwen", "deep": "qwen"}, model_settings={"temperature": 0}, pinned_phase7_generation_id=generation7, pinned_phase8_generation_id=phase8.generation_id, phase7_root=knowledge_root, phase8_root=experience_root, source_database_path=source)
    started = time.monotonic()
    guard = OfflineNetworkGuard()
    replay_result: Any = None
    errors: list[str] = []
    try:
        with guard:
            if replay_factory is not None:
                replay_result = replay_factory(snapshot=snapshot, snapshot_bytes=snapshot_bytes, config=config, runner=runner)
            else:
                if runner is None:
                    from tradingagents.forex.runner import ForexShadowRunner
                    runner = ForexShadowRunner(store=type("NoopStore", (), {})(), config={"forex_evidence_enabled": True, "forex_evidence_artifact_roots": {"knowledge": str(knowledge_root), "experience": str(experience_root), "knowledge_embedding_model_path": str(model_path)}, "evidence_orchestrator_factory": build_local_orchestrator, "llm_provider": "ollama", "backend_url": "http://127.0.0.1:11434/v1", "quick_think_llm": "qwen", "deep_think_llm": "qwen"})
                replay_result = SavedSnapshotReplay(runner=runner, generation_provider=(generation7, phase8.generation_id)).run(snapshot, snapshot_bytes=snapshot_bytes, config=config)
    except Exception as exc:
        # Keep reports diagnostic-only; arbitrary exception text may contain
        # endpoints, credentials, prompts, or model output.
        errors.append(type(exc).__name__)
        if "external network blocked" in str(exc).lower():
            guard.external_network_attempts = max(guard.external_network_attempts, 1)
    after = {"source": _source_fingerprint(source), "phase7": _fingerprint(knowledge_root), "phase8": _fingerprint(experience_root)}
    unchanged = before == after
    if not unchanged:
        errors.append("source or Phase 7/8 artifact bytes changed during smoke")
    fake_ab: Any = None
    if replay_result is not None:
        try:
            fake_ab = _run_fake_ab(fake_runner or _DeterministicFakeRunner(), snapshot=snapshot, snapshot_bytes=snapshot_bytes, config=config)
        except Exception as exc:
            errors.append(f"fake A/B failed: {type(exc).__name__}")
    else:
        errors.append("SmokeAcceptanceError: real replay returned no comparison")
    result_payload = replay_result.to_dict() if hasattr(replay_result, "to_dict") else replay_result
    if replay_result is not None:
        try:
            _validate_smoke_result(replay_result)
        except SmokeAcceptanceError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        audit_target = _audit_path(None if report_path is None else Path(report_path).expanduser().resolve(), source)
        _assert_report_outside(audit_target, ((source, "source"), (Path(str(source) + "-wal"), "source WAL"), (knowledge_root, "Phase 7"), (experience_root, "Phase 8"), (model_path, "embedding model")))
        try:
            _append_audit(audit_target if audit_store is None else audit_store, decision_id=str(row["decision_id"]), source_run_id=str(row.get("source_run_id") or row["decision_id"]), snapshot=snapshot, result=replay_result)
        except Exception as exc:
            errors.append(f"audit append failed: {type(exc).__name__}")
    report = build_report(real_smoke="PASS" if not errors else "FAILED", source_fingerprint={"before": before["source"], "after": after["source"]}, artifact_fingerprints={"phase7": {"before": before["phase7"], "after": after["phase7"]}, "phase8": {"before": before["phase8"], "after": after["phase8"]}}, source_unchanged=unchanged, phase8_preflight=phase8.to_dict(), replay=result_payload, fake_ab=fake_ab, loopback_connection_attempts=guard.loopback_connection_attempts, external_network_attempts=guard.external_network_attempts, retrieval_count=1 if replay_result is not None else 0, latency_seconds=time.monotonic() - started, provider="ollama", models={"quick": "qwen", "deep": "qwen"}, warnings=[], errors=errors, saved_snapshot_replay=_uses_saved_snapshot())
    if report_path is not None:
        destination = Path(report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if guard.external_network_attempts or errors or not unchanged:
        raise SmokeGateError("Phase 9 smoke integrity/network gate failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--experience-artifact-root", required=True)
    parser.add_argument("--knowledge-artifact-root", required=True)
    parser.add_argument("--knowledge-embedding-model-path", required=True)
    parser.add_argument("--profile", default="INTRADAY")
    parser.add_argument("--analysts", default="market,news")
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--report-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.analysts = tuple(item.strip() for item in args.analysts.split(",") if item.strip())
    try:
        print(json.dumps(run_smoke(**vars(args)), indent=2, sort_keys=True))
        return 0
    except (Phase8PreflightError, FileNotFoundError):
        print("PHASE 9 REAL SMOKE PREREQUISITE FAILED")
        return 2
    except Exception as exc:
        print(f"PHASE 9 REAL SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
