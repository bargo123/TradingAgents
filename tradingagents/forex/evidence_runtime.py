"""Read-only Phase 7/8 evidence integration for the forex shadow path."""

from __future__ import annotations

import importlib
import inspect
import json
import multiprocessing
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tradingagents.experience.models import (
    EvidenceBundle,
    EvidenceRequest,
    ExperienceRecord,
    ExperienceSearchResult,
)
from tradingagents.experience.normalization import NormalizationCohortV1, build_profile
from tradingagents.experience.orchestrator import EvidenceOrchestrator
from tradingagents.experience.outcomes import OutcomeStatsCalculator
from tradingagents.experience.query import ExperienceQueryService
from tradingagents.knowledge.embeddings import EmbeddingProvider
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.models import EmbeddingSpec, IndexGeneration, KnowledgeHit
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.reranking import Reranker
from tradingagents.knowledge.vector_index import VectorIndexReader

from .evidence_context import (
    EvidenceBundleStatus,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceQueryPolicy,
    Phase9EvidenceContextBuilder,
)


class EvidenceTimeout(TimeoutError):
    """The isolated evidence worker exceeded its configured deadline."""


@dataclass(frozen=True, slots=True)
class ReadonlyEvidenceRuntimeConfiguration:
    """Serializable child configuration; it contains paths, never services."""

    artifact_roots: tuple[tuple[str, str], ...] = ()
    evidence_timeout_seconds: float = 10.0

    @classmethod
    def from_envelope(cls, envelope: Mapping[str, Any]) -> ReadonlyEvidenceRuntimeConfiguration:
        roots = envelope.get("artifact_roots") or {}
        if not isinstance(roots, Mapping):
            raise TypeError("artifact_roots must be a mapping")
        return cls(
            tuple(sorted((str(key), str(value)) for key, value in roots.items())),
            float(envelope.get("evidence_timeout_seconds", 10.0)),
        )

    def roots(self) -> dict[str, str]:
        return dict(self.artifact_roots)


def approved_readonly_factory(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Mark one top-level factory as an approved read-only child seam."""

    factory.__evidence_runtime_approved__ = True
    return factory


def approved_readonly_component(component: Any) -> Any:
    """Mark a concrete reader/provider wrapper as side-effect free."""

    component.__evidence_runtime_readonly__ = True
    return component


_CLOSED_OBJECT_IDS: set[int] = set()


def _database_path(root: str | Path) -> Path:
    path = Path(root)
    return path if path.name == "catalog.sqlite3" else path / "catalog.sqlite3"


class _ReadonlySQLite:
    def __init__(self, root: str | Path):
        self.database_path = _database_path(root).resolve()
        self.path = self.database_path
        self._connection = self._open()

    def _open(self) -> sqlite3.Connection:
        uri = f"file:{self.database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _read(self):
        try:
            yield self._connection
        finally:
            pass

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False


class ReadonlyKnowledgeCatalog(_ReadonlySQLite):
    """Phase 7 catalog reader that cannot create or migrate its database."""

    @staticmethod
    def _generation(row: sqlite3.Row) -> IndexGeneration:
        return IndexGeneration.from_dict(
            {
                "generation_id": row["generation_id"],
                "vector_location": row["vector_location"],
                "lexical_location": row["lexical_location"],
                "embedding_spec": json.loads(row["embedding_spec_json"]),
                "lexical_index_version": row["lexical_index_version"],
                "lexical_tokenizer_settings": json.loads(row["lexical_tokenizer_settings_json"]),
                "index_version": row["index_version"],
                "population_hash": row["population_hash"],
                "population_identity": row["population_identity"],
                "document_count": row["document_count"],
                "chunk_count": row["chunk_count"],
                "vector_ready": bool(row["vector_ready"]),
                "lexical_ready": bool(row["lexical_ready"]),
                "status": row["status"],
                "created_at": row["created_at"],
                "activated_at": row["activated_at"],
                "component_versions": json.loads(row["component_versions_json"]),
            }
        )

    def active_generation(self) -> IndexGeneration | None:
        try:
            row = self._connection.execute(
                "SELECT * FROM knowledge_index_generations WHERE is_active=1 ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return None if row is None else self._generation(row)

    def document_is_retrieval_ready(self, document_id: str) -> bool:
        try:
            row = self._connection.execute(
                """SELECT 1 FROM knowledge_documents AS document
                   JOIN knowledge_index_generations AS generation
                     ON generation.generation_id=document.projection_generation
                    AND generation.is_active=1
                   WHERE document.document_id=? AND document.active=1
                     AND document.vector_ready=1 AND document.lexical_ready=1
                     AND EXISTS (SELECT 1 FROM knowledge_aliases AS alias
                                 WHERE alias.document_id=document.document_id
                                   AND alias.relation_status='CURRENT')""",
                (document_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return row is not None


class ReadonlyExperienceCatalog(_ReadonlySQLite):
    """Phase 8 catalog reader with in-memory normalization profiles only."""

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._projection_rows = self._load_projection_rows()
        self.feature_vectors = dict(self._projection_rows)
        self.profiles = self._build_profiles()

    @staticmethod
    def _json(value: str | None) -> dict[str, Any]:
        try:
            payload = json.loads(value or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _load_projection_rows(self) -> dict[str, dict[str, Any]]:
        try:
            rows = self._connection.execute(
                "SELECT experience_id, feature_schema_version, projection_json FROM experience_feature_projections"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = ()
        return {
            str(row["experience_id"]): {
                **self._json(row["projection_json"]),
                "experience_id": str(row["experience_id"]),
                "feature_schema_version": row["feature_schema_version"],
            }
            for row in rows
        }

    def _record(self, experience_id: str) -> ExperienceRecord:
        row = self._connection.execute(
            "SELECT * FROM experience_records WHERE experience_id=?", (experience_id,)
        ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        aliases = {
            f"{alias['source_database_id']}:{alias['source_decision_id']}": alias["state"]
            for alias in self._connection.execute(
                "SELECT source_database_id, source_decision_id, state FROM experience_source_aliases WHERE experience_id=?",
                (experience_id,),
            )
        }
        return ExperienceRecord(
            experience_id=row["experience_id"],
            source_database_id=row["source_database_id"],
            source_decision_id=row["source_decision_id"],
            symbol=row["symbol"],
            analysis_snapshot_timestamp=datetime.fromisoformat(row["analysis_snapshot_timestamp"]),
            source_aliases=aliases,
            source_run_id=row["source_run_id"],
            requested_symbol=row["requested_symbol"],
            analysis_profile=row["analysis_profile"],
            analysis_timeframe=row["analysis_timeframe"],
            decision_completed_timestamp=(
                datetime.fromisoformat(row["decision_completed_timestamp"])
                if row["decision_completed_timestamp"]
                else None
            ),
            decision_reference_timestamp=(
                datetime.fromisoformat(row["decision_reference_timestamp"])
                if row["decision_reference_timestamp"]
                else None
            ),
            market_state=self._json(row["market_state_json"]),
            decision_evidence=self._json(row["decision_evidence_json"]),
            provenance=self._json(row["provenance_json"]),
            trust=row["trust"],
            source_decision_fingerprint=row["source_decision_fingerprint"],
            source_evaluation_fingerprints=self._json(row["source_evaluation_fingerprints_json"]),
        )

    def active_records(self) -> tuple[ExperienceRecord, ...]:
        try:
            rows = self._connection.execute(
                "SELECT DISTINCT experience_id FROM experience_source_aliases WHERE state='CURRENT' ORDER BY experience_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return ()
        return tuple(self._record(str(row[0])) for row in rows)

    def historical_records(self) -> tuple[ExperienceRecord, ...]:
        try:
            rows = self._connection.execute(
                "SELECT experience_id FROM experience_records WHERE tombstoned=1 ORDER BY experience_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return ()
        return tuple(self._record(str(row[0])) for row in rows)

    def evaluation_snapshots(self, experience_id: str) -> tuple[dict[str, Any], ...]:
        try:
            rows = self._connection.execute(
                "SELECT * FROM experience_outcome_snapshots WHERE experience_id=? ORDER BY rowid",
                (experience_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return ()
        return tuple(
            {
                **self._json(row["evaluation_json"]),
                "fingerprint": row["fingerprint"],
                "provenance": self._json(row["provenance_json"]),
                "observed_at": row["observed_at"],
            }
            for row in rows
        )

    def active_generation(self) -> dict[str, Any] | None:
        try:
            row = self._connection.execute(
                "SELECT * FROM experience_generations WHERE active=1 ORDER BY published_at DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return {
            "generation_id": row["generation_id"],
            "population_fingerprint": row["population_fingerprint"],
            "metadata": self._json(row["metadata_json"]),
            "published_at": row["published_at"],
        }

    def _build_profiles(self) -> dict[Any, Any]:
        profiles: dict[Any, Any] = {}
        records = self.active_records() + self.historical_records()
        rows: list[dict[str, Any]] = []
        for record in records:
            projection = dict(self._projection_rows.get(record.experience_id, record.market_state or {}))
            cohort_raw = projection.get("cohort")
            if cohort_raw is None:
                continue
            try:
                cohort = cohort_raw if isinstance(cohort_raw, NormalizationCohortV1) else NormalizationCohortV1(*tuple(cohort_raw))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    **projection,
                    "experience_id": record.experience_id,
                    "trust_tier": record.trust,
                    "source_aliases": record.source_aliases,
                    "analysis_snapshot_timestamp": record.analysis_snapshot_timestamp,
                    "decision_completed_timestamp": record.decision_completed_timestamp,
                    "provenance": record.provenance,
                    "feature_fingerprint": projection.get("feature_fingerprint", "persisted"),
                    "cohort": cohort,
                }
            )
        cohorts = {row["cohort"] for row in rows}
        for cohort in cohorts:
            cohort_rows = [row for row in rows if row["cohort"] == cohort]
            profiles[cohort] = build_profile(cohort_rows, cohort)
        return profiles

    def feature_projection(self, experience_id: str) -> Mapping[str, Any] | None:
        return self._projection_rows.get(str(experience_id))


def _bounded_diagnostic(code: str, detail: Any = "") -> dict[str, str]:
    message = " ".join(str(detail).replace("\r", " ").replace("\n", " ").split())[:500]
    return {"code": str(code)[:100], "message": message}


def _generations(provider: Any) -> tuple[str | None, str | None]:
    value = provider() if callable(provider) else provider
    if isinstance(value, Mapping):
        return value.get("knowledge_generation_id", value.get("phase7_generation_id")), value.get("experience_generation_id", value.get("phase8_generation_id"))
    if isinstance(value, (tuple, list)):
        return (value[0] if len(value) > 0 else None, value[1] if len(value) > 1 else None)
    return getattr(value, "knowledge_generation_id", None), getattr(value, "experience_generation_id", None)


def _request_payload(request: EvidenceRequest) -> dict[str, Any]:
    payload = request.to_dict()
    if payload.get("as_of"):
        payload["as_of"] = datetime.fromisoformat(str(payload["as_of"]).replace("Z", "+00:00"))
    payload["trust_tiers"] = tuple(payload.get("trust_tiers") or ())
    return payload


def _factory_descriptor(factory: Callable[..., Any]) -> dict[str, str]:
    module = getattr(factory, "__module__", "")
    qualname = getattr(factory, "__qualname__", "")
    if (
        not module
        or not qualname
        or "<locals>" in qualname
        or "<lambda>" in qualname
        or not getattr(factory, "__evidence_runtime_approved__", False)
    ):
        raise TypeError("orchestrator factory must be a top-level importable callable")
    return {"module": module, "qualname": qualname}


def _resolve_factory(descriptor: Mapping[str, str]) -> Callable[..., Any]:
    value: Any = importlib.import_module(str(descriptor["module"]))
    for part in str(descriptor["qualname"]).split("."):
        value = getattr(value, part)
    if not callable(value) or not getattr(value, "__evidence_runtime_approved__", False):
        raise TypeError("orchestrator factory descriptor is not callable")
    return value


def _make_orchestrator(factory: Callable[..., Any], config: ReadonlyEvidenceRuntimeConfiguration) -> Any:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    return factory(config) if parameters else factory()


def _close_orchestrator(orchestrator: Any) -> None:
    seen: set[int] = set()
    pending = [orchestrator]
    names = (
        "knowledge_service",
        "experience_service",
        "statistics_calculator",
        "knowledge_catalog",
        "experience_catalog",
        "catalog",
    )
    while pending:
        owner = pending.pop()
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                pending.append(value)
        close = getattr(owner, "close", None)
        if callable(close):
            if getattr(owner, "_evidence_runtime_closed", False) or id(owner) in _CLOSED_OBJECT_IDS:
                continue
            close()
            try:
                owner._evidence_runtime_closed = True
            except (AttributeError, TypeError):
                _CLOSED_OBJECT_IDS.add(id(owner))


def _child_query(send_conn: Any, factory_descriptor: Mapping[str, str], envelope: Mapping[str, Any]) -> None:
    orchestrator = None
    try:
        factory = _resolve_factory(factory_descriptor)
        config = ReadonlyEvidenceRuntimeConfiguration.from_envelope(envelope)
        request = EvidenceRequest(**dict(envelope["request"]))
        orchestrator = _make_orchestrator(factory, config)
        _validate_child_orchestrator(orchestrator, config)
        bundle = orchestrator.query(request)
        send_conn.send(("ok", bundle.to_dict() if hasattr(bundle, "to_dict") else bundle))
    except BaseException as exc:  # process boundary transports a bounded typed failure
        kind = "timeout" if isinstance(exc, EvidenceTimeout) else "error"
        send_conn.send((kind, type(exc).__name__, str(exc).replace("\r", " ").replace("\n", " ")[:500]))
    finally:
        if orchestrator is not None:
            _close_orchestrator(orchestrator)
        send_conn.close()


def _validate_child_orchestrator(
    orchestrator: Any, config: ReadonlyEvidenceRuntimeConfiguration | None = None
) -> None:
    """Reject writer/maintenance objects crossing the approved child seam."""

    if not isinstance(orchestrator, EvidenceOrchestrator):
        raise TypeError("approved evidence factory must return one EvidenceOrchestrator")
    if not isinstance(getattr(orchestrator, "knowledge_service", None), KnowledgeQueryService):
        raise TypeError("evidence child requires a typed KnowledgeQueryService")
    if not isinstance(getattr(orchestrator, "experience_service", None), ExperienceQueryService):
        raise TypeError("evidence child requires a typed ExperienceQueryService")
    if not isinstance(getattr(orchestrator, "statistics_calculator", None), OutcomeStatsCalculator):
        raise TypeError("evidence child requires a typed OutcomeStatsCalculator")
    knowledge_catalog = getattr(orchestrator.knowledge_service, "catalog", None)
    experience_catalog = getattr(orchestrator.experience_service, "catalog", None)
    if not isinstance(knowledge_catalog, ReadonlyKnowledgeCatalog):
        raise TypeError("evidence child requires a read-only Phase 7 catalog adapter")
    if not isinstance(experience_catalog, ReadonlyExperienceCatalog):
        raise TypeError("evidence child requires a read-only Phase 8 catalog adapter")
    vector_reader = getattr(orchestrator.knowledge_service, "vector_reader", None)
    lexical_reader = getattr(orchestrator.knowledge_service, "lexical_reader", None)
    embedder = getattr(orchestrator.knowledge_service, "embedder", None)
    if not isinstance(vector_reader, VectorIndexReader) or not getattr(vector_reader, "__evidence_runtime_readonly__", False):
        raise TypeError("evidence child requires an approved read-only vector reader")
    if not isinstance(lexical_reader, LexicalIndexReader) or not getattr(lexical_reader, "__evidence_runtime_readonly__", False):
        raise TypeError("evidence child requires an approved read-only lexical reader")
    if not isinstance(embedder, EmbeddingProvider) or not getattr(embedder, "__evidence_runtime_readonly__", False):
        raise TypeError("evidence child requires an approved read-only embedding provider")

    # Only walk the approved object graph.  Looking at every value reachable
    # from an experience record would be both expensive and too permissive;
    # these are the service/index/provider edges that can own side effects.
    forbidden = (
        "writer", "maintenance", "migration", "importer", "initialize",
        "create_schema", "download", "provision", "trainer", "training",
        "fine_tune", "finetune", "model_provider", "downloader", "model_download",
    )
    pending = [orchestrator]
    seen: set[int] = set()
    allowed_types = (
        EvidenceOrchestrator,
        KnowledgeQueryService,
        ExperienceQueryService,
        OutcomeStatsCalculator,
        ReadonlyKnowledgeCatalog,
        ReadonlyExperienceCatalog,
        VectorIndexReader,
        LexicalIndexReader,
        EmbeddingProvider,
        EmbeddingSpec,
        Reranker,
    )
    edges = (
        "knowledge_service", "experience_service", "statistics_calculator",
        "catalog", "vector_reader", "lexical_reader", "embedder", "reranker",
    )
    while pending:
        owner = pending.pop()
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        if not isinstance(owner, allowed_types) and not getattr(owner, "__evidence_runtime_readonly__", False):
            raise TypeError(
                f"evidence child dependency is not approved: {type(owner).__module__}.{type(owner).__name__}"
            )
        type_name = f"{type(owner).__module__}.{type(owner).__name__}".casefold()
        if any(token in type_name for token in forbidden):
            raise TypeError("evidence child cannot own writer or maintenance components")
        for name in dir(owner):
            lowered = name.casefold()
            if any(token in lowered for token in forbidden):
                raise TypeError("evidence child cannot own writer or maintenance components")
        for name in edges:
            value = getattr(owner, name, None)
            if value is not None:
                pending.append(value)
        # Recurse through instance attributes and container payloads. A
        # writer hidden in ``{"dependencies": [writer]}`` must be rejected,
        # while scalar metadata and immutable value objects remain benign.
        def enqueue(value: Any) -> None:
            if value is None or callable(value) or isinstance(
                value, (str, bytes, bytearray, int, float, complex, bool, Path, datetime, Enum)
            ):
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    enqueue(key)
                    enqueue(item)
                return
            if isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    enqueue(item)
                return
            if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
                if not isinstance(
                    value,
                    (EvidenceBundle, ExperienceRecord, ExperienceSearchResult, IndexGeneration, EmbeddingSpec, KnowledgeHit),
                ) and not getattr(value, "__evidence_runtime_readonly__", False):
                    raise TypeError(
                        f"evidence child dependency is not approved: {type(value).__module__}.{type(value).__name__}"
                    )
                for name in value.__dataclass_fields__:
                    enqueue(getattr(value, name))
                return
            pending.append(value)

        for value in getattr(owner, "__dict__", {}).values():
            enqueue(value)

    if config is not None:
        roots = config.roots()
        if not {"knowledge", "experience"}.issubset(roots):
            raise TypeError("evidence child requires knowledge and experience artifact roots")
        expected = {
            "knowledge": knowledge_catalog.database_path,
            "experience": experience_catalog.database_path,
        }
        for key, database_path in expected.items():
            root = Path(roots[key]).resolve()
            allowed = root if root.name == "catalog.sqlite3" else root / "catalog.sqlite3"
            if database_path.resolve() != allowed:
                raise TypeError("read-only catalog is outside the configured artifact root")


class EvidenceIntegrationService:
    """One-call, best-effort evidence boundary with a lazy disabled path."""

    def __init__(
        self,
        policy: EvidenceQueryPolicy | None,
        orchestrator_factory: Callable[[], Any] | None,
        generation_provider: Callable[[], Any] | Any,
        provider_endpoint: str | Callable[[], str] | None,
        clock: Callable[[], datetime] | None = None,
        process_factory: Callable[..., Any] | None = None,
        artifact_roots: Mapping[str, str | Path] | None = None,
    ) -> None:
        self.policy = policy
        self.orchestrator_factory = orchestrator_factory
        self.generation_provider = generation_provider
        self.provider_endpoint = provider_endpoint
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.process_factory = process_factory
        self.artifact_roots: dict[str, str] = {
            str(key): str(value) for key, value in (artifact_roots or {}).items()
        }
        self.active_evidence_workers = 0
        self.retrieval_count = 0
        self.last_request: Any = None

    @property
    def enabled(self) -> bool:
        return self.policy is not None

    def _fallback(self, as_of: datetime, *, query: Any = None, generations: tuple[Any, Any] = (None, None), code: str = "EVIDENCE_FALLBACK", detail: Any = "") -> EvidenceContext:
        context = Phase9EvidenceContextBuilder().build(
            EvidenceBundle(status="EMPTY"),
            policy=self.policy or EvidenceQueryPolicy(),
            as_of=as_of,
            knowledge_query=query,
            phase7_generation_id=generations[0],
            phase8_generation_id=generations[1],
        )
        return replace(
            context,
            integration_status=EvidenceIntegrationStatus.FALLBACK,
            diagnostics={**dict(context.diagnostics), "integration": _bounded_diagnostic(code, detail)},
        )

    def _query(self, request: Any) -> EvidenceBundle:
        process_factory = self.process_factory or multiprocessing.get_context("spawn").Process
        recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
        envelope = {
            "request": _request_payload(request),
            "artifact_roots": dict(self.artifact_roots),
            "evidence_timeout_seconds": float(self.policy.evidence_timeout_seconds),
        }
        descriptor = _factory_descriptor(self.orchestrator_factory)
        process = process_factory(target=_child_query, args=(send_conn, descriptor, envelope))
        self.active_evidence_workers += 1
        try:
            process.start()
            timeout = max(0.0, float(self.policy.evidence_timeout_seconds if self.policy else 0.0))
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                if process.is_alive():
                    raise EvidenceTimeout("evidence worker survived termination")
                raise EvidenceTimeout("evidence query exceeded timeout")
            if not recv_conn.poll():
                raise RuntimeError("evidence worker returned no result")
            result = recv_conn.recv()
            if result[0] == "timeout":
                raise EvidenceTimeout(result[2])
            if result[0] == "error":
                raise RuntimeError(f"{result[1]}: {result[2]}")
            payload = result[1]
            return payload if isinstance(payload, EvidenceBundle) else EvidenceBundle(**dict(payload))
        finally:
            self.active_evidence_workers -= 1
            recv_conn.close()
            send_conn.close()

    def retrieve(self, snapshot: Any, *, resolved_symbol: str, analysis_profile: str, analysis_timeframe: str) -> EvidenceContext:
        as_of = snapshot.timestamp
        if not self.enabled:
            return EvidenceContext(as_of=as_of, integration_status=EvidenceIntegrationStatus.DISABLED)
        query = None
        generations = (None, None)
        self.retrieval_count += 1
        try:
            request, query = self.policy.build_request(
                snapshot,
                resolved_symbol=resolved_symbol,
                analysis_profile=analysis_profile,
                analysis_timeframe=analysis_timeframe,
                as_of=as_of,
            )
            self.last_request = request
            generations = _generations(self.generation_provider)
            bundle = self._query(request)
            post_generations = _generations(self.generation_provider)
            if post_generations != (None, None):
                generations = post_generations
            context = Phase9EvidenceContextBuilder().build(
                bundle,
                policy=self.policy,
                as_of=as_of,
                knowledge_query=query,
                phase7_generation_id=generations[0],
                phase8_generation_id=generations[1],
            )
            has_text = bool(context.knowledge_items or context.experience_items)
            endpoint = self.provider_endpoint() if callable(self.provider_endpoint) else self.provider_endpoint
            if has_text and not _is_loopback(endpoint):
                return self._fallback(as_of, query=query, generations=generations, code="EVIDENCE_LOCAL_ENDPOINT_REQUIRED")
            if EvidenceBundleStatus(str(bundle.status)) != EvidenceBundleStatus.COMPLETE or not has_text:
                return replace(context, integration_status=EvidenceIntegrationStatus.FALLBACK)
            return context
        except EvidenceTimeout as exc:
            return self._fallback(as_of, query=query, generations=generations, code="EVIDENCE_TIMEOUT", detail=exc)
        except Exception as exc:
            return self._fallback(as_of, query=query, generations=generations, code="ORCHESTRATOR_FAILURE", detail=exc)


def _is_loopback(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    value = str(endpoint).strip()
    parsed = urlparse(value if "://" in value else f"//{value}")
    return (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}


__all__ = [
    "EvidenceIntegrationService",
    "EvidenceTimeout",
    "ReadonlyEvidenceRuntimeConfiguration",
    "ReadonlyExperienceCatalog",
    "ReadonlyKnowledgeCatalog",
    "approved_readonly_component",
    "approved_readonly_factory",
]
