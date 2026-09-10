"""Validated, local-only configuration for the Phase 7 knowledge pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


_DEFAULT_SOURCE_ROOT = Path(r"C:\Users\Zaid barghouthi\Downloads\new books")
_DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "data_cache" / "knowledge"


def _resolved_path(value: str | Path, name: str) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a local filesystem path") from exc


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _nonempty(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _freeze(value: Any) -> Any:
    """Recursively copy config values into immutable, deterministic shapes."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _json_safe(value: Any) -> Any:
    """Render immutable config values as ordinary JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _strict_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise ValueError(f"{name} must be an integer")
    return result


@dataclass(frozen=True, slots=True)
class KnowledgeConfig:
    """Immutable configuration shared by discovery, ingestion, and query.

    The source tree is never used as an artifact destination.  Optional parser
    and embedding artifacts are paths only; adapters decide whether required
    local files are present and raise typed errors when they are not.  This
    keeps constructing a configuration side-effect free and prevents an
    implicit model download.
    """

    source_root: Path = _DEFAULT_SOURCE_ROOT
    artifact_root: Path = _DEFAULT_ARTIFACT_ROOT
    docling_artifacts_path: Path | None = None
    docling_offline: bool = True
    docling_do_ocr: bool = False
    formula_enrichment_enabled: bool = False
    formula_artifacts_path: Path | None = None
    # ``formula_model_path`` is retained as a compatibility spelling for
    # callers that treat formula artifacts as a model directory.
    formula_model_path: Path | None = None
    formula_model_id: str | None = None
    formula_model_version: str | None = None
    formula_artifact_hash: str | None = None

    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    embedding_model_path: Path | None = None
    embedding_dimensions: int = 384
    embedding_runtime: str = "onnx-cpu"
    embedding_normalization: str = "l2"
    embedding_model_version: str = "unknown"
    embedding_artifact_hash: str = ""
    embedding_tokenizer_fingerprint: str = ""
    embedding_max_input_tokens: int = 512
    embedding_special_token_budget: int = 2
    embedding_effective_content_token_limit: int = 510
    embedding_truncation: bool = False
    embedding_corpus_instruction_policy: str = "none-v1"
    embedding_corpus_instruction_version: str = "v1"
    embedding_query_instruction_policy: str = "bge-search-prefix-v1"
    embedding_query_instruction_version: str = "v1"

    offline: bool = True
    worker_count: int = 1
    embedding_batch_size: int = 16

    parser_id: str = "docling"
    parser_version: str = "v1"
    parser_config_hash: str = ""
    chunker_version: str = "structure-v2"
    chunk_soft_token_target: int = 448
    chunk_hard_content_token_limit: int = 510
    chunk_overlap_tokens: int = 64
    lexical_index_version: str = "fts5-v1"
    lexical_tokenizer_settings: Mapping[str, Any] = field(default_factory=dict)
    index_version: str = "index-v1"
    fusion_version: str = "rrf-v1"
    reranker_version: str = "feature-v1"
    candidate_k: int = 100
    query_top_k: int = 10

    def __post_init__(self) -> None:
        source_root = _resolved_path(self.source_root, "source_root")
        if not source_root.is_dir():
            raise ValueError(f"source_root must be an existing directory: {source_root}")
        artifact_root = _resolved_path(self.artifact_root, "artifact_root")
        if _is_within(artifact_root, source_root):
            raise ValueError("artifact_root must be outside source_root")

        docling_path = (
            artifact_root / "docling"
            if self.docling_artifacts_path is None
            else _resolved_path(self.docling_artifacts_path, "docling_artifacts_path")
        )
        docling_path = _resolved_path(docling_path, "docling_artifacts_path")
        if _is_within(docling_path, source_root):
            raise ValueError("docling_artifacts_path must be outside source_root")

        if self.docling_offline is not True:
            raise ValueError("docling_offline must be True for the offline V1 parser")
        if self.docling_do_ocr is not False:
            raise ValueError("OCR is disabled in V1; docling_do_ocr must be False")
        if self.offline is not True:
            raise ValueError("offline must be True for the local-only V1 pipeline")

        if self.formula_artifacts_path is not None and self.formula_model_path is not None:
            first = _resolved_path(self.formula_artifacts_path, "formula_artifacts_path")
            second = _resolved_path(self.formula_model_path, "formula_model_path")
            if first != second:
                raise ValueError("formula_artifacts_path and formula_model_path must match")
            formula_path = first
        else:
            formula_path = self.formula_artifacts_path
            if formula_path is None:
                formula_path = self.formula_model_path
        if formula_path is not None:
            formula_path = _resolved_path(formula_path, "formula_artifacts_path")
            if _is_within(formula_path, source_root):
                raise ValueError("formula_artifacts_path must be outside source_root")
        if self.formula_enrichment_enabled:
            if formula_path is None:
                raise ValueError(
                    "formula enrichment requires an explicit local formula_artifacts_path"
                )
            if not formula_path.is_dir():
                raise ValueError(
                    f"formula_artifacts_path must be an existing local directory: {formula_path}"
                )

        embedding_path = None
        if self.embedding_model_path is not None:
            embedding_path = _resolved_path(self.embedding_model_path, "embedding_model_path")
            if _is_within(embedding_path, source_root):
                raise ValueError("embedding_model_path must be outside source_root")

        workers = _strict_int(self.worker_count, "worker_count")
        if workers < 1 or workers > 6:
            raise ValueError("worker_count must be between 1 and 6")
        batch_size = _strict_int(self.embedding_batch_size, "embedding_batch_size")
        if batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")

        dimensions = _strict_int(self.embedding_dimensions, "embedding_dimensions")
        if dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        model_limit = _strict_int(self.embedding_max_input_tokens, "embedding_max_input_tokens")
        special_budget = _strict_int(
            self.embedding_special_token_budget,
            "embedding_special_token_budget",
        )
        effective_limit = _strict_int(
            self.embedding_effective_content_token_limit,
            "embedding_effective_content_token_limit",
        )
        if model_limit <= 0:
            raise ValueError("embedding_max_input_tokens must be positive")
        if special_budget < 0:
            raise ValueError("embedding_special_token_budget must be non-negative")
        if effective_limit <= 0 or effective_limit > model_limit - special_budget:
            raise ValueError("embedding_effective_content_token_limit exceeds model capacity")
        if self.embedding_truncation is not False:
            raise ValueError("embedding_truncation must be False for the knowledge index")

        soft_target = _strict_int(self.chunk_soft_token_target, "chunk_soft_token_target")
        hard_limit = _strict_int(
            self.chunk_hard_content_token_limit,
            "chunk_hard_content_token_limit",
        )
        overlap = _strict_int(self.chunk_overlap_tokens, "chunk_overlap_tokens")
        if soft_target <= 0:
            raise ValueError("chunk_soft_token_target must be positive")
        if hard_limit <= 0:
            raise ValueError("chunk_hard_content_token_limit must be positive")
        if overlap < 0 or overlap > hard_limit:
            raise ValueError("chunk_overlap_tokens must fit the hard content limit")
        for name in (
            "embedding_model_id",
            "embedding_runtime",
            "embedding_normalization",
            "embedding_model_version",
            "parser_id",
            "parser_version",
            "chunker_version",
            "lexical_index_version",
            "index_version",
            "fusion_version",
            "reranker_version",
        ):
            _nonempty(getattr(self, name), name)
        candidate_k = _strict_int(self.candidate_k, "candidate_k")
        query_top_k = _strict_int(self.query_top_k, "query_top_k")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if query_top_k <= 0:
            raise ValueError("query_top_k must be positive")

        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "artifact_root", artifact_root)
        object.__setattr__(self, "docling_artifacts_path", docling_path)
        object.__setattr__(self, "formula_artifacts_path", formula_path)
        object.__setattr__(self, "formula_model_path", formula_path)
        object.__setattr__(self, "embedding_model_path", embedding_path)
        object.__setattr__(self, "worker_count", workers)
        object.__setattr__(self, "embedding_batch_size", batch_size)
        object.__setattr__(self, "embedding_dimensions", dimensions)
        object.__setattr__(self, "embedding_max_input_tokens", model_limit)
        object.__setattr__(self, "embedding_special_token_budget", special_budget)
        object.__setattr__(self, "embedding_effective_content_token_limit", effective_limit)
        object.__setattr__(self, "chunk_soft_token_target", soft_target)
        object.__setattr__(self, "chunk_hard_content_token_limit", hard_limit)
        object.__setattr__(self, "chunk_overlap_tokens", overlap)
        object.__setattr__(self, "candidate_k", candidate_k)
        object.__setattr__(self, "query_top_k", query_top_k)
        object.__setattr__(
            self,
            "lexical_tokenizer_settings",
            _freeze(self.lexical_tokenizer_settings or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result = self._base_dict()
        result["config_fingerprint"] = self.config_fingerprint
        result["component_fingerprints"] = self.component_fingerprints
        return result

    def _base_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            result[item.name] = _json_safe(getattr(self, item.name))
        return result

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def component_fingerprints(self) -> dict[str, str]:
        """Stable hashes used by later reuse checks for each pipeline stage."""

        payload = self._base_dict()
        groups = {
            "parser": {key: payload[key] for key in payload if key.startswith("docling_") or key.startswith("formula_") or key.startswith("parser_")},
            "chunker": {key: payload[key] for key in payload if key.startswith("chunk_")},
            "embedding": {key: payload[key] for key in payload if key.startswith("embedding_")},
            "index": {key: payload[key] for key in payload if key.startswith("lexical_") or key in {"index_version", "fusion_version", "reranker_version", "candidate_k", "query_top_k"}},
        }
        return {
            name: hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for name, value in groups.items()
        }

    @property
    def config_fingerprint(self) -> str:
        payload = self._base_dict()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Short alias used by catalog callers for the full config identity."""

        return self.config_fingerprint

    @property
    def embedding_spec(self):
        """Build the canonical embedding contract without importing optional adapters."""

        from .models import EmbeddingSpec

        return EmbeddingSpec(
            model_id=self.embedding_model_id,
            resolved_model_version=self.embedding_model_version,
            runtime=self.embedding_runtime,
            artifact_hash=self.embedding_artifact_hash,
            dimensions=self.embedding_dimensions,
            normalization_policy=self.embedding_normalization,
            tokenizer_fingerprint=self.embedding_tokenizer_fingerprint,
            model_max_input_tokens=self.embedding_max_input_tokens,
            special_token_budget=self.embedding_special_token_budget,
            effective_corpus_content_token_limit=self.embedding_effective_content_token_limit,
            corpus_instruction_policy=self.embedding_corpus_instruction_policy,
            corpus_instruction_version=self.embedding_corpus_instruction_version,
            query_instruction_policy=self.embedding_query_instruction_policy,
            query_instruction_version=self.embedding_query_instruction_version,
            truncation=self.embedding_truncation,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeConfig:
        payload = dict(value)
        payload.pop("config_fingerprint", None)
        payload.pop("component_fingerprints", None)
        valid_names = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in payload.items() if key in valid_names})


__all__ = ["KnowledgeConfig"]
