"""Local-only embedding contracts and versioned document-vector artifacts.

This boundary deliberately knows nothing about ingestion, indexes, queries, or
trading.  It accepts already-authoritative :class:`ChunkRecord` values and
ensures that a local ONNX adapter never silently changes their text.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Literal, Protocol

from .config import KnowledgeConfig
from .models import ChunkRecord, EmbeddingSpec


BGE_SEARCH_PREFIX = "Represent this sentence for searching relevant passages: "
_CACHE_SCHEMA_VERSION = "embedding-artifact-v1"


class EmbeddingContractError(RuntimeError):
    """A provider returned vectors or tokenizer data outside the V1 contract."""


class LocalModelUnavailable(RuntimeError):
    """The explicitly configured local embedding model cannot be used offline."""


class EmbeddingInputTooLong(EmbeddingContractError):
    """An exact corpus or query input would exceed the loaded model context."""


class EmbeddingVersionMismatch(EmbeddingContractError):
    """A stored embedding artifact cannot be reused for the active specification."""


class EmbeddingSpecMismatch(EmbeddingContractError):
    """Two complete embedding specifications describe different vector spaces."""


class EmbeddingTokenizer(Protocol):
    """Special-token-aware tokenizer used before every inference call."""

    model_max_input_tokens: int
    special_token_budget: int

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        purpose: str,
        truncation: bool,
    ) -> Sequence[Any] | dict[str, Sequence[Any]]:
        ...


def _token_sequence(encoded: Sequence[Any] | dict[str, Sequence[Any]] | Any) -> Sequence[Any]:
    """Normalize common HuggingFace/tokenizers return shapes without estimating."""

    if hasattr(encoded, "ids"):
        return encoded.ids
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", ())
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, Sequence)
        and encoded
        and not isinstance(encoded, (str, bytes))
        and isinstance(encoded[0], Sequence)
        and not isinstance(encoded[0], (str, bytes))
    ):
        return encoded[0]
    if isinstance(encoded, Sequence):
        return encoded
    raise EmbeddingContractError("configured tokenizer did not return a token sequence")


class EmbeddingProvider:
    """Validation wrapper around a concrete local vector model.

    Subclasses implement only ``_embed_formatted``.  The wrapper performs both
    pre-inference token checks and post-inference vector checks, which keeps
    fake and production providers subject to the same no-truncation contract.
    """

    spec: EmbeddingSpec
    tokenizer: EmbeddingTokenizer
    batch_size: int = 16

    def __init__(
        self,
        *,
        spec: EmbeddingSpec,
        tokenizer: EmbeddingTokenizer,
        batch_size: int = 16,
    ) -> None:
        if not isinstance(spec, EmbeddingSpec):
            raise TypeError("spec must be an EmbeddingSpec")
        if tokenizer is None:
            raise TypeError("tokenizer is required")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        declared_limit = getattr(tokenizer, "model_max_input_tokens", spec.model_max_input_tokens)
        if int(declared_limit) != spec.model_max_input_tokens:
            raise EmbeddingContractError("tokenizer model limit does not match embedding specification")
        self.spec = spec
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: Literal["corpus", "query"] = "corpus",
    ) -> tuple[tuple[float, ...], ...]:
        if purpose not in {"corpus", "query"}:
            raise ValueError("purpose must be 'corpus' or 'query'")
        supplied = tuple(texts)
        if any(not isinstance(text, str) for text in supplied):
            raise TypeError("embedding texts must be strings")
        if not supplied:
            return ()

        formatted = tuple(self._format(text, purpose=purpose) for text in supplied)
        for text in formatted:
            self._validate_input(text, purpose=purpose)

        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(formatted), self.batch_size):
            batch = formatted[start : start + self.batch_size]
            returned = tuple(self._embed_formatted(batch))
            if len(returned) != len(batch):
                raise EmbeddingContractError("provider returned a different number of vectors than inputs")
            vectors.extend(self._validate_vector(vector) for vector in returned)
        return tuple(vectors)

    def _format(self, text: str, *, purpose: Literal["corpus", "query"]) -> str:
        if purpose == "corpus":
            if self.spec.corpus_instruction_policy != "none-v1":
                raise EmbeddingContractError(
                    f"unsupported corpus instruction policy: {self.spec.corpus_instruction_policy}"
                )
            return text
        if self.spec.query_instruction_policy != "bge-search-prefix-v1":
            raise EmbeddingContractError(
                f"unsupported query instruction policy: {self.spec.query_instruction_policy}"
            )
        return BGE_SEARCH_PREFIX + text

    def _validate_input(self, text: str, *, purpose: Literal["corpus", "query"]) -> None:
        # Both calls intentionally use truncation=False.  The first preserves
        # the content-budget proof and the second is the exact model input.
        content = _token_sequence(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                purpose=purpose,
                truncation=False,
            )
        )
        model_input = _token_sequence(
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                purpose=purpose,
                truncation=False,
            )
        )
        if purpose == "corpus" and len(content) > self.spec.effective_corpus_content_token_limit:
            raise EmbeddingInputTooLong(
                "corpus input exceeds effective content token limit "
                f"({len(content)} > {self.spec.effective_corpus_content_token_limit})"
            )
        if len(model_input) > self.spec.model_max_input_tokens:
            raise EmbeddingInputTooLong(
                "input exceeds model token limit "
                f"({len(model_input)} > {self.spec.model_max_input_tokens})"
            )

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        try:
            normalized = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise EmbeddingContractError("provider returned a non-numeric vector") from exc
        if len(normalized) != self.spec.dimensions:
            raise EmbeddingContractError(
                f"provider vector dimensions {len(normalized)} do not match {self.spec.dimensions}"
            )
        if not all(math.isfinite(value) for value in normalized):
            raise EmbeddingContractError("provider vector values must be finite")
        return normalized

    def _embed_formatted(self, texts: tuple[str, ...]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


@dataclass(slots=True)
class _FastEmbedTokenizer:
    """Adapter over FastEmbed's resolved local tokenizer object."""

    raw: Any
    model_max_input_tokens: int
    special_token_budget: int

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        purpose: str,
        truncation: bool,
    ) -> Sequence[Any] | dict[str, Sequence[Any]]:
        if truncation:
            raise EmbeddingContractError("knowledge embeddings require truncation=False")
        try:
            return self.raw.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
        except TypeError:
            # The Rust tokenizers API has no truncation keyword.  It is safe
            # only because this adapter refuses a requested truncation above.
            try:
                return self.raw.encode(text, add_special_tokens=add_special_tokens)
            except TypeError:
                result = self.raw(
                    text,
                    add_special_tokens=add_special_tokens,
                    truncation=False,
                )
                return result


class FastEmbedProvider(EmbeddingProvider):
    """CPU-only FastEmbed adapter which accepts already-provisioned artifacts."""

    def __init__(
        self,
        *,
        spec: EmbeddingSpec,
        tokenizer: EmbeddingTokenizer,
        embedder: Any,
        batch_size: int,
    ) -> None:
        super().__init__(spec=spec, tokenizer=tokenizer, batch_size=batch_size)
        self._embedder = embedder

    @classmethod
    def from_config(cls, config: KnowledgeConfig) -> FastEmbedProvider:
        """Open an existing local artifact without attempting a model download."""

        model_path = config.embedding_model_path
        if model_path is None or not model_path.is_dir() or not any(model_path.iterdir()):
            raise LocalModelUnavailable(
                "embedding_model_path must reference a non-empty pre-provisioned local model"
            )
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise LocalModelUnavailable(
                "FastEmbed is not installed; install tradingagents[knowledge] with local artifacts"
            ) from exc

        constructor = inspect.signature(TextEmbedding)
        parameters = constructor.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "local_files_only" not in parameters and not accepts_kwargs:
            raise LocalModelUnavailable("installed FastEmbed cannot enforce local_files_only")
        if "specific_model_path" not in parameters and not accepts_kwargs:
            raise LocalModelUnavailable(
                "installed FastEmbed cannot bind the configured specific_model_path"
            )
        kwargs: dict[str, Any] = {
            "model_name": config.embedding_model_id,
            "specific_model_path": str(model_path),
            "cache_dir": str(model_path.parent),
            "threads": config.worker_count,
            "providers": ["CPUExecutionProvider"],
            "local_files_only": True,
        }
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        try:
            embedder = TextEmbedding(**kwargs)
        except Exception as exc:  # adapter errors must never trigger a fallback/download path
            raise LocalModelUnavailable(f"could not open local FastEmbed model: {exc}") from exc

        raw_tokenizer = _resolve_tokenizer(embedder)
        model_limit = _resolve_model_limit(embedder, raw_tokenizer, config.embedding_max_input_tokens)
        tokenizer = _FastEmbedTokenizer(
            raw=raw_tokenizer,
            model_max_input_tokens=model_limit,
            special_token_budget=_resolved_special_token_budget(raw_tokenizer),
        )
        dimensions = _resolve_dimensions(embedder, config.embedding_dimensions)
        artifact_hash = config.embedding_artifact_hash or _directory_hash(model_path)
        version = _resolved_version(config)
        spec = EmbeddingSpec(
            model_id=config.embedding_model_id,
            resolved_model_version=version,
            runtime=config.embedding_runtime,
            artifact_hash=artifact_hash,
            dimensions=dimensions,
            normalization_policy=config.embedding_normalization,
            tokenizer_fingerprint=config.embedding_tokenizer_fingerprint
            or _tokenizer_asset_fingerprint(model_path),
            model_max_input_tokens=model_limit,
            special_token_budget=tokenizer.special_token_budget,
            effective_corpus_content_token_limit=(
                model_limit - tokenizer.special_token_budget
            ),
            corpus_instruction_policy=config.embedding_corpus_instruction_policy,
            corpus_instruction_version=config.embedding_corpus_instruction_version,
            query_instruction_policy=config.embedding_query_instruction_policy,
            query_instruction_version=config.embedding_query_instruction_version,
            truncation=False,
        )
        if spec.model_id == "BAAI/bge-small-en-v1.5" and (
            spec.model_max_input_tokens != 512
            or spec.special_token_budget != 2
            or spec.effective_corpus_content_token_limit != 510
        ):
            raise EmbeddingContractError("BGE-small V1 must resolve to 512 model / 510 corpus tokens")
        return cls(
            spec=spec,
            tokenizer=tokenizer,
            embedder=embedder,
            batch_size=config.embedding_batch_size,
        )

    def _embed_formatted(self, texts: tuple[str, ...]) -> Sequence[Sequence[float]]:
        try:
            return tuple(self._embedder.embed(list(texts), batch_size=self.batch_size))
        except Exception as exc:
            raise EmbeddingContractError(f"local FastEmbed inference failed: {exc}") from exc


def _resolve_tokenizer(embedder: Any) -> Any:
    for owner in (embedder, getattr(embedder, "model", None)):
        tokenizer = getattr(owner, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    raise LocalModelUnavailable("local FastEmbed model did not expose its resolved tokenizer")


def _resolve_model_limit(embedder: Any, tokenizer: Any, fallback: int) -> int:
    for owner in (tokenizer, getattr(embedder, "model", None), embedder):
        for name in ("model_max_length", "max_length", "max_input_tokens"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and 0 < value <= 100_000:
                return value
    return int(fallback)


def _resolved_special_token_budget(tokenizer: Any) -> int:
    adapter = _FastEmbedTokenizer(raw=tokenizer, model_max_input_tokens=1, special_token_budget=0)
    plain = len(_token_sequence(adapter.encode("", add_special_tokens=False, purpose="corpus", truncation=False)))
    with_special = len(_token_sequence(adapter.encode("", add_special_tokens=True, purpose="corpus", truncation=False)))
    budget = with_special - plain
    if budget < 0:
        raise EmbeddingContractError("tokenizer special-token budget cannot be negative")
    return budget


def _resolve_dimensions(embedder: Any, fallback: int) -> int:
    for owner in (getattr(embedder, "model", None), embedder):
        for name in ("embedding_size", "dimensions", "dimension"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return int(fallback)


def _resolved_version(config: KnowledgeConfig) -> str:
    if config.embedding_model_version != "unknown":
        return config.embedding_model_version
    try:
        from importlib.metadata import version

        return "fastembed-" + version("fastembed")
    except Exception:
        return "fastembed-unknown"


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        digest.update(entry.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tokenizer_asset_fingerprint(model_path: Path) -> str:
    """Hash resolved tokenizer/config bytes, not only their filenames.

    FastEmbed's ONNX artifact can contain weights unrelated to tokenization.  A
    dedicated tokenizer/config digest records the files that change how text
    becomes input IDs, while ``artifact_hash`` retains full-model identity.
    """

    names = {
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
    assets = sorted(
        (
            entry
            for entry in model_path.rglob("*")
            if entry.is_file()
            and (
                entry.name.lower() in names
                or "tokenizer" in entry.name.lower()
                or "vocab" in entry.name.lower()
            )
        ),
        key=lambda item: item.as_posix(),
    )
    if not assets:
        raise LocalModelUnavailable("local model has no tokenizer/config assets to fingerprint")
    digest = hashlib.sha256()
    for asset in assets:
        digest.update(asset.relative_to(model_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with asset.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


class EmbeddingArtifactStore:
    """One exact document-vector artifact per model/version/specification."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_or_compute(
        self,
        chunks: Sequence[ChunkRecord],
        provider: EmbeddingProvider,
    ) -> tuple[tuple[float, ...], ...]:
        if not chunks:
            return ()
        grouped: OrderedDict[str, list[ChunkRecord]] = OrderedDict()
        for chunk in chunks:
            grouped.setdefault(chunk.document_id, []).append(chunk)
        vectors_by_chunk: dict[str, tuple[float, ...]] = {}
        for document_chunks in grouped.values():
            try:
                vectors = self.load(document_chunks, provider)
            except (FileNotFoundError, EmbeddingVersionMismatch):
                vectors = provider.embed(tuple(chunk.text for chunk in document_chunks), purpose="corpus")
                self._write(document_chunks, provider.spec, vectors)
                vectors = self.load(document_chunks, provider)
            vectors_by_chunk.update(
                {chunk.chunk_id: vector for chunk, vector in zip(document_chunks, vectors, strict=True)}
            )
        return tuple(vectors_by_chunk[chunk.chunk_id] for chunk in chunks)

    def load(
        self,
        chunks: Sequence[ChunkRecord],
        provider: EmbeddingProvider,
    ) -> tuple[tuple[float, ...], ...]:
        if not chunks:
            return ()
        self._validate_document_chunks(chunks)
        vector_path, sidecar_path = self._paths(chunks[0].document_id, provider.spec)
        if not vector_path.is_file() or not sidecar_path.is_file():
            if self._has_incompatible_document_artifact(chunks[0].document_id, provider.spec):
                raise EmbeddingVersionMismatch("embedding specification/cache mismatch")
            raise FileNotFoundError(vector_path)
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingVersionMismatch("embedding artifact sidecar is unreadable") from exc
        expected = self._metadata(chunks, provider.spec)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise EmbeddingVersionMismatch(f"embedding specification/cache mismatch: {key}")
        try:
            import numpy as np

            vectors = np.load(vector_path, allow_pickle=False)
        except Exception as exc:
            raise EmbeddingVersionMismatch("embedding artifact vectors are unreadable") from exc
        if vectors.ndim != 2 or vectors.shape != (len(chunks), provider.spec.dimensions):
            raise EmbeddingVersionMismatch("embedding artifact dimensions do not match specification")
        result = tuple(tuple(float(value) for value in row) for row in vectors.tolist())
        for vector in result:
            provider._validate_vector(vector)
        return result

    def _write(
        self,
        chunks: Sequence[ChunkRecord],
        spec: EmbeddingSpec,
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self._validate_document_chunks(chunks)
        if len(vectors) != len(chunks):
            raise EmbeddingContractError("cannot cache a different number of vectors than chunks")
        vector_path, sidecar_path = self._paths(chunks[0].document_id, spec)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np
        except ImportError as exc:
            raise LocalModelUnavailable("numpy is required by the local embedding artifact format") from exc
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(chunks), spec.dimensions) or not np.isfinite(array).all():
            raise EmbeddingContractError("cached vectors violate the embedding dimension or finite-value contract")
        self._atomic_npy_write(vector_path, array, np)
        self._atomic_json_write(sidecar_path, self._metadata(chunks, spec))

    def _paths(self, document_id: str, spec: EmbeddingSpec) -> tuple[Path, Path]:
        model_parts = _safe_model_parts(spec.model_id)
        version = _safe_path_component(spec.resolved_model_version, "resolved model version")
        document = _safe_path_component(document_id, "document id")
        directory = self.root.joinpath(*model_parts, version)
        fingerprint = _embedding_spec_fingerprint(spec)
        return (
            directory / f"{document}.{fingerprint}.npy",
            directory / f"{document}.{fingerprint}.json",
        )

    def _has_incompatible_document_artifact(self, document_id: str, spec: EmbeddingSpec) -> bool:
        """Distinguish a first generation from an incompatible reuse attempt."""

        _, expected_sidecar = self._paths(document_id, spec)
        document = _safe_path_component(document_id, "document id")
        for candidate in expected_sidecar.parent.glob(f"{document}.*.json"):
            if candidate != expected_sidecar:
                return True
        return False

    @staticmethod
    def _metadata(chunks: Sequence[ChunkRecord], spec: EmbeddingSpec) -> dict[str, Any]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "document_id": chunks[0].document_id,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "source_hashes": [chunk.source_hash for chunk in chunks],
            "chunk_content_hashes": [_chunk_content_hash(chunk) for chunk in chunks],
            "embedding_spec": spec.to_dict(),
            "embedding_spec_fingerprint": _embedding_spec_fingerprint(spec),
            "embedding_model_version": spec.resolved_model_version,
            "embedding_artifact_hash": spec.artifact_hash,
            "dimensions": spec.dimensions,
        }

    @staticmethod
    def _validate_document_chunks(chunks: Sequence[ChunkRecord]) -> None:
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1 or not next(iter(document_ids), ""):
            raise ValueError("one embedding artifact must contain chunks from exactly one document")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if not all(chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("embedding artifact chunk IDs must be non-empty and unique")

    @staticmethod
    def _atomic_npy_write(path: Path, array: Any, np: Any) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)


def _safe_model_parts(model_id: str) -> tuple[str, ...]:
    raw = str(model_id).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("embedding model id must be a relative model identifier")
    return tuple(_safe_path_component(part, "embedding model id") for part in path.parts)


def _safe_path_component(value: str, name: str) -> str:
    result = str(value).strip()
    if not result or result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"{name} must be one safe path component")
    return result


def _chunk_content_hash(chunk: ChunkRecord) -> str:
    if chunk.content_hash:
        return chunk.content_hash
    return "sha256:" + hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def _embedding_spec_fingerprint(spec: EmbeddingSpec) -> str:
    """Stable complete-spec identity used as part of the cache storage key."""

    return hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest()


__all__ = [
    "BGE_SEARCH_PREFIX",
    "EmbeddingArtifactStore",
    "EmbeddingContractError",
    "EmbeddingInputTooLong",
    "EmbeddingProvider",
    "EmbeddingSpecMismatch",
    "EmbeddingTokenizer",
    "EmbeddingVersionMismatch",
    "FastEmbedProvider",
    "LocalModelUnavailable",
]
