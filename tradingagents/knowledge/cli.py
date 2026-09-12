"""Explicit, local-only operator commands for the Phase 7 knowledge index.

This module intentionally owns a separate ``knowledge`` console entry point.
It neither imports nor extends the stock TradingAgents command surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .catalog import KnowledgeCatalog
from .config import KnowledgeConfig
from .discovery import SourceScanner
from .docling_parser import DoclingDocumentParser as DocumentParser
from .embeddings import FastEmbedProvider as EmbeddingProvider
from .ingestion import IngestionMode, KnowledgeIngestor
from .lexical_index import LexicalIndexReader, LexicalIndexWriter  # noqa: F401 - CLI test seam
from .models import ContentType, IngestionState, KnowledgeQuery
from .query import KnowledgeQueryService
from .vector_index import VectorIndexReader, VectorIndexWriter  # noqa: F401 - CLI test seam

_EXIT_OK = 0
_EXIT_FAILURE = 1
_EXIT_USAGE = 2
_CATALOG_NAME = "catalog.sqlite3"


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _json_value(value: Any) -> Any:
    """Convert public contract values to stable machine-readable JSON."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and type(value).__module__.startswith("enum"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _emit(value: Any, *, as_json: bool) -> None:
    payload = _json_value(value)
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    if isinstance(payload, list):
        if not payload:
            print("(none)")
            return
        for item in payload:
            print(json.dumps(item, sort_keys=True))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                print(f"{str(key).upper()}: {json.dumps(value, sort_keys=True)}")
            else:
                print(f"{str(key).upper()}: {value}")
        return
    print(payload)


def _default_artifact_root() -> Path:
    # Constructing a config is unnecessary (and invalid without its source
    # directory) for metadata-only commands.
    return Path(__file__).resolve().parents[2] / "data_cache" / "knowledge"


def _artifact_root(value: str | None) -> Path:
    return _path(value) if value is not None else _path(_default_artifact_root())


def _open_catalog(artifact_root: Path, *, allow_missing: bool) -> KnowledgeCatalog | None:
    """Open an existing catalog without creating metadata-command artifacts."""

    path = artifact_root / _CATALOG_NAME
    if not path.is_file() and allow_missing:
        return None
    return KnowledgeCatalog(path)


def _all_state_counts(resources: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(resource.state.value) for resource in resources)
    return {state.value: int(counts.get(state.value, 0)) for state in IngestionState}


def _metadata_status(catalog: KnowledgeCatalog | None) -> dict[str, Any]:
    if catalog is None:
        return {
            "artifact_root_initialized": False,
            "resource_counts": _all_state_counts(()),
            "active_generation": None,
            "latest_run": None,
        }
    resources = catalog.list_resources()
    runs = catalog.list_runs()
    generation = catalog.active_generation()
    return {
        "artifact_root_initialized": True,
        "resource_counts": _all_state_counts(resources),
        "active_generation": generation,
        "latest_run": runs[-1] if runs else None,
    }


def _metadata_command(args: argparse.Namespace) -> int:
    artifact_root = _artifact_root(args.artifact_root)
    catalog = _open_catalog(artifact_root, allow_missing=True)
    if args.command == "status":
        _emit(_metadata_status(catalog), as_json=args.json)
        return _EXIT_OK
    if args.command == "list":
        state = IngestionState(args.state) if args.state else None
        resources = () if catalog is None else catalog.list_resources(state)
        _emit(resources, as_json=args.json)
        return _EXIT_OK
    if args.command == "document":
        document = None if catalog is None else catalog.get_document(args.document_id)
        chunks = () if document is None or catalog is None else catalog.chunks_for_document(args.document_id)
        aliases = ()
        if document is not None and catalog is not None:
            aliases = tuple(
                {"resource": resource, "alias": alias}
                for resource in catalog.list_resources()
                if (alias := catalog.get_alias(resource.resource_id)) is not None
                and alias.document_id == document.document_id
            )
        _emit({"document": document, "aliases": aliases, "chunks": chunks}, as_json=args.json)
        return _EXIT_OK
    if args.command == "quarantine":
        records = () if catalog is None else catalog.list_quarantine()
        _emit(records, as_json=args.json)
        return _EXIT_OK
    raise ValueError(f"unsupported metadata command: {args.command}")


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either local path contains the other."""

    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _safe_query_source_root(artifact_root: Path, model_path: Path | None) -> Path:
    """Find an existing, disjoint path solely to satisfy config validation.

    Search never scans this directory.  The selection avoids creating a
    directory or touching source files while still retaining KnowledgeConfig's
    source/artifact separation invariant for the local provider adapter.
    """

    candidates = (Path.home(), Path.cwd(), Path(os.getenv("TEMP", "")))
    for candidate in candidates:
        try:
            candidate = candidate.resolve(strict=True)
            if not _paths_overlap(candidate, artifact_root) and (
                model_path is None or not _paths_overlap(candidate, model_path)
            ):
                return candidate
        except (OSError, RuntimeError):
            continue
    raise ValueError(
        "cannot find a local path disjoint from artifact_root and embedding_model_path "
        "for read-only query configuration"
    )


def _query_config(artifact_root: Path, embedding_spec: Any) -> KnowledgeConfig:
    """Build a read-only local provider config from the active exact spec."""

    model_path = os.getenv("KNOWLEDGE_EMBEDDING_MODEL_PATH")
    if not model_path:
        candidate = artifact_root / "models" / "embedding"
        model_path = str(candidate) if candidate.is_dir() else None
    resolved_model_path = _path(model_path) if model_path is not None else None
    return KnowledgeConfig(
        source_root=_safe_query_source_root(artifact_root, resolved_model_path),
        artifact_root=artifact_root,
        embedding_model_id=embedding_spec.model_id,
        embedding_model_path=resolved_model_path,
        embedding_dimensions=embedding_spec.dimensions,
        embedding_runtime=embedding_spec.runtime,
        embedding_normalization=embedding_spec.normalization_policy,
        embedding_model_version=embedding_spec.resolved_model_version,
        embedding_artifact_hash=embedding_spec.artifact_hash,
        embedding_tokenizer_fingerprint=embedding_spec.tokenizer_fingerprint,
        embedding_max_input_tokens=embedding_spec.model_max_input_tokens,
        embedding_special_token_budget=embedding_spec.special_token_budget,
        embedding_effective_content_token_limit=embedding_spec.effective_corpus_content_token_limit,
        embedding_truncation=embedding_spec.truncation,
        embedding_corpus_instruction_policy=embedding_spec.corpus_instruction_policy,
        embedding_corpus_instruction_version=embedding_spec.corpus_instruction_version,
        embedding_query_instruction_policy=embedding_spec.query_instruction_policy,
        embedding_query_instruction_version=embedding_spec.query_instruction_version,
    )


def _search(args: argparse.Namespace) -> int:
    artifact_root = _artifact_root(args.artifact_root)
    catalog = _open_catalog(artifact_root, allow_missing=False)
    assert catalog is not None
    generation = catalog.active_generation()
    if generation is None:
        raise ValueError("no active matched knowledge generation; run 'knowledge index' explicitly")

    # No parser, scanner, ingestor, or writer is constructed here.  The
    # service compares this provider's complete spec before any dense read.
    embedder = EmbeddingProvider.from_config(_query_config(artifact_root, generation.embedding_spec))
    vector_reader = VectorIndexReader(generation.vector_location)
    lexical_reader = LexicalIndexReader(generation.lexical_location)
    service = KnowledgeQueryService(vector_reader, lexical_reader, catalog, embedder)
    content_types = (ContentType(args.content_type),) if args.content_type else ()
    request = KnowledgeQuery(
        text=args.text,
        top_k=args.top_k,
        content_types=content_types,
        document_ids=(args.document_id,) if args.document_id else (),
    )
    _emit(service.search(request), as_json=args.json)
    return _EXIT_OK


def _ingestion_config(args: argparse.Namespace) -> KnowledgeConfig:
    values: dict[str, Any] = {"artifact_root": _artifact_root(args.artifact_root)}
    if args.source_root is not None:
        values["source_root"] = _path(args.source_root)
    if args.docling_artifacts_path is not None:
        values["docling_artifacts_path"] = _path(args.docling_artifacts_path)
    model_path = getattr(args, "embedding_model_path", None) or os.getenv(
        "KNOWLEDGE_EMBEDDING_MODEL_PATH"
    )
    if model_path is None:
        candidate = values["artifact_root"] / "models" / "embedding"
        if candidate.is_dir():
            model_path = str(candidate)
    if model_path is not None:
        values["embedding_model_path"] = _path(model_path)
    return KnowledgeConfig(**values)


def _index(args: argparse.Namespace) -> int:
    config = _ingestion_config(args)
    # Component construction is explicit and only occurs for the two mutating
    # commands.  Sources are subsequently opened by the scanner/parser as
    # read-only inputs; all generated files remain below artifact_root.
    embedder = EmbeddingProvider.from_config(config)
    ingestor = KnowledgeIngestor(
        config,
        scanner=SourceScanner(config),
        parser=DocumentParser(config),
        embedder=embedder,
    )
    mode = IngestionMode.REBUILD if args.command == "rebuild" else IngestionMode.INCREMENTAL
    summary = ingestor.run(mode)
    _emit(summary, as_json=args.json)
    return _EXIT_OK if str(summary.state) == "SUCCEEDED" else _EXIT_FAILURE


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _add_artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", metavar="PATH", help="knowledge artifact directory")


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge",
        description="Explicit local-only HFT knowledge indexing and retrieval.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "rebuild"):
        command = commands.add_parser(name, help="explicitly build a local knowledge index")
        command.add_argument("--source-root", metavar="PATH", help="read-only document source directory")
        _add_artifact_argument(command)
        command.add_argument("--docling-artifacts-path", metavar="PATH", help="prefetched local Docling assets")
        command.add_argument(
            "--embedding-model-path",
            metavar="PATH",
            help="prefetched local FastEmbed/ONNX model (or KNOWLEDGE_EMBEDDING_MODEL_PATH)",
        )
        _add_json_argument(command)
    status = commands.add_parser("status", help="show catalog status without parsing or indexing")
    _add_artifact_argument(status)
    _add_json_argument(status)
    listing = commands.add_parser("list", help="list catalog resources without parsing or indexing")
    _add_artifact_argument(listing)
    listing.add_argument("--state", choices=[state.value for state in IngestionState], help="catalog state")
    _add_json_argument(listing)
    search = commands.add_parser("search", help="search the active local generation; never indexes")
    search.add_argument("text", metavar="TEXT", help="query text")
    _add_artifact_argument(search)
    search.add_argument("--top-k", type=_positive_int, default=10, help="maximum result count (default: 10)")
    search.add_argument("--content-type", choices=[item.value for item in ContentType], help="filter one content type")
    search.add_argument("--document-id", metavar="ID", help="filter one document")
    _add_json_argument(search)
    document = commands.add_parser("document", help="show document/chunk provenance from the catalog")
    document.add_argument("document_id", metavar="DOCUMENT_ID")
    _add_artifact_argument(document)
    _add_json_argument(document)
    quarantine = commands.add_parser("quarantine", help="show bounded catalog diagnostics")
    _add_artifact_argument(quarantine)
    _add_json_argument(quarantine)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone knowledge command and return a process exit code."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"status", "list", "document", "quarantine"}:
            return _metadata_command(args)
        if args.command == "search":
            return _search(args)
        return _index(args)
    except Exception as exc:  # CLI boundary: retain a concise, nonzero error code.
        print(f"KNOWLEDGE ERROR: {exc}", file=sys.stderr)
        return _EXIT_USAGE if isinstance(exc, ValueError) else _EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
