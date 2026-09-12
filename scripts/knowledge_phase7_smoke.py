"""Bounded, real, offline-only Phase 7 parser/index/query closure smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DOCLING_MANIFEST = "docling-artifacts.json"
_EMBEDDING_MANIFEST = "embedding-artifacts.json"


class NetworkAttempt(RuntimeError):
    """A smoke dependency tried to open a network connection."""


class _OfflineNetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._allow_socketpair = False
        self._original_socketpair = socket.socketpair
        self._original_connect = socket.socket.connect
        self._original_create_connection = socket.create_connection

    def _blocked(self, address: Any, *args: Any, **kwargs: Any) -> None:
        self.attempts.append(str(address))
        raise NetworkAttempt(f"offline smoke blocked network attempt: {address}")

    def _socketpair(self, *args: Any, **kwargs: Any) -> Any:
        self._allow_socketpair = True
        try:
            return self._original_socketpair(*args, **kwargs)
        finally:
            self._allow_socketpair = False

    def __enter__(self) -> _OfflineNetworkGuard:
        def blocked_socket(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
            if (
                self._allow_socketpair
                and isinstance(address, tuple)
                and len(address) >= 2
                and address[0] in {"127.0.0.1", "::1"}
            ):
                # Windows' asyncio ProactorEventLoop implements socketpair()
                # via a temporary loopback listener. This is an in-process
                # transport, not external network access, and must remain
                # usable by LanceDB.
                self._original_connect(sock, address, *args, **kwargs)
                return
            self._blocked(address, *args, **kwargs)

        socket.socket.connect = blocked_socket  # type: ignore[method-assign]
        socket.socketpair = self._socketpair  # type: ignore[assignment]
        socket.create_connection = self._blocked  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        socket.socket.connect = self._original_connect  # type: ignore[method-assign]
        socket.socketpair = self._original_socketpair  # type: ignore[assignment]
        socket.create_connection = self._original_create_connection  # type: ignore[assignment]


class _SelectedScanner:
    def __init__(self, selected: tuple[Any, ...]) -> None:
        self.selected = selected

    def discover(self) -> tuple[Any, ...]:
        return self.selected


def _snapshot(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": metadata.st_size,
        "modified_ns": metadata.st_mtime_ns,
        "mode": metadata.st_mode,
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _require_local_artifacts(docling_path: Path, embedding_path: Path) -> None:
    if not docling_path.is_dir():
        raise RuntimeError(f"missing local Docling artifact directory: {docling_path}")
    if not embedding_path.is_dir():
        raise RuntimeError(f"missing local embedding model directory: {embedding_path}")
    _load_json(docling_path / _DOCLING_MANIFEST, "Docling artifact manifest")
    _load_json(embedding_path / _EMBEDDING_MANIFEST, "embedding artifact manifest")
    if not any(embedding_path.rglob("*.onnx")):
        raise RuntimeError("local embedding artifact directory has no ONNX model")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_paths(source_root: Path, artifact_root: Path, report_path: Path) -> None:
    if _is_within(artifact_root, source_root):
        raise ValueError("smoke artifact_root must be outside source_root")
    if _is_within(report_path, source_root):
        raise ValueError("report_path must be outside source_root")


def _select_resources(resources: tuple[Any, ...], *, limit: int) -> tuple[Any, ...]:
    """Select only source-hashed PDF/EPUB records for the bounded real run."""

    return tuple(
        resource
        for resource in resources
        if getattr(getattr(resource, "state", None), "value", None) == "HASHED"
        and resource.path.suffix.casefold() in {".pdf", ".epub"}
    )[:limit]


def _ocr_policy_report(
    configured_do_ocr: bool,
    selected: tuple[Any, ...],
    observed_pdf_options: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Prove OCR is disabled even when the bounded set contains EPUBs only."""

    if configured_do_ocr is not False:
        raise RuntimeError("Docling OCR policy must be configured false")
    pdf_count = sum(resource.path.suffix.casefold() == ".pdf" for resource in selected)
    if pdf_count and len(observed_pdf_options) != pdf_count:
        raise RuntimeError("Docling PDF option observations do not cover every selected PDF")
    if any(option.get("do_ocr") is not False for option in observed_pdf_options):
        raise RuntimeError("Docling PDF options did not prove do_ocr=False")
    return {
        "configured_do_ocr": configured_do_ocr,
        "pdf_resource_count": pdf_count,
        "observed_pdf_option_count": len(observed_pdf_options),
    }


def _first_query(chunks: tuple[Any, ...]) -> str:
    for chunk in chunks:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", chunk.text)
        if candidates:
            return candidates[0]
    raise RuntimeError("indexed chunks have no lexical query term")


def _hit_fields(hit: Any) -> dict[str, Any]:
    return {
        key: getattr(hit, key)
        for key in (
            "chunk_id", "document_id", "source_hash", "source_filename", "source_relative_path",
            "content_type", "page", "page_start", "page_end", "epub_spine_item", "anchor",
            "section_path", "parser_version", "chunker_version", "index_version",
            "score", "semantic_score", "lexical_score", "fused_score", "rerank_score",
        )
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mandatory bounded Phase 7 offline real smoke.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--docling-artifacts-path", required=True)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--resource-limit", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="required; smoke refuses any network access")
    parser.add_argument("--report-path", help="optional JSON report path outside the source tree")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline or os.environ.get("KNOWLEDGE_OFFLINE") != "1":
        print("SMOKE ERROR: --offline and KNOWLEDGE_OFFLINE=1 are both required", file=sys.stderr)
        return 2
    if not 1 <= args.resource_limit <= 3:
        print("SMOKE ERROR: --resource-limit must be between 1 and 3", file=sys.stderr)
        return 2
    source_root = Path(args.source_root).expanduser().resolve(strict=False)
    artifact_root = Path(args.artifact_root).expanduser().resolve(strict=False)
    docling_path = Path(args.docling_artifacts_path).expanduser().resolve(strict=False)
    embedding_path = Path(args.embedding_model_path).expanduser().resolve(strict=False)
    if not source_root.is_dir():
        print(f"SMOKE ERROR: source root does not exist: {source_root}", file=sys.stderr)
        return 2
    try:
        report_path = (
            Path(args.report_path).expanduser().resolve(strict=False)
            if args.report_path
            else artifact_root / "smoke-report.json"
        )
        _validate_paths(source_root, artifact_root, report_path)
        _require_local_artifacts(docling_path, embedding_path)
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tradingagents.knowledge.catalog import KnowledgeCatalog
        from tradingagents.knowledge.config import KnowledgeConfig
        from tradingagents.knowledge.discovery import SourceScanner
        from tradingagents.knowledge.docling_parser import DoclingDocumentParser
        from tradingagents.knowledge.embeddings import FastEmbedProvider
        from tradingagents.knowledge.ingestion import IngestionMode, KnowledgeIngestor
        from tradingagents.knowledge.lexical_index import LexicalIndexReader
        from tradingagents.knowledge.models import KnowledgeQuery
        from tradingagents.knowledge.query import KnowledgeQueryService
        from tradingagents.knowledge.vector_index import VectorIndexReader

        config = KnowledgeConfig(
            source_root=source_root,
            artifact_root=artifact_root,
            docling_artifacts_path=docling_path,
            embedding_model_path=embedding_path,
            worker_count=1,
            embedding_batch_size=16,
        )
        all_resources = SourceScanner(config).discover()
        selected = _select_resources(all_resources, limit=args.resource_limit)
        if not selected:
            raise RuntimeError("no supported PDF/EPUB resources found under the approved source root")
        before = tuple(_snapshot(resource.path) for resource in selected)
        parser = DoclingDocumentParser(config)
        observed_pdf_options: list[dict[str, Any]] = []
        original_pdf_options = parser.build_pdf_pipeline_options

        def observed_pdf_options_factory(resource_id: str | None = None) -> Any:
            options = original_pdf_options(resource_id)
            observed_pdf_options.append({"resource_id": resource_id, "do_ocr": getattr(options, "do_ocr", None)})
            return options

        parser.build_pdf_pipeline_options = observed_pdf_options_factory  # type: ignore[method-assign]
        with _OfflineNetworkGuard() as network:
            embedder = FastEmbedProvider.from_config(config)
            ingestor = KnowledgeIngestor(
                config,
                scanner=_SelectedScanner(selected),
                parser=parser,
                embedder=embedder,
            )
            summary = ingestor.run(IngestionMode.REBUILD)
            if str(summary.state) != "SUCCEEDED":
                raise RuntimeError(f"bounded ingestion did not succeed: {summary.to_dict()}")
            catalog = KnowledgeCatalog(artifact_root / "catalog.sqlite3")
            generation = catalog.active_generation()
            if generation is None:
                raise RuntimeError("bounded ingestion published no active generation")
            chunks = tuple(
                chunk
                for document_id in catalog.current_document_ids()
                for chunk in catalog.chunks_for_document(document_id)
            )
            if not chunks:
                raise RuntimeError("bounded ingestion produced no searchable chunks")
            vector = VectorIndexReader(generation.vector_location)
            lexical = LexicalIndexReader(generation.lexical_location)
            query = _first_query(chunks)
            service = KnowledgeQueryService(vector, lexical, catalog, embedder)
            started = time.perf_counter()
            hits = service.search(KnowledgeQuery(text=query, top_k=min(3, len(chunks))))
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
        if network.attempts:
            raise NetworkAttempt(f"network-attempt count was {len(network.attempts)}")
        if not hits:
            raise RuntimeError("hybrid query returned no provenance-complete hit")
        after = tuple(_snapshot(resource.path) for resource in selected)
        if before != after:
            raise RuntimeError("source hashes or filesystem metadata changed during smoke")
        ocr_policy = _ocr_policy_report(
            config.docling_do_ocr,
            selected,
            observed_pdf_options,
        )
        report = {
            "state": "COMPLETE",
            "resources": {"selected": [resource.relative_path for resource in selected], "before": before, "after": after},
            "parser": {"id": parser.parser_id, "docling_version": _package_version("docling"), "pdf_options": observed_pdf_options, "ocr_policy": ocr_policy, "formula_enrichment": "disabled"},
            "scan_classifications": dict(summary.to_dict()["counts"]),
            "chunker": {"version": config.chunker_version, "soft_target": config.chunk_soft_token_target, "effective_content_limit": embedder.spec.effective_corpus_content_token_limit, "model_input_limit": embedder.spec.model_max_input_tokens, "tokenizer_fingerprint": embedder.spec.tokenizer_fingerprint},
            "embedding_spec": embedder.spec.to_dict(),
            "generation": {"id": generation.generation_id, "population_hash": generation.population_hash, "vector_rows": len(vector.rows()), "lexical_rows": lexical.row_count(), "matched_population": vector.metadata().get("population_hash") == lexical.metadata().get("population_hash") == generation.population_hash},
            "hybrid_query": {"text": query, "result_count": len(hits), "latency_ms": latency_ms, "provenance": [_hit_fields(hit) for hit in hits]},
            "network_attempt_count": 0,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, default=str, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps(report, default=str, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"SMOKE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
