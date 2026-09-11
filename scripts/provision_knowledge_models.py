"""Explicit, setup-only provisioning for the optional Phase 7 local models.

This command is deliberately outside the normal ``knowledge`` command surface.
It is the only Phase 7 command that permits a network connection.  Indexing,
searching, and the real smoke all consume only the manifests and model files it
writes, with their runtime adapters configured ``local_files_only=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DOCLING_MANIFEST = "docling-artifacts.json"
_DOCLING_SCHEMA = "docling-artifacts-v1"
_EMBEDDING_MANIFEST = "embedding-artifacts.json"
_EMBEDDING_SCHEMA = "embedding-artifacts-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, *, exclude: Iterable[Path] = ()) -> tuple[Path, ...]:
    excluded = {path.resolve(strict=False) for path in exclude}
    return tuple(
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.resolve(strict=False) not in excluded
    )


def _tree_hash(root: Path, *, exclude: Iterable[Path] = ()) -> str:
    digest = hashlib.sha256()
    for path in _files(root, exclude=exclude):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _artifact_rows(root: Path, *, exclude: Iterable[Path] = ()) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
        for path in _files(root, exclude=exclude)
    ]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{name} is not installed in this interpreter; install tradingagents[knowledge] first"
        ) from exc


def _require_directory(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not path.name:
        raise ValueError(f"{name} must be a specific local directory")
    return path


def _ensure_outside(first: Path, second: Path, first_name: str, second_name: str) -> None:
    try:
        first.relative_to(second)
    except ValueError:
        try:
            second.relative_to(first)
        except ValueError:
            return
    raise ValueError(f"{first_name} and {second_name} must not overlap")


def _validate_destinations(artifact_root: Path, docling_path: Path, embedding_path: Path) -> None:
    """Keep setup-only writes inside the one operator-supplied artifact root."""

    for path, name in ((docling_path, "docling_artifacts_path"), (embedding_path, "embedding_model_path")):
        try:
            relative = path.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(f"{name} must be beneath artifact_root") from exc
        if not relative.parts:
            raise ValueError(f"{name} must be a child directory beneath artifact_root")
    _ensure_outside(docling_path, embedding_path, "docling_artifacts_path", "embedding_model_path")


def _download_docling(destination: Path) -> None:
    try:
        from docling.utils.model_downloader import download_models
    except ImportError as exc:
        raise RuntimeError("Docling model downloader is unavailable in this interpreter") from exc

    destination.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(download_models)
    values: dict[str, Any] = {
        "output_dir": destination,
        "artifacts_path": destination,
        "cache_dir": destination,
        "force": False,
        # V1 never performs OCR and does not need RapidOCR assets. Formula
        # enrichment is separately disabled unless later explicitly configured.
        "with_rapidocr": False,
        "with_code_formula": False,
    }
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = values if accepts_kwargs else {key: value for key, value in values.items() if key in signature.parameters}
    try:
        result = download_models(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Docling artifact provisioning failed: {exc}") from exc
    # Different supported Docling releases either write into output_dir or
    # return an artifact directory.  Copy a returned directory only if it is
    # outside the configured, operator-visible cache.
    if isinstance(result, (str, Path)):
        returned = Path(result).expanduser().resolve(strict=False)
        if returned.is_dir() and returned != destination and not destination.exists():
            shutil.copytree(returned, destination)
    if not _files(destination):
        raise RuntimeError("Docling downloader completed without local artifacts")


def _model_path_from_embedder(embedder: Any) -> Path | None:
    for owner in (embedder, getattr(embedder, "model", None)):
        if owner is None:
            continue
        for name in ("model_path", "_model_path", "specific_model_path"):
            value = getattr(owner, name, None)
            if value is not None and Path(value).is_dir():
                return Path(value).resolve(strict=False)
    return None


def _download_embedding(model_id: str, destination: Path) -> None:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("FastEmbed is unavailable in this interpreter") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    # The constructor is intentionally used only here.  It can resolve the
    # FastEmbed registry/network artifact; normal runtime always provides both
    # ``specific_model_path`` and ``local_files_only=True``.
    try:
        downloaded = TextEmbedding(
            model_name=model_id,
            cache_dir=str(destination.parent),
            threads=1,
            providers=["CPUExecutionProvider"],
        )
        tuple(downloaded.embed(["phase seven local artifact verification"], batch_size=1))
    except Exception as exc:
        raise RuntimeError(f"FastEmbed artifact provisioning failed: {exc}") from exc

    source = _model_path_from_embedder(downloaded)
    if source is None:
        candidates = [path for path in destination.parent.iterdir() if path.is_dir() and path != destination]
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        source = next((path for path in candidates if any(path.rglob("*.onnx"))), None)
    if source is None or not source.is_dir():
        raise RuntimeError("FastEmbed did not expose a local model directory")
    if source != destination:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    if not any(destination.rglob("*.onnx")):
        raise RuntimeError("provisioned FastEmbed directory has no ONNX model")


def _write_docling_manifest(destination: Path) -> dict[str, Any]:
    manifest = {
        "schema_version": _DOCLING_SCHEMA,
        "docling_version": _package_version("docling"),
        "artifacts": _artifact_rows(destination, exclude=(destination / _DOCLING_MANIFEST,)),
    }
    if not manifest["artifacts"]:
        raise RuntimeError("Docling artifact cache is empty")
    (destination / _DOCLING_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return manifest


def _write_embedding_manifest(model_id: str, destination: Path) -> dict[str, Any]:
    # Import the project adapter only after the network-enabled download and
    # re-open it using its normal offline-only contract.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tradingagents.knowledge.config import KnowledgeConfig
    from tradingagents.knowledge.embeddings import FastEmbedProvider

    scratch = destination.parent / ".provision-config-source"
    scratch.mkdir(exist_ok=True)
    provider = FastEmbedProvider.from_config(
        KnowledgeConfig(
            source_root=scratch,
            artifact_root=destination.parent / ".provision-artifacts",
            embedding_model_id=model_id,
            embedding_model_path=destination,
            worker_count=1,
            embedding_batch_size=1,
        )
    )
    spec = provider.spec.to_dict()
    manifest_path = destination / _EMBEDDING_MANIFEST
    manifest = {
        "schema_version": _EMBEDDING_SCHEMA,
        "model_id": model_id,
        "fastembed_version": _package_version("fastembed"),
        "onnxruntime_version": _package_version("onnxruntime"),
        "artifact_hash": _tree_hash(destination, exclude=(manifest_path,)),
        "tokenizer_fingerprint": spec["tokenizer_fingerprint"],
        "dimensions": spec["dimensions"],
        "embedding_spec": spec,
        "artifacts": _artifact_rows(destination, exclude=(manifest_path,)),
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicitly provision Phase 7 local Docling and FastEmbed artifacts.")
    parser.add_argument("--artifact-root", required=True, help="setup-only artifact parent outside the book source")
    parser.add_argument("--docling-artifacts-path", required=True, help="destination for Docling local assets")
    parser.add_argument("--embedding-model-id", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-model-path", required=True, help="destination for the local ONNX embedding model")
    parser.add_argument("--allow-network", action="store_true", help="required acknowledgement for this setup-only download")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_network:
        print("PROVISION ERROR: --allow-network is required; normal knowledge commands never download", file=sys.stderr)
        return 2
    try:
        artifact_root = _require_directory(args.artifact_root, "artifact_root")
        docling_path = _require_directory(args.docling_artifacts_path, "docling_artifacts_path")
        embedding_path = _require_directory(args.embedding_model_path, "embedding_model_path")
        _validate_destinations(artifact_root, docling_path, embedding_path)
        artifact_root.mkdir(parents=True, exist_ok=True)
        _download_docling(docling_path)
        docling = _write_docling_manifest(docling_path)
        _download_embedding(args.embedding_model_id, embedding_path)
        embedding = _write_embedding_manifest(args.embedding_model_id, embedding_path)
        report = {
            "state": "PROVISIONED",
            "docling": {"version": docling["docling_version"], "artifact_count": len(docling["artifacts"])},
            "embedding": {
                "model_id": args.embedding_model_id,
                "artifact_hash": embedding["artifact_hash"],
                "tokenizer_fingerprint": embedding["tokenizer_fingerprint"],
                "dimensions": embedding["dimensions"],
            },
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"PROVISION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
