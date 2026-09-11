"""Explicit, setup-only provisioning for the optional Phase 7 local models.

This command is deliberately outside the normal ``knowledge`` command surface.
It is the only Phase 7 command that permits a network connection.  Indexing,
searching, and the real smoke all consume only the manifests and model files it
writes, with their runtime adapters configured ``local_files_only=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DOCLING_MANIFEST = "docling-artifacts.json"
_DOCLING_SCHEMA = "docling-artifacts-v1"
_EMBEDDING_MANIFEST = "embedding-artifacts.json"
_EMBEDDING_SCHEMA = "embedding-artifacts-v1"
_PROVISIONING_MANIFEST = "provisioning-manifest.json"


def _setup_environment() -> dict[str, str]:
    """Return setup-only environment defaults that avoid fragile Xet transfers."""

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    environment.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return environment


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


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


def _validate_source_exclusion(source_root: Path, artifact_root: Path) -> None:
    if not source_root.is_dir():
        raise ValueError(f"source_root must be an existing directory: {source_root}")
    try:
        artifact_root.relative_to(source_root)
    except ValueError:
        try:
            source_root.relative_to(artifact_root)
        except ValueError:
            return
    raise ValueError("artifact_root must be outside source_root")


class ProvisioningError(RuntimeError):
    """Typed setup failure with safe scalar diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        exception_type: str = "ProvisioningError",
        failure_class: str = "unknown",
        http_status: int | None = None,
        repository: str | None = None,
        file: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.exception_type = exception_type
        self.failure_class = failure_class
        self.http_status = http_status
        self.repository = repository
        self.file = file

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "exception_type": self.exception_type,
            "failure_class": self.failure_class,
            "http_status": self.http_status,
            "repository": self.repository,
            "file": self.file,
            "message": str(self),
        }


def _run_staged(target: Path, operation: Callable[[Path], dict[str, Any] | None]) -> dict[str, Any] | None:
    """Run one artifact stage in a disposable directory, then promote atomically."""

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial-{uuid.uuid4().hex}")
    backup = target.with_name(f"{target.name}.previous-{uuid.uuid4().hex}")
    try:
        partial.mkdir(parents=True, exist_ok=False)
        result = operation(partial)
        if not partial.is_dir() or not _files(partial):
            raise RuntimeError("stage produced no validated artifacts")
        if target.exists():
            target.rename(backup)
        partial.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
        return result
    except Exception:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise


def _retry_docling_download(destination: Path, download: Callable[[], None]) -> None:
    """Retry once after deleting only an incomplete, unmanifested setup cache."""

    for attempt in range(2):
        destination.mkdir(parents=True, exist_ok=True)
        try:
            download()
            if not _files(destination):
                raise RuntimeError("Docling downloader completed without local artifacts")
            return
        except Exception:
            manifest = destination / _DOCLING_MANIFEST
            if attempt == 0 and not manifest.is_file():
                shutil.rmtree(destination)
                continue
            raise


def _download_docling(destination: Path) -> None:
    executable = Path(sys.executable).with_name("docling-tools.exe")
    command = [
        str(executable if executable.is_file() else "docling-tools"),
        "models",
        "download",
        "--output-dir",
        str(destination),
        "--quiet",
        "layout",
        "tableformer",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
            env=_setup_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ProvisioningError(
            "Docling model download timed out",
            stage="docling",
            exception_type=type(exc).__name__,
            failure_class="timeout",
            repository="docling-project/docling-layout-heron",
        ) from exc
    except OSError as exc:
        raise ProvisioningError(
            "Docling model downloader could not start",
            stage="docling",
            exception_type=type(exc).__name__,
            failure_class="filesystem_or_executable",
            repository="docling-project/docling-layout-heron",
        ) from exc
    if completed.returncode != 0:
        output = f"{completed.stdout}\n{completed.stderr}"[-2000:]
        status_match = re.search(r"HTTP(?:/\d(?:\.\d)?)?[^\d]*(\d{3})", output)
        status = int(status_match.group(1)) if status_match else None
        failure_class = "native_process_exit" if completed.returncode < 0 else "downloader_error"
        raise ProvisioningError(
            f"Docling model download failed (returncode={completed.returncode})",
            stage="docling",
            exception_type="SubprocessExit",
            failure_class=failure_class,
            http_status=status,
            repository="docling-project/docling-layout-heron/tableformer",
            file=None,
        )


def _model_path_from_embedder(embedder: Any) -> Path | None:
    for owner in (embedder, getattr(embedder, "model", None)):
        if owner is None:
            continue
        for name in ("model_path", "_model_path", "specific_model_path"):
            value = getattr(owner, name, None)
            if value is not None and Path(value).is_dir():
                return Path(value).resolve(strict=False)
    return None


def _resolve_fastembed_model_dir(root: Path) -> Path | None:
    """Resolve FastEmbed's downloaded snapshot to the directory it can open."""

    if any(root.glob("*.onnx")):
        return root
    snapshots = sorted((path for path in root.glob("snapshots/*") if path.is_dir()), key=lambda p: p.as_posix())
    for snapshot in reversed(snapshots):
        if any(snapshot.glob("*.onnx")):
            return snapshot
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
    previous_environment = os.environ.copy()
    os.environ.update(_setup_environment())
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
    finally:
        os.environ.clear()
        os.environ.update(previous_environment)

    source = _model_path_from_embedder(downloaded)
    if source is None:
        candidates = [path for path in destination.parent.iterdir() if path.is_dir() and path != destination]
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        source = next(
            (resolved for path in candidates if (resolved := _resolve_fastembed_model_dir(path)) is not None),
            None,
        )
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
    # Force real ONNX inference while network access is still explicitly
    # enabled for setup; the normal provider will reopen this exact directory
    # with local_files_only=True.
    tuple(provider.embed(("phase7 provisioning probe",), purpose="query"))
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


def _provision_docling_stage(destination: Path, source_root: Path, artifact_root: Path) -> dict[str, Any]:
    _download_docling(destination)
    manifest = _write_docling_manifest(destination)
    from tradingagents.knowledge.config import KnowledgeConfig
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    config = KnowledgeConfig(
        source_root=source_root,
        artifact_root=artifact_root,
        docling_artifacts_path=destination,
    )
    options = DoclingDocumentParser(config).build_pdf_pipeline_options("provision")
    if getattr(options, "do_ocr", None) is not False:
        raise ProvisioningError(
            "Docling pipeline did not prove do_ocr=False",
            stage="docling",
            failure_class="configuration",
        )
    return manifest


def _provision_embedding_stage(model_id: str, destination: Path) -> dict[str, Any]:
    _download_embedding(model_id, destination)
    return _write_embedding_manifest(model_id, destination)


def _verify_artifacts(
    source_root: Path,
    artifact_root: Path,
    docling_path: Path,
    embedding_path: Path,
    model_id: str,
) -> dict[str, Any]:
    docling_manifest = _load_json(docling_path / _DOCLING_MANIFEST, "Docling artifact manifest")
    embedding_manifest = _load_json(embedding_path / _EMBEDDING_MANIFEST, "embedding artifact manifest")
    from tradingagents.knowledge.config import KnowledgeConfig
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser
    from tradingagents.knowledge.embeddings import FastEmbedProvider

    config = KnowledgeConfig(
        source_root=source_root,
        artifact_root=artifact_root,
        docling_artifacts_path=docling_path,
        embedding_model_id=model_id,
        embedding_model_path=embedding_path,
    )
    options = DoclingDocumentParser(config).build_pdf_pipeline_options("verify")
    embedder = FastEmbedProvider.from_config(config)
    tuple(embedder.embed(("phase7 offline verification",), purpose="query"))
    return {
        "state": "VERIFIED",
        "docling": {"version": docling_manifest.get("docling_version"), "do_ocr": getattr(options, "do_ocr", None)},
        "embedding": {
            "model_id": model_id,
            "artifact_hash": embedding_manifest.get("artifact_hash"),
            "dimensions": embedder.spec.dimensions,
            "tokenizer_fingerprint": embedder.spec.tokenizer_fingerprint,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicitly provision Phase 7 local Docling and FastEmbed artifacts.")
    parser.add_argument("command", nargs="?", choices=("provision", "verify"))
    parser.add_argument("stage", nargs="?", choices=("docling", "embedding", "all"))
    parser.add_argument("--source-root", required=True, help="approved read-only source root excluded from setup artifacts")
    parser.add_argument("--artifact-root", required=True, help="setup-only artifact parent outside the book source")
    parser.add_argument("--docling-artifacts-path", help="destination for Docling local assets")
    parser.add_argument("--embedding-model-id", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-model-path", help="destination for the local ONNX embedding model")
    parser.add_argument("--allow-network", action="store_true", help="required acknowledgement for this setup-only download")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "provision"
    stage = args.stage or "all"
    if command == "provision" and not args.allow_network:
        print("PROVISION ERROR: --allow-network is required; normal knowledge commands never download", file=sys.stderr)
        return 2
    try:
        source_root = _require_directory(args.source_root, "source_root")
        artifact_root = _require_directory(args.artifact_root, "artifact_root")
        docling_path = _require_directory(
            args.docling_artifacts_path or str(artifact_root / "docling"), "docling_artifacts_path"
        )
        embedding_path = _require_directory(
            args.embedding_model_path or str(artifact_root / "embeddings" / args.embedding_model_id.replace("/", "--")),
            "embedding_model_path",
        )
        _validate_source_exclusion(source_root, artifact_root)
        _validate_destinations(artifact_root, docling_path, embedding_path)
        artifact_root.mkdir(parents=True, exist_ok=True)
        if command == "verify":
            report = _verify_artifacts(source_root, artifact_root, docling_path, embedding_path, args.embedding_model_id)
        else:
            results: dict[str, Any] = {}
            errors: dict[str, dict[str, Any]] = {}
            if stage in {"docling", "all"}:
                try:
                    results["docling"] = _run_staged(
                        docling_path, lambda path: _provision_docling_stage(path, source_root, artifact_root)
                    )
                except Exception as exc:
                    errors["docling"] = exc.to_dict() if isinstance(exc, ProvisioningError) else {"message": str(exc), "exception_type": type(exc).__name__}
            if stage in {"embedding", "all"}:
                try:
                    results["embedding"] = _run_staged(
                        embedding_path, lambda path: _provision_embedding_stage(args.embedding_model_id, path)
                    )
                except Exception as exc:
                    errors["embedding"] = exc.to_dict() if isinstance(exc, ProvisioningError) else {"message": str(exc), "exception_type": type(exc).__name__}
            if errors:
                print(json.dumps({"state": "FAILED", "stages": results, "errors": errors}, sort_keys=True), file=sys.stderr)
                return 1
            report = {"state": "PROVISIONED", "docling": results.get("docling"), "embedding": results.get("embedding")}
            (artifact_root / _PROVISIONING_MANIFEST).write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"PROVISION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
