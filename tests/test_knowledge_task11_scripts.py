"""Task 11 command-boundary regressions for explicit setup and offline smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provision_requires_explicit_network_opt_in(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    source = tmp_path / "source"
    source.mkdir()

    result = provision.main(
        [
            "--source-root",
            str(source),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--docling-artifacts-path",
            str(tmp_path / "docling"),
            "--embedding-model-path",
            str(tmp_path / "embedding"),
        ]
    )

    assert result != 0
    assert not (tmp_path / "artifacts").exists()


def test_provision_rejects_artifact_root_inside_the_mandatory_source_root(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="artifact_root must be outside source_root"):
        provision._validate_source_exclusion(source, source / "artifacts")


def test_provision_retries_once_after_cleaning_an_unmanifested_docling_partial(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    destination = tmp_path / "artifacts" / "docling"
    attempts: list[int] = []

    def download() -> None:
        attempts.append(1)
        destination.mkdir(parents=True, exist_ok=True)
        if len(attempts) == 1:
            (destination / "partial.incomplete").write_text("partial", encoding="utf-8")
            raise OSError("transient downloader failure")
        (destination / "complete.bin").write_bytes(b"complete")

    provision._retry_docling_download(destination, download)

    assert len(attempts) == 2
    assert not (destination / "partial.incomplete").exists()
    assert (destination / "complete.bin").read_bytes() == b"complete"


def test_smoke_fails_before_parsing_when_local_artifacts_are_missing(tmp_path: Path) -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "book.pdf").write_bytes(b"%PDF-1.4\n")

    result = smoke.main(
        [
            "--source-root",
            str(source),
            "--artifact-root",
            str(tmp_path / "smoke-artifacts"),
            "--docling-artifacts-path",
            str(tmp_path / "missing-docling"),
            "--embedding-model-path",
            str(tmp_path / "missing-embedding"),
            "--offline",
        ]
    )

    assert result != 0
    assert not (tmp_path / "smoke-artifacts").exists()


def test_smoke_reports_only_public_knowledge_hit_provenance_fields() -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    from tradingagents.knowledge.models import KnowledgeHit

    fields = smoke._hit_fields(
        KnowledgeHit(
            chunk_id="chunk",
            document_id="document",
            source_hash="sha256:source",
            source_filename="book.pdf",
            source_relative_path="book.pdf",
            page=7,
            epub_spine_item="chapter.xhtml",
            anchor="section",
            parser_version="docling-2",
            chunker_version="structure-v2",
            index_version="index-v1",
        )
    )

    assert fields["page"] == 7
    assert fields["epub_spine_item"] == "chapter.xhtml"
    assert fields["anchor"] == "section"
    assert fields["index_version"] == "index-v1"


def test_provision_requires_destinations_beneath_the_artifact_root(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    artifact_root = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="beneath artifact_root"):
        provision._validate_destinations(
            artifact_root,
            tmp_path / "outside-docling",
            artifact_root / "embeddings" / "bge-small-en-v1.5",
        )


def test_smoke_selects_only_hashed_supported_resources() -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    from tradingagents.knowledge.models import IngestionState

    resources = (
        SimpleNamespace(state=IngestionState.DISCOVERED, path=Path("unhashed.pdf")),
        SimpleNamespace(state=IngestionState.HASHED, path=Path("book.pdf")),
        SimpleNamespace(state=IngestionState.HASHED, path=Path("chapter.epub")),
        SimpleNamespace(state=IngestionState.HASHED, path=Path("notes.txt")),
    )

    selected = smoke._select_resources(resources, limit=3)

    assert [resource.path.name for resource in selected] == ["book.pdf", "chapter.epub"]


def test_smoke_rejects_report_path_inside_source_root(tmp_path: Path) -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="report_path must be outside source_root"):
        smoke._validate_paths(source, tmp_path / "artifacts", source / "smoke-report.json")


def test_epub_only_smoke_records_explicit_ocr_disabled_policy() -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")

    policy = smoke._ocr_policy_report(
        False,
        (SimpleNamespace(path=Path("chapter.epub")),),
        (),
    )

    assert policy == {
        "configured_do_ocr": False,
        "pdf_resource_count": 0,
        "observed_pdf_option_count": 0,
    }


def test_provision_parser_exposes_independent_artifact_stages() -> None:
    provision = _load_script("provision_knowledge_models.py")
    parser = provision.build_parser()

    args = parser.parse_args(["provision", "docling", "--source-root", "source", "--artifact-root", "artifacts", "--allow-network"])

    assert args.command == "provision"
    assert args.stage == "docling"


def test_stage_failure_does_not_publish_partial_target(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    target = tmp_path / "artifacts" / "docling"
    calls: list[Path] = []

    def fail(stage: Path) -> dict[str, object]:
        calls.append(stage)
        (stage / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        provision._run_staged(target, fail)

    assert len(calls) == 1
    assert not target.exists()
    assert not list(target.parent.glob("docling.partial-*"))


def test_embedding_provision_resolves_fastembed_snapshot_layout(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    root = tmp_path / "cache" / "models--qdrant--bge-small-en-v1.5-onnx-q"
    snapshot = root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_optimized.onnx").write_bytes(b"onnx")

    assert provision._resolve_fastembed_model_dir(root) == snapshot


def test_setup_environment_disables_hf_xet_transfer_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    provision = _load_script("provision_knowledge_models.py")
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)

    environment = provision._setup_environment()

    assert environment["HF_HUB_DISABLE_XET"] == "1"
    assert environment["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_fastembed_dimension_probe_falls_back_when_model_property_is_unimplemented() -> None:
    from tradingagents.knowledge.embeddings import _resolve_dimensions

    class Model:
        @property
        def embedding_size(self) -> int:
            raise NotImplementedError

    class Embedder:
        model = Model()

    assert _resolve_dimensions(Embedder(), 384) == 384


def test_docling_setup_requests_only_required_non_ocr_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "layout.bin").write_bytes(b"layout")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(provision.subprocess, "run", fake_run)
    destination = tmp_path / "docling"
    provision._download_docling(destination)

    command = captured["command"]
    assert command[-2:] == ["layout", "tableformer"]
    assert "ocr" not in [str(value).casefold() for value in command]
    assert captured["env"]["HF_HUB_DISABLE_XET"] == "1"


def test_failed_all_stage_does_not_publish_success_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provision = _load_script("provision_knowledge_models.py")
    source = tmp_path / "source"
    source.mkdir()

    monkeypatch.setattr(provision, "_provision_docling_stage", lambda *args: (_ for _ in ()).throw(RuntimeError("docling failed")))
    monkeypatch.setattr(provision, "_provision_embedding_stage", lambda *args: {"ok": True})

    artifact_root = tmp_path / "artifacts"
    assert provision.main([
        "provision", "all", "--source-root", str(source), "--artifact-root", str(artifact_root), "--allow-network"
    ]) != 0
    assert not (artifact_root / "provisioning-manifest.json").exists()
