"""Run a saved-snapshot, non-persisting Phase 9 A/B replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.experience.source_reader import ReadonlySourceReader
from tradingagents.forex.evidence_replay import (
    EvidenceReplayConfig,
    SavedSnapshotCodec,
    SavedSnapshotReplay,
    SnapshotReplayError,
)


def _analysts(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one analyst is required")
    return names


def _load_generation(root: Path, *, phase: int) -> str | None:
    """Read an active generation ID without constructing a writer catalog."""

    from tradingagents.forex.evidence_runtime import (
        ReadonlyExperienceCatalog,
        ReadonlyKnowledgeCatalog,
    )

    catalog = (ReadonlyKnowledgeCatalog if phase == 7 else ReadonlyExperienceCatalog)(root)
    try:
        generation = catalog.active_generation()
        if generation is None:
            return None
        return str(generation.get("generation_id") if isinstance(generation, dict) else generation.generation_id)
    finally:
        close = getattr(catalog, "close", None)
        if callable(close):
            close()


def run_replay(
    source_db: str | Path,
    decision_id: str,
    *,
    experience_artifact_root: str | Path,
    knowledge_artifact_root: str | Path,
    knowledge_embedding_model_path: str | Path,
    profile: str = "INTRADAY",
    analysts: tuple[str, ...] = ("market", "news"),
    offline: bool = False,
    report_path: str | Path | None = None,
    replay: SavedSnapshotReplay | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Replay one persisted decision and return a privacy-safe report.

    The source is opened by :class:`ReadonlySourceReader`; ``SavedSnapshotReplay``
    receives the exact source bytes and is required to call ``analyze`` only.
    ``runner`` is an injectable seam for local acceptance fixtures.
    """

    if not offline:
        raise ValueError("--offline is required")
    if os.environ.get("KNOWLEDGE_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("KNOWLEDGE_OFFLINE, HF_HUB_OFFLINE, and TRANSFORMERS_OFFLINE must all be 1")
    source = Path(source_db).expanduser().resolve()
    experience_root = Path(experience_artifact_root).expanduser().resolve()
    knowledge_root = Path(knowledge_artifact_root).expanduser().resolve()
    model_path = Path(knowledge_embedding_model_path).expanduser().resolve()
    for path, label in ((source, "source database"), (experience_root, "Phase 8 artifact root"), (knowledge_root, "Phase 7 artifact root"), (model_path, "embedding model")):
        if not path.exists():
            raise SnapshotReplayError(f"{label} does not exist: {path}")
    if report_path is not None:
        destination = Path(report_path).expanduser().resolve()
        for root, label in ((source, "source"), (Path(str(source) + "-wal"), "source WAL"), (experience_root, "Phase 8"), (knowledge_root, "Phase 7"), (model_path, "embedding model")):
            try:
                destination.relative_to(root)
            except ValueError:
                continue
            raise SnapshotReplayError(f"report path must be outside {label} artifacts")
    # Replays pin only an already accepted, published Phase 8 generation.
    from scripts.phase9_evidence_smoke import (
        OfflineNetworkGuard,
        _fingerprint,
        _resolve_verified_phase7_generation,
        _source_fingerprint,
        resolve_verified_phase8_root,
    )

    phase8_preflight = resolve_verified_phase8_root(experience_root)
    phase7_preflight = _resolve_verified_phase7_generation(knowledge_root)
    reader = ReadonlySourceReader(source)
    snapshot_store = reader.read_snapshot()
    row = next((item for item in snapshot_store.decisions if str(item.get("decision_id")) == str(decision_id)), None)
    if row is None:
        raise SnapshotReplayError(f"source decision not found: {decision_id}")
    snapshot = SavedSnapshotCodec.from_source_row(row)
    encoded = row.get("snapshot_json")
    snapshot_bytes = encoded if isinstance(encoded, bytes) else str(encoded).encode("utf-8")
    phase7_generation = str(phase7_preflight.generation_id)
    phase8_generation = phase8_preflight.generation_id
    config = EvidenceReplayConfig(
        source_decision_id=str(decision_id),
        profile=profile,
        analysts=tuple(analysts),
        provider="ollama",
        models={"quick": "qwen", "deep": "qwen"},
        model_settings={"temperature": 0},
        pinned_phase7_generation_id=phase7_generation,
        pinned_phase8_generation_id=phase8_generation,
        phase7_root=knowledge_root,
        phase8_root=experience_root,
        source_database_path=source,
    )
    if replay is None:
        if runner is None:
            from scripts.phase9_evidence_smoke import build_local_orchestrator
            from tradingagents.forex.runner import ForexShadowRunner

            runner = ForexShadowRunner(
                store=type("NoopStore", (), {})(),
                config={
                    "forex_evidence_enabled": True,
                    "forex_evidence_artifact_roots": {
                        "knowledge": str(knowledge_root),
                        "experience": str(experience_root),
                        "knowledge_embedding_model_path": str(model_path),
                    },
                    "evidence_orchestrator_factory": build_local_orchestrator,
                    "llm_provider": "ollama",
                    "backend_url": "http://127.0.0.1:11434/v1",
                    "quick_think_llm": "qwen",
                    "deep_think_llm": "qwen",
                },
            )
        replay = SavedSnapshotReplay(runner=runner, generation_provider=(phase7_generation, phase8_generation))
    before = {"source": _source_fingerprint(source), "phase7": _fingerprint(knowledge_root), "phase8": _fingerprint(experience_root)}
    guard = OfflineNetworkGuard()
    failure: Exception | None = None
    result: Any = None
    try:
        with guard:
            result = replay.run(snapshot, snapshot_bytes=snapshot_bytes, config=config)
    except Exception as exc:
        if "external network blocked" in str(exc).lower():
            guard.external_network_attempts = max(guard.external_network_attempts, 1)
        failure = exc
    after = {"source": _source_fingerprint(source), "phase7": _fingerprint(knowledge_root), "phase8": _fingerprint(experience_root)}
    if before != after:
        raise SnapshotReplayError("source or Phase 7/8 artifacts changed during replay")
    if guard.external_network_attempts:
        raise SnapshotReplayError("external network attempt blocked during replay") from failure
    if failure is not None:
        raise failure
    payload = result.to_dict()
    payload["source_unchanged"] = True
    payload["external_network_attempts"] = guard.external_network_attempts
    payload["loopback_connection_attempts"] = guard.loopback_connection_attempts
    if report_path is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--experience-artifact-root", required=True)
    parser.add_argument("--knowledge-artifact-root", required=True)
    parser.add_argument("--knowledge-embedding-model-path", required=True)
    parser.add_argument("--profile", default="INTRADAY")
    parser.add_argument("--analysts", type=_analysts, default=("market", "news"))
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--report-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(run_replay(**vars(args)), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # command boundary is intentionally fail-closed
        print(f"PHASE 9 REPLAY FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
