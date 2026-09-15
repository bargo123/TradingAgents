"""Bounded, read-only Phase 11A acceptance against a real Phase 7 artifact.

The harness deliberately stops at source planning.  A production teacher is
never inferred and no lesson is fabricated when one is not configured.  It is
safe to run repeatedly because it only reads the Phase 7 catalog/projection
files and writes an optional report outside that root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Direct script invocation should use the checkout's package, without
    # importing optional parser/embedding/provider dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.distillation.models import canonical_json
from tradingagents.distillation.phase7 import Phase7KnowledgeSource
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.teacher import teacher_from_environment


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
            size += len(part)
    return {"exists": True, "bytes": size, "sha256": digest.hexdigest()}


def _artifact_fingerprint(root: Path) -> dict[str, Any]:
    """Hash the bounded Phase 7 files relevant to the adapter.

    Projection files can be large, so the fingerprint is a deterministic map
    of file names and SHA-256s rather than a second in-memory copy of them.
    """

    if not root.exists():
        return {"exists": False}
    files = {}
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: str(p)):
        try:
            relative = path.relative_to(root).as_posix()
            files[relative] = _file_fingerprint(path)
        except (OSError, ValueError):
            # A disappearing file is a visible fingerprint mismatch after the
            # read; it is not silently treated as unchanged.
            files[str(path)] = {"exists": False, "error": "UNREADABLE"}
    return {"exists": True, "files": files}


def _source_counts(source: Phase7KnowledgeSource) -> tuple[int, int]:
    blocks = tuple(source.blocks())
    documents_method = getattr(source, "documents", None)
    if callable(documents_method):
        document_count = len(tuple(documents_method()))
    else:
        document_count = len({block.ref.document_id for block in blocks})
    return document_count, len(blocks)


def run_real_acceptance(
    phase7_root: str | Path,
    *,
    topics: tuple[str, ...] = (),
    expected_generation_id: str | None = None,
    planner_config: PlannerConfig | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plan bounded packets from a real Phase 7 generation.

    The returned mapping contains only counts, identifiers, fingerprints and
    bounded diagnostics.  It never includes source text, prompts, or teacher
    output.  A configured teacher is reported as metadata only; callers must
    explicitly invoke the distillation factory to generate a dataset.
    """

    root = Path(phase7_root).resolve()
    before = _artifact_fingerprint(root)
    payload: dict[str, Any] = {
        "status": "FAILED",
        "phase7_root": str(root),
        "topics": list(topics),
        "network_attempts": 0,
        "mt5_calls": 0,
        "llm_calls": 0,
    }
    try:
        source = Phase7KnowledgeSource.open(root, expected_generation_id)
        config = planner_config or PlannerConfig()
        plan = SourcePacketPlanner.plan(source, topics, config)
        document_count, chunk_count = _source_counts(source)
        teacher = teacher_from_environment({})
        configured = teacher.__class__.__name__ != "UnconfiguredTeacher"
        payload.update(
            {
                "status": "PLANNED",
                "generation_id": source.generation_id,
                "source_fingerprints": source.source_fingerprints,
                "documents": document_count,
                "chunks": chunk_count,
                "quarantined_chunks": len(getattr(source, "quarantined_chunks", ())),
                "packets": len(plan.packets),
                "plan_request_fingerprint": plan.request_fingerprint,
                "diagnostics": list(plan.diagnostics),
                "teacher_configured": configured,
                "generation_status": (
                    "TEACHER_NOT_RUN" if configured else "DISTILLATION_TEACHER_NOT_CONFIGURED"
                ),
            }
        )
    except Exception as exc:
        payload.update({"status": type(exc).__name__, "error": str(exc)[:512]})
    after = _artifact_fingerprint(root)
    payload["fingerprints_before"] = before
    payload["fingerprints_after"] = after
    payload["source_unchanged"] = before == after
    if not payload["source_unchanged"] and payload["status"] == "PLANNED":
        payload["status"] = "SOURCE_MUTATED"
    if report_path is not None:
        destination = Path(report_path).resolve()
        if destination == root or root in destination.parents:
            raise ValueError("acceptance report must be outside the Phase 7 root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-root", type=Path, required=True)
    parser.add_argument("--expected-generation-id")
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--max-estimated-tokens", type=int, default=3_000)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_real_acceptance(
        args.phase7_root,
        topics=tuple(args.topic),
        expected_generation_id=args.expected_generation_id,
        planner_config=PlannerConfig(
            max_blocks=args.max_blocks,
            max_chars=args.max_chars,
            max_estimated_tokens=args.max_estimated_tokens,
        ),
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "PLANNED" and result.get("source_unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
