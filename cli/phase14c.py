"""Explicit local-only Phase 14C book-discovery operator interface.

Importing this module constructs no model, embedder, trading reader, or MT5
session.  The artifact-only status path intentionally has no imports from the
knowledge or self-enhancement execution modules.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_EXIT_OK = 0
_EXIT_FAILURE = 1
_EXIT_USAGE = 2
_STATUS_REPORT_FIELDS = frozenset(
    {
        "status",
        "generation_id",
        "generation_fingerprint",
        "population_hash",
        "source_manifest_fingerprint",
        "raw_hit_count",
        "unique_hit_count",
        "query_count",
        "selected_group_count",
        "deferred_group_count",
        "unclassified_chunk_count",
        "candidate_run_count",
        "completed_candidate_count",
        "not_run_candidate_count",
        "failed_candidate_count",
        "llm_calls",
        "conservative_estimated_teacher_calls",
        "runtime_seconds",
        "conservative_estimated_runtime_seconds",
        "runtime_estimate_basis",
        "artifact_integrity",
        "citation_validation_status",
        "source_unchanged",
        "coverage_complete",
        "selected_groups_by_family",
        "deferred_groups_by_family",
        "reason_code_counts",
    }
)


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _safe_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _STATUS_REPORT_FIELDS:
            continue
        if isinstance(item, Mapping):
            safe_mapping = {
                str(inner_key): inner_value
                for inner_key, inner_value in item.items()
                if isinstance(inner_key, str)
                and inner_key.replace("_", "").isalnum()
                and (type(inner_value) is int or isinstance(inner_value, str))
                and type(inner_value) is int
            }
            result[key] = dict(sorted(safe_mapping.items()))
        elif type(item) is str:
            if key == "runtime_estimate_basis":
                if item == "selected groups x 15 maximum local extraction attempts per group x timeout":
                    result[key] = item
            elif key in {"generation_fingerprint", "source_manifest_fingerprint"}:
                if re.fullmatch(r"[0-9a-f]{64}", item):
                    result[key] = item
            elif key == "population_hash":
                if re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", item):
                    result[key] = item
            elif key in {"citation_validation_status", "artifact_integrity"}:
                if item in {"VALID", "INVALID", "NOT_RUN", "UNKNOWN"}:
                    result[key] = item
            elif item and len(item) <= 160 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", item):
                result[key] = item
        elif item is None or type(item) in {int, float, bool}:
            if isinstance(item, float) and not math.isfinite(item):
                continue
            result[key] = item
    return result


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _read_status_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path.stat().st_size > 4_000_000:
        raise ValueError("status artifact exceeds its size limit")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    return value if isinstance(value, dict) else None


def _status(artifact_root: Path, *, as_json: bool) -> int:
    """Read only the two Phase 14C artifact JSON files; missing is not an error."""

    root = _path(artifact_root)
    if not root.is_dir():
        payload = {"status": "MISSING", "artifact_root": str(root)}
    else:
        manifest: dict[str, Any] | None = None
        report: dict[str, Any] | None = None
        try:
            manifest = _read_status_json(root / "run-manifest.json")
            report = _read_status_json(root / "phase14c-report.json")
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {"status": "INVALID", "artifact_root": str(root)}
        else:
            if manifest is None:
                status = "EMPTY"
            else:
                status = str(manifest.get("status", "INVALID"))
                if status not in {"IN_PROGRESS", "INTERRUPTED", "COMPLETE", "FAILED"}:
                    status = "INVALID"
            manifest_view = None
            if manifest is not None:
                identity_fingerprint = manifest.get("identity_fingerprint")
                if not isinstance(identity_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", identity_fingerprint):
                    identity_fingerprint = None
                manifest_view = {
                    "schema_version": (
                        manifest.get("schema_version")
                        if manifest.get("schema_version") == "phase14c-run.v1"
                        else None
                    ),
                    "status": status,
                    "identity_fingerprint": identity_fingerprint,
                }
            payload = {
                "status": status,
                "artifact_root": str(root),
                "manifest": manifest_view,
                "report": _safe_report(report),
            }
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(f"PHASE14C STATUS: {payload['status']}")
        print(f"ARTIFACT ROOT: {payload['artifact_root']}")
        for key, value in _safe_report(payload.get("report")).items():
            print(f"{key.upper()}: {value}")
    return _EXIT_OK


def plan_phase14c(**kwargs: Any) -> Mapping[str, Any]:
    """Lazy runner import keeps status metadata-only and dependency-free."""

    from tradingagents.self_enhancement.phase14c_runner import plan_phase14c as plan

    return plan(**kwargs)


def create_local_teacher(args: argparse.Namespace) -> Any:
    """Construct the native local Ollama extractor only after runner preflight."""

    import httpx

    from tradingagents.self_enhancement.book_atomic_extraction import AtomicStrategyExtractor

    transport = httpx.Client(trust_env=False, timeout=args.timeout_seconds)
    try:
        return AtomicStrategyExtractor(
            args.ollama_endpoint,
            args.model,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens=args.max_output_tokens,
            context_tokens=args.context_tokens,
            model_version=args.model_version,
            cache_path=args.atomic_cache_path,
            transport=transport,
        )
    except Exception:
        transport.close()
        raise


def evaluate_phase14c_candidates(*args: Any, **kwargs: Any) -> Any:
    from tradingagents.self_enhancement.phase14c_runner import (
        evaluate_phase14c_candidates as evaluate,
    )

    return evaluate(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase14c",
        description="Explicit local-only Phase 14C book discovery; never starts trading.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="read only a Phase 14C manifest and report")
    status.add_argument("--artifact-root", required=True, metavar="PATH")
    status.add_argument("--json", action="store_true")

    plan = commands.add_parser(
        "plan-only",
        help="pin Phase 7, retrieve and select evidence locally without Ollama or artifacts",
    )
    plan.add_argument("--knowledge-root", required=True, metavar="PATH")
    plan.add_argument("--expected-generation", required=True)
    plan.add_argument("--expected-fingerprint", required=True)
    plan.add_argument("--expected-population-hash", required=True)
    plan.add_argument("--embedding-model-path", required=True, metavar="PATH")
    plan.add_argument("--timeout-seconds", type=float, default=300.0)
    plan.add_argument("--top-k-per-formulation", type=_positive_int, default=10)
    plan.add_argument("--json", action="store_true")

    discover = commands.add_parser(
        "discover",
        help="explicitly run bounded, source-grounded local Phase 14C extraction",
    )
    _add_pinned_knowledge_options(discover)
    _add_source_options(discover)
    discover.add_argument("--phase14c-artifact-root", required=True, metavar="NEW_OR_RESUME_PATH")
    discover.add_argument("--embedding-model-path", required=True, metavar="PATH")
    discover.add_argument("--atomic-cache-path", required=True, metavar="EXTERNAL_PATH")
    discover.add_argument("--model", required=True, help="explicit qwen3.5:2b-compatible local model")
    discover.add_argument("--model-version", required=True, help="immutable local model digest/revision")
    discover.add_argument("--ollama-endpoint", required=True, help="native loopback Ollama base URL")
    discover.add_argument("--timeout-seconds", type=float, default=300.0)
    discover.add_argument("--max-output-tokens", type=_positive_int, default=512)
    discover.add_argument("--context-tokens", type=_positive_int, default=8192)
    discover.add_argument("--resume", action="store_true")
    discover.add_argument("--json", action="store_true")

    evaluate = commands.add_parser(
        "evaluate",
        help="replay complete validated candidates without model or embedder setup",
    )
    evaluate.add_argument("--artifact-root", required=True, metavar="PATH")
    _add_source_options(evaluate)
    evaluate.add_argument("--source-commit", required=True)
    evaluate.add_argument("--max-candidate-runs", type=_positive_int, default=5)
    evaluate.add_argument("--json", action="store_true")
    return parser


def _add_pinned_knowledge_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-root", required=True, metavar="PATH")
    parser.add_argument("--expected-generation", required=True)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--expected-population-hash", required=True)


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase14a-db", required=True, metavar="PATH")
    parser.add_argument("--hft-db", required=True, metavar="PATH")
    parser.add_argument("--demo-db", required=True, metavar="PATH")


def _validate_local_endpoint(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("Ollama endpoint must be a native loopback base URL") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Ollama endpoint must be a native loopback base URL")
    return f"http://{parsed.netloc}"


def _validate_teacher_settings(args: argparse.Namespace) -> None:
    if not isinstance(args.model, str) or not args.model.startswith("qwen3.5:2b"):
        raise ValueError("model must be an explicit qwen3.5:2b-compatible local identifier")
    if not isinstance(args.model_version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", args.model_version
    ):
        raise ValueError("model_version must be an explicit immutable local identifier")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be bounded between 0 and 600")
    if args.max_output_tokens > 8192:
        raise ValueError("max_output_tokens must not exceed 8192")
    if not 512 <= args.context_tokens <= 32768:
        raise ValueError("context_tokens must be between 512 and 32768")
    args.ollama_endpoint = _validate_local_endpoint(args.ollama_endpoint)


def _plan(args: argparse.Namespace) -> int:
    result = plan_phase14c(
        knowledge_root=_path(args.knowledge_root),
        expected_generation_id=args.expected_generation,
        expected_generation_fingerprint=args.expected_fingerprint,
        expected_population_hash=args.expected_population_hash,
        embedding_model_path=_path(args.embedding_model_path),
        timeout_seconds=args.timeout_seconds,
        top_k_per_formulation=args.top_k_per_formulation,
    )
    if result.get("llm_calls") != 0:
        raise ValueError("plan-only must report exactly zero LLM calls")
    print(json.dumps(dict(result), sort_keys=True, separators=(",", ":")))
    return _EXIT_OK


def _discover(args: argparse.Namespace) -> int:
    _validate_teacher_settings(args)
    from tradingagents.self_enhancement import phase14c_runner as runner
    from tradingagents.self_enhancement.phase14c_models import Phase14CSourcePaths
    from tradingagents.self_enhancement.phase14c_runner import (
        validate_discovery_paths,
    )

    source_paths = Phase14CSourcePaths(args.phase14a_db, args.hft_db, args.demo_db)
    knowledge, embedding, artifact, cache = validate_discovery_paths(
        knowledge_root=_path(args.knowledge_root),
        embedding_model_path=_path(args.embedding_model_path),
        artifact_root=_path(args.phase14c_artifact_root),
        atomic_cache_path=_path(args.atomic_cache_path),
        source_paths=source_paths,
        allow_existing_artifact_root=args.resume,
    )
    run_discovery = getattr(runner, "run_phase14c_discovery", None)
    if not callable(run_discovery):
        raise ValueError("Phase 14C discovery orchestration is not available in this build")
    pin, inventory = runner.read_phase7_inventory(
        knowledge,
        expected_generation_id=args.expected_generation,
        expected_generation_fingerprint=args.expected_fingerprint,
        expected_population_hash=args.expected_population_hash,
    )
    preflight = runner.preflight_phase14_sources(
        source_paths.phase14a_path,
        source_paths.hft_path,
        source_paths.demo_path,
    )
    if not preflight.ready:
        raise ValueError(f"Phase 14 replay preflight is not READY: {preflight.reason_code.value}")
    identity = run_discovery(
        artifact_root=artifact,
        knowledge_root=knowledge,
        embedding_model_path=embedding,
        atomic_cache_path=cache,
        source_paths=source_paths,
        source_fingerprints=dict(preflight.source_fingerprints),
        generation_pin=pin,
        inventory=inventory,
        expected_generation_id=args.expected_generation,
        expected_generation_fingerprint=args.expected_fingerprint,
        expected_population_hash=args.expected_population_hash,
        endpoint=args.ollama_endpoint,
        model=args.model,
        model_version=args.model_version,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        context_tokens=args.context_tokens,
        resume=args.resume,
        teacher_factory=lambda: create_local_teacher(args),
    )
    print(json.dumps(dict(identity), sort_keys=True, separators=(",", ":")))
    return _EXIT_OK


def _evaluate(args: argparse.Namespace) -> int:
    from tradingagents.self_enhancement.phase14c_models import Phase14CSourcePaths
    from tradingagents.self_enhancement.phase14c_runner import load_complete_discovery_artifacts

    artifact_root = _path(args.artifact_root)
    source_paths = Phase14CSourcePaths(args.phase14a_db, args.hft_db, args.demo_db)
    missing = [path for path in source_paths.to_dict().values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("all explicit Phase 14 source databases must exist")
    specs, discovery_report = load_complete_discovery_artifacts(artifact_root)
    records = evaluate_phase14c_candidates(
        specs,
        artifact_root=artifact_root / "evaluations",
        source_paths=source_paths,
        source_commit=args.source_commit,
        max_candidate_runs=args.max_candidate_runs,
    )
    result = {
        "status": "EVALUATED",
        "discovery_status": discovery_report["status"],
        "validated_spec_count": len(specs),
        "candidate_record_count": len(records),
        "completed_candidate_count": sum(record.status.value == "COMPLETED" for record in records),
        "not_run_candidate_count": sum(record.status.value == "NOT_RUN" for record in records),
        "failed_candidate_count": sum(record.status.value == "FAILED" for record in records),
        "records": [
            {
                "spec_id": record.spec_id,
                "candidate_id": record.candidate_id,
                "status": record.status.value,
                "reason_code": record.reason_code.value,
                "candidate_state": record.candidate_state,
            }
            for record in records
        ],
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return _EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "status":
            return _status(Path(args.artifact_root), as_json=args.json)
        if args.command == "plan-only":
            if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 600:
                raise ValueError("timeout_seconds must be bounded between 0 and 600")
            return _plan(args)
        if args.command == "discover":
            return _discover(args)
        if args.command == "evaluate":
            if not 1 <= args.max_candidate_runs <= 10:
                raise ValueError("max_candidate_runs must be between 1 and 10")
            return _evaluate(args)
        raise ValueError("unsupported Phase 14C command")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"PHASE14C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE if isinstance(exc, (ValueError, FileExistsError)) else _EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
