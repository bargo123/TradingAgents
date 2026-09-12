"""Explicit, deterministic command line boundary for Phase 8 Experience Memory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .catalog import ExperienceCatalog
from .importer import ExperienceImporter, ExperienceRebuilder
from .models import EvidenceRequest, ExperienceQuery, OutcomeStatsRequest, TrustTier
from .orchestrator import EvidenceOrchestrator
from .outcomes import OutcomeStatsCalculator
from .query import ExperienceQueryService

_CATALOG = "catalog.sqlite3"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(v) for v in value]
    return value


def _emit(value: Any) -> None:
    print(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")))


def _root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else Path("data_cache/experience").resolve()


def _existing_catalog(root: Path) -> ExperienceCatalog | None:
    # Metadata commands must not create an artifact directory or database.
    if not (root / _CATALOG).is_file():
        return None
    return ExperienceCatalog(root)


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _trust(values: list[str] | None, default: tuple[TrustTier, ...] = ()) -> tuple[TrustTier, ...]:
    return tuple(TrustTier(v) for v in (values if values else default))


def _filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol")
    parser.add_argument("--profile", "--analysis-profile", dest="analysis_profile")
    parser.add_argument("--timeframe", "--analysis-timeframe", dest="analysis_timeframe")
    parser.add_argument(
        "--trust-tier",
        "--trust",
        dest="trust_tiers",
        action="append",
        choices=[t.value for t in TrustTier],
    )
    parser.add_argument("--as-of")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experience", description="Local Phase 8 Experience Memory commands."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="import read-only source decisions")
    imp.add_argument("--source-db", action="append", required=True)
    imp.add_argument("--artifact-root")
    imp.add_argument("--json", action="store_true")
    reb = commands.add_parser("rebuild", help="rebuild the active experience generation")
    reb.add_argument("--artifact-root")
    reb.add_argument("--json", action="store_true")
    status = commands.add_parser("status", help="show artifact metadata")
    status.add_argument("--artifact-root")
    status.add_argument("--json", action="store_true")
    listing = commands.add_parser("list", help="list retained records")
    listing.add_argument("--artifact-root")
    listing.add_argument("--state", choices=("active", "historical"))
    listing.add_argument("--trust-tier", dest="trust_tier", choices=[t.value for t in TrustTier])
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show", help="show one record")
    show.add_argument("experience_id")
    show.add_argument("--artifact-root")
    show.add_argument("--json", action="store_true")
    similar = commands.add_parser("similar", help="search numeric market-state projections")
    similar.add_argument("--market-state-json", required=True)
    similar.add_argument("--artifact-root")
    similar.add_argument("--top-k", type=_positive, default=10)
    _filters(similar)
    similar.add_argument("--json", action="store_true")
    stats = commands.add_parser("stats", help="calculate descriptive outcome statistics")
    stats.add_argument("--basis", required=True)
    stats.add_argument("--horizon-seconds", type=_positive, required=True)
    stats.add_argument("--experience-id", action="append", required=True)
    stats.add_argument("--artifact-root")
    stats.add_argument(
        "--trust-tier", dest="trust_tiers", action="append", choices=[t.value for t in TrustTier]
    )
    stats.add_argument("--as-of")
    stats.add_argument("--json", action="store_true")
    quarantine = commands.add_parser("quarantine", help="show bounded diagnostics")
    quarantine.add_argument("--artifact-root")
    quarantine.add_argument("--json", action="store_true")
    evidence = commands.add_parser("evidence", help="compose knowledge and experience evidence")
    evidence.add_argument("--question")
    evidence.add_argument("--market-state-json")
    evidence.add_argument("--artifact-root")
    evidence.add_argument("--knowledge-artifact-root")
    evidence.add_argument("--knowledge-top-k", type=_positive, default=10)
    evidence.add_argument("--experience-top-k", type=_positive, default=50)
    evidence.add_argument("--basis")
    evidence.add_argument("--horizon-seconds", type=_positive)
    _filters(evidence)
    evidence.add_argument("--json", action="store_true")
    return parser


def _similar_service(args: argparse.Namespace) -> ExperienceQueryService:
    catalog = _existing_catalog(_root(args.artifact_root))
    if catalog is None:
        return ExperienceQueryService(())
    return ExperienceQueryService(catalog)


def _build_knowledge_service(artifact_root: Path) -> Any:
    """Construct only Phase 7's existing read/query boundary.

    Imports are deliberately lazy: metadata, numeric similarity, and
    market-state-only evidence never load Phase 7 embedding/index modules.
    """
    from tradingagents.knowledge.catalog import KnowledgeCatalog
    from tradingagents.knowledge.config import KnowledgeConfig
    from tradingagents.knowledge.embeddings import FastEmbedProvider
    from tradingagents.knowledge.lexical_index import LexicalIndexReader
    from tradingagents.knowledge.query import KnowledgeQueryService
    from tradingagents.knowledge.vector_index import VectorIndexReader

    catalog = KnowledgeCatalog(artifact_root / "catalog.sqlite3")
    generation = catalog.active_generation()
    if generation is None:
        raise ValueError("no active Phase 7 knowledge generation")
    spec = generation.embedding_spec
    model_path = artifact_root / "models" / "embedding"
    config = KnowledgeConfig(
        source_root=Path.cwd(),
        artifact_root=artifact_root,
        embedding_model_id=spec.model_id,
        embedding_model_path=model_path if model_path.is_dir() else None,
        embedding_dimensions=spec.dimensions,
        embedding_runtime=spec.runtime,
        embedding_normalization=spec.normalization_policy,
        embedding_model_version=spec.resolved_model_version,
        embedding_artifact_hash=spec.artifact_hash,
        embedding_tokenizer_fingerprint=spec.tokenizer_fingerprint,
        embedding_max_input_tokens=spec.model_max_input_tokens,
        embedding_special_token_budget=spec.special_token_budget,
        embedding_effective_content_token_limit=spec.effective_corpus_content_token_limit,
        embedding_truncation=spec.truncation,
        embedding_corpus_instruction_policy=spec.corpus_instruction_policy,
        embedding_corpus_instruction_version=spec.corpus_instruction_version,
        embedding_query_instruction_policy=spec.query_instruction_policy,
        embedding_query_instruction_version=spec.query_instruction_version,
    )
    embedder = FastEmbedProvider.from_config(config)
    return KnowledgeQueryService(
        VectorIndexReader(generation.vector_location),
        LexicalIndexReader(generation.lexical_location),
        catalog,
        embedder,
    )


def _query(args: argparse.Namespace, state: dict[str, Any]) -> Any:
    tiers = _trust(args.trust_tiers, (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED))
    return ExperienceQuery(
        market_state=state,
        top_k=args.top_k,
        symbol=args.symbol,
        analysis_profile=args.analysis_profile,
        analysis_timeframe=args.analysis_timeframe,
        trust_tiers=tiers,
        as_of=_as_of(args.as_of),
    )


def _run(args: argparse.Namespace) -> Any:
    root = _root(getattr(args, "artifact_root", None))
    command = args.command
    if command == "status":
        catalog = _existing_catalog(root)
        if catalog is None:
            return {
                "active_generation": None,
                "artifact_root_initialized": False,
                "experience_counts": {"active": 0, "historical": 0},
                "quarantine_count": 0,
            }
        return {
            "active_generation": catalog.active_generation(),
            "artifact_root_initialized": True,
            "experience_counts": {
                "active": len(catalog.active_records()),
                "historical": len(catalog.historical_records()),
            },
            "quarantine_count": catalog.quarantine_count(),
        }
    if command == "import":
        report = ExperienceImporter(ExperienceCatalog(root)).import_sources(args.source_db)
        return report
    if command == "rebuild":
        return ExperienceRebuilder(ExperienceCatalog(root)).rebuild()
    catalog = _existing_catalog(root)
    if command == "quarantine":
        if catalog is None:
            return []
        with catalog._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT source_database_id,decision_id,fingerprint,reason,observed_at FROM experience_quarantine ORDER BY quarantine_id"
                )
            ]
    if command == "list":
        if catalog is None:
            return []
        records = (
            catalog.historical_records()
            if args.state == "historical"
            else catalog.active_records()
            if args.state == "active"
            else catalog.active_records() + catalog.historical_records()
        )
        return [r for r in records if not args.trust_tier or str(r.trust) == args.trust_tier]
    if command == "show":
        if catalog is None:
            return None
        records = catalog.active_records() + catalog.historical_records()
        return next((r for r in records if r.experience_id == args.experience_id), None)
    if command == "similar":
        state = json.loads(Path(args.market_state_json).read_text(encoding="utf-8"))
        return _similar_service(args).search(_query(args, state))
    if command == "stats":
        if catalog is None:
            return {
                "eligible_count": 0,
                "excluded_counts": {"EXPERIENCE_MEMORY_UNAVAILABLE": len(args.experience_id)},
            }
        return OutcomeStatsCalculator(catalog).calculate(
            OutcomeStatsRequest(
                tuple(args.experience_id),
                args.basis,
                args.horizon_seconds,
                trust_tiers=_trust(args.trust_tiers, (TrustTier.TIER_A_HIGH_TRUST,)),
                as_of=_as_of(args.as_of),
            )
        )
    if command == "evidence":
        state = (
            json.loads(Path(args.market_state_json).read_text(encoding="utf-8"))
            if args.market_state_json
            else None
        )
        service = _similar_service(args) if state is not None else ExperienceQueryService(())
        knowledge = (
            _build_knowledge_service(_root(args.knowledge_artifact_root))
            if args.question and args.knowledge_artifact_root
            else None
        )
        bundle = EvidenceOrchestrator(
            knowledge,
            service,
            OutcomeStatsCalculator(catalog) if catalog else OutcomeStatsCalculator(()),
        ).query(
            EvidenceRequest(
                research_question=args.question,
                market_state=state,
                experience_top_k=args.experience_top_k,
                knowledge_top_k=args.knowledge_top_k,
                symbol=args.symbol,
                analysis_profile=args.analysis_profile,
                analysis_timeframe=args.analysis_timeframe,
                evaluation_basis=args.basis,
                horizon_seconds=args.horizon_seconds,
                as_of=_as_of(args.as_of),
                trust_tiers=_trust(args.trust_tiers),
            )
        )
        return bundle
    raise ValueError(f"unsupported command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        _emit(_run(args))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": {"type": type(exc).__name__, "message": str(exc)[:500]}},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2 if isinstance(exc, (ValueError, argparse.ArgumentError)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
