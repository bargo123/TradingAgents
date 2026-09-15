"""Bounded, read-only acceptance helper for the published real Phase 10 empty set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Keep ``python scripts/phase11_real_empty_acceptance.py`` equivalent to the
# installed/module entry point without importing any optional training stack.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.datasets.writer import validate_generation
from tradingagents.finetuning.formatting import SFTFormatter
from tradingagents.finetuning.preparation import prepare_generation
from tradingagents.finetuning.tokenization import TokenizationPolicy

GENERATION_ID = "generation-d78d143c760b021f48200255"
_FINGERPRINT_FIELDS = {
    "source_id",
    "canonical_path",
    "schema_fingerprint",
    "file_sha256",
    "snapshot_fingerprint",
    "contract_version",
}


def default_generation_path() -> Path:
    """Locate the known real generation in the checkout's sibling data cache."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "data_cache" / "phase10-real-acceptance-20260914" / GENERATION_ID,
        repo_root / "data_cache" / "phase10-real-acceptance-20260914-v2" / GENERATION_ID,
        repo_root.parent / "data_cache" / "phase10-real-acceptance-20260914" / GENERATION_ID,
        repo_root.parent / "data_cache" / "phase10-real-acceptance-20260914-v2" / GENERATION_ID,
        repo_root.parent.parent / "data_cache" / "phase10-real-acceptance-20260914" / GENERATION_ID,
        repo_root.parent.parent / "data_cache" / "phase10-real-acceptance-20260914-v2" / GENERATION_ID,
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def run_real_empty_acceptance(generation_path: str | Path | None = None) -> dict[str, Any]:
    """Validate the immutable generation and stop before tokenizer/model setup."""
    path = Path(generation_path) if generation_path is not None else default_generation_path()
    if not path.is_dir():
        return {
            "status": "REAL_GENERATION_UNAVAILABLE",
            "generation_id": GENERATION_ID,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "message": "known real Phase 10 generation is unavailable",
        }
    if path.name != GENERATION_ID:
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "message": "generation identity does not match the authoritative generation",
        }
    try:
        report = validate_generation(path)
    except Exception as exc:
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "message": f"generation validation failed: {type(exc).__name__}",
        }
    if not report.valid:
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "errors": list(report.errors)[:8],
        }
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "message": f"manifest read failed: {type(exc).__name__}",
        }
    fingerprints = manifest.get("source_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {"phase56", "phase8", "phase9"} or any(
        not isinstance(value, dict) or set(value) != _FINGERPRINT_FIELDS
        for value in fingerprints.values()
    ):
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": "NOT_RUN",
            "training_attempted": False,
            "message": "source fingerprints are incomplete",
        }
    # An empty generation is handled by preparation before TokenizationPolicy.load,
    # so this loader is deliberately an assertion rather than a fallback.
    policy = TokenizationPolicy(tokenizer_loader=lambda: (_ for _ in ()).throw(AssertionError("tokenizer constructed")))
    result = prepare_generation(path, path.parent / ".phase11-real-empty-no-write", SFTFormatter(), policy)
    if result.status != "EMPTY_ELIGIBLE_SET":
        return {
            "status": "PHASE10_INVALID",
            "generation_id": path.name,
            "training_status": result.status,
            "training_attempted": False,
            "message": "empty acceptance did not short-circuit preparation",
        }
    return {
        "status": "EMPTY_ELIGIBLE_SET",
        "generation_id": path.name,
        "training_status": "NO_TRAINING_ATTEMPTED",
        "training_attempted": False,
        "source_fingerprints": fingerprints,
        "source_fingerprint_phases": sorted(fingerprints),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_real_empty_acceptance(args.generation)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] in {"EMPTY_ELIGIBLE_SET", "REAL_GENERATION_UNAVAILABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
