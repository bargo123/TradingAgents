from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import EmptyEligibleSetError, Phase10InvalidError
from .formatting import SFTFormatter
from .models import SFT_FORMAT_VERSION, canonical_json
from .phase10 import Phase10Generation
from .tokenization import TokenizationPolicy
from .tokenization import tokenize_example


@dataclass(frozen=True, slots=True)
class PreparationResult:
    status: str
    manifest: dict[str, Any]
    output_root: Path


def _jsonl(rows) -> bytes:
    return b"".join((canonical_json(r) + "\n").encode("utf-8") for r in rows)


def _row(ex):
    return {"example_id": ex.example_id, "split": ex.split, "messages": [dict(m) for m in ex.messages], "target": dict(ex.target), "version": ex.version}


def prepare_generation(generation, output_root, formatter: SFTFormatter, tokenizer_policy: TokenizationPolicy) -> PreparationResult:
    try:
        gen = generation if isinstance(generation, Phase10Generation) else Phase10Generation.open(generation)
    except EmptyEligibleSetError:
        return PreparationResult("EMPTY_ELIGIBLE_SET", {"status": "EMPTY_ELIGIBLE_SET"}, Path(output_root))
    except Exception as exc:
        if isinstance(exc, Phase10InvalidError):
            raise
        raise Phase10InvalidError(str(exc)) from exc
    out = Path(output_root).resolve(); prepared = out / "prepared"; prepared.mkdir(parents=True, exist_ok=True)
    def prepare(rows, split):
        output = []
        for source in sorted(rows, key=lambda r: r.example_id):
            item = formatter.format(source, split)
            tokenize_example(item, tokenizer_policy)
            output.append(_row(item))
        return output
    train = prepare(gen.train, "train")
    validation = prepare(gen.validation, "validation")
    files = {"train.sft.jsonl": _jsonl(train), "validation.sft.jsonl": _jsonl(validation)}
    manifest = {
        "status": "PREPARED", "generation_id": gen.generation_id,
        "source_fingerprints": gen.source_fingerprints,
        "formatter_version": formatter.policy_version, "format_version": SFT_FORMAT_VERSION,
        "tokenization_policy_version": tokenizer_policy.policy_version,
        "tokenizer": tokenizer_policy.fingerprint(),
        "train_count": len(train), "validation_count": len(validation),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for name, data in files.items()},
    }
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    files["manifest.json"] = manifest_bytes
    stage = Path(tempfile.mkdtemp(prefix=".prepared-staging-", dir=str(out)))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        for name in ("train.sft.jsonl", "validation.sft.jsonl", "manifest.json"):
            os.replace(stage / name, prepared / name)
    finally:
        try: stage.rmdir()
        except OSError: pass
    return PreparationResult("PREPARED", manifest, out)


__all__ = ["PreparationResult", "prepare_generation"]
