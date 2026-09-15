"""Standalone, offline CLI for the Phase 11 fine-tuning boundary."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import EmptyEligibleSetError, Phase10InvalidError, SequenceTooLongError
from .models import Phase11Status, TrainingConfig, canonical_json
from .provenance import BaseModelRevisionUnpinnedError, ProvenanceError

_MAX_ITEMS = 32


def _emit(payload: dict[str, Any]) -> None:
    print(canonical_json(payload))


def _failure(status: Phase11Status | str, message: str) -> dict[str, Any]:
    return {"status": status.value if isinstance(status, Phase11Status) else status, "message": str(message)[:4096]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finetune", description="Offline Phase 11 LoRA/QLoRA commands")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate a Phase 10 generation and write SFT rows")
    prepare.add_argument("--generation", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--base-model")
    prepare.add_argument("--base-model-revision")
    prepare.add_argument("--tokenizer")
    prepare.add_argument("--template", choices=("native", "fallback-v1"), default="native")
    prepare.add_argument("--json", action="store_true")

    train = commands.add_parser("train", help="train an adapter from prepared SFT rows")
    train.add_argument("--prepared", required=True)
    train.add_argument("--base-model")
    train.add_argument("--base-model-revision")
    train.add_argument("--config", metavar="JSON")
    train.add_argument("--output-root", required=True)
    train.add_argument("--json", action="store_true")

    validate = commands.add_parser("validate", help="validate one immutable run")
    validate.add_argument("--run", required=True)
    validate.add_argument("--json", action="store_true")

    inspect = commands.add_parser("inspect", help="inspect generation metadata without ML imports")
    inspect.add_argument("--generation", required=True)
    inspect.add_argument("--json", action="store_true")

    status = commands.add_parser("status", help="show finetuning artifact metadata")
    status.add_argument("--output-root", required=True)
    status.add_argument("--json", action="store_true")
    return parser


def _inspect(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        generation_id = manifest.get("dataset_id", path.name)
        status = manifest.get("status", "UNKNOWN")
        counts = manifest.get("split_counts", {})
        payload = {"status": status, "generation_id": generation_id,
                   "train_count": counts.get("train", 0) if isinstance(counts, dict) else 0,
                   "validation_count": counts.get("validation", 0) if isinstance(counts, dict) else 0}
        return 0, payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return 1, _failure(Phase11Status.PHASE10_INVALID, f"invalid generation metadata: {type(exc).__name__}")


def _prepare(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    # Importing Phase10Generation is safe; it has no optional ML imports. This
    # validation deliberately precedes tokenizer/Transformers construction.
    from .phase10 import Phase10Generation

    try:
        generation = Phase10Generation.open(args.generation)
    except EmptyEligibleSetError:
        return 0, _failure(Phase11Status.EMPTY_ELIGIBLE_SET, "no eligible training rows") | {"generation": str(Path(args.generation).resolve())}
    except (Phase10InvalidError, OSError, ValueError, TypeError):
        return 1, _failure(Phase11Status.PHASE10_INVALID, "Phase 10 generation is invalid")

    try:
        from transformers import AutoTokenizer

        tokenizer_path = args.tokenizer or args.base_model
        if not tokenizer_path:
            return 1, _failure(Phase11Status.TOKENIZER_INVALID, "tokenizer or base model is required")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=args.base_model_revision,
            local_files_only=True,
        )
        from .formatting import SFTFormatter
        from .preparation import prepare_generation
        from .tokenization import TokenizationPolicy

        policy = TokenizationPolicy(tokenizer=tokenizer, max_length=4096, template=None if args.template == "native" else "fallback-v1")
        result = prepare_generation(generation, args.output_root, SFTFormatter(), policy)
        return 0, {"status": result.status, "output_root": str(result.output_root), "manifest": result.manifest}
    except (ImportError, ModuleNotFoundError) as exc:
        return 1, _failure(Phase11Status.TRAINING_DEPENDENCY_MISSING, f"optional dependency missing: {exc.name or 'transformers'}")
    except SequenceTooLongError as exc:
        return 1, _failure(Phase11Status.SEQUENCE_TOO_LONG, str(exc))
    except Exception as exc:
        return 1, _failure(Phase11Status.TOKENIZER_INVALID, f"preparation failed: {type(exc).__name__}")


def _config(path: str, base_model: str, revision: str | None = None) -> TrainingConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    value = dict(value)
    value["base_model"] = base_model
    if revision is not None:
        value["base_model_revision"] = revision
    from .models import LoraConfig, QloraConfig

    if isinstance(value.get("lora"), dict):
        value["lora"] = LoraConfig(**value["lora"])
    if isinstance(value.get("qlora"), dict):
        value["qlora"] = QloraConfig(**value["qlora"])
    return TrainingConfig(**value)


def _train(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if not args.base_model:
        return 1, _failure(Phase11Status.TRAINING_FAILED, "explicit base model is required")
    if not args.config:
        return 1, _failure(Phase11Status.TRAINING_FAILED, "explicit config is required")
    try:
        config = _config(args.config, args.base_model, args.base_model_revision)
        from .preparation import validate_prepared

        prepared_root = Path(args.prepared).resolve()
        folder = prepared_root / "prepared" if (prepared_root / "prepared").is_dir() else prepared_root
        prepared_manifest = validate_prepared(folder)

        from transformers import AutoTokenizer

        from .provenance import ModelProvenance
        from .tokenization import TokenizationPolicy
        from .training import train

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            revision=config.base_model_revision,
            local_files_only=True,
        )
        policy = TokenizationPolicy(tokenizer=tokenizer, max_length=config.max_sequence_length)
        provenance = ModelProvenance.inspect(
            args.base_model,
            revision=config.base_model_revision,
            tokenizer_path=args.base_model,
        )
        resolved_config = config.to_dict()
        resolved_config["model_provenance"] = provenance.to_dict()
        adapter_tmp = Path(tempfile.mkdtemp(prefix="phase11-adapter-"))
        try:
            result = train(
                args.prepared,
                tokenizer_policy=policy,
                config=config,
                output_dir=adapter_tmp,
                provenance=provenance,
            )
            payload = _failure(result.status, result.message)
            if result.metrics is not None:
                payload["metrics"] = result.metrics.to_dict()
            if result.status == Phase11Status.COMPLETE:
                from .artifacts import publish_run

                run_path = publish_run(
                    args.output_root,
                    config=resolved_config,
                    dataset=prepared_manifest,
                    metrics=result.metrics.to_dict() if result.metrics else {},
                    prepared={
                        name: folder / name
                        for name in ("manifest.json", "train.sft.jsonl", "validation.sft.jsonl")
                    },
                    adapter=adapter_tmp,
                    request={
                        "generation_id": prepared_manifest.get("generation_id"),
                        "formatter_version": prepared_manifest.get("formatter_version"),
                        "tokenization_policy_version": prepared_manifest.get("tokenization_policy_version"),
                        "base_model": provenance.fingerprint,
                        "config": resolved_config,
                    },
                )
                payload["run_path"] = str(run_path)
        finally:
            shutil.rmtree(adapter_tmp, ignore_errors=True)
        return (0 if result.status == Phase11Status.COMPLETE else 1), payload
    except (ImportError, ModuleNotFoundError) as exc:
        return 1, _failure(Phase11Status.TRAINING_DEPENDENCY_MISSING, f"optional dependency missing: {exc.name or 'training stack'}")
    except BaseModelRevisionUnpinnedError as exc:
        return 1, _failure(Phase11Status.BASE_MODEL_REVISION_UNPINNED, str(exc))
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ProvenanceError):
        return 1, _failure(Phase11Status.TRAINING_FAILED, "invalid training configuration or prepared data")
    except Exception as exc:
        return 1, _failure(Phase11Status.TRAINING_FAILED, f"training failed: {type(exc).__name__}")


def _validate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    try:
        from .validation import validate_run
        report = validate_run(args.run)
        payload = {"valid": report.valid, "status": report.status.value, "errors": list(report.errors[:_MAX_ITEMS]), "warnings": list(report.warnings[:_MAX_ITEMS])}
        return (0 if report.valid else 1), payload
    except Exception:
        return 1, _failure(Phase11Status.ADAPTER_INVALID, "run validation failed")


def _status(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    root = Path(args.output_root).resolve()
    runs = []
    if root.is_dir():
        for path in sorted(root.iterdir(), key=lambda p: p.name)[:_MAX_ITEMS]:
            if path.is_dir() and (path / "run_manifest.json").is_file():
                try:
                    data = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                    runs.append({"run_id": path.name, "status": data.get("status", "INVALID")})
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    runs.append({"run_id": path.name, "status": "INVALID"})
    return 0, {"status": "OK" if runs else "EMPTY", "output_root": str(root), "runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        handler = {"prepare": _prepare, "train": _train, "validate": _validate, "inspect": lambda a: _inspect(Path(a.generation)), "status": _status}[args.command]
        code, payload = handler(args)
    except SystemExit:
        raise
    except Exception as exc:
        code, payload = 1, _failure(Phase11Status.TRAINING_FAILED, f"command failed: {type(exc).__name__}")
    _emit(payload)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
