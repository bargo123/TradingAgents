from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scripts.phase11_real_empty_acceptance import run_real_empty_acceptance


def _enabled_real_generation() -> Path | None:
    if os.environ.get("PHASE11_REAL_ACCEPTANCE") != "1":
        return None
    configured = os.environ.get("PHASE11_REAL_GENERATION")
    if configured:
        return Path(configured)
    from scripts.phase11_real_empty_acceptance import default_generation_path

    candidate = default_generation_path()
    return candidate if candidate.is_dir() else None


def test_real_empty_generation_short_circuits_before_training(monkeypatch) -> None:
    generation = _enabled_real_generation()
    if generation is None:
        pytest.skip("opt in with PHASE11_REAL_ACCEPTANCE=1 and PHASE11_REAL_GENERATION")

    # Every heavyweight or external constructor is forbidden in this acceptance.
    def forbidden(*args, **kwargs):
        raise AssertionError("constructor called")
    import tradingagents.finetuning.training as training

    monkeypatch.setattr(training, "train", forbidden)
    monkeypatch.setattr(training, "_load_optional_stack", forbidden)
    before = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in generation.iterdir() if item.is_file()}
    result = run_real_empty_acceptance(generation)
    after = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in generation.iterdir() if item.is_file()}
    assert result["status"] == "EMPTY_ELIGIBLE_SET"
    assert result["training_status"] == "NO_TRAINING_ATTEMPTED"
    assert result["training_attempted"] is False
    assert result["source_fingerprint_phases"] == ["phase56", "phase8", "phase9"]
    assert before == after


def test_unavailable_real_generation_is_bounded_and_does_not_construct_models(monkeypatch, tmp_path: Path) -> None:
    import tradingagents.finetuning.training as training

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (_ for _ in ()).throw(AssertionError("ML stack loaded")))
    result = run_real_empty_acceptance(tmp_path / "missing-generation")
    assert result["status"] == "REAL_GENERATION_UNAVAILABLE"
    assert result["training_status"] == "NOT_RUN"
    assert result["training_attempted"] is False
