"""Point-in-time, deterministic projection of Phase 10 rows for SFT."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ContractError
from .models import SFT_FORMAT_VERSION, SFTExample, canonical_json

_ACTION = {"BUY", "SELL", "HOLD"}
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SENSITIVE = re.compile(r"(?:secret|password|credential|api[_ -]?key|access[_ -]?token|private[_ -]?key|prompt|completion|reasoning|chain[_ -]?of[_ -]?thought|\bcot\b|scratch)", re.I)
_NON_MODEL = re.compile(
    r"(?:^|_)(?:outcome|evaluation|future|objective|target|observation|entry|exit|pnl|profit|roi|reward|label)(?:$|_)",
    re.I,
)

_DECISION = frozenset({
    "action", "resolved_symbol", "requested_symbol", "analysis_profile", "analysis_timeframe",
    "decision_context_status", "normalization_status", "analysis_snapshot_timestamp",
    "snapshot_timestamp", "decision_completed_timestamp", "valid_for_seconds", "valid_until",
})
_MARKET = frozenset({"symbol", "point", "digits", "snapshot_timestamp", "analysis_snapshot_timestamp", "quote", "snapshot", "features", "bars", "indicators", "current_state"})
_RESEARCH = frozenset({
    "context_integrity", "evidence_use_status", "evidence_audit_status",
    "bundle_status", "integration_status", "integration",
    "evidence_refs", "evidence_refs_used", "refs_used", "used_evidence",
    "available_evidence", "evidence_refs_rejected", "refs_rejected",
    "selected_counts", "dropped_counts", "tables", "equations", "definitions",
    "figure_captions", "research_report", "risk", "current_state",
    "knowledge_query_fingerprint", "query_policy_version",
    "knowledge_generation_id", "experience_generation_id",
})
_PROVENANCE = frozenset({"source_id", "canonical_path", "schema_fingerprint", "file_sha256", "snapshot_fingerprint", "contract_version", "phase56", "phase8", "phase9"})


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_source(value: Any, *, key: str = "", reject_non_model: bool = True) -> None:
    if key and _SENSITIVE.search(key):
        raise ContractError(f"forbidden sensitive field: {key}")
    if reject_non_model and key and _NON_MODEL.search(key):
        raise ContractError(f"future/outcome field is not model input: {key}")
    if isinstance(value, Mapping):
        for name, child in value.items():
            _safe_source(child, key=str(name), reject_non_model=reject_non_model)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _safe_source(child, reject_non_model=reject_non_model)
    elif isinstance(value, str):
        if _SENSITIVE.search(value):
            raise ContractError("forbidden sensitive value")
        if len(value) > 256 * 1024:
            raise ContractError("source text exceeds bound")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ContractError("non-finite source value")


def _project_nested(value: Any, *, depth: int = 0) -> Any:
    """Copy bounded JSON-like state while rejecting future/sensitive fields."""
    if depth > 6:
        raise ContractError("source nesting exceeds bound")
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and len(value) > 16 * 1024:
            raise ContractError("source text exceeds bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite source value")
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ContractError("source mapping exceeds bound")
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if _SENSITIVE.search(key):
                raise ContractError(f"forbidden sensitive field: {key}")
            if _NON_MODEL.search(key):
                raise ContractError(f"future/outcome field is not model input: {key}")
            result[key] = _project_nested(value[raw_key], depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ContractError("source sequence exceeds bound")
        return [_project_nested(item, depth=depth + 1) for item in value]
    raise ContractError(f"source value is not JSON-safe: {type(value).__name__}")


def _project(value: Mapping[str, Any], allowed: frozenset[str], *, reject_unknown: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("source section must be a mapping")
    for key, child in value.items():
        if _SENSITIVE.search(str(key)):
            raise ContractError(f"forbidden sensitive field: {key}")
        if reject_unknown and key not in allowed:
            raise ContractError(f"unknown source field: {key}")
        if key in allowed:
            _safe_source(child, key=str(key))
    return {
        str(key): _project_nested(value[key])
        for key in sorted(value, key=lambda item: str(item))
        if key in allowed
    }


def _refs(*sections: Mapping[str, Any]) -> list[str]:
    found: set[str] = set()
    for section in sections:
        for key in ("evidence_refs", "evidence_refs_used", "refs_used", "used_evidence", "available_evidence"):
            raw = section.get(key, ())
            if isinstance(raw, str):
                raw = [raw]
            if raw is None:
                continue
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise ContractError("evidence references must be a sequence")
            for ref in raw:
                if not isinstance(ref, str) or not _REF.fullmatch(ref):
                    raise ContractError("invalid evidence reference")
                found.add(ref)
    return sorted(found)


def _refs_from(section: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    found: set[str] = set()
    for key in keys:
        raw = section.get(key, ())
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple, set, frozenset)):
            raise ContractError("evidence references must be a sequence")
        for ref in raw:
            if not isinstance(ref, str) or not _REF.fullmatch(ref):
                raise ContractError("invalid evidence reference")
            found.add(ref)
    return sorted(found)


_EVIDENCE_STATUSES = {"USED", "NONE_RELEVANT", "UNAVAILABLE", "DISABLED", "INCOMPLETE", "INVALID"}
_REJECTION_REASONS = {
    "CONFLICTS_WITH_CURRENT_STATE", "LOW_RELEVANCE", "INSUFFICIENT_SAMPLE", "DIAGNOSTIC_ONLY", "REDUNDANT",
}


def _rejected(section: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = section.get("evidence_refs_rejected", section.get("refs_rejected", ()))
    if raw is None:
        raise ContractError("rejected evidence references must be a sequence")
    if not isinstance(raw, (list, tuple)):
        raise ContractError("rejected evidence references must be a sequence")
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"ref", "reason"}:
            raise ContractError("rejected evidence references need ref and reason")
        ref, reason = item["ref"], item["reason"]
        if not isinstance(ref, str) or not _REF.fullmatch(ref):
            raise ContractError("invalid rejected evidence reference")
        if not isinstance(reason, str) or reason not in _REJECTION_REASONS:
            raise ContractError("invalid rejected evidence reason")
        result.append({"ref": ref, "reason": reason})
    if len({item["ref"] for item in result}) != len(result):
        raise ContractError("rejected evidence references must be unique")
    return sorted(result, key=lambda item: item["ref"])


def parse_target(target: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse the closed assistant target schema without accepting extensions."""
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError as exc:
            raise ContractError("target is not valid JSON") from exc
    allowed = {"action", "evidence_refs", "evidence_use_status", "evidence_refs_rejected"}
    required = {"action", "evidence_refs"}
    if not isinstance(target, Mapping) or not required.issubset(target) or set(target) - allowed:
        raise ContractError("target contains unsupported or missing fields")
    action = target["action"]
    if not isinstance(action, str) or action not in _ACTION:
        raise ContractError("target action must be BUY, SELL, or HOLD")
    evidence = target["evidence_refs"]
    if not isinstance(evidence, (list, tuple)) or any(not isinstance(x, str) or not _REF.fullmatch(x) for x in evidence):
        raise ContractError("invalid evidence references")
    normalized = list(evidence)
    if normalized != sorted(set(normalized)):
        raise ContractError("evidence references must be unique and sorted")
    output: dict[str, Any] = {"action": action, "evidence_refs": normalized}
    status = target.get("evidence_use_status")
    if status is not None:
        if not isinstance(status, str) or status not in _EVIDENCE_STATUSES:
            raise ContractError("invalid evidence_use_status")
        output["evidence_use_status"] = status
        if status == "NONE_RELEVANT" and normalized:
            raise ContractError("NONE_RELEVANT cannot contain used evidence")
    if "evidence_refs_rejected" in target:
        rejected = _rejected(target)
        raw_rejected = target["evidence_refs_rejected"]
        if [item["ref"] for item in raw_rejected] != [item["ref"] for item in rejected]:
            raise ContractError("rejected evidence references must be sorted")
        if set(normalized) & {item["ref"] for item in rejected}:
            raise ContractError("evidence references cannot be both used and rejected")
        output["evidence_refs_rejected"] = rejected
    return output


@dataclass(frozen=True, slots=True)
class SFTFormatter:
    policy_version: str = SFT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != SFT_FORMAT_VERSION:
            raise ContractError("unsupported formatter policy version")

    def format(self, row: Any, split: str | None = None) -> SFTExample:
        example_id = _value(row, "example_id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ContractError("row example_id is required")
        split = split or _value(row, "split", "train")
        if split not in {"train", "validation"}:
            raise ContractError("invalid SFT split")
        decision = _value(row, "decision", {})
        market = _value(row, "market", {})
        research = _value(row, "research", {})
        provenance = _value(row, "provenance", {})
        # Trust and outcome are deliberately inspected only for forbidden keys,
        # then dropped. Their values can never reach either prompt or target.
        _safe_source(_value(row, "trust", {}), reject_non_model=False)
        _safe_source(_value(row, "outcome", {}), reject_non_model=False)
        d = _project(decision, _DECISION)
        m = _project(market, _MARKET)
        r = _project(research, _RESEARCH)
        p = _project(provenance, _PROVENANCE)
        action = d.get("action")
        if not isinstance(action, str) or action not in _ACTION:
            raise ContractError("source action must be BUY, SELL, or HOLD")
        used = _refs_from(r, ("evidence_refs_used", "refs_used", "used_evidence", "evidence_refs"))
        rejected = _rejected(r)
        target_payload: dict[str, Any] = {"action": action, "evidence_refs": used}
        status = r.get("evidence_use_status")
        if status is not None:
            if not isinstance(status, str) or status not in _EVIDENCE_STATUSES:
                raise ContractError("invalid evidence_use_status")
            target_payload["evidence_use_status"] = status
        if rejected:
            target_payload["evidence_refs_rejected"] = rejected
        target = parse_target(target_payload)
        user = {"decision": d, "market": m, "research": r, "provenance": p}
        messages = (
            {"role": "system", "content": "Use only the supplied point-in-time state and evidence references."},
            {"role": "user", "content": canonical_json(user)},
            {"role": "assistant", "content": canonical_json(target)},
        )
        return SFTExample(example_id=example_id, split=split, messages=tuple(messages), target=target)

    format_example = format


__all__ = ["SFTFormatter", "parse_target"]
