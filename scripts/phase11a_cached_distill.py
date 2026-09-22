from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tradingagents.distillation.cli import _load_plan, _revalidate_plan
from tradingagents.distillation.factory import DistillationFactory
from tradingagents.distillation.models import canonical_hash, canonical_json
from tradingagents.distillation.teacher import (
    TEACHER_TRANSPORT_SCHEMA_VERSION,
    TeacherResult,
    teacher_from_environment,
)

CACHE_VERSION = "phase11a-teacher-cache.v1"


class CachedTeacher:
    def __init__(self, inner, cache_root: Path):
        self.inner = inner
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.provider = getattr(inner, "provider", "")
        self.model = getattr(inner, "model", "")
        self.config = getattr(inner, "config", None)

        self.cache_hits = 0
        self.live_calls = 0

    def _path(self, packet) -> Path:
        return self.cache_root / f"{packet.packet_id}.json"

    def _teacher_signature(self) -> dict:
        config = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": getattr(self.inner, "endpoint", ""),
            "max_retries": getattr(self.inner, "max_retries", None),
            "config": config,
            "transport_schema_version": TEACHER_TRANSPORT_SCHEMA_VERSION,
        }

    def _binding(self, packet) -> dict:
        teacher_signature = self._teacher_signature()
        return {
            "cache_version": CACHE_VERSION,
            "packet_id": packet.packet_id,
            "packet_fingerprint": canonical_hash(packet),
            "teacher_signature": teacher_signature,
            "teacher_signature_fingerprint": canonical_hash(teacher_signature),
        }

    def generate(self, packet, config):
        path = self._path(packet)
        expected = self._binding(packet)

        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))

            for key in (
                "cache_version",
                "packet_id",
                "packet_fingerprint",
                "teacher_signature_fingerprint",
            ):
                if value.get(key) != expected.get(key):
                    raise RuntimeError(f"CACHE_BINDING_MISMATCH:{key}")

            if value.get("teacher_signature") != expected["teacher_signature"]:
                raise RuntimeError("CACHE_BINDING_MISMATCH:teacher_signature")

            self.cache_hits += 1
            return TeacherResult(
                candidate=value.get("candidate"),
                provider=value.get("provider", self.provider),
                model=value.get("model", self.model),
                error_code=value.get("error_code"),
                diagnostics={"cache_hit": True},
            )

        result = self.inner.generate(packet, config)
        self.live_calls += 1

        # Transient provider failures must be retried on resume.
        if result.error_code == "TEACHER_FAILED":
            return result

        payload = {
            **expected,
            "provider": result.provider,
            "model": result.model,
            "error_code": result.error_code,
            "candidate": result.candidate,
        }

        temp = path.with_suffix(".tmp")
        temp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temp, path)

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()

    plan = _revalidate_plan(_load_plan(Path(args.plan)))

    teacher = teacher_from_environment()
    if teacher.__class__.__name__ == "UnconfiguredTeacher":
        raise RuntimeError("Phase 11A teacher is not configured")

    cached = CachedTeacher(teacher, Path(args.cache_root))

    try:
        report = DistillationFactory().distill(
            plan,
            cached,
            Path(args.output_root),
        )
    finally:
        print(
            json.dumps(
                {
                    "cache_hits": cached.cache_hits,
                    "live_teacher_calls": cached.live_calls,
                    "cached_packets": len(list(Path(args.cache_root).glob("*.json"))),
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "generation": str(report.generation),
                "accepted": report.accepted,
                "excluded": report.excluded,
                "teacher_calls": report.teacher_calls,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
