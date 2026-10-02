"""Read-only bridge from the frozen Phase 8 catalog into Phase 14 findings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import FindingKind, LearningFinding
from .store import SelfEnhancementStore


def import_phase8_observations(phase8_root: str | Path, store: SelfEnhancementStore) -> int:
    """Copy only auditable Phase 8 source observations into isolated findings.

    The Phase 8 catalog is opened with SQLite ``mode=ro``.  No outcome text,
    vectors, or trust semantics are merged into the Phase 14 trade catalog.
    """

    root = Path(phase8_root).expanduser().resolve()
    catalog = root / "catalog.sqlite3"
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    imported = 0
    with sqlite3.connect(f"file:{catalog.as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"experience_records"}
        if not required.issubset(tables):
            raise ValueError("Phase 8 catalog is missing experience_records")
        rows = db.execute(
            "SELECT experience_id,source_database_id,source_decision_id,source_decision_fingerprint,symbol,trust,tombstoned FROM experience_records ORDER BY experience_id"
        ).fetchall()
        for row in rows:
            if int(row["tombstoned"] or 0) != 0:
                continue
            if str(row["trust"]) not in {"TIER_A_HIGH_TRUST", "TIER_B_LIMITED"}:
                continue
            finding = LearningFinding(
                finding_id=f"phase8-observation-{row['experience_id']}",
                kind=FindingKind.OBSERVATION,
                statement=f"Phase 8 verified experience observation for {row['symbol']}",
                evidence_ids=(str(row["experience_id"]),),
                provenance={
                    "source": "PHASE8_EXPERIENCE_MEMORY",
                    "source_database_id": str(row["source_database_id"]),
                    "source_decision_id": str(row["source_decision_id"]),
                    "source_decision_fingerprint": str(row["source_decision_fingerprint"]),
                    "trust": str(row["trust"]),
                },
            )
            store.record_finding(finding)
            imported += 1
    return imported


__all__ = ["import_phase8_observations"]
