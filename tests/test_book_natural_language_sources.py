"""Genuine source fixtures are addresses, never synthetic trading evidence."""
import json
import sqlite3
from pathlib import Path

import pytest

CASES_PATH = Path(__file__).parent / "fixtures/book_natural_language_source_cases.json"
CATALOG = Path("C:/p7fast/catalog.sqlite3")


def source_cases():
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_source_cases_have_exact_offsets_and_hashes():
    cases = source_cases()
    assert len({case["case_id"] for case in cases}) == len(cases)
    if not CATALOG.exists():
        pytest.skip("pinned Phase 7 catalog absent")
    with sqlite3.connect(CATALOG.as_uri() + "?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        for case in cases:
            row = con.execute(
                "SELECT document_id, source_hash, projection_generation, provenance_json "
                "FROM knowledge_chunks WHERE chunk_id=? AND active=1",
                (case["evidence"]["chunk_id"],),
            ).fetchone()
            e = case["evidence"]
            assert row is not None
            assert row[:3] == (e["document_id"], e["source_hash"], e["generation_id"])
            text = json.loads(row[3])["text"]
            assert text[e["start_offset"]:e["end_offset"]] == e["quote"]


def test_source_cases_separate_support_from_execution():
    cases = source_cases()
    assert {c["case_id"] for c in cases} >= {"target_10_pip", "stop_limit_20_pips", "chart_bar_entry", "development_stop_limit"}
    assert all(c["expected_status"] != "EXECUTABLE_ELIGIBLE" for c in cases)
    assert next(c for c in cases if c["case_id"] == "chart_bar_entry")["expected_status"] == "REJECTED"
