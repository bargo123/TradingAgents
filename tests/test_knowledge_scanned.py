"""Deterministic scan-quarantine contract tests."""

from __future__ import annotations

from tradingagents.knowledge.models import (
    ContentType,
    DocumentMetadata,
    EquationMetadata,
    ParsedBlock,
    ParsedDocument,
)


def make_parsed_document(
    *, page_text_chars: tuple[int, ...], image_pages: tuple[bool, ...]
) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc_" + "11" * 32,
        source_hash="11" * 32,
        metadata=DocumentMetadata(format="pdf"),
        parser_id="fixture",
        parser_version="1",
        parser_config_hash="cfg",
        parser_provenance={
            "page_text_chars": page_text_chars,
            "image_pages": image_pages,
        },
    )


def test_scan_detector_marks_image_only_pdf_needs_ocr():
    from tradingagents.knowledge.models import IngestionState
    from tradingagents.knowledge.scanned import ScannedDetector

    decision = ScannedDetector().classify(
        make_parsed_document(page_text_chars=(0, 2, 0, 1), image_pages=(True, True, True, True))
    )

    assert decision.state is IngestionState.NEEDS_OCR
    assert decision.text_bearing_pages == 0
    assert decision.image_bearing_pages == 4


def test_scan_detector_requires_image_evidence_before_quarantining_empty_document():
    from tradingagents.knowledge.models import IngestionState
    from tradingagents.knowledge.scanned import ScannedDetector

    decision = ScannedDetector().classify(
        make_parsed_document(page_text_chars=(0, 0), image_pages=(False, False))
    )

    assert decision.state is IngestionState.PARSED


def test_nonempty_parser_native_equation_is_text_bearing_below_character_threshold():
    from tradingagents.knowledge.scanned import ScannedDetector

    document = ParsedDocument(
        document_id="doc_" + "22" * 32,
        source_hash="22" * 32,
        metadata=DocumentMetadata(format="pdf"),
        blocks=(
            ParsedBlock(
                block_id="equation",
                content_type=ContentType.EQUATION,
                equation=EquationMetadata(parser_native="x=y"),
                page_start=1,
            ),
        ),
        parser_id="fixture",
        parser_version="1",
        parser_config_hash="cfg",
    )

    decision = ScannedDetector().classify(document)

    assert decision.text_bearing_pages == 1
