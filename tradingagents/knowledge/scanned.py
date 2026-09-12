"""Deterministic, OCR-free image-only document classification."""

from __future__ import annotations

from dataclasses import dataclass

from .models import IngestionState, ParsedBlock, ParsedDocument

_MIN_TEXT_CHARS = 64


@dataclass(frozen=True, slots=True)
class ScanDecision:
    """Scalar scan evidence only; it never carries or creates OCR text."""

    state: IngestionState
    document_id: str
    source_hash: str
    total_pages: int
    text_bearing_pages: int
    image_bearing_pages: int
    parser_id: str
    parser_version: str
    parser_config_hash: str
    scanner_version: str = "scan-v1"

    @property
    def total_items(self) -> int:
        """EPUB callers use the same scalar field for spine items."""

        return self.total_pages

    @property
    def text_bearing_items(self) -> int:
        return self.text_bearing_pages

    @property
    def image_bearing_items(self) -> int:
        return self.image_bearing_pages


class ScannedDetector:
    """Classify parser inventories using the fixed V1 64-character rule."""

    version = "scan-v1"
    minimum_text_chars = _MIN_TEXT_CHARS

    def classify(self, document: ParsedDocument) -> ScanDecision:
        text_counts, image_flags = self._inventory(document)
        total = max(len(text_counts), len(image_flags))
        if total == 0:
            return ScanDecision(
                state=IngestionState.PARSED,
                document_id=document.document_id,
                source_hash=document.source_hash,
                total_pages=0,
                text_bearing_pages=0,
                image_bearing_pages=0,
                parser_id=document.parser_id,
                parser_version=document.parser_version,
                parser_config_hash=document.parser_config_hash,
                scanner_version=self.version,
            )

        text_counts.extend([0] * (total - len(text_counts)))
        image_flags.extend([False] * (total - len(image_flags)))
        text_bearing = sum(count >= self.minimum_text_chars for count in text_counts)
        image_bearing = sum(image_flags)
        text_ratio = text_bearing / total
        image_ratio = image_bearing / total
        needs_ocr = (text_ratio < 0.20 and image_ratio >= 0.80) or (
            sum(text_counts) == 0 and image_bearing > 0
        )
        return ScanDecision(
            state=IngestionState.NEEDS_OCR if needs_ocr else IngestionState.PARSED,
            document_id=document.document_id,
            source_hash=document.source_hash,
            total_pages=total,
            text_bearing_pages=text_bearing,
            image_bearing_pages=image_bearing,
            parser_id=document.parser_id,
            parser_version=document.parser_version,
            parser_config_hash=document.parser_config_hash,
            scanner_version=self.version,
        )

    def _inventory(self, document: ParsedDocument) -> tuple[list[int], list[bool]]:
        provenance = document.parser_provenance
        text_key = "spine_text_chars" if document.format == "epub" else "page_text_chars"
        image_key = "image_spine_items" if document.format == "epub" else "image_pages"
        direct_text = provenance.get(text_key)
        direct_images = provenance.get(image_key)
        if isinstance(direct_text, (tuple, list)) or isinstance(direct_images, (tuple, list)):
            text_counts = [max(0, int(value)) for value in (direct_text or ())]
            image_flags = [bool(value) for value in (direct_images or ())]
            return text_counts, image_flags

        ordered_locations: list[object] = []
        text_counts_by_location: dict[object, int] = {}
        image_by_location: dict[object, bool] = {}
        for block in document.blocks:
            location = self._location(block, len(ordered_locations))
            if location not in text_counts_by_location:
                ordered_locations.append(location)
                text_counts_by_location[location] = 0
                image_by_location[location] = False
            text_counts_by_location[location] += self._block_text_chars(block)
            image_by_location[location] = image_by_location[location] or bool(
                block.metadata.get("image_bearing", False)
            )
        return (
            [text_counts_by_location[location] for location in ordered_locations],
            [image_by_location[location] for location in ordered_locations],
        )

    @staticmethod
    def _location(block: ParsedBlock, fallback: int) -> object:
        if block.page_start is not None:
            return ("page", block.page_start)
        if block.epub_spine_item:
            return ("spine", block.epub_spine_item)
        return ("item", fallback)

    @staticmethod
    def _block_text_chars(block: ParsedBlock) -> int:
        text = block.text
        has_structured_text = False
        if block.equation is not None:
            equation_text = " ".join(
                value
                for value in (
                    block.equation.latex,
                    block.equation.mathml,
                    block.equation.parser_native,
                    block.equation.plain_text,
                )
                if value
            )
            text += equation_text
            has_structured_text = bool(equation_text.strip())
        if block.table is not None:
            table_text = " ".join(block.table.headers)
            table_text += " ".join(cell for row in block.table.cells for cell in row)
            text += table_text
            has_structured_text = has_structured_text or bool(table_text.strip())
        count = len("".join(text.split()))
        # The V1 contract treats a real equation/table text block as textual
        # evidence even when its rendered form is shorter than 64 characters.
        return max(count, _MIN_TEXT_CHARS) if has_structured_text else count


__all__ = ["ScanDecision", "ScannedDetector"]
