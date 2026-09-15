"""Small offline Phase 11A source and teacher fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.distillation.models import SourceBlock, SourceRef


@dataclass(frozen=True)
class FixtureSource:
    generation_id: str = "phase7-fixture-generation"
    source_fingerprints: dict[str, str] | None = None
    _blocks: tuple[SourceBlock, ...] = ()

    def __post_init__(self) -> None:
        if self.source_fingerprints is None:
            object.__setattr__(
                self,
                "source_fingerprints",
                {"catalog": "sha256:catalog-fixture", "generation": "sha256:generation-fixture"},
            )

    def blocks(self) -> tuple[SourceBlock, ...]:
        return self._blocks


def fixture_source() -> FixtureSource:
    rows: list[SourceBlock] = []
    entries = (
        ("ofi", "Order flow imbalance measures changes at the best bid and ask.", "DEFINITION"),
        ("liq", "Liquidity describes available depth and the cost of immediacy.", "PROSE"),
        (
            "impact",
            "Market impact increases when aggressive orders consume shallow depth.",
            "PROSE",
        ),
        (
            "inventory",
            "Inventory risk links position size, volatility, and adverse selection.",
            "PROSE",
        ),
        ("vol", "Volatility changes the distribution of short-horizon price moves.", "PROSE"),
        (
            "eq",
            "microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)",
            "EQUATION",
        ),
        ("table", "Horizon Accuracy 1 0.71", "TABLE"),
    )
    for index, (chunk_id, text, content_type) in enumerate(entries):
        document = f"doc-{index // 2 + 1}"
        section = ("Microstructure", chunk_id)
        ref = SourceRef(
            document_id=document,
            source_filename=f"fixture-{document}.pdf",
            source_hash=f"sha256:{document}-hash",
            chunk_id=chunk_id,
            generation_id="phase7-fixture-generation",
            page=index + 1,
            chapter="Chapter 1",
            section=" / ".join(section),
            content_type=content_type,
        )
        rows.append(
            SourceBlock(
                ref=ref,
                text=text,
                content_type=content_type,
                reading_order=index,
                section_path=section,
                metadata={
                    "equation_metadata": {"parser_native": text}
                    if content_type == "EQUATION"
                    else None,
                    "table_metadata": {"caption": "Prediction accuracy"}
                    if content_type == "TABLE"
                    else None,
                },
            )
        )
    return FixtureSource(_blocks=tuple(rows))


def grounded_candidate(
    packet, *, lesson_type: str = "DEFINITION", topic: str = "order flow imbalance"
) -> dict:
    ref = packet.blocks[0].ref
    return {
        "lesson_type": lesson_type,
        "topic": topic,
        "difficulty": "FOUNDATIONAL",
        "system_instruction": "Explain the supplied market-microstructure evidence.",
        "user_instruction": "What does this concept measure and what limitation should be remembered?",
        "assistant_target": "It describes a bounded market-microstructure relationship; interpretation depends on the stated assumptions and available depth.",
        "source_refs": [ref.chunk_id],
        "grounding_claims": [
            {
                "claim": "The supplied source defines the concept and its scope.",
                "source_refs": [ref.chunk_id],
            }
        ],
    }
