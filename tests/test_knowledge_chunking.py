from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tradingagents.knowledge.models import (
    ContentType,
    DocumentMetadata,
    EmbeddingSpec,
    EquationMetadata,
    ParsedBlock,
    ParsedDocument,
    TableMetadata,
)


@dataclass
class FakeTokenizer:
    model_max_input_tokens: int
    special_token_budget: int
    calls: list[dict[str, object]] = field(default_factory=list)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        purpose: str,
        truncation: bool,
    ) -> tuple[str, ...]:
        self.calls.append(
            {
                "add_special_tokens": add_special_tokens,
                "purpose": purpose,
                "truncation": truncation,
            }
        )
        tokens = tuple(text.split())
        if add_special_tokens:
            tokens = ("<s>",) * self.special_token_budget + tokens
        return tokens

    def decode(self, tokens: tuple[str, ...] | list[str]) -> str:
        return " ".join(token for token in tokens if token != "<s>")


def make_embedding_spec(**overrides: object) -> EmbeddingSpec:
    values: dict[str, object] = {
        "model_id": "BAAI/bge-small-en-v1.5",
        "resolved_model_version": "fixture-v1",
        "artifact_hash": "fixture-artifact",
        "tokenizer_fingerprint": "fixture-tokenizer",
        "model_max_input_tokens": 512,
        "special_token_budget": 2,
        "effective_corpus_content_token_limit": 510,
        "truncation": False,
    }
    values.update(overrides)
    return EmbeddingSpec(**values)


def make_document(blocks: tuple[ParsedBlock, ...]) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc_" + "ab" * 32,
        source_hash="ab" * 32,
        metadata=DocumentMetadata(
            title="Microstructure fixture",
            authors=("A. Author",),
            document_type="paper",
            format="pdf",
            source_filename="fixture.pdf",
            source_relative_path="fixture.pdf",
        ),
        blocks=blocks,
        parser_id="fixture-parser",
        parser_version="1.0",
        parser_config_hash="parser-config",
    )


def make_structured_document() -> ParsedDocument:
    return make_document(
        (
            ParsedBlock(
                block_id="intro",
                content_type=ContentType.PROSE,
                text="A short introduction.",
                reading_order=0,
                page_start=1,
                section_path=("Introduction",),
            ),
            ParsedBlock(
                block_id="equation",
                content_type=ContentType.EQUATION,
                text="p_t = m_t + lambda I_t",
                reading_order=1,
                page_start=4,
                section_path=("Results", "Prediction"),
                equation=EquationMetadata(
                    latex=r"p_t = m_t + \lambda I_t",
                    variable_definitions=("I_t is inventory imbalance.",),
                ),
            ),
            ParsedBlock(
                block_id="table",
                content_type=ContentType.TABLE,
                text="Horizon Accuracy 1 0.71",
                reading_order=2,
                page_start=4,
                section_path=("Results", "Prediction"),
                table=TableMetadata(
                    caption="Prediction accuracy",
                    headers=("Horizon", "Accuracy"),
                    cells=(("1", "0.71"),),
                ),
            ),
        )
    )


def make_long_section_document(tokens: int = 900, *, indivisible: bool = False) -> ParsedDocument:
    text = " ".join(f"t{index}" for index in range(tokens))
    if indivisible:
        return make_document(
            (
                ParsedBlock(
                    block_id="too-large-table",
                    content_type=ContentType.TABLE,
                    text=text,
                    reading_order=0,
                    section_path=("Methods",),
                    table=TableMetadata(
                        caption="Indivisible",
                        headers=("Only column",),
                        cells=((text,),),
                    ),
                ),
            )
        )
    return make_document(
        (
            ParsedBlock(
                block_id="methods",
                content_type=ContentType.PROSE,
                text=text,
                reading_order=0,
                page_start=7,
                section_path=("Methods",),
            ),
        )
    )


def make_large_table_document() -> ParsedDocument:
    return make_document(
        (
            ParsedBlock(
                block_id="large-table",
                content_type=ContentType.TABLE,
                text="large table",
                reading_order=0,
                section_path=("Results",),
                table=TableMetadata(
                    caption="Prediction accuracy",
                    headers=("Horizon", "Accuracy"),
                    cells=tuple((str(index), " ".join(["score"] * 80)) for index in range(12)),
                    notes=("Scores are out of sample.",),
                ),
            ),
        )
    )


def make_large_equation_definition_document() -> ParsedDocument:
    return make_document(
        (
            ParsedBlock(
                block_id="large-equation",
                content_type=ContentType.EQUATION,
                text="risk = inventory times volatility",
                reading_order=0,
                section_path=("Risk",),
                equation=EquationMetadata(
                    latex="risk = inventory * volatility",
                    variable_definitions=tuple(
                        f"variable_{index} means " + " ".join(["value"] * 35)
                        for index in range(4)
                    ),
                ),
            ),
        )
    )


def bge_chunker_fixture():
    from tradingagents.knowledge.chunking import ChunkPolicy, StructureAwareChunker

    spec = make_embedding_spec()
    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    return spec, tokenizer, StructureAwareChunker(ChunkPolicy.for_embedding(spec), tokenizer)


def max_overlap_tokens(chunks: tuple[object, ...]) -> int:
    texts = [item.text.split() for item in chunks]
    overlaps = []
    for previous, current in zip(texts, texts[1:], strict=False):
        maximum = min(len(previous), len(current))
        overlaps.append(
            max(
                (size for size in range(maximum + 1) if previous[-size:] == current[:size]),
                default=0,
            )
        )
    return max(overlaps, default=0)


def test_policy_for_bge_uses_real_model_limit_and_no_truncation():
    from tradingagents.knowledge.chunking import ChunkPolicy

    policy = ChunkPolicy.for_embedding(make_embedding_spec())

    assert policy.soft_token_target == 448
    assert policy.hard_content_token_limit == 510
    assert policy.overlap_tokens == 64
    assert policy.version == "structure-v2"


def test_policy_for_embedding_preserves_explicit_corpus_instruction_budget():
    from tradingagents.knowledge.chunking import ChunkPolicy

    policy = ChunkPolicy.for_embedding(
        make_embedding_spec(
            model_max_input_tokens=128,
            special_token_budget=2,
            effective_corpus_content_token_limit=120,
        )
    )

    assert policy.hard_content_token_limit == 120
    assert policy.corpus_instruction_tokens == 6


def test_equation_keeps_variable_definition_and_table_keeps_header():
    _, _, chunker = bge_chunker_fixture()
    chunks = chunker.chunk(make_structured_document())

    equation = next(item for item in chunks if item.content_type is ContentType.EQUATION)
    table = next(item for item in chunks if item.content_type is ContentType.TABLE)
    assert "inventory" in equation.text.lower()
    assert table.table_metadata["headers"] == ["Horizon", "Accuracy"]
    assert table.table_metadata["caption"] == "Prediction accuracy"


def test_equation_absorbs_adjacent_definition_block_in_the_same_section():
    _, _, chunker = bge_chunker_fixture()
    document = make_document(
        (
            ParsedBlock(
                block_id="equation",
                content_type=ContentType.EQUATION,
                text="risk = inventory * volatility",
                reading_order=0,
                section_path=("Risk",),
                equation=EquationMetadata(latex="risk = inventory * volatility"),
            ),
            ParsedBlock(
                block_id="definition",
                content_type=ContentType.DEFINITION,
                text="inventory is the signed position held by the market maker.",
                reading_order=1,
                section_path=("Risk",),
            ),
        )
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content_type is ContentType.EQUATION
    assert "signed position" in chunks[0].text
    assert chunks[0].equation_metadata["variable_definitions"]
    assert chunks[0].block_range == (0, 1)


def test_adjacent_prose_paragraphs_in_one_section_share_a_structural_chunk():
    _, _, chunker = bge_chunker_fixture()
    document = make_document(
        (
            ParsedBlock(
                block_id="p-1",
                content_type=ContentType.PROSE,
                text="First Methods paragraph.",
                reading_order=0,
                page_start=3,
                section_path=("Methods",),
            ),
            ParsedBlock(
                block_id="p-2",
                content_type=ContentType.PROSE,
                text="Second Methods paragraph.",
                reading_order=1,
                page_start=3,
                section_path=("Methods",),
            ),
        )
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].block_range == (0, 1)
    assert chunks[0].text == "First Methods paragraph.\n\nSecond Methods paragraph."


def test_adjacent_paragraphs_flush_before_the_next_whole_paragraph_exceeds_target():
    _, _, chunker = bge_chunker_fixture()
    first_paragraph = "P1_SENTINEL " + " ".join(f"first_{index}" for index in range(299))
    second_paragraph = "P2_SENTINEL " + " ".join(f"second_{index}" for index in range(299))
    document = make_document(
        (
            ParsedBlock(
                block_id="p-1",
                content_type=ContentType.PROSE,
                text=first_paragraph,
                reading_order=0,
                section_path=("Methods",),
            ),
            ParsedBlock(
                block_id="p-2",
                content_type=ContentType.PROSE,
                text=second_paragraph,
                reading_order=1,
                section_path=("Methods",),
            ),
        )
    )

    chunks = chunker.chunk(document)

    assert [item.block_range for item in chunks] == [(0, 0), (1, 1)]
    assert all(not ("P1_SENTINEL" in item.text and "P2_SENTINEL" in item.text) for item in chunks)
    assert [item.text for item in chunks] == [first_paragraph, second_paragraph]


def test_parser_marked_heading_is_not_emitted_but_body_keeps_its_section_path():
    _, _, chunker = bge_chunker_fixture()
    heading = ParsedBlock(
        block_id="heading",
        content_type=ContentType.PROSE,
        text="Methods",
        reading_order=0,
        section_path=("Methods",),
        metadata={"is_heading": True},
    )
    body = ParsedBlock(
        block_id="body",
        content_type=ContentType.PROSE,
        text="The body discusses inventory risk.",
        reading_order=1,
        section_path=("Methods",),
    )

    assert chunker.chunk(make_document((heading,))) == ()
    chunks = chunker.chunk(make_document((heading, body)))
    assert len(chunks) == 1
    assert chunks[0].text == body.text
    assert chunks[0].section_path == ("Methods",)
    assert chunks[0].block_range == (1, 1)


def test_equation_absorbs_immediately_preceding_and_trailing_definition_sequences():
    _, _, chunker = bge_chunker_fixture()
    document = make_document(
        (
            ParsedBlock(
                block_id="before-1",
                content_type=ContentType.DEFINITION,
                text="inventory is the signed position.",
                reading_order=0,
                section_path=("Risk",),
            ),
            ParsedBlock(
                block_id="before-2",
                content_type=ContentType.DEFINITION,
                text="volatility is the forecast scale.",
                reading_order=1,
                section_path=("Risk",),
            ),
            ParsedBlock(
                block_id="equation",
                content_type=ContentType.EQUATION,
                text="risk = inventory * volatility",
                reading_order=2,
                section_path=("Risk",),
                equation=EquationMetadata(latex="risk = inventory * volatility"),
            ),
            ParsedBlock(
                block_id="after",
                content_type=ContentType.DEFINITION,
                text="risk is marked in currency units.",
                reading_order=3,
                section_path=("Risk",),
            ),
        )
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content_type is ContentType.EQUATION
    assert chunks[0].block_range == (0, 3)
    assert all(marker in chunks[0].text for marker in ("signed position", "forecast scale", "currency units"))
    assert chunks[0].equation_metadata["variable_definitions"]


def test_definition_sequence_is_not_stolen_across_a_section_boundary():
    _, _, chunker = bge_chunker_fixture()
    other_section_definition = ParsedBlock(
        block_id="other-section",
        content_type=ContentType.DEFINITION,
        text="alpha is unrelated.",
        reading_order=0,
        section_path=("Other",),
    )
    equation = ParsedBlock(
        block_id="equation",
        content_type=ContentType.EQUATION,
        text="risk = inventory",
        reading_order=1,
        section_path=("Risk",),
        equation=EquationMetadata(latex="risk = inventory"),
    )
    chunks = chunker.chunk(make_document((other_section_definition, equation)))

    assert [item.content_type for item in chunks] == [ContentType.DEFINITION, ContentType.EQUATION]
    assert "alpha is unrelated" not in chunks[1].text
    assert chunks[1].block_range == (1, 1)


def test_definition_sequence_is_not_stolen_across_an_unrelated_block():
    _, _, chunker = bge_chunker_fixture()
    definition = ParsedBlock(
        block_id="definition",
        content_type=ContentType.DEFINITION,
        text="inventory is the signed position.",
        reading_order=0,
        section_path=("Risk",),
    )
    intervening_prose = ParsedBlock(
        block_id="intervening",
        content_type=ContentType.PROSE,
        text="This explanation separates the definition from the formula.",
        reading_order=1,
        section_path=("Risk",),
    )
    equation = ParsedBlock(
        block_id="equation",
        content_type=ContentType.EQUATION,
        text="risk = inventory",
        reading_order=2,
        section_path=("Risk",),
        equation=EquationMetadata(latex="risk = inventory"),
    )

    chunks = chunker.chunk(make_document((definition, intervening_prose, equation)))

    assert [item.content_type for item in chunks] == [
        ContentType.DEFINITION,
        ContentType.PROSE,
        ContentType.EQUATION,
    ]
    assert "signed position" not in chunks[-1].text
    assert chunks[-1].block_range == (2, 2)


def test_chunk_ids_and_order_are_repeatable_and_normalize_hash_input():
    from tradingagents.knowledge.chunking import deterministic_chunk_id

    _, _, chunker = bge_chunker_fixture()
    first = chunker.chunk(make_structured_document())
    second = chunker.chunk(make_structured_document())

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.chunk_ordinal for item in first] == list(range(len(first)))
    assert deterministic_chunk_id("doc", "hash", "v", ContentType.PROSE, (), (0, 0), 0, "caf\u00e9\r\n") == deterministic_chunk_id(
        "doc", "hash", "v", ContentType.PROSE, (), (0, 0), 0, "cafe\u0301\n"
    )


def test_long_prose_splits_only_inside_section_and_uses_bounded_overlap():
    from tradingagents.knowledge.chunking import ChunkPolicy, StructureAwareChunker

    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec())
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_long_section_document())

    assert all(item.embedding_input_tokens <= 512 for item in chunks)
    assert all(item.content_tokens <= 510 for item in chunks)
    assert max_overlap_tokens(chunks) <= 64
    assert all(item.section_path == ("Methods",) for item in chunks)
    assert all(call["add_special_tokens"] is True for call in tokenizer.calls)
    assert all(call["purpose"] == "corpus" for call in tokenizer.calls)
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_chunk_at_model_capacity_is_allowed_but_over_capacity_fails_without_truncation():
    from tradingagents.knowledge.chunking import (
        ChunkPolicy,
        ChunkTooLargeForEmbedding,
        StructureAwareChunker,
    )

    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec())
    allowed = make_long_section_document(tokens=510)
    assert StructureAwareChunker(policy, tokenizer).chunk(allowed)

    with pytest.raises(ChunkTooLargeForEmbedding):
        StructureAwareChunker(policy, tokenizer).chunk(make_long_section_document(tokens=511, indivisible=True))


def test_large_table_splits_by_complete_rows_and_repeats_context():
    from tradingagents.knowledge.chunking import ChunkPolicy, StructureAwareChunker

    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec())
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_large_table_document())

    assert len(chunks) > 1
    assert all(item.table_metadata["headers"] for item in chunks)
    assert all(item.table_metadata["caption"] == "Prediction accuracy" for item in chunks)
    assert all(item.table_metadata["row_group_is_complete"] for item in chunks)


def test_equation_definition_bundle_is_preserved_when_splitting():
    from tradingagents.knowledge.chunking import ChunkPolicy, StructureAwareChunker

    tokenizer = FakeTokenizer(model_max_input_tokens=128, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(
        make_embedding_spec(
            model_max_input_tokens=128,
            effective_corpus_content_token_limit=126,
        )
    )
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_large_equation_definition_document())

    assert len(chunks) > 1
    assert all(item.equation_metadata["variable_definitions"] for item in chunks)


def test_chunker_requires_explicit_policy_and_tokenizer():
    from tradingagents.knowledge.chunking import StructureAwareChunker

    with pytest.raises(TypeError):
        StructureAwareChunker()  # type: ignore[call-arg]
