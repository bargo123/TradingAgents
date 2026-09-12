from __future__ import annotations

import json


def _fail_if_constructed(*args, **kwargs):
    raise AssertionError("forbidden Phase 7 writer/parser/embedder was constructed")


def test_parser_exposes_all_experience_commands():
    from tradingagents.experience.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "import",
        "rebuild",
        "status",
        "list",
        "show",
        "similar",
        "stats",
        "quarantine",
        "evidence",
    ):
        assert command in help_text


def test_status_emits_stable_json_without_constructing_phase7_components(
    monkeypatch, tmp_path, capsys
):
    from tradingagents.experience import cli

    monkeypatch.setattr(
        "tradingagents.knowledge.docling_parser.DoclingDocumentParser", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.embeddings.FastEmbedProvider", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.vector_index.VectorIndexWriter", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.lexical_index.LexicalIndexWriter", _fail_if_constructed
    )

    assert cli.main(["status", "--artifact-root", str(tmp_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "active_generation": None,
        "artifact_root_initialized": False,
        "experience_counts": {"active": 0, "historical": 0},
        "quarantine_count": 0,
    }
    assert not (tmp_path / "catalog.sqlite3").exists()


def test_similar_market_state_does_not_construct_embedder(monkeypatch, tmp_path, capsys):
    from tradingagents.experience import cli

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"values": [1.0], "mask": [True]}), encoding="utf-8")
    monkeypatch.setattr(
        "tradingagents.knowledge.embeddings.FastEmbedProvider", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.docling_parser.DoclingDocumentParser", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.vector_index.VectorIndexWriter", _fail_if_constructed
    )
    monkeypatch.setattr(
        "tradingagents.knowledge.lexical_index.LexicalIndexWriter", _fail_if_constructed
    )
    monkeypatch.setattr(cli, "_similar_service", lambda args: _EmptyService())

    assert (
        cli.main(
            [
                "similar",
                "--market-state-json",
                str(state),
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["hits"] == []


class _EmptyService:
    def search(self, query):
        from tradingagents.experience.models import ExperienceSearchResult

        return ExperienceSearchResult()


def test_cli_help_has_no_trading_or_model_options():
    from tradingagents.experience.cli import build_parser

    text = build_parser().format_help()
    assert "--mt5" not in text
    assert "--model" not in text
    assert "--order" not in text


def test_evidence_question_uses_phase7_read_only_service_when_root_supplied(
    monkeypatch, tmp_path, capsys
):
    from tradingagents.experience import cli

    calls = []

    class KnowledgeStub:
        def search(self, request):
            calls.append(request.text)
            return ()

    monkeypatch.setattr(
        cli, "_build_knowledge_service", lambda root: (calls.append(root), KnowledgeStub())[1]
    )
    assert (
        cli.main(
            [
                "evidence",
                "--question",
                "order flow",
                "--knowledge-artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert calls == [tmp_path.resolve(), "order flow"]
    assert json.loads(capsys.readouterr().out)["status"] == "EMPTY"


def test_similar_omitted_trust_uses_tier_a_and_b():
    from tradingagents.experience import cli

    args = cli.build_parser().parse_args(["similar", "--market-state-json", "state.json"])
    query = cli._query(args, {"values": [1.0], "mask": [True]})
    assert query.trust_tiers == (cli.TrustTier.TIER_A_HIGH_TRUST, cli.TrustTier.TIER_B_LIMITED)


def test_stats_omitted_trust_uses_tier_a(monkeypatch, tmp_path):
    from tradingagents.experience import cli
    from tradingagents.experience.catalog import ExperienceCatalog

    ExperienceCatalog(tmp_path)
    captured = {}

    class Calculator:
        def __init__(self, catalog):
            pass

        def calculate(self, request):
            captured["request"] = request
            return {}

    monkeypatch.setattr(cli, "OutcomeStatsCalculator", Calculator)
    args = cli.build_parser().parse_args(
        [
            "stats",
            "--basis",
            "ANALYSIS_SNAPSHOT",
            "--horizon-seconds",
            "300",
            "--experience-id",
            "exp-1",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    cli._run(args)
    assert captured["request"].trust_tiers == (cli.TrustTier.TIER_A_HIGH_TRUST,)
