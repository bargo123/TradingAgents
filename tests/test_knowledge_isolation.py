"""Static import and packaging boundaries for the local knowledge package."""

from __future__ import annotations

from pathlib import Path

from tradingagents.knowledge.quality import imported_modules_under

ROOT = Path(__file__).resolve().parents[1]


def test_knowledge_package_has_no_trading_or_external_experience_imports():
    """Would fail if knowledge begins depending on execution, agents, or experience stores."""
    forbidden = {
        "tradingagents.forex",
        "tradingagents.graph",
        "tradingagents.agents",
        "tradingagents.experience",
        "tradingagents.memory",
        "tradingagents.stock",
        "cli.forex_watch",
        "cli.forex_shadow",
        "cli.forex_evaluate",
    }

    assert forbidden.isdisjoint(imported_modules_under(ROOT / "tradingagents" / "knowledge"))


def test_isolation_scanner_resolves_relative_imports_against_the_scanned_package(tmp_path):
    """Would fail if a relative import could hide a forbidden parent package."""
    package = tmp_path / "tradingagents"
    knowledge = package / "knowledge"
    nested = knowledge / "nested"
    nested.mkdir(parents=True)
    for directory in (package, knowledge, nested):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (knowledge / "relative_forex.py").write_text(
        "from ..forex import broker\n", encoding="utf-8"
    )
    (nested / "relative_graph.py").write_text(
        "from ...graph import workflow\n", encoding="utf-8"
    )
    (nested / "relative_experience.py").write_text(
        "from ...experience import store\n", encoding="utf-8"
    )

    imports = imported_modules_under(knowledge)

    assert {"tradingagents.forex", "tradingagents.graph", "tradingagents.experience"}.issubset(
        imports
    )


def test_knowledge_package_has_no_network_client_imports():
    """Would fail if ordinary knowledge imports gained a direct network client."""
    forbidden_roots = {
        "requests",
        "httpx",
        "urllib",
        "socket",
        "aiohttp",
        "websockets",
        "MetaTrader5",
    }
    imports = imported_modules_under(ROOT / "tradingagents" / "knowledge")

    assert forbidden_roots.isdisjoint({name.split(".", 1)[0] for name in imports})


def test_knowledge_entrypoint_stays_separate_from_stock_and_forex_paths():
    """Would fail if the knowledge console command were wired into the stock/forex entry points."""
    imports = imported_modules_under(ROOT / "tradingagents" / "knowledge")
    entrypoints = imported_modules_under(ROOT / "tradingagents" / "knowledge", ROOT / "pyproject.toml")

    assert all(not name.startswith("cli.") for name in imports)
    assert {"tradingagents.knowledge.cli", "cli.main"}.issubset(entrypoints)
