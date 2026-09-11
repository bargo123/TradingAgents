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
