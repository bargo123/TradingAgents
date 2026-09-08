"""Forex-safe wrapper for broad global news."""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.yfinance_news import forex_global_news_context

from .profile import MACRO_EVENT_UNAVAILABLE


def _unavailable_report(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    text = value.strip().casefold()
    return text.startswith((
        "no global news found",
        "error fetching global news",
        "data_unavailable:",
        "no_data_available:",
    ))


@tool("get_global_news")
def get_forex_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back"] = None,
    limit: Annotated[int | None, "Maximum number of articles"] = None,
) -> str:
    """Retrieve broad macro/global news without ticker or issuer context."""
    config = get_config()
    queries = config.get("forex_global_news_queries")
    with forex_global_news_context(queries):
        result = route_to_vendor("get_global_news", curr_date, look_back_days, limit)
    return MACRO_EVENT_UNAVAILABLE if _unavailable_report(result) else str(result)


__all__ = ["get_forex_global_news"]
