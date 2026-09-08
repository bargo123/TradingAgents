"""Read-only MT5 connection and market-data smoke check."""

from __future__ import annotations

import argparse
import sys

from tradingagents.dataflows.mt5.errors import Mt5ProviderError
from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar
from tradingagents.dataflows.mt5.provider import MT5Provider


def _format_bar(bar: Mt5Bar) -> str:
    return (
        f"{bar.timestamp.isoformat()} O={bar.open:.5f} H={bar.high:.5f} "
        f"L={bar.low:.5f} C={bar.close:.5f} V={bar.tick_volume}"
    )


def render_report(
    snapshot: ForexMarketSnapshot,
    terminal_name: str | None = None,
    requested_symbol: str | None = None,
) -> str:
    """Render a deterministic, credential-free human-readable report."""
    account = snapshot.account
    lines = [
        "MT5 CONNECTED",
        f"Terminal: {terminal_name or 'Unknown'} (connected)",
        f"Account server: {account.server if account else 'None'}",
        f"Currency: {account.currency if account else 'None'}",
        f"Balance: {account.balance if account else 'None'}",
        f"Equity: {account.equity if account else 'None'}",
        f"Free margin: {account.free_margin if account else 'None'}",
        f"Requested symbol: {requested_symbol or snapshot.symbol}",
        f"Resolved symbol: {snapshot.symbol}",
        f"Bid: {snapshot.bid:g}",
        f"Ask: {snapshot.ask:g}",
        f"Spread: {snapshot.spread:g}",
        f"Spread points: {snapshot.spread_points:g}",
        f"Last candle: {_format_bar(snapshot.m5_candles[-1])}",
        "Open positions:",
    ]
    if snapshot.positions:
        lines.extend(
            f"  {position.ticket} {position.symbol} volume={position.volume} profit={position.profit}"
            for position in snapshot.positions
        )
    else:
        lines.append("  None")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--terminal-path", default=None)
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be greater than zero")
    provider = MT5Provider(terminal_path=args.terminal_path)
    try:
        provider.initialize()
        snapshot = provider.get_market_snapshot(args.symbol, count=args.count)
        terminal = provider.get_terminal_info()
        print(render_report(snapshot, terminal_name=terminal.name, requested_symbol=args.symbol))
        return 0
    except Mt5ProviderError as exc:
        print(f"MT5 ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        provider.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
