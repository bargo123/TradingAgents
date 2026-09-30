from tradingagents.dataflows.mt5.models import Mt5Position


def test_mt5_position_retains_ownership_metadata():
    position = Mt5Position(
        ticket=11,
        symbol="EURUSD",
        magic=12012012,
        comment="TradingAgents-P12D-DEMO",
    )

    assert position.magic == 12012012
    assert position.comment == "TradingAgents-P12D-DEMO"
