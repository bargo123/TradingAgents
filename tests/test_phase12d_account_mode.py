from types import SimpleNamespace

from tradingagents.dataflows.mt5.models import Mt5AccountInfo
from tradingagents.dataflows.mt5.provider import MT5Provider


def test_mt5_account_info_preserves_authoritative_trade_mode():
    account = Mt5AccountInfo(login=7, server="Broker-Demo", currency="USD", trade_mode=0)

    assert account.trade_mode == 0


def test_mt5_provider_maps_account_trade_mode():
    class FakeApi:
        def terminal_info(self):
            return SimpleNamespace(connected=True)

        def account_info(self):
            return SimpleNamespace(
                login=7,
                server="Broker-Demo",
                currency="USD",
                trade_mode=0,
                balance=1000.0,
                equity=1000.0,
                profit=0.0,
                margin=0.0,
                margin_free=1000.0,
                leverage=100,
            )

    provider = MT5Provider(api=FakeApi())
    provider._initialized = True

    assert provider.get_account_info().trade_mode == 0
