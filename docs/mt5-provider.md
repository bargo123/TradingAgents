# Read-only MT5 provider

Install the optional MetaTrader 5 dependency and run the connection smoke check:

```powershell
pip install -e ".[mt5]"
python scripts/test_mt5_connection.py --symbol EURUSD
```

The provider captures normalized M1, M5, M15, and H1 candles, UTC tick/spread
data, account information, and positions for the selected symbol.

| Timeframe | MT5 series |
| --- | --- |
| M1 | 1-minute candles |
| M5 | 5-minute candles |
| M15 | 15-minute candles |
| H1 | 1-hour candles |

Symbol resolution first uses an exact case-insensitive match. If there is no
exact match, a single flexible prefix/suffix broker decoration around the
six-letter currency pair is accepted (for example `EURUSD.raw` or `mEURUSD`).
Multiple plausible variants are rejected as ambiguous; no match is reported as
not found.

Initialization reports `MT5 initialization failed`, `MT5 terminal is not
connected`, or `MT5 account is not connected`, including the terminal's error
detail when available. A non-default terminal can be selected with:

```powershell
python scripts/test_mt5_connection.py --symbol EURUSD --terminal-path "C:\\Path\\terminal64.exe"
```

The CLI accepts `--count N` (which must be positive). It accepts no credentials
and always shuts down the provider after the check. For the optional live
integration test, set the guard and run the dedicated integration test:

```powershell
$env:RUN_MT5_INTEGRATION = "1"
pytest tests/test_mt5_integration.py -m integration
```

**Phase 3 is read-only and does not register LangGraph tools.**
