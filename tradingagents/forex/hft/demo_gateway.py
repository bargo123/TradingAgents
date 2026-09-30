"""The single broker-mutation boundary for Phase 12D DEMO execution."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

from .demo_models import DemoAccountSnapshot, DemoOrderIntent, DemoOrderResult
from .demo_store import DemoExecutionStore


class DemoExecutionForbidden(RuntimeError):
    """Raised whenever authoritative account verification is not DEMO."""

    code = "DEMO_EXECUTION_FORBIDDEN"

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"{self.code}: {self.reason}")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class VerifiedDemoExecutionGateway:
    """Validate the account and request before the only broker order call."""

    def __init__(
        self,
        provider: Any,
        broker_api: Any,
        store: DemoExecutionStore,
        *,
        run_id: str,
        demo_trade_mode: int,
        magic: int,
        comment: str = "TradingAgents-P12D-DEMO",
        volume_cap: float = 0.01,
        gate: Any | None = None,
    ) -> None:
        if not callable(getattr(provider, "get_account_info", None)):
            raise TypeError("provider must expose get_account_info")
        if not callable(getattr(provider, "get_terminal_info", None)):
            raise TypeError("provider must expose get_terminal_info")
        if not callable(getattr(broker_api, "order_send", None)):
            raise TypeError("broker API must expose order_send")
        if not isinstance(store, DemoExecutionStore):
            raise TypeError("store must be DemoExecutionStore")
        if isinstance(demo_trade_mode, bool) or not isinstance(demo_trade_mode, int) or demo_trade_mode < 0:
            raise ValueError("demo_trade_mode must be a non-negative integer")
        if isinstance(magic, bool) or not isinstance(magic, int) or magic <= 0:
            raise ValueError("magic must be a positive integer")
        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("comment must be non-empty")
        if float(volume_cap) <= 0:
            raise ValueError("volume_cap must be positive")
        self.provider = provider
        self.broker_api = broker_api
        self.store = store
        self.run_id = str(run_id)
        self.demo_trade_mode = demo_trade_mode
        self.magic = magic
        self.comment = comment.strip()[:31]
        self.volume_cap = float(volume_cap)
        self.gate = gate

    def _operation(self, name: str):
        if self.gate is None:
            return nullcontext()
        return self.gate.acquire(name)

    @staticmethod
    def read_account(provider: Any) -> DemoAccountSnapshot:
        """Read normalized account metadata without constructing a gateway."""

        if not callable(getattr(provider, "get_account_info", None)):
            raise DemoExecutionForbidden("account_info is unavailable")
        if not callable(getattr(provider, "get_terminal_info", None)):
            raise DemoExecutionForbidden("terminal_info is unavailable")
        account = provider.get_account_info()
        terminal = provider.get_terminal_info()
        if account is None or terminal is None:
            raise DemoExecutionForbidden("account_info or terminal_info unavailable")
        observed = datetime.now(timezone.utc)
        return DemoAccountSnapshot(
            login=_field(account, "login"),
            server=_field(account, "server"),
            company=_field(terminal, "company"),
            currency=_field(account, "currency"),
            trade_mode=_field(account, "trade_mode"),
            balance=_field(account, "balance"),
            equity=_field(account, "equity"),
            margin=_field(account, "margin"),
            free_margin=_field(account, "free_margin", _field(account, "margin_free")),
            margin_level=_field(account, "margin_level"),
            observed_at=observed,
        )

    @staticmethod
    def require_demo_account(account: DemoAccountSnapshot, demo_trade_mode: int) -> None:
        if not account.is_demo(demo_trade_mode):
            raise DemoExecutionForbidden(
                f"authoritative trade_mode={account.trade_mode!r} expected={demo_trade_mode}"
            )

    def _read_account(self) -> DemoAccountSnapshot:
        return self.read_account(self.provider)

    def _require_demo(self, account: DemoAccountSnapshot) -> None:
        self.require_demo_account(account, self.demo_trade_mode)

    def verify_demo_account(self) -> DemoAccountSnapshot:
        with self._operation("demo_account_info"):
            account = self._read_account()
        self._require_demo(account)
        return account

    def _request(self, intent: DemoOrderIntent, *, position_ticket: int | None = None) -> dict[str, Any]:
        order_type = getattr(
            self.broker_api,
            "ORDER_TYPE_BUY" if intent.direction == "LONG" else "ORDER_TYPE_SELL",
            None,
        )
        if order_type is None:
            raise DemoExecutionForbidden("broker BUY/SELL order type is unavailable")
        request: dict[str, Any] = {
            "action": getattr(self.broker_api, "TRADE_ACTION_DEAL", 1),
            "symbol": intent.symbol,
            "volume": intent.volume,
            "type": order_type,
            "price": intent.requested_price,
            "sl": intent.stop_loss,
            "tp": intent.take_profit,
            "deviation": intent.deviation_points,
            "magic": self.magic,
            "comment": self.comment,
            "type_time": getattr(self.broker_api, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(self.broker_api, "ORDER_FILLING_IOC", 0),
        }
        if position_ticket is not None:
            request["position"] = int(position_ticket)
        return request

    def _retcode(self, name: str, default: int) -> int:
        value = getattr(self.broker_api, name, default)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else default

    def _classify(self, retcode: int | None) -> str:
        if retcode is None:
            return "UNKNOWN"
        mapping = {
            self._retcode("TRADE_RETCODE_DONE", 10009): "FILLED",
            self._retcode("TRADE_RETCODE_DONE_PARTIAL", 10010): "PARTIAL",
            self._retcode("TRADE_RETCODE_REQUOTE", 10004): "REQUOTE",
            self._retcode("TRADE_RETCODE_PRICE_CHANGED", 10020): "PRICE_CHANGED",
            self._retcode("TRADE_RETCODE_MARKET_CLOSED", 10018): "MARKET_CLOSED",
            self._retcode("TRADE_RETCODE_NO_MONEY", 10019): "NO_MONEY",
            self._retcode("TRADE_RETCODE_INVALID_STOPS", 10016): "INVALID_STOPS",
            self._retcode("TRADE_RETCODE_CONNECTION", 10031): "CONNECTION_ERROR",
            self._retcode("TRADE_RETCODE_REJECT", 10006): "REJECTED",
            self._retcode("TRADE_RETCODE_ERROR", 10011): "REJECTED",
        }
        return mapping.get(retcode, "UNKNOWN")

    @staticmethod
    def _broker_timestamp(result: Any) -> datetime | None:
        raw = _field(result, "time_msc")
        if raw in (None, 0):
            raw = _field(result, "time")
            divisor = 1
        else:
            divisor = 1000
        if raw in (None, 0):
            return None
        try:
            return datetime.fromtimestamp(float(raw) / divisor, timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return None

    def _persist_result(
        self,
        intent: DemoOrderIntent,
        request: Mapping[str, Any],
        result: Any,
        *,
        broker_order_sent: bool,
    ) -> DemoOrderResult:
        retcode = _field(result, "retcode")
        if isinstance(retcode, bool) or not isinstance(retcode, int):
            retcode = None
        classification = self._classify(retcode)
        order_ticket = _field(result, "order")
        deal_ticket = _field(result, "deal")
        fill_price = _field(result, "price")
        fill_volume = _field(result, "volume")
        order_ticket = order_ticket if isinstance(order_ticket, int) and order_ticket > 0 else None
        deal_ticket = deal_ticket if isinstance(deal_ticket, int) and deal_ticket > 0 else None
        fill_price = fill_price if isinstance(fill_price, (int, float)) and fill_price > 0 else None
        fill_volume = fill_volume if isinstance(fill_volume, (int, float)) and fill_volume > 0 else None
        comment = _field(result, "comment")
        timestamp = self._broker_timestamp(result)
        self.store.record_order_result(
            intent_id=intent.intent_id,
            classification=classification,
            retcode=retcode,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            fill_price=fill_price,
            fill_volume=fill_volume,
            broker_comment=None if comment is None else str(comment),
            broker_timestamp=timestamp,
            request_payload=request,
            broker_order_sent=broker_order_sent,
        )
        return DemoOrderResult(
            intent_id=intent.intent_id,
            classification=classification,
            retcode=retcode,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            fill_price=fill_price,
            fill_volume=fill_volume,
            broker_comment=None if comment is None else str(comment),
            broker_timestamp=timestamp,
            request_payload=request,
            broker_order_sent=broker_order_sent,
        )

    def submit(
        self,
        intent: DemoOrderIntent,
        *,
        position_ticket: int | None = None,
        exit_reason: str | None = None,
    ) -> DemoOrderResult:
        if not isinstance(intent, DemoOrderIntent):
            raise TypeError("intent must be DemoOrderIntent")
        account = self.verify_demo_account()
        if intent.account_trade_mode != account.trade_mode:
            raise DemoExecutionForbidden("order intent account trade mode is not current")
        owned_position = None
        if position_ticket is not None:
            owned = self.store.read_owned_positions(intent.symbol)
            owned_position = next(
                (item for item in owned if int(item.get("ticket", -1)) == int(position_ticket) and item.get("state") == "OPEN"),
                None,
            )
            if owned_position is None:
                raise DemoExecutionForbidden("position is not ledger-owned")
        request = self._request(intent, position_ticket=position_ticket)
        self.store.record_order_intent(self.run_id, intent)
        # Keep the final account check and the broker call inside one serialized
        # MT5 operation so no reconnect or competing consumer can intervene.
        with self._operation("demo_order_send"):
            final_account = self._read_account()
            self._require_demo(final_account)
            if final_account.trade_mode != account.trade_mode:
                self._persist_result(
                    intent,
                    request,
                    {"retcode": None, "comment": "account trade mode changed"},
                    broker_order_sent=False,
                )
                raise DemoExecutionForbidden("account trade mode changed before order_send")
            if position_ticket is not None:
                get_positions = getattr(self.provider, "get_positions", None)
                if not callable(get_positions):
                    self.store.record_reconciliation(
                        status="BROKER_STATE_UNKNOWN",
                        reason="provider does not expose position reads",
                        details={"ticket": position_ticket},
                    )
                    raise DemoExecutionForbidden("broker position state unavailable")
                broker_positions = get_positions(intent.symbol)
                if not any(int(_field(item, "ticket", -1)) == int(position_ticket) for item in broker_positions):
                    self.store.record_reconciliation(
                        status="RECONCILIATION_REQUIRED",
                        reason="owned broker position is missing",
                        details={"ticket": position_ticket, "symbol": intent.symbol},
                    )
                    raise DemoExecutionForbidden("owned broker position is unavailable")
            try:
                result = self.broker_api.order_send(request)
            except Exception as exc:
                return self._persist_result(
                    intent,
                    request,
                    {"retcode": None, "comment": type(exc).__name__},
                    broker_order_sent=True,
                )
            if result is None:
                parsed = self._persist_result(
                    intent,
                    request,
                    {"retcode": None, "comment": "empty broker result"},
                    broker_order_sent=True,
                )
                return parsed
            parsed = self._persist_result(intent, request, result, broker_order_sent=True)
            if (
                position_ticket is not None
                and exit_reason
                and parsed.classification in {"FILLED", "PARTIAL"}
                and owned_position is not None
            ):
                self.store.record_exit(
                    ticket=position_ticket,
                    reason=exit_reason,
                    realized_pnl=None,
                    payload={
                        "intent_id": intent.intent_id,
                        "classification": parsed.classification,
                        "deal_ticket": parsed.deal_ticket,
                    },
                )
                self.store.record_position({**owned_position, "state": "CLOSED"})
            return parsed


__all__ = ["DemoExecutionForbidden", "VerifiedDemoExecutionGateway"]
