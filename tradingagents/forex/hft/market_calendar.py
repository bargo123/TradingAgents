"""Strict broker/symbol session calendar used only for market lifecycle state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class CalendarStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


_WEEKDAYS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "calendar_version",
    "broker_server",
    "symbol",
    "timezone",
    "valid_from",
    "valid_through",
    "weekly_sessions",
    "holidays",
    "date_overrides",
}


@dataclass(frozen=True, slots=True)
class _Session:
    opens: time
    closes: time


@dataclass(frozen=True, slots=True)
class BrokerSessionCalendar:
    """Immutable, explicitly versioned sessions for one broker symbol.

    Calendar times are local wall times in ``timezone``. Overnight intervals
    are rejected so callers must represent their local-date boundaries
    explicitly rather than relying on implicit day rollover.
    """

    calendar_version: str
    broker_server: str
    symbol: str
    timezone: ZoneInfo
    valid_from: date
    valid_through: date
    weekly_sessions: Mapping[int, tuple[_Session, ...]]
    holidays: frozenset[date]
    date_overrides: Mapping[date, tuple[_Session, ...]]

    @classmethod
    def from_json(cls, path: str | Path) -> BrokerSessionCalendar:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("calendar file is unavailable or invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
            raise ValueError("calendar fields do not match schema version 1")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported calendar schema version")

        version = _required_string(payload, "calendar_version")
        broker = _required_string(payload, "broker_server")
        symbol = _required_string(payload, "symbol")
        timezone_name = _required_string(payload, "timezone")
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("calendar timezone is not available") from exc
        valid_from = _parse_date(payload["valid_from"], "valid_from")
        valid_through = _parse_date(payload["valid_through"], "valid_through")
        if valid_through < valid_from:
            raise ValueError("calendar coverage range is reversed")

        weekly_raw = payload["weekly_sessions"]
        if not isinstance(weekly_raw, dict) or set(weekly_raw) - set(_WEEKDAYS):
            raise ValueError("weekly_sessions must use weekday keys mon through sun")
        weekly: dict[int, tuple[_Session, ...]] = {}
        for day_name, raw_sessions in weekly_raw.items():
            weekly[_WEEKDAYS[day_name]] = _parse_sessions(raw_sessions)

        raw_holidays = payload["holidays"]
        if not isinstance(raw_holidays, list):
            raise ValueError("holidays must be a list of dates")
        holidays = tuple(_parse_date(item, "holiday") for item in raw_holidays)
        if len(holidays) != len(set(holidays)):
            raise ValueError("duplicate holiday date")

        raw_overrides = payload["date_overrides"]
        if not isinstance(raw_overrides, dict):
            raise ValueError("date_overrides must map dates to sessions")
        overrides: dict[date, tuple[_Session, ...]] = {}
        for raw_day, raw_sessions in raw_overrides.items():
            day = _parse_date(raw_day, "date override")
            if day in overrides:
                raise ValueError("duplicate date override")
            overrides[day] = _parse_sessions(raw_sessions)
        if set(holidays) & set(overrides):
            raise ValueError("a date cannot be both a holiday and an override")

        return cls(
            calendar_version=version,
            broker_server=broker,
            symbol=symbol,
            timezone=timezone,
            valid_from=valid_from,
            valid_through=valid_through,
            weekly_sessions=MappingProxyType(weekly),
            holidays=frozenset(holidays),
            date_overrides=MappingProxyType(overrides),
        )

    def status_at(
        self, at_utc: datetime, *, broker_server: str, symbol: str
    ) -> CalendarStatus:
        if (
            not isinstance(at_utc, datetime)
            or at_utc.tzinfo is None
            or at_utc.utcoffset() is None
            or at_utc.utcoffset().total_seconds() != 0
            or broker_server != self.broker_server
            or symbol != self.symbol
        ):
            return CalendarStatus.UNKNOWN
        local = at_utc.astimezone(self.timezone)
        local_day = local.date()
        if not self.valid_from <= local_day <= self.valid_through:
            return CalendarStatus.UNKNOWN
        if local_day in self.holidays:
            return CalendarStatus.CLOSED
        sessions = self.date_overrides.get(local_day)
        if sessions is None:
            sessions = self.weekly_sessions.get(local.weekday(), ())
        minute = local.hour * 60 + local.minute
        return (
            CalendarStatus.OPEN
            if any(_contains(session, minute) for session in sessions)
            else CalendarStatus.CLOSED
        )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"calendar {key} must be a non-empty string")
    return value


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"calendar {field} must be an ISO date")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"calendar {field} must be an ISO date") from exc
    if result.isoformat() != value:
        raise ValueError(f"calendar {field} must be a canonical ISO date")
    return result


def _parse_sessions(value: Any) -> tuple[_Session, ...]:
    if not isinstance(value, list):
        raise ValueError("sessions must be a list")
    sessions: list[_Session] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"open", "close"}:
            raise ValueError("each session must contain only open and close")
        opens = _parse_time(item["open"])
        closes = _parse_time(item["close"])
        if closes <= opens:
            raise ValueError("overnight or zero-length sessions are unsupported")
        sessions.append(_Session(opens, closes))
    sessions.sort(key=lambda session: session.opens)
    if any(left.closes > right.opens for left, right in zip(sessions, sessions[1:], strict=False)):
        raise ValueError("overlapping sessions are invalid")
    return tuple(sessions)


def _parse_time(value: Any) -> time:
    if not isinstance(value, str):
        raise ValueError("session time must use HH:MM")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("session time must use HH:MM") from exc
    if parsed.tzinfo is not None or parsed.second or parsed.microsecond or parsed.isoformat(timespec="minutes") != value:
        raise ValueError("session time must use canonical HH:MM")
    return parsed


def _contains(session: _Session, minute: int) -> bool:
    start = session.opens.hour * 60 + session.opens.minute
    end = session.closes.hour * 60 + session.closes.minute
    return start <= minute < end


__all__ = ["BrokerSessionCalendar", "CalendarStatus"]
