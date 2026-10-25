"""Timeframe parsing and UTC helpers."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
_PANDAS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_timeframe(timeframe: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip())
    if not match:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return int(match.group(1)), match.group(2)


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    count, unit = parse_timeframe(timeframe)
    return timedelta(**{_UNITS[unit]: count})


def pandas_rule(timeframe: str) -> str:
    count, unit = parse_timeframe(timeframe)
    return f"{count}{_PANDAS[unit]}"


def as_utc(value: datetime | pd.Timestamp | str | int | float) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        stamp = pd.Timestamp(value, unit="ms", tz="UTC")
    else:
        stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def to_utc_datetime(value: datetime | pd.Timestamp) -> datetime:
    stamp = as_utc(value)
    return stamp.to_pydatetime()
