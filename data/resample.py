"""Causal OHLCV resampling. Incomplete bins are dropped."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.timeframes import as_utc, pandas_rule, timeframe_to_timedelta


def ohlcv_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame
    frame["timestamp"] = frame["timestamp"].map(lambda value: as_utc(value))
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def drop_incomplete(frame: pd.DataFrame, timeframe: str, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    duration = timeframe_to_timedelta(timeframe)
    deadline = as_utc(now)
    mask = frame["timestamp"].map(lambda ts: as_utc(ts) + duration <= deadline)
    return frame.loc[mask].reset_index(drop=True)


def resample_ohlcv(frame: pd.DataFrame, target_timeframe: str, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    duration = timeframe_to_timedelta(target_timeframe)
    indexed = frame.set_index("timestamp").sort_index()
    aggregated = (
        indexed.resample(pandas_rule(target_timeframe), label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )
    deadline = as_utc(now)
    complete = aggregated[[stamp + duration <= deadline for stamp in aggregated.index]]
    complete = complete.reset_index()
    return complete
