"""Indicator calculations.

RSI, MACD, EMA, ATR, Bollinger Bands, and ADX come from the `ta` library
(the maintained alternative to pandas-ta). Volume profile, point of control,
and value area are computed here because `ta` does not provide them.
All of these indicators are causal: the value at a bar uses only that bar
and earlier bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from core.models import IndicatorSet, TimeframeView, VolumeBin
from core.timeframes import to_utc_datetime


def _finite(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def compute_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV with indicator columns aligned to each bar."""

    out = frame.copy()
    if out.empty:
        return out
    close = out["close"]
    high = out["high"]
    low = out["low"]
    out["rsi"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    out["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(close, window=26).ema_indicator()
    out["ema_50"] = EMAIndicator(close, window=50).ema_indicator() if len(out) >= 2 else np.nan
    out["ema_200"] = EMAIndicator(close, window=200).ema_indicator() if len(out) >= 2 else np.nan
    try:
        out["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range() if len(out) > 14 else np.nan
    except (IndexError, ValueError, ZeroDivisionError):
        out["atr"] = np.nan
    try:
        out["adx"] = ADXIndicator(high, low, close, window=14).adx() if len(out) > 20 else np.nan
    except (IndexError, ValueError, ZeroDivisionError):
        out["adx"] = np.nan
    bands = BollingerBands(close, window=20, window_dev=2)
    out["bb_upper"] = bands.bollinger_hband()
    out["bb_mid"] = bands.bollinger_mavg()
    out["bb_lower"] = bands.bollinger_lband()
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    return out


def volume_profile(frame: pd.DataFrame, bins: int = 12) -> tuple[list[VolumeBin], float | None, float | None, float | None]:
    """Price bins plus point of control and 70% value area."""

    if frame.empty:
        return [], None, None, None
    prices = frame["close"].to_numpy(dtype=float)
    volumes = frame["volume"].to_numpy(dtype=float)
    if len(prices) == 0 or float(np.nansum(volumes)) <= 0:
        return [], None, None, None
    low = float(np.nanmin(prices))
    high = float(np.nanmax(prices))
    if not math.isfinite(low) or not math.isfinite(high):
        return [], None, None, None
    if math.isclose(low, high):
        total = float(np.nansum(volumes))
        bin_row = VolumeBin(price=low, volume=total, pct=1.0)
        return [bin_row], low, high, low
    counts, edges = np.histogram(prices, bins=bins, range=(low, high), weights=volumes)
    total = float(counts.sum()) or 1.0
    rows: list[VolumeBin] = []
    for index, volume in enumerate(counts):
        mid = float((edges[index] + edges[index + 1]) / 2)
        rows.append(VolumeBin(price=mid, volume=float(volume), pct=float(volume / total)))
    poc_index = int(np.argmax(counts))
    poc = rows[poc_index].price
    selected = {poc_index}
    covered = float(counts[poc_index])
    left = poc_index - 1
    right = poc_index + 1
    while covered / total < 0.70 and (left >= 0 or right < len(counts)):
        left_volume = float(counts[left]) if left >= 0 else -1
        right_volume = float(counts[right]) if right < len(counts) else -1
        if right_volume > left_volume:
            selected.add(right)
            covered += max(right_volume, 0)
            right += 1
        else:
            selected.add(left)
            covered += max(left_volume, 0)
            left -= 1
    chosen = sorted(selected)
    val = float(edges[chosen[0]])
    vah = float(edges[chosen[-1] + 1])
    return rows, poc, vah, val


def indicator_set_at(frame: pd.DataFrame, index: int, include_profile: bool = True) -> IndicatorSet:
    row = frame.iloc[index]
    previous = frame.iloc[index - 1] if index >= 1 else None
    previous2 = frame.iloc[index - 2] if index >= 2 else None
    profile: list[VolumeBin] = []
    poc = vah = val = None
    if include_profile:
        start = max(0, index - 167)
        profile, poc, vah, val = volume_profile(frame.iloc[start : index + 1])
    return IndicatorSet(
        rsi=_finite(row.get("rsi")),
        macd=_finite(row.get("macd")),
        macd_signal=_finite(row.get("macd_signal")),
        macd_hist=_finite(row.get("macd_hist")),
        macd_hist_prev=_finite(previous.get("macd_hist")) if previous is not None else None,
        macd_hist_prev2=_finite(previous2.get("macd_hist")) if previous2 is not None else None,
        ema_12=_finite(row.get("ema_12")),
        ema_26=_finite(row.get("ema_26")),
        ema_50=_finite(row.get("ema_50")),
        ema_200=_finite(row.get("ema_200")),
        atr=_finite(row.get("atr")),
        adx=_finite(row.get("adx")),
        bb_upper=_finite(row.get("bb_upper")),
        bb_mid=_finite(row.get("bb_mid")),
        bb_lower=_finite(row.get("bb_lower")),
        volume_sma_20=_finite(row.get("volume_sma_20")),
        poc=poc,
        vah=vah,
        val=val,
        volume_profile=profile,
    )


def timeframe_view_from_frame(
    raw: pd.DataFrame,
    indicators: pd.DataFrame,
    timeframe: str,
    index: int | None = None,
    include_profile: bool = False,
) -> TimeframeView:
    if indicators.empty:
        return TimeframeView(timeframe=timeframe)
    pointer = len(indicators) - 1 if index is None else index
    row = indicators.iloc[pointer]
    timestamp = row["timestamp"]
    py_time = to_utc_datetime(timestamp) if timestamp is not None else None
    return TimeframeView(
        timeframe=timeframe,
        timestamp=py_time,
        open=_finite(row.get("open")),
        high=_finite(row.get("high")),
        low=_finite(row.get("low")),
        close=_finite(row.get("close")),
        volume=_finite(row.get("volume")),
        indicators=indicator_set_at(indicators, pointer, include_profile=include_profile),
    )
