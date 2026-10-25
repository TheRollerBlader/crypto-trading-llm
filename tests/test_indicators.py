from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from core.timeframes import timeframe_to_timedelta
from data.indicators import compute_indicator_frame, volume_profile
from data.resample import drop_incomplete, resample_ohlcv


def _frame(prices: list[float], spread: float = 1.0) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for index, price in enumerate(prices):
        rows.append(
            {
                "timestamp": start + pd.Timedelta(hours=index),
                "open": price,
                "high": price + spread,
                "low": price - spread,
                "close": price,
                "volume": 10 + index,
            }
        )
    return pd.DataFrame(rows)


def test_flat_prices_have_zero_width_bands_and_zero_atr():
    prices = [50.0] * 80
    frame = compute_indicator_frame(_frame(prices, spread=0))
    last = frame.iloc[-1]
    assert last["ema_12"] == pytest.approx(50)
    assert last["ema_50"] == pytest.approx(50)
    assert last["macd"] == pytest.approx(0, abs=1e-8)
    assert last["atr"] == pytest.approx(0, abs=1e-8)
    assert last["bb_upper"] == pytest.approx(last["bb_lower"])


def test_rising_rsi_is_above_70():
    prices = [float(20 + index) for index in range(40)]
    frame = compute_indicator_frame(_frame(prices))
    assert frame.iloc[-1]["rsi"] > 70


def test_indicator_at_row_ignores_future_bars():
    rng = np.random.default_rng(4)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 90))
    raw = _frame(prices.tolist())
    full = compute_indicator_frame(raw)
    prefix = compute_indicator_frame(raw.iloc[:50].reset_index(drop=True))
    assert full.iloc[49]["rsi"] == pytest.approx(prefix.iloc[-1]["rsi"])
    assert full.iloc[49]["ema_26"] == pytest.approx(prefix.iloc[-1]["ema_26"])
    assert full.iloc[49]["macd_hist"] == pytest.approx(prefix.iloc[-1]["macd_hist"])
    assert full.iloc[49]["atr"] == pytest.approx(prefix.iloc[-1]["atr"])


def test_volume_profile_percentages_sum_to_one():
    frame = _frame([10, 11, 10.5, 12, 11.5, 13, 12.5, 14])
    bins, poc, vah, val = volume_profile(frame, bins=4)
    assert poc is not None
    assert val <= poc <= vah
    assert sum(item.pct for item in bins) == pytest.approx(1)
    assert sum(item.volume for item in bins) == pytest.approx(frame["volume"].sum())


def test_incomplete_higher_timeframe_is_dropped():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for index in range(4):
        price = 100 + index
        rows.append(
            {
                "timestamp": start + pd.Timedelta(hours=index),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1,
            }
        )
    frame = pd.DataFrame(rows)
    now = start + timedelta(hours=3)
    resampled = resample_ohlcv(frame, "4h", now.to_pydatetime())
    assert resampled.empty
    finished = resample_ohlcv(frame, "4h", (start + timedelta(hours=4)).to_pydatetime())
    assert len(finished) == 1
    assert finished.iloc[0]["open"] == 100
    assert finished.iloc[0]["close"] == 103
    kept = drop_incomplete(frame, "1h", (start + timedelta(hours=2)).to_pydatetime())
    assert len(kept) == 2
    assert timeframe_to_timedelta("15m") == timedelta(minutes=15)
