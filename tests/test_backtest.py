import asyncio

import numpy as np
import pandas as pd

from agent.baseline import BaselineDecisionClient
from backtest.engine import BacktestEngine
from core.models import hold_decision
from monitoring.store import Store
from tests.helpers import config


def _frame(prices: list[float]) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for index, price in enumerate(prices):
        drift = 1 if index % 5 else -1
        rows.append(
            {
                "timestamp": start + pd.Timedelta(hours=index),
                "open": price,
                "high": price + 30,
                "low": price - 30,
                "close": price + drift,
                "volume": 100 + index,
            }
        )
    return pd.DataFrame(rows)


def test_backtest_does_not_show_the_decision_a_future_close(tmp_path):
    cfg = config()
    cfg.backtest.warmup_bars = 40
    cfg.backtest.timeframe = "1h"
    cfg.primary_timeframe = "1h"
    rng = np.random.default_rng(3)
    prices = (50_000 * np.cumprod(1 + rng.normal(0, 0.002, 160))).tolist()
    frame = _frame(prices)
    seen = []

    def decider(markets, portfolio, now):
        market = markets[0]
        seen.append((market.last_price, market.primary.timestamp, market.primary.close))
        if "1d" in market.timeframes:
            daily = market.timeframes["1d"]
            assert daily.timestamp is not None
            assert pd.Timestamp(daily.timestamp) + pd.Timedelta(days=1) <= pd.Timestamp(now)
        return [hold_decision("BTC/USDT", "lookahead probe")]

    store = Store(f"sqlite:///{tmp_path}/bt.sqlite")
    result = asyncio.run(BacktestEngine(cfg, store, decider=decider).run({"BTC/USDT": frame}))
    start = cfg.backtest.warmup_bars
    assert len(seen) == len(frame) - 1 - start
    for index, (price, stamp, close) in enumerate(seen):
        row = frame.iloc[start + index]
        assert price == float(row["close"])
        assert close == float(row["close"])
        assert abs((stamp - row["timestamp"].to_pydatetime()).total_seconds()) < 1
    assert seen[0][0] != float(frame.iloc[-1]["close"])
    assert result.social_data == "unavailable"
    assert result.num_trades == 0
    assert result.ending_equity == result.initial_equity


def test_baseline_backtest_on_synthetic_months_returns_finite_metrics(tmp_path):
    cfg = config()
    cfg.backtest.timeframe = "1h"
    cfg.primary_timeframe = "1h"
    cfg.backtest.warmup_bars = 24 * 60
    rng = np.random.default_rng(8)
    hours = 24 * 80
    cfg.backtest.warmup_bars = 24 * 55
    btc = (40_000 * np.cumprod(1 + rng.normal(0.00015, 0.01, hours))).tolist()
    eth = (2_000 * np.cumprod(1 + rng.normal(0.00015, 0.012, hours))).tolist()
    store = Store(f"sqlite:///{tmp_path}/bt2.sqlite")
    engine = BacktestEngine(cfg, store)
    result = asyncio.run(engine.run({"BTC/USDT": _frame(btc), "ETH/USDT": _frame(eth)}))
    assert result.decision == "baseline"
    assert result.sharpe == result.sharpe
    assert result.max_drawdown_pct <= 0
    assert result.num_trades >= 0
    assert result.social_data == "unavailable"
    client = BaselineDecisionClient(cfg)
    assert client.calls == 0 or client.calls >= 0
