"""Replay closed candles without lookahead.

Decisions at the close of bar i fill at the open of bar i+1. Stops are checked
on bar i+1 after the fill. If both a stop and a target are touched, the stop
wins. A gap through the stop fills at the open, then slippage. Social data is
not available: there is no historical social archive in this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agent.baseline import BaselineDecisionClient
from core.config import AppConfig
from core.models import MarketSnapshot
from core.timeframes import as_utc, timeframe_to_timedelta, to_utc_datetime
from data.indicators import compute_indicator_frame, timeframe_view_from_frame
from data.resample import resample_ohlcv
from execution.broker import Broker
from execution.simulated import SimulatedExchange
from monitoring.store import Store
from risk.manager import RiskManager


@dataclass
class BacktestResult:
    total_return_pct: float
    buy_hold_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float | None
    num_trades: int
    fees_paid: float
    ending_equity: float
    initial_equity: float
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    social_data: str = "unavailable"
    decision: str = "baseline"
    notes: list[str] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, config: AppConfig, store: Store, decider=None):
        self.config = config
        self.store = store
        self.decider = decider
        self.baseline = BaselineDecisionClient(config)
        self.seen_prices: list[float] = []
        self.seen_timestamps: list[datetime] = []

    async def run(self, frames: dict[str, pd.DataFrame]) -> BacktestResult:
        timeframe = self.config.backtest.timeframe
        duration = timeframe_to_timedelta(timeframe)
        prepared = {symbol: _normalize(frame) for symbol, frame in frames.items()}
        end = max(to_utc_datetime(frame["timestamp"].iloc[-1]) + duration for frame in prepared.values())
        indicators = {symbol: compute_indicator_frame(frame) for symbol, frame in prepared.items()}
        higher = {
            symbol: {
                "4h": compute_indicator_frame(resample_ohlcv(frame, "4h", end)),
                "1d": compute_indicator_frame(resample_ohlcv(frame, "1d", end)),
            }
            for symbol, frame in prepared.items()
        }
        first = next(iter(prepared.values()))
        start = min(self.config.backtest.warmup_bars, len(first) - 2)
        exchange = SimulatedExchange(
            quote=self.config.exchange.quote_currency,
            initial_quote=self.config.exchange.initial_quote,
            fee_bps=self.config.risk.fee_bps,
            slippage_bps=self.config.risk.slippage_bps,
            base_timeframe=timeframe,
        )
        for symbol, frame in prepared.items():
            exchange.last[symbol] = float(frame.iloc[start]["close"])
        broker = Broker(exchange, self.config, self.store, dry_run=False)
        risk = RiskManager(self.config)
        initial = self.config.exchange.initial_quote
        curve: list[tuple[str, float]] = []
        for index in range(start, len(first) - 1):
            now = to_utc_datetime(first.iloc[index]["timestamp"]) + duration
            markets = []
            for symbol, frame in prepared.items():
                row_time = to_utc_datetime(frame.iloc[index]["timestamp"])
                self.seen_timestamps.append(row_time)
                close = float(frame.iloc[index]["close"])
                self.seen_prices.append(close)
                exchange.last[symbol] = close
                exchange.set_market_fill(symbol, float(frame.iloc[index + 1]["open"]))
                markets.append(
                    _snapshot(symbol, now, close, indicators[symbol], frame, index, higher[symbol], timeframe)
                )
            state = self.store.load_risk_state(initial, now)
            equity = await broker.equity({market.symbol: market.last_price for market in markets})
            state = _roll_day(state, equity, now)
            if equity > state.peak_equity:
                state.peak_equity = equity
            portfolio = broker.portfolio(equity, await broker.quote_balance(), state)
            if self.decider is not None:
                decisions = self.decider(markets, portfolio, now)
            else:
                decisions = self.baseline.decide(markets, portfolio, now)
            verdicts = risk.review(decisions, portfolio, markets, now)
            for verdict in verdicts:
                if not verdict.approved or verdict.action.value == "HOLD":
                    continue
                if state.halted and verdict.action.value != "CLOSE":
                    continue
                result = await broker.execute(verdict, f"bt{index}{verdict.symbol}", equity, now)
                if result.status == "filled" and result.action.value in {"BUY", "SELL"}:
                    state.trades_today += 1
                if result.realized_pnl is not None and result.realized_pnl < 0:
                    state.last_loss_at = now
            ranges = {}
            marks = {}
            for symbol, frame in prepared.items():
                nxt = frame.iloc[index + 1]
                marks[symbol] = float(nxt["close"])
                ranges[symbol] = {"high": float(nxt["high"]), "low": float(nxt["low"]), "open": float(nxt["open"])}
            fill_time = to_utc_datetime(first.iloc[index + 1]["timestamp"]) + duration
            await broker.reconcile(marks, ranges, now=fill_time)
            equity_after = await broker.equity(marks)
            if equity_after > state.peak_equity:
                state.peak_equity = equity_after
            if state.day_start_equity > 0 and equity_after < state.day_start_equity:
                daily_loss = (state.day_start_equity - equity_after) / state.day_start_equity * 100
            else:
                daily_loss = 0.0
            if state.peak_equity > 0 and equity_after < state.peak_equity:
                drawdown = (state.peak_equity - equity_after) / state.peak_equity * 100
            else:
                drawdown = 0.0
            if drawdown >= self.config.risk.max_drawdown_pct:
                state.halted = True
                state.halt_reason = "max_drawdown"
            elif daily_loss >= self.config.risk.max_daily_loss_pct:
                state.halted = True
                state.halt_reason = "max_daily_loss"
            state.realized_pnl = broker.realized_pnl
            self.store.save_risk_state(state)
            self.store.add_equity(now, equity_after)
            curve.append((now.isoformat(), equity_after))
        ending = curve[-1][1] if curve else initial
        trades = self.store.closed_trades()
        wins = [trade for trade in trades if trade.realized_pnl > 0]
        sharpe, max_dd = _risk_stats(curve)
        fee_rate = self.config.risk.fee_bps / 10_000
        slip = self.config.risk.slippage_bps / 10_000
        buy_hold = _buy_and_hold(prepared, start, initial, fee_rate, slip)
        return BacktestResult(
            total_return_pct=(ending / initial - 1) * 100,
            buy_hold_return_pct=buy_hold,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            win_rate=(len(wins) / len(trades)) if trades else None,
            num_trades=len(trades),
            fees_paid=sum(trade["fee"] for trade in exchange.trades),
            ending_equity=ending,
            initial_equity=initial,
            equity_curve=curve,
            decision=self.config.backtest.decision,
            notes=[
                "Social data was unavailable. No historical social archive was replayed.",
                "The default path is the MOM-03 baseline, not an LLM, so this is not evidence about model skill.",
                "Stops are pessimistic: same-bar stop and target fills the stop, and gaps fill at the open plus slippage.",
                "Indicator values at bar i were checked to depend only on bars up to i.",
            ],
        )


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = out["timestamp"].map(lambda value: as_utc(value))
    return out.sort_values("timestamp").reset_index(drop=True)


def _snapshot(symbol, now, close, indicators, raw, index, higher, timeframe) -> MarketSnapshot:
    primary = timeframe_view_from_frame(raw, indicators, timeframe, index=index, include_profile=False)
    views = {timeframe: primary}
    for name, frame in higher.items():
        if frame.empty:
            continue
        pointer = _last_complete(frame, name, now)
        if pointer is None:
            continue
        views[name] = timeframe_view_from_frame(frame, frame, name, index=pointer, include_profile=False)
    return MarketSnapshot(symbol=symbol, as_of=now, last_price=close, stale=False, primary=primary, timeframes=views)


def _last_complete(frame: pd.DataFrame, timeframe: str, now: datetime) -> int | None:
    duration = timeframe_to_timedelta(timeframe)
    deadline = as_utc(now)
    matches = [index for index, stamp in enumerate(frame["timestamp"]) if as_utc(stamp) + duration <= deadline]
    if not matches:
        return None
    return matches[-1]


def _risk_stats(curve: list[tuple[str, float]]) -> tuple[float, float]:
    if len(curve) < 3:
        return 0.0, 0.0
    index = pd.to_datetime([point[0] for point in curve], utc=True)
    equity = pd.Series([point[1] for point in curve], index=index)
    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    sharpe = 0.0
    if len(returns) >= 2 and float(returns.std()) > 0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(365))
    values = equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / np.where(peak == 0, 1, peak)
    return sharpe, float(drawdown.min() * 100)


def _roll_day(state, equity: float, now: datetime):
    day = now.astimezone(timezone.utc).date().isoformat()
    if state.day != day:
        if state.halt_reason == "max_daily_loss":
            state.halted = False
            state.halt_reason = None
        state.day = day
        state.day_start_equity = equity
        state.trades_today = 0
    return state


def _buy_and_hold(frames: dict[str, pd.DataFrame], start: int, initial: float, fee_rate: float, slippage: float) -> float:
    if not frames:
        return 0.0
    alloc = initial / len(frames)
    final = 0.0
    for frame in frames.values():
        entry = float(frame.iloc[start + 1]["open"]) * (1 + slippage)
        qty = alloc / (entry * (1 + fee_rate))
        exit_price = float(frame.iloc[-1]["close"]) * (1 - slippage)
        final += qty * exit_price * (1 - fee_rate)
    return (final / initial - 1) * 100
