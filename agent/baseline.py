"""Deterministic MOM-03 baseline from the strategy knowledge base.

This is not the LLM and it is not the full eleven-strategy ensemble.
Rules implemented, long-only:
- close > EMA(50)
- MACD histogram expanding and positive, MACD line > 0
- 52 <= RSI <= 68
- volume > 1.5 x SMA(20)
- daily close > daily EMA(50)
- no new entries from Saturday 00:00 UTC through Sunday 20:00 UTC
- stop at entry - 1.8 x ATR
- size risks about 0.50% of equity, then the risk manager caps it
- exit when RSI > 72, histogram < 0, or 12 bars pass with profit < 0.5 x ATR
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.config import AppConfig
from core.models import (
    Action,
    EntryType,
    MarketSnapshot,
    PortfolioState,
    Position,
    TradeDecision,
    hold_decision,
)
from core.timeframes import timeframe_to_timedelta


class BaselineDecisionClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.calls = 0

    async def complete(self, system, user: str, tool: dict, cache: bool) -> dict:
        raise RuntimeError("BaselineDecisionClient is used through decide_from_context, not the LLM prompt")

    def decide(self, markets: list[MarketSnapshot], portfolio: PortfolioState, now: datetime) -> list[TradeDecision]:
        self.calls += 1
        decisions = []
        for market in markets:
            position = portfolio.position_for(market.symbol)
            decisions.append(self._one(market, position, portfolio.equity, now))
        return decisions

    def _one(self, market: MarketSnapshot, position: Position | None, equity: float, now: datetime) -> TradeDecision:
        symbol = market.symbol
        price = market.last_price
        indicators = market.primary.indicators
        if price is None or price <= 0:
            return hold_decision(symbol, "MOM-03 baseline has no price")
        if _missing(indicators.ema_50, indicators.rsi, indicators.macd, indicators.macd_hist, indicators.atr, indicators.volume_sma_20):
            return hold_decision(symbol, "MOM-03 indicators are still warming up")
        daily = market.timeframes.get("1d")
        if position is None:
            return self._entry(market, equity, now, daily)
        return self._exit(market, position, now)

    def _entry(self, market: MarketSnapshot, equity: float, now: datetime, daily) -> TradeDecision:
        symbol = market.symbol
        price = float(market.last_price)
        indicators = market.primary.indicators
        if _weekend_block(now):
            return hold_decision(symbol, "MOM-03 skips the weekend liquidity window")
        if daily is None or daily.close is None or daily.indicators.ema_50 is None:
            return hold_decision(symbol, "MOM-03 daily EMA filter is unavailable")
        if daily.close <= daily.indicators.ema_50:
            return hold_decision(symbol, "MOM-03 daily close is not above EMA(50)")
        hist = indicators.macd_hist
        prev = indicators.macd_hist_prev
        prev2 = indicators.macd_hist_prev2
        volume = market.primary.volume
        if _missing(hist, prev, prev2, volume):
            return hold_decision(symbol, "MOM-03 needs three histogram bars and volume")
        expanding = hist > 0 and hist > prev > prev2 and indicators.macd > 0
        rsi_ok = 52 <= indicators.rsi <= 68
        trend = price > indicators.ema_50
        volume_ok = volume > 1.5 * indicators.volume_sma_20
        if not (expanding and rsi_ok and trend and volume_ok):
            return hold_decision(symbol, "MOM-03 entry conditions are not all true")
        atr = float(indicators.atr)
        stop = price - 1.8 * atr
        if stop <= 0 or stop >= price:
            return hold_decision(symbol, "MOM-03 could not place a protective stop")
        distance_pct = (price - stop) / price * 100
        if distance_pct <= 0:
            return hold_decision(symbol, "MOM-03 stop distance is invalid")
        size_pct = min(self.config.risk.max_position_size_pct, (0.50 / distance_pct) * 100)
        take_profit = price + 3.0 * atr
        return TradeDecision(
            symbol=symbol,
            action=Action.BUY,
            size_pct=round(size_pct, 4),
            entry_type=EntryType.MARKET,
            stop_loss=stop,
            take_profit=take_profit,
            confidence=0.66,
            reasoning=(
                f"MOM-03 long: close {price:.4f} above EMA50 {indicators.ema_50:.4f}, "
                f"expanding positive MACD histogram, RSI {indicators.rsi:.1f}, "
                f"volume above 1.5x SMA20, daily close above daily EMA50. "
                f"Stop {stop:.4f} is 1.8 ATR. Size targets 0.5% equity risk."
            ),
            strategy_referenced="MOM-03",
        )

    def _exit(self, market: MarketSnapshot, position: Position, now: datetime) -> TradeDecision:
        indicators = market.primary.indicators
        price = float(market.last_price)
        reasons = []
        if indicators.rsi is not None and indicators.rsi > 72:
            reasons.append(f"RSI {indicators.rsi:.1f} > 72")
        if indicators.macd_hist is not None and indicators.macd_hist < 0:
            reasons.append("MACD histogram crossed below zero")
        if indicators.atr is not None and position.opened_at is not None:
            held = (now - _aware(position.opened_at)) / timeframe_to_timedelta(self.config.primary_timeframe)
            profit = price - position.entry_price
            if held >= 12 and profit < 0.5 * indicators.atr:
                reasons.append("12-bar time stop with profit below 0.5 ATR")
        if not reasons:
            return hold_decision(market.symbol, "MOM-03 position remains valid; protective stop stays on the exchange")
        return TradeDecision(
            symbol=market.symbol,
            action=Action.CLOSE,
            size_pct=0,
            entry_type=EntryType.MARKET,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            confidence=0.7,
            reasoning="MOM-03 exit: " + "; ".join(reasons),
            strategy_referenced="MOM-03",
        )


def _missing(*values) -> bool:
    return any(value is None for value in values)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _weekend_block(now: datetime) -> bool:
    moment = _aware(now)
    if moment.weekday() == 5:
        return True
    return moment.weekday() == 6 and moment.hour < 20
