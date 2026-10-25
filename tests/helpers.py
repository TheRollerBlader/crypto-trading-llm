"""Shared fixtures for unit and pipeline tests."""

from __future__ import annotations

from datetime import datetime, timezone

from core.config import AppConfig, load_config
from core.models import (
    Action,
    EntryType,
    IndicatorSet,
    MarketSnapshot,
    PortfolioState,
    TimeframeView,
    TradeDecision,
)
from core.timeframes import utcnow


def config() -> AppConfig:
    loaded = load_config()
    return loaded.model_copy(deep=True)


def portfolio(**overrides) -> PortfolioState:
    state = PortfolioState(
        equity=10_000,
        quote_balance=10_000,
        positions=[],
        day_start_equity=10_000,
        peak_equity=10_000,
        trades_today=0,
    )
    return state.model_copy(update=overrides)


def snapshot(symbol: str = "BTC/USDT", price: float = 100, atr: float = 2, stale: bool = False) -> MarketSnapshot:
    indicators = IndicatorSet(atr=atr, rsi=55, ema_50=price * 0.99, macd=1, macd_hist=0.2, macd_hist_prev=0.1, macd_hist_prev2=0.05, volume_sma_20=10)
    view = TimeframeView(
        timeframe="1h",
        timestamp=utcnow(),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=20,
        indicators=indicators,
    )
    return MarketSnapshot(
        symbol=symbol,
        as_of=utcnow(),
        last_price=price,
        stale=stale,
        primary=view,
        timeframes={"1h": view},
    )


def decision(
    symbol: str = "BTC/USDT",
    action: Action = Action.BUY,
    size: float = 5,
    stop: float | None = 95,
    take: float | None = 110,
    confidence: float = 0.8,
    price: float | None = None,
) -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
        action=action,
        size_pct=0 if action in (Action.HOLD, Action.CLOSE) else size,
        entry_type=EntryType.MARKET,
        stop_loss=stop,
        take_profit=take,
        confidence=0 if action is Action.HOLD else confidence,
        reasoning="test decision",
        strategy_referenced="MOM-03",
    )


def aware(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)
