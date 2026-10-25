"""Pydantic schemas shared across the agent, risk manager, and broker."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class EntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


def _finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class TradeDecision(BaseModel):
    """One symbol decision. The model proposes this; it does not place orders."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    action: Action
    size_pct: float = Field(ge=0, le=100)
    entry_type: EntryType = EntryType.MARKET
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)
    strategy_referenced: str = Field(min_length=1)

    @field_validator("action", "entry_type", mode="before")
    @classmethod
    def _uppercase(cls, value):
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("size_pct", "confidence")
    @classmethod
    def _finite_required(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @field_validator("stop_loss", "take_profit", "limit_price")
    @classmethod
    def _finite_optional(cls, value: float | None, info) -> float | None:
        if value is None:
            return None
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def _normalize(self) -> TradeDecision:
        if self.action is Action.HOLD:
            self.size_pct = 0
        if self.action in (Action.BUY, Action.SELL) and self.size_pct <= 0:
            raise ValueError("BUY and SELL require size_pct > 0")
        if (
            self.entry_type is EntryType.LIMIT
            and self.action in (Action.BUY, Action.SELL)
            and self.limit_price is None
        ):
            raise ValueError("LIMIT entries require limit_price")
        return self


class VolumeBin(BaseModel):
    price: float
    volume: float
    pct: float


class IndicatorSet(BaseModel):
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    macd_hist_prev: float | None = None
    macd_hist_prev2: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    atr: float | None = None
    adx: float | None = None
    bb_upper: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    volume_sma_20: float | None = None
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    volume_profile: list[VolumeBin] = Field(default_factory=list)


class TimeframeView(BaseModel):
    timeframe: str
    timestamp: datetime | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    indicators: IndicatorSet = Field(default_factory=IndicatorSet)


class OrderBookLevel(BaseModel):
    price: float
    amount: float


class OrderBookSnapshot(BaseModel):
    bid: float | None = None
    ask: float | None = None
    spread_bps: float | None = None
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    symbol: str
    as_of: datetime
    last_price: float | None
    stale: bool
    primary: TimeframeView
    timeframes: dict[str, TimeframeView] = Field(default_factory=dict)
    order_book: OrderBookSnapshot = Field(default_factory=OrderBookSnapshot)
    funding_rate: float | None = None
    volume_24h: float | None = None
    warnings: list[str] = Field(default_factory=list)


class Engagement(BaseModel):
    likes: int = 0
    reposts: int = 0
    comments: int = 0

    @property
    def total(self) -> int:
        return self.likes + self.reposts + self.comments


class SocialPost(BaseModel):
    source: str
    text: str
    author: str
    timestamp: datetime
    engagement: Engagement = Field(default_factory=Engagement)
    symbols: list[str] = Field(default_factory=list)
    url: str | None = None


class AssetSentiment(BaseModel):
    symbol: str
    post_count: int
    bullish: int
    bearish: int
    neutral: int
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    volume_spike: bool
    notable_accounts: list[str]
    spam_filtered: int
    summary: str
    scorer: str


class Position(BaseModel):
    symbol: str
    qty: float
    entry_price: float
    entry_fee: float = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    stop_order_id: str | None = None
    tp_order_id: str | None = None
    opened_at: datetime
    entry_order_id: str | None = None


class PortfolioState(BaseModel):
    equity: float
    quote_balance: float
    positions: list[Position] = Field(default_factory=list)
    day_start_equity: float
    peak_equity: float
    trades_today: int = 0
    last_loss_at: datetime | None = None
    halted: bool = False
    halt_reason: str | None = None

    def position_for(self, symbol: str) -> Position | None:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    def notionals(self, marks: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for position in self.positions:
            mark = marks.get(position.symbol, position.entry_price)
            result[position.symbol] = position.qty * mark
        return result


class RiskVerdict(BaseModel):
    symbol: str
    action: Action
    size_pct: float
    stop_loss: float | None
    take_profit: float | None
    approved: bool
    reasons: list[str]
    halt: bool = False
    halt_reason: str | None = None
    reference_price: float | None = None
    original_action: Action
    original_size_pct: float


class ExecutionResult(BaseModel):
    symbol: str
    status: str
    action: Action
    protected: bool = False
    order_id: str | None = None
    stop_order_id: str | None = None
    tp_order_id: str | None = None
    qty: float = 0
    price: float | None = None
    fee: float = 0
    realized_pnl: float | None = None
    error: str | None = None
    reasons: list[str] = Field(default_factory=list)


class ClosedTrade(BaseModel):
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    opened_at: datetime
    closed_at: datetime
    reason: str


class CycleRecord(BaseModel):
    cycle_id: str
    started_at: datetime
    mode: str
    dry_run: bool
    halted: bool
    inputs_summary: dict[str, Any]
    decisions: list[TradeDecision]
    verdicts: list[RiskVerdict]
    executions: list[ExecutionResult]
    errors: list[str] = Field(default_factory=list)
    equity: float | None = None


class StrategyContext(BaseModel):
    mode: str
    text: str
    sources: list[str]
    chunk_count: int


def hold_decision(symbol: str, reason: str) -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
        action=Action.HOLD,
        size_pct=0,
        entry_type=EntryType.MARKET,
        confidence=0,
        reasoning=reason,
        strategy_referenced="safety_fallback",
    )
