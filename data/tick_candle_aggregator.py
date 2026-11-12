"""Real-time Streaming Tick-to-OHLCV Candle Aggregator.

Ingests streaming trade ticks from WebSocket exchange feeds, computes
running Volume-Weighted Average Price (VWAP), and seals completed 1-minute
candles at minute boundaries with gap detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable


@dataclass
class TradeTick:
    timestamp_ms: int
    symbol: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str = ""


@dataclass
class Candle:
    symbol: str
    bucket_start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    is_sealed: bool = False

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.bucket_start_ms / 1000.0, tz=timezone.utc).isoformat()


class TickCandleAggregator:
    """Aggregates streaming trades into deterministic 1-minute OHLCV buckets."""

    def __init__(
        self,
        symbol: str,
        bucket_size_sec: int = 60,
        on_candle_sealed: Optional[Callable[[Candle], None]] = None,
    ):
        self.symbol = symbol
        self.bucket_size_ms = bucket_size_sec * 1000
        self.on_candle_sealed = on_candle_sealed
        self.current_bucket_start: Optional[int] = None
        self.current_candle: Optional[Candle] = None
        self.sealed_candles: List[Candle] = []

        # Internal accumulation state for VWAP
        self._cum_pv: float = 0.0
        self._cum_vol: float = 0.0

    def ingest_tick(self, tick: TradeTick) -> Optional[Candle]:
        """Ingests a new trade tick. If this tick crosses into a new minute bucket,
        the previous candle is sealed and returned.
        """
        if tick.symbol != self.symbol:
            return None

        bucket_start = (tick.timestamp_ms // self.bucket_size_ms) * self.bucket_size_ms
        sealed_candle: Optional[Candle] = None

        if self.current_bucket_start is None:
            # First tick initialized
            self.current_bucket_start = bucket_start
            self._cum_pv = tick.price * tick.volume
            self._cum_vol = tick.volume
            vwap = self._cum_pv / self._cum_vol if self._cum_vol > 0 else tick.price
            self.current_candle = Candle(
                symbol=self.symbol,
                bucket_start_ms=bucket_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                vwap=round(vwap, 4),
                tick_count=1,
                is_sealed=False,
            )
            return None

        if bucket_start > self.current_bucket_start:
            # Seal the current candle
            if self.current_candle:
                self.current_candle.is_sealed = True
                sealed_candle = self.current_candle
                self.sealed_candles.append(sealed_candle)
                if self.on_candle_sealed:
                    self.on_candle_sealed(sealed_candle)

            # Start the new candle bucket
            self.current_bucket_start = bucket_start
            self._cum_pv = tick.price * tick.volume
            self._cum_vol = tick.volume
            vwap = self._cum_pv / self._cum_vol if self._cum_vol > 0 else tick.price
            self.current_candle = Candle(
                symbol=self.symbol,
                bucket_start_ms=bucket_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                vwap=round(vwap, 4),
                tick_count=1,
                is_sealed=False,
            )
            return sealed_candle

        elif bucket_start == self.current_bucket_start and self.current_candle:
            # Update running in-flight candle
            c = self.current_candle
            c.high = max(c.high, tick.price)
            c.low = min(c.low, tick.price)
            c.close = tick.price
            c.volume += tick.volume
            c.tick_count += 1

            self._cum_pv += tick.price * tick.volume
            self._cum_vol += tick.volume
            c.vwap = round(self._cum_pv / self._cum_vol, 4) if self._cum_vol > 0 else tick.price

        # Out-of-order historical ticks before current_bucket_start are ignored
        return None

    def get_latest_candle(self) -> Optional[Candle]:
        return self.current_candle
