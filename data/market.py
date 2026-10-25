"""Fetch closed candles, order books, and funding, then attach indicators."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from core.config import AppConfig
from core.models import MarketSnapshot, OrderBookLevel, OrderBookSnapshot
from core.timeframes import timeframe_to_timedelta, utcnow
from data.indicators import compute_indicator_frame, timeframe_view_from_frame
from data.resample import drop_incomplete, ohlcv_frame, resample_ohlcv

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(self, exchange, config: AppConfig):
        self.exchange = exchange
        self.config = config

    async def snapshots(self, symbols: list[str], now: datetime | None = None) -> list[MarketSnapshot]:
        moment = now or utcnow()
        return [await self.snapshot(symbol, moment) for symbol in symbols]

    async def snapshot(self, symbol: str, now: datetime) -> MarketSnapshot:
        warnings: list[str] = []
        frames: dict[str, pd.DataFrame] = {}
        primary = self.config.primary_timeframe
        try:
            rows = await self.exchange.fetch_ohlcv(symbol, primary, limit=self.config.candle_limit)
            frames[primary] = drop_incomplete(ohlcv_frame(rows), primary, now)
        except Exception as exc:
            logger.warning("ohlcv failed for %s %s: %s", symbol, primary, exc)
            warnings.append(f"ohlcv {primary} failed")
            frames[primary] = ohlcv_frame([])

        for timeframe in self.config.timeframes:
            if timeframe == primary:
                continue
            try:
                rows = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.config.candle_limit)
                fetched = drop_incomplete(ohlcv_frame(rows), timeframe, now)
            except Exception as exc:
                logger.info("ohlcv %s %s unavailable: %s", symbol, timeframe, exc)
                fetched = ohlcv_frame([])
                warnings.append(f"ohlcv {timeframe} failed")
            if len(fetched) < 30 and not frames[primary].empty:
                fetched = resample_ohlcv(frames[primary], timeframe, now)
                warnings.append(f"{timeframe} resampled from {primary}")
            frames[timeframe] = fetched

        primary_frame = frames.get(primary, ohlcv_frame([]))
        if primary_frame.empty:
            return MarketSnapshot(
                symbol=symbol,
                as_of=now,
                last_price=None,
                stale=True,
                primary=timeframe_view_from_frame(primary_frame, primary_frame, primary),
                warnings=warnings + ["no closed candles"],
            )

        indicator_frames = {name: compute_indicator_frame(frame) for name, frame in frames.items() if not frame.empty}
        primary_indicators = indicator_frames[primary]
        views = {}
        for name, frame in indicator_frames.items():
            views[name] = timeframe_view_from_frame(
                frames[name],
                frame,
                name,
                include_profile=(name == primary),
            )
        last_closed = primary_indicators.iloc[-1]
        closed_at = last_closed["timestamp"] + timeframe_to_timedelta(primary)
        age = now - closed_at.to_pydatetime()
        stale = age > self.config.stale_candle_multiple * timeframe_to_timedelta(primary)

        last_price = float(last_closed["close"])
        book = OrderBookSnapshot()
        funding = None
        volume_24h = None
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            if ticker.get("last") is not None:
                last_price = float(ticker["last"])
            volume_24h = _optional_float(ticker.get("quoteVolume"))
        except Exception as exc:
            warnings.append(f"ticker failed: {exc.__class__.__name__}")
        try:
            book = _order_book(await self.exchange.fetch_order_book(symbol, limit=10))
        except Exception as exc:
            warnings.append(f"order book failed: {exc.__class__.__name__}")
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            if isinstance(funding, dict):
                funding = _optional_float(funding.get("fundingRate"))
        except Exception:
            funding = None
            warnings.append("funding rate unavailable")

        return MarketSnapshot(
            symbol=symbol,
            as_of=now,
            last_price=last_price,
            stale=stale,
            primary=views[primary],
            timeframes=views,
            order_book=book,
            funding_rate=funding if isinstance(funding, float) else None,
            volume_24h=volume_24h,
            warnings=warnings,
        )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_book(raw: dict) -> OrderBookSnapshot:
    bids = [OrderBookLevel(price=float(price), amount=float(amount)) for price, amount in raw.get("bids", [])[:10]]
    asks = [OrderBookLevel(price=float(price), amount=float(amount)) for price, amount in raw.get("asks", [])[:10]]
    bid = bids[0].price if bids else None
    ask = asks[0].price if asks else None
    spread = None
    if bid and ask and bid > 0:
        spread = (ask - bid) / bid * 10_000
    return OrderBookSnapshot(bid=bid, ask=ask, spread_bps=spread, bids=bids, asks=asks)
