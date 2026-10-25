"""Read-only historical candles for backtests.

This client has no API key and refuses order placement. It is not the paper
broker and it is not used when paper mode checks trading endpoints.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.safety import SafetyError
from core.timeframes import as_utc, timeframe_to_timedelta


# Binance.com returns HTTP 451 in some regions. Later venues are public
# market-data fallbacks. None of them are given an API key.
PUBLIC_VENUES: list[tuple[str, dict[str, str]]] = [
    ("binance", {"BTC/USDT": "BTC/USDT", "ETH/USDT": "ETH/USDT"}),
    ("binanceus", {"BTC/USDT": "BTC/USDT", "ETH/USDT": "ETH/USDT"}),
    ("coinbase", {"BTC/USDT": "BTC/USD", "ETH/USDT": "ETH/USD"}),
]


class ReadOnlyMarketData:
    """Public candles for one ccxt exchange. create_order always raises."""

    def __init__(self, exchange_id: str = "binance"):
        import ccxt

        if not hasattr(ccxt, exchange_id):
            raise SafetyError(f"ccxt has no exchange named {exchange_id}")
        self._exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self._exchange.apiKey = ""
        self._exchange.secret = ""
        self.exchange_id = exchange_id

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int = 1000):
        return self._exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    def create_order(self, *args, **kwargs):
        raise SafetyError("The historical data client cannot place orders.")

    def close(self) -> None:
        close = getattr(self._exchange, "close", None)
        if close:
            close()


def _cache_path(root: Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return root / "data" / "cache" / f"{safe}_{timeframe}.csv"


def _fetch_range(exchange_id: str, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_ms = int(as_utc(start).timestamp() * 1000)
    end_ms = int(as_utc(end).timestamp() * 1000)
    client = ReadOnlyMarketData(exchange_id)
    rows: list[list[float]] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            batch = client.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
    finally:
        client.close()
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame = frame[(frame["timestamp"] >= as_utc(start)) & (frame["timestamp"] <= as_utc(end))]
    duration = timeframe_to_timedelta(timeframe)
    frame = frame[frame["timestamp"] + duration <= pd.Timestamp.now(tz="UTC")]
    return frame.reset_index(drop=True)


def download_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    root: Path,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Download public OHLCV and cache it as CSV. No account and no orders."""

    path = _cache_path(root, symbol, timeframe)
    start_ms = int(as_utc(start).timestamp() * 1000)
    end_ms = int(as_utc(end).timestamp() * 1000)
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        if not cached.empty:
            first = int(cached["timestamp"].iloc[0].timestamp() * 1000)
            last = int(cached["timestamp"].iloc[-1].timestamp() * 1000)
            if first <= start_ms and last >= end_ms - 3_600_000:
                mask = (cached["timestamp"] >= as_utc(start)) & (cached["timestamp"] <= as_utc(end))
                venue_path = path.with_suffix(".venue")
                venue = venue_path.read_text(encoding="utf-8").strip() if venue_path.exists() else "cached public candles"
                return cached.loc[mask].reset_index(drop=True), f"{symbol} loaded from cache ({venue}, {mask.sum()} bars)."

    errors: list[str] = []
    for venue, mapping in PUBLIC_VENUES:
        mapped = mapping.get(symbol)
        if not mapped:
            continue
        try:
            frame = _fetch_range(venue, mapped, timeframe, start, end)
        except Exception as exc:
            errors.append(f"{venue} {mapped}: {exc.__class__.__name__}")
            continue
        if len(frame) < 24:
            errors.append(f"{venue} {mapped}: only {len(frame)} closed bars")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        path.with_suffix(".venue").write_text(f"{venue} {mapped}", encoding="utf-8")
        note = (
            f"{symbol} uses {len(frame)} closed {timeframe} bars from the {venue} public API "
            f"({mapped}). No API key was sent. This downloader cannot place orders."
        )
        if mapped != symbol:
            note += f" {mapped} prices were replayed under the label {symbol}."
        return frame, note
    raise SafetyError("No public candle source returned history. " + "; ".join(errors))


async def download_many(
    symbols: list[str], timeframe: str, start: datetime, end: datetime, root: Path
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    loop = asyncio.get_running_loop()
    frames: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    for symbol in symbols:
        frame, note = await loop.run_in_executor(None, download_ohlcv, symbol, timeframe, start, end, root, True)
        frames[symbol] = frame
        notes.append(note)
    return frames, notes


def default_window(months: int = 6) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = pd.Timestamp(end) - pd.Timedelta(days=int(months * 30.5))
    return start.to_pydatetime(), end
