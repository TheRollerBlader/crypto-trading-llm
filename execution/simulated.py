"""Local paper exchange used in tests and when no sandbox keys are configured.

Fills are immediate for market orders. Resting stops and take-profits trigger
from prices the caller supplies. This is not a Binance testnet connection.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from core.timeframes import timeframe_to_timedelta


class SimulatedExchange:
    def __init__(
        self,
        quote: str = "USDT",
        initial_quote: float = 10_000.0,
        fee_bps: float = 10,
        slippage_bps: float = 5,
        base_timeframe: str = "1h",
    ):
        self.id = "simulated"
        self.sandbox = True
        self.quote = quote
        self.fee_rate = fee_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.base_timeframe = base_timeframe
        self.balances: dict[str, dict[str, float]] = {
            quote: {"free": initial_quote, "used": 0.0, "total": initial_quote}
        }
        self.orders: dict[str, dict] = {}
        self.by_client: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.candles: dict[str, list[dict]] = {}
        self.last: dict[str, float] = {}
        self.fill_prices: dict[str, float] = {}
        self.visible_until: datetime | None = None
        self.fail_stop_orders = False
        self.fail_market_sells = False
        self._next_id = 1
        self.trading_urls = ["simulated://paper"]

    def seed(self, symbol: str, prices: list[float], start: datetime, volume: float = 1_000.0, spread: float = 0.0) -> None:
        moment = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        step = timeframe_to_timedelta(self.base_timeframe)
        self.candles[symbol] = []
        for price in prices:
            self.candles[symbol].append(
                self._candle(moment, price, price + spread, max(price - spread, 0.0), price, volume)
            )
            moment += step
        self.last[symbol] = prices[-1]

    def append_candle(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        volume: float = 1_000.0,
        high: float | None = None,
        low: float | None = None,
        open_: float | None = None,
    ) -> None:
        moment = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        self.candles.setdefault(symbol, []).append(
            self._candle(moment, open_ if open_ is not None else price, high if high is not None else price, low if low is not None else price, price, volume)
        )
        self.last[symbol] = price

    def set_market_fill(self, symbol: str, price: float) -> None:
        self.fill_prices[symbol] = price

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        rows = self.candles.get(symbol, [])
        if self.visible_until is not None:
            rows = [row for row in rows if row["timestamp"] <= self.visible_until]
        if timeframe != self.base_timeframe and rows:
            frame = pd.DataFrame(rows)
            from data.resample import resample_ohlcv

            now = rows[-1]["timestamp"] + timeframe_to_timedelta(self.base_timeframe)
            frame = resample_ohlcv(frame, timeframe, now)
            rows = frame.to_dict("records")
        rows = rows[-limit:]
        output = []
        for row in rows:
            stamp = row["timestamp"]
            millis = int(stamp.timestamp() * 1000)
            output.append([millis, row["open"], row["high"], row["low"], row["close"], row["volume"]])
        return output

    async def fetch_ticker(self, symbol: str):
        last = self.last[symbol]
        return {"symbol": symbol, "last": last, "bid": last * 0.9999, "ask": last * 1.0001, "quoteVolume": 1_000_000.0}

    async def fetch_order_book(self, symbol: str, limit: int = 10):
        last = self.last[symbol]
        bids = [[last * (1 - 0.0001 * (i + 1)), 1.0] for i in range(limit)]
        asks = [[last * (1 + 0.0001 * (i + 1)), 1.0] for i in range(limit)]
        return {"bids": bids, "asks": asks}

    async def fetch_funding_rate(self, symbol: str):
        return None

    async def fetch_balance(self):
        currencies = {}
        free = {}
        total = {}
        for name, row in self.balances.items():
            currencies[name] = {"free": row["free"], "used": row["used"], "total": row["total"]}
            free[name] = row["free"]
            total[name] = row["total"]
        return {**currencies, "free": free, "total": total}

    async def fetch_open_orders(self, symbol: str | None = None):
        orders = [self._public(order) for order in self.orders.values() if order["status"] == "open"]
        if symbol:
            orders = [order for order in orders if order["symbol"] == symbol]
        return orders

    async def fetch_order(self, order_id: str, symbol: str):
        order = self.orders[str(order_id)]
        return self._public(order)

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price, params: dict | None = None):
        params = params or {}
        client_id = params.get("clientOrderId")
        if client_id and client_id in self.by_client:
            existing = self._public(self.by_client[client_id])
            existing["duplicate"] = True
            return existing
        if order_type == "stop_loss_limit" and self.fail_stop_orders:
            raise RuntimeError("injected stop failure api_key=SUPERSECRET123456")
        if order_type == "market" and side == "sell" and self.fail_market_sells:
            raise RuntimeError("injected sell failure")
        order = {
            "id": str(self._next_id),
            "clientOrderId": client_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": float(amount),
            "filled": 0.0,
            "remaining": float(amount),
            "average": None,
            "price": None if price is None else float(price),
            "stopPrice": params.get("stopPrice"),
            "status": "open",
            "fee": 0.0,
            "timestamp": int(time.time() * 1000),
            "duplicate": False,
        }
        self._next_id += 1
        self.orders[order["id"]] = order
        if client_id:
            self.by_client[client_id] = order
        if order_type == "market":
            reference = self.fill_prices.get(symbol, self.last[symbol])
            self._fill(order, reference, apply_slippage=True)
        return self._public(order)

    async def cancel_order(self, order_id: str, symbol: str):
        order = self.orders[str(order_id)]
        if order["status"] == "open":
            order["status"] = "canceled"
        return self._public(order)

    async def process_price(self, symbol: str, price: float, high: float | None = None, low: float | None = None, open_: float | None = None):
        self.last[symbol] = price
        high = price if high is None else high
        low = price if low is None else low
        open_ = price if open_ is None else open_
        resting = [order for order in self.orders.values() if order["symbol"] == symbol and order["status"] == "open"]
        stops = [order for order in resting if order["type"] == "stop_loss_limit"]
        limits = [order for order in resting if order["type"] == "limit"]
        for order in stops:
            stop = float(order["stopPrice"])
            if order["side"] == "sell" and low <= stop:
                reference = open_ if open_ < stop else stop
                self._fill(order, reference, apply_slippage=True)
                self._cancel_siblings(limits)
                return
            if order["side"] == "buy" and high >= stop:
                reference = open_ if open_ > stop else stop
                self._fill(order, reference, apply_slippage=True)
                self._cancel_siblings(limits)
                return
        for order in limits:
            limit_price = float(order["price"])
            if order["side"] == "sell" and high >= limit_price:
                self._fill(order, limit_price, apply_slippage=False)
                self._cancel_siblings(stops)
                return
            if order["side"] == "buy" and low <= limit_price:
                self._fill(order, limit_price, apply_slippage=False)
                self._cancel_siblings(stops)
                return

    async def close(self) -> None:
        return None

    def _cancel_siblings(self, orders: list[dict]) -> None:
        for order in orders:
            if order["status"] == "open":
                order["status"] = "canceled"

    def _fill(self, order: dict, reference: float, apply_slippage: bool) -> None:
        slip = self.slippage if apply_slippage else 0.0
        if order["side"] == "buy":
            fill = reference * (1 + slip)
            notional = order["amount"] * fill
            fee = notional * self.fee_rate
            quote = self._bucket(self.quote)
            if quote["free"] + 1e-8 < notional + fee:
                order["status"] = "rejected"
                raise RuntimeError("insufficient quote balance")
            quote["free"] -= notional + fee
            quote["total"] = quote["free"]
            base = self._bucket(order["symbol"].split("/")[0])
            base["free"] += order["amount"]
            base["total"] = base["free"]
        else:
            fill = reference * (1 - slip)
            notional = order["amount"] * fill
            fee = notional * self.fee_rate
            base_name = order["symbol"].split("/")[0]
            base = self._bucket(base_name)
            if base["free"] + 1e-8 < order["amount"]:
                order["status"] = "rejected"
                raise RuntimeError("insufficient base balance")
            base["free"] -= order["amount"]
            base["total"] = base["free"]
            quote = self._bucket(self.quote)
            quote["free"] += notional - fee
            quote["total"] = quote["free"]
        order["status"] = "closed"
        order["filled"] = order["amount"]
        order["remaining"] = 0.0
        order["average"] = fill
        order["fee"] = fee
        self.trades.append(
            {
                "symbol": order["symbol"],
                "side": order["side"],
                "qty": order["amount"],
                "price": fill,
                "fee": fee,
                "order_id": order["id"],
            }
        )

    def _bucket(self, currency: str) -> dict[str, float]:
        if currency not in self.balances:
            self.balances[currency] = {"free": 0.0, "used": 0.0, "total": 0.0}
        return self.balances[currency]

    def _public(self, order: dict) -> dict:
        return {
            "id": order["id"],
            "clientOrderId": order["clientOrderId"],
            "symbol": order["symbol"],
            "type": order["type"],
            "side": order["side"],
            "amount": order["amount"],
            "filled": order["filled"],
            "remaining": order["remaining"],
            "average": order["average"],
            "price": order["price"],
            "stopPrice": order.get("stopPrice"),
            "status": order["status"],
            "fee": {"cost": order["fee"], "currency": self.quote},
            "duplicate": order.get("duplicate", False),
            "timestamp": order["timestamp"],
        }

    @staticmethod
    def _candle(timestamp: datetime, open_: float, high: float, low: float, close: float, volume: float) -> dict:
        return {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
