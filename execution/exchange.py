"""ccxt order client.

Paper mode calls set_sandbox_mode before any order. The caller must run
`authorize` on `trading_urls` and close this client if that check fails.
"""

from __future__ import annotations

from core.config import AppConfig
from core.safety import SafetyError, active_trading_urls


class CcxtExchange:
    def __init__(self, config: AppConfig, api_key: str, secret: str, sandbox: bool):
        import ccxt.async_support as ccxt

        exchange_id = config.exchange.id
        if not hasattr(ccxt, exchange_id):
            raise SafetyError(f"ccxt has no exchange named {exchange_id}")
        klass = getattr(ccxt, exchange_id)
        self.client = klass(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "timeout": 20000,
                "options": {"defaultType": config.exchange.market_type},
            }
        )
        self.id = exchange_id
        self.sandbox = sandbox
        if sandbox:
            if not getattr(self.client, "has", {}).get("sandbox", True):
                # ccxt still exposes set_sandbox_mode for binance. Call it whenever requested.
                pass
            self.client.set_sandbox_mode(True)
        self.trading_urls = active_trading_urls(self.client.urls)

    async def close(self) -> None:
        await self.client.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_order_book(self, symbol: str, limit: int = 10):
        return await self.client.fetch_order_book(symbol, limit=limit)

    async def fetch_ticker(self, symbol: str):
        return await self.client.fetch_ticker(symbol)

    async def fetch_funding_rate(self, symbol: str):
        if not self.client.has.get("fetchFundingRate"):
            return None
        try:
            return await self.client.fetch_funding_rate(symbol)
        except Exception:
            return None

    async def fetch_balance(self):
        return await self.client.fetch_balance()

    async def fetch_open_orders(self, symbol: str | None = None):
        return await self.client.fetch_open_orders(symbol)

    async def fetch_order(self, order_id: str, symbol: str):
        return await self.client.fetch_order(order_id, symbol)

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price, params: dict | None = None):
        params = dict(params or {})
        if order_type == "stop_loss_limit":
            # Binance spot implements protective stops as stop-limit orders.
            if "stopPrice" not in params:
                raise SafetyError("stop order is missing stopPrice")
            return await self.client.create_order(symbol, "limit", side, amount, price, params)
        return await self.client.create_order(symbol, order_type, side, amount, price, params)

    async def cancel_order(self, order_id: str, symbol: str):
        return await self.client.cancel_order(order_id, symbol)
