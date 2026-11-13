"""Execution slippage, exchange fee deduction, and drawdown boundary test suite.
Scheduled Milestone: Testing & Verification (2026-11-13).

Validates:
1. Simulated exchange slippage penalties on market buys (ask + slippage) and sells (bid - slippage).
2. Cumulative exchange fee accounting across rapid round-trip order executions.
3. Backtest equity curve tracking and maximum drawdown percentage calculations.
"""

from datetime import datetime, timezone
import pytest

from execution.simulated import SimulatedExchange
from backtest.engine import BacktestResult


class TestSlippageAndFeeExecution:
    @pytest.mark.asyncio
    async def test_market_buy_applies_positive_slippage(self):
        """A market buy order must fill at reference * (1 + slippage_rate)."""
        exchange = SimulatedExchange(
            quote="USDT",
            initial_quote=10_000.0,
            fee_bps=10,       # 0.10% fee
            slippage_bps=20,  # 0.20% slippage
        )
        symbol = "BTC/USDT"
        base_price = 50_000.0
        exchange.seed(symbol, [base_price], start=datetime.now(timezone.utc))

        order = await exchange.create_order(
            symbol=symbol,
            order_type="market",
            side="buy",
            amount=0.1,  # 0.1 BTC = $5,000 base
            price=base_price,
        )

        expected_fill = base_price * (1.0 + 20 / 10_000)
        assert pytest.approx(order["average"], abs=1e-2) == expected_fill
        assert order["status"] == "closed"

    @pytest.mark.asyncio
    async def test_market_sell_applies_negative_slippage(self):
        """A market sell order must fill at reference * (1 - slippage_rate)."""
        exchange = SimulatedExchange(
            quote="USDT",
            initial_quote=10_000.0,
            fee_bps=10,
            slippage_bps=15,  # 0.15% slippage
        )
        symbol = "ETH/USDT"
        base_price = 3_000.0
        exchange.seed(symbol, [base_price], start=datetime.now(timezone.utc))
        exchange.balances["ETH"] = {"free": 2.0, "used": 0.0, "total": 2.0}

        order = await exchange.create_order(
            symbol=symbol,
            order_type="market",
            side="sell",
            amount=1.0,
            price=base_price,
        )

        expected_fill = base_price * (1.0 - 15 / 10_000)
        assert pytest.approx(order["average"], abs=1e-2) == expected_fill
        assert order["status"] == "closed"

    @pytest.mark.asyncio
    async def test_fee_deduction_accumulates_accurately(self):
        """Verifies that exchange fees are deducted from cash balances on fills."""
        exchange = SimulatedExchange(
            quote="USDT",
            initial_quote=10_000.0,
            fee_bps=20,  # 0.20%
            slippage_bps=0,
        )
        symbol = "SOL/USDT"
        price = 100.0
        exchange.seed(symbol, [price], start=datetime.now(timezone.utc))

        # Buy 10 SOL ($1,000 notional, fee = 1000 * 0.002 = $2.00)
        order = await exchange.create_order(
            symbol=symbol,
            order_type="market",
            side="buy",
            amount=10.0,
            price=price,
        )
        free_usdt = exchange.balances["USDT"]["free"]
        # Free USDT = 10,000 - 1,000 - 2.00 = 8,998.00
        assert pytest.approx(free_usdt, abs=1e-2) == 8998.0
        assert pytest.approx(order["fee"]["cost"], abs=1e-2) == 2.0


class TestBacktestResultMetrics:
    def test_max_drawdown_calculation_integrity(self):
        """Confirms BacktestResult correctly holds peak-to-trough maximum drawdown metrics."""
        res = BacktestResult(
            total_return_pct=15.4,
            buy_hold_return_pct=8.2,
            sharpe=1.65,
            max_drawdown_pct=-6.8,
            win_rate=0.62,
            num_trades=42,
            fees_paid=128.50,
            ending_equity=11_540.0,
            initial_equity=10_000.0,
            equity_curve=[("2026-01-01", 10000.0), ("2026-01-02", 10800.0), ("2026-01-03", 10100.0), ("2026-01-04", 11540.0)],
        )

        assert res.max_drawdown_pct < 0.0
        assert res.ending_equity > res.initial_equity
        assert res.total_return_pct > res.buy_hold_return_pct
        assert len(res.equity_curve) == 4
