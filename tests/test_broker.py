import asyncio
from datetime import datetime, timezone

from core.models import Action, RiskVerdict
from execution.broker import Broker
from execution.simulated import SimulatedExchange
from monitoring.store import Store
from tests.helpers import config


def _verdict(symbol="BTC/USDT", action=Action.BUY, size=5, stop=49_000, price=50_000, approved=True):
    return RiskVerdict(
        symbol=symbol,
        action=action,
        size_pct=size if action is not Action.CLOSE else 0,
        stop_loss=stop,
        take_profit=price * 1.04,
        approved=approved,
        reasons=["test"],
        reference_price=price,
        original_action=action,
        original_size_pct=size,
    )


def _broker(tmp_path, exchange=None, dry_run=False):
    cfg = config()
    exchange = exchange or SimulatedExchange(initial_quote=10_000, fee_bps=cfg.risk.fee_bps, slippage_bps=cfg.risk.slippage_bps)
    exchange.last["BTC/USDT"] = 50_000
    exchange.last["ETH/USDT"] = 3_000
    store = Store(f"sqlite:///{tmp_path}/broker.sqlite")
    return Broker(exchange, cfg, store, dry_run=dry_run), exchange, store


def test_buy_is_protected_idempotent_and_reconciles(tmp_path):
    broker, exchange, store = _broker(tmp_path)
    now = datetime.now(timezone.utc)

    async def scenario():
        first = await broker.execute(_verdict(), "cycle-1", 10_000, now)
        second = await broker.execute(_verdict(), "cycle-1", 10_000, now)
        equity = await broker.equity({"BTC/USDT": 50_000})
        notes = await broker.reconcile({"BTC/USDT": 50_000})
        return first, second, equity, notes

    first, second, equity, notes = asyncio.run(scenario())
    assert first.status == "filled"
    assert first.protected
    assert first.stop_order_id
    assert second.status == "duplicate"
    assert exchange.balances["BTC"]["total"] == first.qty
    assert abs(broker.positions["BTC/USDT"].qty - exchange.balances["BTC"]["total"]) < 1e-8
    assert equity < 10_000
    assert notes == [] or all("reconciled" not in note for note in notes)
    assert len(store.get_positions()) == 1


def test_stop_gap_closes_the_position_and_pnl_matches_equity(tmp_path):
    broker, exchange, store = _broker(tmp_path)
    now = datetime.now(timezone.utc)

    async def scenario():
        await broker.execute(_verdict(), "cycle-1", 10_000, now)
        await exchange.process_price("BTC/USDT", 47_000)
        await broker.reconcile({"BTC/USDT": 47_000})
        equity = await broker.equity({"BTC/USDT": 47_000})
        return equity

    equity = asyncio.run(scenario())
    assert "BTC/USDT" not in broker.positions
    assert exchange.balances.get("BTC", {}).get("total", 0) == 0
    trades = store.closed_trades()
    assert len(trades) == 1
    assert trades[0].reason == "stop"
    assert equity == pytest_approx(10_000 + broker.realized_pnl)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=0.05)


def test_failed_stop_unwinds_the_entry(tmp_path):
    exchange = SimulatedExchange(initial_quote=10_000, fee_bps=10, slippage_bps=5)
    exchange.fail_stop_orders = True
    broker, exchange, store = _broker(tmp_path, exchange)
    now = datetime.now(timezone.utc)
    result = asyncio.run(broker.execute(_verdict(), "cycle-1", 10_000, now))
    assert result.status == "unwound"
    assert result.protected
    assert "SUPERSECRET" not in (result.error or "")
    assert broker.positions == {}
    assert exchange.balances.get("BTC", {}).get("total", 0) == 0


def test_failed_unwind_halts_and_tracks_the_position(tmp_path):
    exchange = SimulatedExchange(initial_quote=10_000, fee_bps=10, slippage_bps=5)
    exchange.fail_stop_orders = True
    exchange.fail_market_sells = True
    broker, exchange, _store = _broker(tmp_path, exchange)
    result = asyncio.run(broker.execute(_verdict(), "cycle-1", 10_000, datetime.now(timezone.utc)))
    assert result.status == "error"
    assert result.protected is False
    assert broker.halt_requested
    assert broker.positions["BTC/USDT"].stop_loss is None
    assert exchange.balances["BTC"]["total"] > 0


def test_close_cancels_the_stop_and_flattens(tmp_path):
    broker, exchange, store = _broker(tmp_path)
    now = datetime.now(timezone.utc)

    async def scenario():
        opened = await broker.execute(_verdict(), "cycle-1", 10_000, now)
        closed = await broker.execute(
            _verdict(action=Action.CLOSE, stop=49_000, price=50_500),
            "cycle-2",
            10_000,
            now,
        )
        return opened, closed

    opened, closed = asyncio.run(scenario())
    assert opened.protected
    assert closed.status == "closed"
    assert broker.positions == {}
    assert store.closed_trades()[0].reason == "signal"
    assert exchange.balances.get("BTC", {}).get("total", 0) == 0


def test_dry_run_places_nothing(tmp_path):
    broker, exchange, _store = _broker(tmp_path, dry_run=True)
    result = asyncio.run(broker.execute(_verdict(), "cycle-1", 10_000, datetime.now(timezone.utc)))
    assert result.status == "dry_run"
    assert exchange.trades == []
    assert broker.positions == {}
