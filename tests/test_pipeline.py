import asyncio
from datetime import datetime, timedelta, timezone

from agent.brain import Brain, MappingDecisionClient
from core.loop import TradingLoop
from core.models import Action
from core.safety import RuntimeOptions
from data.social import SocialCollectResult
from execution.broker import Broker
from execution.simulated import SimulatedExchange
from monitoring.store import Store
from risk.manager import RiskManager
from signals.sentiment import SentimentService
from strategy.context_loader import ContextLoader
from tests.helpers import config


class QuietSocial:
    async def collect(self, request):
        return SocialCollectResult([], [], ["twitter", "reddit", "instagram"])


def _system(tmp_path, dry_run: bool):
    cfg = config()
    cfg.exchange.id = "simulated"
    cfg.primary_timeframe = "1h"
    cfg.timeframes = ["1h"]
    cfg.symbols = ["BTC/USDT", "ETH/USDT"]
    cfg.cycle_interval_minutes = 0
    docs = tmp_path / "strategy_context"
    docs.mkdir()
    (docs / "mom.md").write_text("# MOM-03\n\nUse a stop on every entry.\n", encoding="utf-8")
    cfg.strategy.directory = str(docs)
    exchange = SimulatedExchange(
        initial_quote=10_000,
        fee_bps=cfg.risk.fee_bps,
        slippage_bps=cfg.risk.slippage_bps,
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exchange.seed("BTC/USDT", [50_000] * 120, start, volume=2_000)
    exchange.seed("ETH/USDT", [3_000] * 120, start, volume=2_000)
    store = Store(f"sqlite:///{tmp_path}/pipe.sqlite", tmp_path / "cycles.jsonl")
    calls = {"n": 0}

    def responder(call, user):
        calls["n"] += 1
        if dry_run or calls["n"] == 1:
            payload = [_buy("BTC/USDT", 50_000, 49_000), _buy("ETH/USDT", 3_000, 2_940)]
        else:
            payload = [_hold("BTC/USDT"), _hold("ETH/USDT")]
        return {"decisions": payload}

    options = RuntimeOptions(mode="paper", dry_run=dry_run, symbols=cfg.symbols, decision="llm")
    loop = TradingLoop(
        cfg,
        exchange,
        Brain(cfg, MappingDecisionClient(responder)),
        RiskManager(cfg),
        Broker(exchange, cfg, store, dry_run=dry_run),
        QuietSocial(),
        SentimentService(cfg),
        ContextLoader(cfg, tmp_path),
        store,
        options,
        project_root=tmp_path,
    )
    return loop, exchange, store, start + timedelta(hours=120), calls


def _buy(symbol, price, stop):
    return {
        "symbol": symbol,
        "action": "BUY",
        "size_pct": 5,
        "entry_type": "MARKET",
        "limit_price": None,
        "stop_loss": stop,
        "take_profit": price * 1.04,
        "confidence": 0.8,
        "reasoning": "test entry with a protective stop",
        "strategy_referenced": "MOM-03",
    }


def _hold(symbol):
    return {
        "symbol": symbol,
        "action": "HOLD",
        "size_pct": 0,
        "entry_type": "MARKET",
        "limit_price": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": 0,
        "reasoning": "nothing new",
        "strategy_referenced": "MOM-03",
    }


def test_dry_run_completes_twenty_cycles_without_orders(tmp_path):
    loop, exchange, store, start, _calls = _system(tmp_path, dry_run=True)
    records = asyncio.run(_run(loop, exchange, start, crash_on=None))
    assert len(records) == 20
    assert store.cycle_count() == 20
    assert exchange.trades == []
    assert {item.symbol for item in records[0].decisions} == {"BTC/USDT", "ETH/USDT"}


def test_twenty_cycles_fill_stops_and_reconcile_pnl(tmp_path):
    loop, exchange, store, start, calls = _system(tmp_path, dry_run=False)
    records = asyncio.run(_run(loop, exchange, start, crash_on=10))
    assert len(records) == 20
    assert calls["n"] >= 1
    filled = [item for item in records[0].executions if item.status == "filled"]
    assert {item.symbol for item in filled} == {"BTC/USDT", "ETH/USDT"}
    assert all(item.protected and item.stop_order_id for item in filled)
    assert exchange.balances.get("BTC", {}).get("total", 0) == 0
    assert exchange.balances.get("ETH", {}).get("total", 0) == 0
    assert loop.broker.positions == {}
    trades = store.closed_trades()
    assert len(trades) == 2
    assert {trade.reason for trade in trades} == {"stop"}
    equity = records[-1].equity
    assert equity == __import__("pytest").approx(10_000 + loop.broker.realized_pnl, abs=0.05)
    assert equity < 10_000


def test_kill_switch_closes_positions_and_skips_the_model(tmp_path):
    loop, exchange, _store, start, calls = _system(tmp_path, dry_run=False)
    asyncio.run(loop.run_once(now=start))
    assert loop.broker.positions
    kill = tmp_path / "KILL_SWITCH"
    kill.write_text("close_all\n", encoding="utf-8")
    before = calls["n"]
    record = asyncio.run(loop.run_once(now=start + timedelta(hours=1)))
    assert record.halted
    assert calls["n"] == before
    assert loop.broker.positions == {}
    assert record.executions
    assert all(item.action is Action.CLOSE for item in record.executions)


async def _run(loop, exchange, start, crash_on):
    records = []
    for index in range(20):
        now = start + timedelta(hours=index)
        btc = 47_000 if crash_on is not None and index == crash_on else 50_000
        eth = 2_800 if crash_on is not None and index == crash_on else 3_000
        exchange.append_candle("BTC/USDT", btc, now - timedelta(hours=1), volume=2_000)
        exchange.append_candle("ETH/USDT", eth, now - timedelta(hours=1), volume=2_000)
        records.append(await loop.run_once(now=now))
    return records
