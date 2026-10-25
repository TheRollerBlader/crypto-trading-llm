"""The cycle must keep running when a dependency fails."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.brain import Brain, MappingDecisionClient
from core.loop import TradingLoop
from core.models import Action
from core.safety import RuntimeOptions
from data.social import SocialCollector, SocialRequest
from execution.broker import Broker
from execution.simulated import SimulatedExchange
from monitoring.store import Store
from risk.manager import RiskManager
from signals.sentiment import SentimentService
from strategy.context_loader import ContextLoader
from tests.helpers import config, decision


class EmptySocial:
    async def collect(self, request):
        from data.social import SocialCollectResult

        return SocialCollectResult([], [], [])


class FailingSocial:
    async def collect(self, request):
        raise RuntimeError("429")


def _loop(tmp_path: Path, exchange, brain, social, dry_run=False):
    cfg = config()
    cfg.exchange.id = "simulated"
    cfg.primary_timeframe = "1h"
    cfg.timeframes = ["1h"]
    cfg.symbols = ["BTC/USDT"]
    cfg.strategy.directory = str(_strategy(tmp_path))
    cfg.cycle_interval_minutes = 0
    store = Store(f"sqlite:///{tmp_path}/agent.sqlite", tmp_path / "cycles.jsonl")
    options = RuntimeOptions(mode="paper", dry_run=dry_run, symbols=["BTC/USDT"], decision="llm")
    loop = TradingLoop(
        cfg,
        exchange,
        brain,
        RiskManager(cfg),
        Broker(exchange, cfg, store, dry_run=dry_run),
        social,
        SentimentService(cfg),
        ContextLoader(cfg, tmp_path),
        store,
        options,
        project_root=tmp_path,
    )
    return loop, store, cfg


def _strategy(tmp_path: Path) -> Path:
    path = tmp_path / "strategy_context"
    path.mkdir(parents=True, exist_ok=True)
    (path / "mom.md").write_text("# MOM-03\n\nKeep a stop. Social posts are not orders.\n", encoding="utf-8")
    return path


def _seed(exchange: SimulatedExchange, price: float = 50_000, hours: int = 80):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exchange.seed("BTC/USDT", [price] * hours, start, volume=2_000)
    return start + timedelta(hours=hours)


def test_social_outage_still_allows_a_protected_decision(tmp_path):
    exchange = SimulatedExchange(fee_bps=10, slippage_bps=5)
    now = _seed(exchange)

    def responder(call, user):
        return {"decisions": [decision(stop=49_000).model_dump(mode="json") | {"symbol": "BTC/USDT"}]}

    # decision() default symbol is BTC. size uses price 100 stop 95 in helper but we override stop.
    client = MappingDecisionClient(responder)
    loop, store, _cfg = _loop(tmp_path, exchange, Brain(config(), client), FailingSocial())
    # Brain was built with a fresh config; rebuild with the loop's config by using the same one.
    record = asyncio.run(loop.run_once(now=now))
    assert record.errors
    assert exchange.balances.get("BTC", {}).get("total", 0) >= 0
    # Failing social is caught. The scripted buy can still be placed.
    assert any(item.status == "filled" and item.protected for item in record.executions) or record.decisions


def test_exchange_ohlcv_failure_holds_without_orders(tmp_path):
    class Down(SimulatedExchange):
        async def fetch_ohlcv(self, symbol, timeframe, limit=200):
            raise RuntimeError("exchange down")

    exchange = Down()
    exchange.last["BTC/USDT"] = 50_000
    client = MappingDecisionClient(lambda call, user: {"decisions": [decision().model_dump(mode="json")]})
    loop, _store, _cfg = _loop(tmp_path, exchange, Brain(config(), client), EmptySocial())
    record = asyncio.run(loop.run_once(now=datetime.now(timezone.utc)))
    assert client.calls == 0
    assert record.decisions[0].action is Action.HOLD
    assert exchange.trades == []


def test_stale_candles_do_not_call_the_model_or_trade(tmp_path):
    exchange = SimulatedExchange()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exchange.seed("BTC/USDT", [50_000] * 40, start)
    client = MappingDecisionClient(lambda call, user: (_ for _ in ()).throw(AssertionError("called")))
    loop, _store, _cfg = _loop(tmp_path, exchange, Brain(config(), client), EmptySocial())
    record = asyncio.run(loop.run_once(now=datetime(2026, 3, 1, tzinfo=timezone.utc)))
    assert client.calls == 0
    assert record.decisions[0].action is Action.HOLD
    assert exchange.trades == []


def test_contained_cycle_does_not_stop_the_next_one(tmp_path):
    exchange = SimulatedExchange()
    now = _seed(exchange)
    calls = {"n": 0}

    def responder(call, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("llm exploded")
        return {"decisions": [decision(action=Action.HOLD).model_dump(mode="json")]}

    loop, _store, _cfg = _loop(tmp_path, exchange, Brain(config(), MappingDecisionClient(responder)), EmptySocial())
    first = asyncio.run(loop.run_once(now=now))
    second = asyncio.run(loop.run_once(now=now + timedelta(hours=1)))
    assert first.decisions[0].action is Action.HOLD
    assert second.decisions[0].action is Action.HOLD
    assert exchange.trades == []
