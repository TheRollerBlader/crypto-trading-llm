"""Run the paper loop for a fixed number of cycles and write a markdown report.

The default exchange is the local simulator, so this can run without API keys.
It is not a live-trading soak.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agent.brain import AnthropicDecisionClient, Brain
from core.config import load_config, project_root, resolve_path
from core.loop import TradingLoop
from core.safety import RuntimeOptions, anthropic_api_key
from data.social import SocialCollector, build_sources
from execution.broker import Broker
from execution.simulated import SimulatedExchange
from monitoring.report import performance_markdown
from monitoring.store import Store
from risk.manager import RiskManager
from signals.sentiment import SentimentService
from strategy.context_loader import ContextLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper soak on the local simulator or a caller-supplied loop")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    parser.add_argument("--report", default="reports/soak_report.md")
    parser.add_argument("--decision", choices=["baseline", "llm"], default="baseline")
    return parser.parse_args()


async def run() -> Path:
    args = parse_args()
    root = project_root()
    config = load_config(root / "config.yaml")
    config.exchange.id = "simulated"
    config.symbols = [piece.strip() for piece in args.symbols.split(",") if piece.strip()]
    config.primary_timeframe = "1h"
    config.cycle_interval_minutes = 0
    if args.decision == "llm" and not anthropic_api_key():
        raise SystemExit("ANTHROPIC_API_KEY is required for --decision llm")
    database = root / "logs" / "soak.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    store = Store(f"sqlite:///{database}", root / "logs" / "soak.jsonl")
    exchange = SimulatedExchange(
        quote=config.exchange.quote_currency,
        initial_quote=config.exchange.initial_quote,
        fee_bps=config.risk.fee_bps,
        slippage_bps=config.risk.slippage_bps,
        base_timeframe="1h",
    )
    start = datetime.now(timezone.utc) - timedelta(hours=400)
    rng = np.random.default_rng(11)
    for symbol, origin in zip(config.symbols, [50_000.0, 3_000.0]):
        prices = origin * np.cumprod(1 + rng.normal(0.0004, 0.008, 320))
        exchange.seed(symbol, prices.tolist(), start)
    options = RuntimeOptions(mode="paper", dry_run=False, symbols=config.symbols, decision=args.decision, cycles=args.cycles)
    loop = TradingLoop(
        config,
        exchange,
        Brain(config, AnthropicDecisionClient(config)),
        RiskManager(config),
        Broker(exchange, config, store, dry_run=False),
        SocialCollector(build_sources(config, root / ".cache" / "social")),
        SentimentService(config),
        ContextLoader(config, root),
        store,
        options,
        project_root=root,
    )
    moment = start + timedelta(hours=320)
    for _ in range(args.cycles):
        moment += timedelta(hours=1)
        for symbol in config.symbols:
            last = exchange.last[symbol] * (1 + float(rng.normal(0.0002, 0.006)))
            exchange.append_candle(symbol, last, moment - timedelta(hours=1))
        await loop.run_once(now=moment)
    stats = store.stats()
    initial = config.exchange.initial_quote
    ending = stats["equity"] if stats["equity"] is not None else initial
    report = performance_markdown(
        "Paper soak report",
        {
            "total_return_pct": (ending / initial - 1) * 100,
            "buy_hold_return_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "win_rate": stats["win_rate"],
            "num_trades": stats["closed_trades"],
            "fees_paid": None,
            "ending_equity": ending,
        },
        [
            f"Cycles completed: {stats['cycles']}.",
            "Exchange: local simulator, not Binance Spot Testnet.",
            "Decision engine: " + args.decision + ".",
            "Social sources without credentials were skipped.",
            "This soak does not estimate live fill quality.",
        ],
    )
    destination = Path(args.report)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report)
    return destination


if __name__ == "__main__":
    asyncio.run(run())
