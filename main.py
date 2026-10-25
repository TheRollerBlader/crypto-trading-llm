"""CLI for paper trading, backtests, the kill switch, and the monitor.

Live trading stays off unless config, environment, --enable-live, an interactive
terminal, and the confirmation phrase all agree. Paper mode then still refuses
production order endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.brain import AnthropicDecisionClient, Brain
from core.config import load_config, project_root, resolve_path
from core.logutil import setup_logging
from core.loop import TradingLoop
from core.safety import (
    CONFIRMATION_PHRASE,
    SafetyError,
    RuntimeOptions,
    anthropic_api_key,
    authorize,
    live_credentials,
    paper_credentials,
    require_paper_keys,
)
from data.social import SocialCollector, build_sources
from execution.broker import Broker
from execution.exchange import CcxtExchange
from execution.simulated import SimulatedExchange
from monitoring.store import Store
from risk.manager import RiskManager
from signals.sentiment import SentimentService
from strategy.context_loader import ContextLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-trading crypto agent. Live trading is locked.")
    parser.add_argument("--mode", choices=["paper", "backtest"], default="paper")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols, for example BTC/USDT,ETH/USDT")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions and risk verdicts without sending orders")
    parser.add_argument("--cycles", type=int, default=0, help="Stop after this many paper cycles. 0 runs until interrupted.")
    parser.add_argument("--decision", choices=["llm", "baseline"], default="")
    parser.add_argument("--exchange", default="", help="Override config exchange id, for example simulated or binance")
    parser.add_argument("--start", default="", help="Backtest start date, YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Backtest end date, YYYY-MM-DD")
    parser.add_argument("--timeframe", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--enable-live", action="store_true", help="Part of the live-trading gate. Not enough on its own.")
    parser.add_argument("--kill", action="store_true", help="Create the kill switch file so the running loop halts")
    parser.add_argument("--close-all", action="store_true", help="With --kill, ask the loop to close paper positions")
    parser.add_argument("--clear-kill", action="store_true")
    parser.add_argument("--clear-halt", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root()
    config = load_config(root / "config.yaml")
    if args.exchange:
        config.exchange.id = args.exchange
    if args.symbols:
        config.symbols = [piece.strip() for piece in args.symbols.split(",") if piece.strip()]
    if args.timeframe:
        config.backtest.timeframe = args.timeframe
        config.primary_timeframe = args.timeframe
    log_path = resolve_path(root, config.monitoring.log_path)
    setup_logging(log_path)
    try:
        if args.kill:
            return _kill(root, config, args.close_all)
        if args.clear_kill:
            path = root / config.monitoring.kill_switch_file
            path.unlink(missing_ok=True)
            print(f"Removed {path.name}")
            return 0
        if args.clear_halt:
            _store(root, config).clear_halt()
            print("Cleared the drawdown and daily-loss halt latch.")
            return 0
        if args.status:
            _print_status(_store(root, config))
            return 0
        if args.dashboard:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(root / "monitoring" / "dashboard.py")], check=False)
            return 0
        decision = args.decision or ("baseline" if args.mode == "backtest" else "llm")
        options = RuntimeOptions(
            mode=args.mode,
            dry_run=args.dry_run,
            symbols=config.symbols,
            cycles=args.cycles or None,
            decision=decision,
            enable_live=args.enable_live,
            interactive=sys.stdin.isatty(),
        )
        if args.mode == "backtest":
            return asyncio.run(_backtest(config, options, root, args))
        if args.enable_live:
            if not sys.stdin.isatty():
                raise SafetyError("Live confirmation needs an interactive terminal.")
            print(f"Type {CONFIRMATION_PHRASE} to confirm live trading. Anything else aborts.")
            options.confirmation = input("> ").strip()
        return asyncio.run(_paper(config, options, root))
    except SafetyError as exc:
        print(f"Safety stop: {exc}", file=sys.stderr)
        return 2


def _store(root: Path, config) -> Store:
    database = resolve_path(root, config.monitoring.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    jsonl = resolve_path(root, config.monitoring.jsonl_path)
    return Store(f"sqlite:///{database}", jsonl)


def _kill(root: Path, config, close_all: bool) -> int:
    path = root / config.monitoring.kill_switch_file
    path.write_text("close_all\n" if close_all else "halt\n", encoding="utf-8")
    print(f"Kill switch written to {path}. The next cycle will halt and will not open trades.")
    if close_all:
        print("The next cycle will also close paper positions.")
    return 0


def _print_status(store: Store) -> None:
    stats = store.stats()
    win = "n/a" if stats["win_rate"] is None else f"{stats['win_rate'] * 100:.1f}%"
    print(f"cycles={stats['cycles']} equity={stats['equity']} realized_pnl={stats['realized_pnl']:.4f}")
    print(f"closed_trades={stats['closed_trades']} win_rate={win}")
    if not stats["positions"]:
        print("positions=none")
    for position in stats["positions"]:
        print(f"position {position['symbol']} qty={position['qty']} entry={position['entry_price']} stop={position['stop_loss']}")


async def _paper(config, options: RuntimeOptions, root: Path) -> int:
    authorize(config, options, check_urls=False)
    if options.decision == "llm" and not anthropic_api_key():
        raise SafetyError("ANTHROPIC_API_KEY is not set. Re-run with --decision baseline or add the key to .env.")
    require_paper_keys(config, options)
    exchange = _build_exchange(config, options)
    if not isinstance(exchange, SimulatedExchange):
        try:
            authorize(config, options, exchange.trading_urls, check_urls=True)
        except SafetyError:
            await exchange.close()
            raise
    store = _store(root, config)
    brain = Brain(config, AnthropicDecisionClient(config))
    loop = TradingLoop(
        config=config,
        exchange=exchange,
        brain=brain,
        risk=RiskManager(config),
        broker=Broker(exchange, config, store, dry_run=options.dry_run),
        social=SocialCollector(build_sources(config, root / ".cache" / "social")),
        sentiment=SentimentService(config),
        context_loader=ContextLoader(config, root),
        store=store,
        options=options,
        project_root=root,
    )
    cycles = options.cycles
    try:
        await loop.run(max_cycles=cycles)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        await exchange.close()
    return 0


def _build_exchange(config, options: RuntimeOptions):
    if config.exchange.id.lower() == "simulated":
        return SimulatedExchange(
            quote=config.exchange.quote_currency,
            initial_quote=config.exchange.initial_quote,
            fee_bps=config.risk.fee_bps,
            slippage_bps=config.risk.slippage_bps,
            base_timeframe=config.primary_timeframe,
        )
    label = authorize(config, options, check_urls=False)
    if label == "live":
        key, secret = live_credentials()
        sandbox = False
    else:
        key, secret = paper_credentials()
        sandbox = True
    return CcxtExchange(config, key, secret, sandbox=sandbox)


async def _backtest(config, options: RuntimeOptions, root: Path, args) -> int:
    authorize(config, options, check_urls=False)
    from backtest.engine import BacktestEngine
    from data.history import default_window, download_many
    from monitoring.report import performance_markdown

    start, end = default_window(6)
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    if args.end:
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    frames, source_notes = await download_many(config.symbols, config.backtest.timeframe, start, end, root)
    if any(frame.empty for frame in frames.values()):
        raise SafetyError("Historical download returned no candles. Check the network and the symbol names.")
    store = _store(root, config)
    engine = BacktestEngine(config, store)
    result = await engine.run(frames)
    report = performance_markdown(
        "Backtest report",
        {
            "total_return_pct": result.total_return_pct,
            "buy_hold_return_pct": result.buy_hold_return_pct,
            "sharpe": result.sharpe,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "num_trades": result.num_trades,
            "fees_paid": result.fees_paid,
            "ending_equity": result.ending_equity,
        },
        source_notes + result.notes,
    )
    destination = Path(args.report) if args.report else root / "reports" / "backtest_report.md"
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
