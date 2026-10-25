"""One decision cycle: data, sentiment, LLM, risk, then optional paper orders."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent.baseline import BaselineDecisionClient
from agent.brain import Brain, CycleContext
from core.config import AppConfig
from core.models import CycleRecord, hold_decision
from core.redact import redact
from core.safety import SafetyError
from core.timeframes import utcnow
from data.market import MarketDataService
from monitoring.store import RiskState

logger = logging.getLogger(__name__)


def make_cycle_id(now: datetime, minutes: int, symbols: list[str]) -> str:
    bucket = int(now.timestamp()) // max(minutes, 1) // 60
    raw = f"{bucket}|{','.join(symbols)}|{now.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TradingLoop:
    def __init__(
        self,
        config: AppConfig,
        exchange,
        brain: Brain,
        risk,
        broker,
        social,
        sentiment,
        context_loader,
        store,
        options,
        project_root: Path | None = None,
    ):
        self.config = config
        self.exchange = exchange
        self.brain = brain
        self.baseline = BaselineDecisionClient(config)
        self.risk = risk
        self.broker = broker
        self.social = social
        self.sentiment = sentiment
        self.context_loader = context_loader
        self.store = store
        self.options = options
        self.market = MarketDataService(exchange, config)
        self.project_root = project_root or Path.cwd()

    async def run(self, max_cycles: int | None = None) -> list[CycleRecord]:
        records = []
        count = 0
        while max_cycles is None or count < max_cycles:
            try:
                records.append(await self.run_once())
            except SafetyError:
                raise
            except Exception as exc:
                logger.exception("cycle crashed and was contained")
                now = utcnow()
                record = CycleRecord(
                    cycle_id=make_cycle_id(now, self.config.cycle_interval_minutes, self.symbols),
                    started_at=now,
                    mode=self.options.mode,
                    dry_run=self.options.dry_run,
                    halted=True,
                    inputs_summary={"error": redact(str(exc))},
                    decisions=[],
                    verdicts=[],
                    executions=[],
                    errors=[redact(str(exc))],
                )
                self.store.save_cycle(record)
                records.append(record)
            count += 1
            if max_cycles is not None and count >= max_cycles:
                break
            await asyncio.sleep(max(self.config.cycle_interval_minutes, 0) * 60)
        return records

    @property
    def symbols(self) -> list[str]:
        return self.options.symbols or self.config.symbols

    async def run_once(self, now: datetime | None = None) -> CycleRecord:
        moment = now or utcnow()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        symbols = self.symbols
        cycle_id = make_cycle_id(moment, self.config.cycle_interval_minutes, symbols)
        errors: list[str] = []
        markets = await self.market.snapshots(symbols, moment)
        marks = {market.symbol: market.last_price for market in markets if market.last_price is not None}
        notes = await self.broker.reconcile(marks)
        equity = await self.broker.equity(marks) if marks else self.config.exchange.initial_quote
        quote = await self.broker.quote_balance()
        state = self._roll_risk(equity, moment)
        if self.broker.halt_requested:
            state.halted = True
            state.halt_reason = self.broker.halt_reason or "unprotected_position"
        kill = self._kill_switch()
        if kill is not None:
            executions = []
            if "close_all" in kill:
                executions = await self.broker.close_all(marks, cycle_id, moment)
            record = self._record(
                cycle_id,
                moment,
                markets,
                [],
                [],
                executions,
                ["kill switch is engaged"],
                True,
                equity,
                social_failed=[],
                social_skipped=[],
                notes=notes,
            )
            self.store.save_cycle(record)
            self.store.add_equity(moment, equity)
            return record

        if any(market.stale or market.last_price is None for market in markets):
            decisions = [hold_decision(symbol, "stale or missing market data") for symbol in symbols]
            record = self._record(cycle_id, moment, markets, decisions, [], [], ["stale market data"], state.halted, equity, [], [], notes)
            self.store.save_cycle(record)
            self.store.add_equity(moment, equity)
            self.store.save_risk_state(state)
            return record

        posts = []
        failed: list[str] = []
        skipped: list[str] = []
        social_note = "social data collected"
        try:
            from data.social import SocialRequest

            request = SocialRequest(
                symbols,
                self.config.social.cashtags,
                self.config.social.accounts,
                self.config.social.subreddits,
                self.config.social.max_posts_per_source,
            )
            collected = await self.social.collect(request)
            posts = collected.posts
            failed = collected.failed
            skipped = collected.skipped
            if failed:
                social_note = "some social sources failed: " + ",".join(failed)
            elif skipped and not posts:
                social_note = "social sources were not configured"
        except Exception as exc:
            errors.append(f"social collector failed: {exc.__class__.__name__}")
            social_note = "social data unavailable"
        sentiment = []
        try:
            sentiment = await self.sentiment.summarize(symbols, posts)
        except Exception as exc:
            errors.append(f"sentiment failed: {exc.__class__.__name__}")
        query = " ".join(symbols) + " entry exit stop risk regime momentum"
        strategy = self.context_loader.load(query)
        portfolio = self.broker.portfolio(equity, quote, state)
        if self.options.decision == "baseline":
            decisions = self.baseline.decide(markets, portfolio, moment)
        else:
            decisions = await self.brain.decide(
                CycleContext(
                    symbols=symbols,
                    strategy=strategy,
                    portfolio=portfolio,
                    markets=markets,
                    sentiment=sentiment,
                    posts=posts,
                    history=self.store.recent_decisions(10),
                    social_note=social_note,
                )
            )
        verdicts = self.risk.review(decisions, portfolio, markets, moment)
        if any(verdict.halt and verdict.halt_reason in {"max_daily_loss", "max_drawdown"} for verdict in verdicts):
            state.halted = True
            state.halt_reason = next(verdict.halt_reason for verdict in verdicts if verdict.halt_reason)
        executions = []
        for verdict in verdicts:
            if verdict.action.value == "HOLD" or not verdict.approved:
                continue
            if state.halted and verdict.action.value not in {"CLOSE"}:
                continue
            result = await self.broker.execute(verdict, cycle_id, equity, moment)
            executions.append(result)
            if result.status in {"filled"} and result.action.value in {"BUY", "SELL"}:
                state.trades_today += 1
            if result.realized_pnl is not None and result.realized_pnl < 0:
                state.last_loss_at = moment
            if self.broker.halt_requested:
                state.halted = True
                state.halt_reason = self.broker.halt_reason
        equity_after = await self.broker.equity(marks)
        if equity_after > state.peak_equity:
            state.peak_equity = equity_after
        state.realized_pnl = self.broker.realized_pnl
        self.store.save_risk_state(state)
        record = self._record(
            cycle_id,
            moment,
            markets,
            decisions,
            verdicts,
            executions,
            errors,
            state.halted,
            equity_after,
            failed,
            skipped,
            notes,
            strategy.mode,
            sentiment,
        )
        self.store.save_cycle(record)
        self.store.add_equity(moment, equity_after)
        return record

    def _roll_risk(self, equity: float, now: datetime) -> RiskState:
        state = self.store.load_risk_state(equity, now)
        day = now.astimezone(timezone.utc).date().isoformat()
        if state.day != day:
            if state.halt_reason == "max_daily_loss":
                state.halted = False
                state.halt_reason = None
            state.day = day
            state.day_start_equity = equity
            state.trades_today = 0
        if equity > state.peak_equity:
            state.peak_equity = equity
        if state.day_start_equity <= 0:
            state.day_start_equity = equity
        return state

    def _kill_switch(self) -> str | None:
        path = Path(self.config.monitoring.kill_switch_file)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _record(self, cycle_id, moment, markets, decisions, verdicts, executions, errors, halted, equity, social_failed, social_skipped, notes, strategy_mode="n/a", sentiment=None) -> CycleRecord:
        summary = {
            "symbols": {
                market.symbol: {
                    "price": market.last_price,
                    "rsi": market.primary.indicators.rsi,
                    "macd_hist": market.primary.indicators.macd_hist,
                    "ema_50": market.primary.indicators.ema_50,
                    "atr": market.primary.indicators.atr,
                    "stale": market.stale,
                }
                for market in markets
            },
            "sentiment": [item.summary for item in (sentiment or [])],
            "social_failed": social_failed,
            "social_skipped": social_skipped,
            "strategy_mode": strategy_mode,
            "reconcile": notes,
        }
        return CycleRecord(
            cycle_id=cycle_id,
            started_at=moment,
            mode=self.options.mode,
            dry_run=self.options.dry_run,
            halted=halted,
            inputs_summary=summary,
            decisions=decisions,
            verdicts=verdicts,
            executions=executions,
            errors=errors,
            equity=equity,
        )
