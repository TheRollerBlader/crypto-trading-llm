"""Veto or resize every decision before the broker sees it.

Stops are mandatory on entries. Missing stops are attached from ATR when a
reference price exists. If a stop cannot be attached, the decision becomes HOLD
and no order is emitted. Circuit breakers block new risk. CLOSE remains
available so an existing position can be reduced.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.config import AppConfig
from core.models import Action, MarketSnapshot, PortfolioState, RiskVerdict, TradeDecision


class RiskManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.risk = config.risk

    def review(
        self,
        decisions: list[TradeDecision],
        portfolio: PortfolioState,
        markets: list[MarketSnapshot],
        now: datetime,
    ) -> list[RiskVerdict]:
        marks = {market.symbol: market.last_price for market in markets if market.last_price is not None}
        atrs = {market.symbol: market.primary.indicators.atr for market in markets}
        halt, halt_reason = self._circuit(portfolio)
        trades_left = self.risk.max_trades_per_day - portfolio.trades_today
        exposure = sum(portfolio.notionals(marks).values())
        notionals = portfolio.notionals(marks)
        verdicts: list[RiskVerdict] = []
        for decision in decisions:
            verdict, trades_left, exposure = self._one(
                decision,
                portfolio,
                marks,
                atrs,
                now,
                halt,
                halt_reason,
                trades_left,
                exposure,
                notionals,
            )
            if verdict.approved and verdict.action is Action.BUY:
                added = portfolio.equity * verdict.size_pct / 100
                notionals[decision.symbol] = notionals.get(decision.symbol, 0.0) + added
            elif verdict.approved and verdict.action is Action.SELL:
                sold = portfolio.equity * verdict.size_pct / 100
                notionals[decision.symbol] = max(0.0, notionals.get(decision.symbol, 0.0) - sold)
                exposure -= sold
            verdicts.append(verdict)
        return verdicts

    def _circuit(self, portfolio: PortfolioState) -> tuple[bool, str | None]:
        if portfolio.halted:
            return True, portfolio.halt_reason or "halted"
        daily_loss = 0.0
        if portfolio.day_start_equity > 0 and portfolio.equity < portfolio.day_start_equity:
            daily_loss = (portfolio.day_start_equity - portfolio.equity) / portfolio.day_start_equity * 100
        drawdown = 0.0
        if portfolio.peak_equity > 0 and portfolio.equity < portfolio.peak_equity:
            drawdown = (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity * 100
        daily_hit = daily_loss >= self.risk.max_daily_loss_pct
        drawdown_hit = drawdown >= self.risk.max_drawdown_pct
        if drawdown_hit:
            return True, "max_drawdown"
        if daily_hit:
            return True, "max_daily_loss"
        return False, None

    def _one(
        self,
        decision: TradeDecision,
        portfolio: PortfolioState,
        marks: dict[str, float],
        atrs: dict[str, float | None],
        now: datetime,
        halt: bool,
        halt_reason: str | None,
        trades_left: int,
        exposure: float,
        notionals: dict[str, float],
    ) -> tuple[RiskVerdict, int, float]:
        reasons: list[str] = []
        price = marks.get(decision.symbol)
        position = portfolio.position_for(decision.symbol)

        def reject(action: Action, reason: str) -> tuple[RiskVerdict, int, float]:
            reasons.append(reason)
            return (
                RiskVerdict(
                    symbol=decision.symbol,
                    action=action,
                    size_pct=0,
                    stop_loss=None,
                    take_profit=decision.take_profit,
                    approved=action is Action.CLOSE,
                    reasons=reasons,
                    halt=halt,
                    halt_reason=halt_reason,
                    reference_price=price,
                    original_action=decision.action,
                    original_size_pct=decision.size_pct,
                ),
                trades_left,
                exposure,
            )

        if decision.action is Action.HOLD:
            return reject(Action.HOLD, "hold")

        if decision.action is Action.CLOSE:
            if position is None or position.qty <= 0:
                return reject(Action.HOLD, "close ignored because there is no position")
            reasons.append("close reduces risk")
            if halt:
                reasons.append(f"halt is active ({halt_reason}) and close is still allowed")
            return (
                RiskVerdict(
                    symbol=decision.symbol,
                    action=Action.CLOSE,
                    size_pct=0,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    approved=True,
                    reasons=reasons,
                    halt=halt,
                    halt_reason=halt_reason,
                    reference_price=price,
                    original_action=decision.action,
                    original_size_pct=decision.size_pct,
                ),
                trades_left,
                exposure,
            )

        if halt:
            return reject(Action.HOLD, f"circuit breaker {halt_reason}")
        if decision.action in (Action.BUY, Action.SELL) and decision.confidence < self.risk.min_confidence:
            return reject(Action.HOLD, "confidence below minimum")
        if self._in_cooldown(portfolio, now) and decision.action in (Action.BUY, Action.SELL):
            return reject(Action.HOLD, "cooldown after a losing trade")
        if trades_left <= 0:
            return reject(Action.HOLD, "max trades per day")
        if price is None or price <= 0:
            return reject(Action.HOLD, "no reference price, so a stop cannot be checked")

        if decision.action is Action.SELL and not self.risk.allow_short and (position is None or position.qty <= 0):
            return reject(Action.HOLD, "shorting is disabled")

        if decision.action is Action.SELL and position is not None:
            position_notional = notionals.get(decision.symbol, position.qty * price)
            requested = portfolio.equity * decision.size_pct / 100
            sell_notional = min(requested, position_notional)
            if sell_notional <= 0:
                return reject(Action.HOLD, "nothing to sell")
            size_pct = sell_notional / portfolio.equity * 100
            reasons.append("sell reduces an existing long")
            return (
                RiskVerdict(
                    symbol=decision.symbol,
                    action=Action.SELL,
                    size_pct=size_pct,
                    stop_loss=decision.stop_loss or position.stop_loss,
                    take_profit=decision.take_profit or position.take_profit,
                    approved=True,
                    reasons=reasons,
                    halt=False,
                    halt_reason=None,
                    reference_price=price,
                    original_action=decision.action,
                    original_size_pct=decision.size_pct,
                ),
                trades_left - 1,
                exposure,
            )

        stop, stop_reasons = self._stop_for_buy(decision, price, atrs.get(decision.symbol))
        reasons.extend(stop_reasons)
        if stop is None:
            return reject(Action.HOLD, "entry rejected because no valid stop could be attached")

        size_pct = decision.size_pct
        if size_pct > self.risk.max_position_size_pct:
            size_pct = self.risk.max_position_size_pct
            reasons.append("resized to max position size")
        existing = notionals.get(decision.symbol, 0.0)
        room_position = self.risk.max_position_size_pct / 100 * portfolio.equity - existing
        room_total = self.risk.max_total_exposure_pct / 100 * portfolio.equity - exposure
        requested = size_pct / 100 * portfolio.equity
        allowed = min(requested, room_position, room_total)
        if allowed + 1e-9 < requested:
            reasons.append("resized to the remaining exposure room")
        if portfolio.equity <= 0 or allowed / portfolio.equity * 100 < self.risk.min_trade_pct:
            return reject(Action.HOLD, "remaining risk room is below the minimum trade size")
        size_pct = allowed / portfolio.equity * 100

        take_profit = decision.take_profit
        if take_profit is not None and take_profit <= price:
            reasons.append("ignored take-profit on the wrong side")
            take_profit = None
        if take_profit is None and atrs.get(decision.symbol):
            take_profit = price + self.risk.default_take_profit_atr_mult * float(atrs[decision.symbol])
            reasons.append("take-profit attached from ATR")

        reasons.append(
            f"fees and slippage budget is {2 * (self.risk.fee_bps + self.risk.slippage_bps):.1f} bps round trip"
        )
        return (
            RiskVerdict(
                symbol=decision.symbol,
                action=Action.BUY,
                size_pct=size_pct,
                stop_loss=stop,
                take_profit=take_profit,
                approved=True,
                reasons=reasons,
                halt=False,
                halt_reason=None,
                reference_price=price,
                original_action=decision.action,
                original_size_pct=decision.size_pct,
            ),
            trades_left - 1,
            exposure + portfolio.equity * size_pct / 100,
        )

    def _stop_for_buy(self, decision: TradeDecision, price: float, atr: float | None) -> tuple[float | None, list[str]]:
        notes: list[str] = []
        stop = decision.stop_loss
        if stop is None:
            if atr is None or atr <= 0:
                return None, ["no stop and no ATR"]
            stop = price - self.risk.default_stop_atr_mult * atr
            notes.append("stop attached by the risk manager")
        if stop >= price:
            return None, notes + ["stop is on the wrong side of price"]
        distance_pct = (price - stop) / price * 100
        round_trip_pct = 2 * (self.risk.fee_bps + self.risk.slippage_bps) / 100
        if distance_pct < round_trip_pct:
            return None, notes + ["stop is tighter than fees and slippage"]
        if distance_pct < self.risk.min_stop_distance_pct:
            return None, notes + ["stop is inside the minimum distance"]
        if distance_pct > self.risk.max_stop_distance_pct:
            stop = price * (1 - self.risk.max_stop_distance_pct / 100)
            notes.append("stop tightened to the maximum distance")
        if stop >= price or stop <= 0:
            return None, notes + ["stop is not a protective price"]
        return stop, notes

    def _in_cooldown(self, portfolio: PortfolioState, now: datetime) -> bool:
        if portfolio.last_loss_at is None:
            return False
        last = portfolio.last_loss_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        moment = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        elapsed = (moment - last).total_seconds() / 60
        return elapsed < self.risk.cooldown_after_loss_minutes
