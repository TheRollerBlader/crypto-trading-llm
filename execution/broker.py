"""Place orders only after a risk verdict. Entries are unwound if the stop fails."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone

from core.config import AppConfig
from core.models import Action, ClosedTrade, ExecutionResult, PortfolioState, Position, RiskVerdict
from core.redact import redact

logger = logging.getLogger(__name__)


def client_order_id(cycle_id: str, symbol: str, action: str, kind: str) -> str:
    raw = f"{cycle_id}|{symbol}|{action}|{kind}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:32]
    return f"{kind[:1]}{digest}"[:36]


def floor_qty(qty: float, step: float = 1e-6) -> float:
    if qty <= 0:
        return 0.0
    return round(math.floor(qty / step) * step, 8)


class Broker:
    def __init__(self, exchange, config: AppConfig, store, dry_run: bool = False):
        self.exchange = exchange
        self.config = config
        self.store = store
        self.dry_run = dry_run
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.halt_requested = False
        self.halt_reason: str | None = None
        self._as_of = datetime.now(timezone.utc)
        self._results: dict[str, ExecutionResult] = {}
        for position in store.get_positions():
            self.positions[position.symbol] = position

    async def equity(self, marks: dict[str, float]) -> float:
        balance = await self.exchange.fetch_balance()
        quote = self.config.exchange.quote_currency
        total = float((balance.get(quote) or {}).get("total") or 0)
        seen = set()
        for symbol, mark in marks.items():
            base = symbol.split("/")[0]
            if base in seen:
                continue
            seen.add(base)
            total += float((balance.get(base) or {}).get("total") or 0) * mark
        for position in self.positions.values():
            base = position.symbol.split("/")[0]
            if base not in seen and position.symbol not in marks:
                total += position.qty * position.entry_price
        return total

    def portfolio(self, equity: float, quote_balance: float, risk_state) -> PortfolioState:
        return PortfolioState(
            equity=equity,
            quote_balance=quote_balance,
            positions=list(self.positions.values()),
            day_start_equity=risk_state.day_start_equity,
            peak_equity=risk_state.peak_equity,
            trades_today=risk_state.trades_today,
            last_loss_at=risk_state.last_loss_at,
            halted=risk_state.halted,
            halt_reason=risk_state.halt_reason,
        )

    async def quote_balance(self) -> float:
        balance = await self.exchange.fetch_balance()
        quote = self.config.exchange.quote_currency
        return float((balance.get(quote) or {}).get("total") or 0)

    async def reconcile(
        self,
        marks: dict[str, float],
        ranges: dict[str, dict] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        notes: list[str] = []
        ranges = ranges or {}
        self._as_of = now or datetime.now(timezone.utc)
        for symbol in list(self.positions):
            price = marks.get(symbol)
            bar = ranges.get(symbol, {})
            if price is not None and hasattr(self.exchange, "process_price"):
                await self.exchange.process_price(
                    symbol,
                    price,
                    high=bar.get("high"),
                    low=bar.get("low"),
                    open_=bar.get("open"),
                )
            position = self.positions.get(symbol)
            if position is None:
                continue
            closed = await self._protection_filled(position)
            if closed:
                notes.append(closed)
        balance = await self.exchange.fetch_balance()
        for symbol, position in list(self.positions.items()):
            base = symbol.split("/")[0]
            remote = float((balance.get(base) or {}).get("total") or 0)
            if abs(remote - position.qty) > 1e-5:
                notes.append(f"reconciled {symbol} local qty {position.qty:.8f} to exchange qty {remote:.8f}")
                if remote <= 1e-8:
                    self._drop_position(symbol)
                else:
                    position.qty = remote
                    self.store.save_position(position)
        return notes

    async def execute(
        self,
        verdict: RiskVerdict,
        cycle_id: str,
        equity: float,
        now: datetime,
        force: bool = False,
    ) -> ExecutionResult:
        if verdict.action is Action.HOLD or not verdict.approved:
            return ExecutionResult(
                symbol=verdict.symbol,
                status="skipped",
                action=verdict.action,
                reasons=verdict.reasons,
            )
        if self.dry_run and not force:
            return ExecutionResult(
                symbol=verdict.symbol,
                status="dry_run",
                action=verdict.action,
                protected=False,
                reasons=verdict.reasons,
            )
        if verdict.action is Action.BUY and (verdict.stop_loss is None or verdict.reference_price is None):
            return ExecutionResult(
                symbol=verdict.symbol,
                status="rejected",
                action=verdict.action,
                protected=False,
                error="refusing entry without a stop and a reference price",
                reasons=verdict.reasons,
            )
        key = client_order_id(cycle_id, verdict.symbol, verdict.action.value, "e")
        if key in self._results:
            previous = self._results[key].model_copy(update={"status": "duplicate"})
            return previous
        try:
            if verdict.action is Action.BUY:
                result = await self._buy(verdict, key, equity, now)
            elif verdict.action is Action.CLOSE:
                result = await self._close(verdict, key, now, reason="signal")
            elif verdict.action is Action.SELL:
                result = await self._sell(verdict, key, equity, now)
            else:
                result = ExecutionResult(symbol=verdict.symbol, status="skipped", action=verdict.action, reasons=verdict.reasons)
        except Exception as exc:
            logger.exception("order failed for %s", verdict.symbol)
            result = ExecutionResult(
                symbol=verdict.symbol,
                status="error",
                action=verdict.action,
                protected=False,
                error=redact(str(exc)),
                reasons=verdict.reasons,
            )
        self._results[key] = result
        return result

    async def close_all(self, marks: dict[str, float], cycle_id: str, now: datetime) -> list[ExecutionResult]:
        results = []
        for symbol, position in list(self.positions.items()):
            verdict = RiskVerdict(
                symbol=symbol,
                action=Action.CLOSE,
                size_pct=0,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                approved=True,
                reasons=["kill switch close"],
                reference_price=marks.get(symbol, position.entry_price),
                original_action=Action.CLOSE,
                original_size_pct=0,
            )
            results.append(await self.execute(verdict, cycle_id + symbol, 0, now, force=True))
        return results

    async def _buy(self, verdict: RiskVerdict, client_id: str, equity: float, now: datetime) -> ExecutionResult:
        price = float(verdict.reference_price)
        slip = self.config.risk.slippage_bps / 10_000
        fee = self.config.risk.fee_bps / 10_000
        budget = equity * verdict.size_pct / 100
        qty = floor_qty(budget / (price * (1 + slip) * (1 + fee)))
        if qty <= 0:
            return ExecutionResult(symbol=verdict.symbol, status="rejected", action=Action.BUY, error="quantity rounded to zero", reasons=verdict.reasons)
        order = await self.exchange.create_order(verdict.symbol, "market", "buy", qty, None, {"clientOrderId": client_id})
        if order.get("duplicate"):
            return ExecutionResult(symbol=verdict.symbol, status="duplicate", action=Action.BUY, order_id=str(order["id"]), protected=True, reasons=verdict.reasons)
        if order.get("status") != "closed" or float(order.get("filled") or 0) <= 0:
            return ExecutionResult(symbol=verdict.symbol, status="error", action=Action.BUY, error="entry was not filled", order_id=str(order.get("id")), reasons=verdict.reasons)
        filled = float(order["filled"])
        fill_price = float(order["average"])
        entry_fee = float((order.get("fee") or {}).get("cost") or 0)
        stop_id = client_order_id(client_id, verdict.symbol, "stop", "s")
        stop_price = float(verdict.stop_loss)
        try:
            stop = await self.exchange.create_order(
                verdict.symbol,
                "stop_loss_limit",
                "sell",
                filled,
                stop_price * 0.999,
                {"stopPrice": stop_price, "clientOrderId": stop_id},
            )
        except Exception as exc:
            return await self._unwind(verdict, filled, redact(str(exc)), now, fill_price, entry_fee)
        tp_id = None
        if verdict.take_profit:
            try:
                tp = await self.exchange.create_order(
                    verdict.symbol,
                    "limit",
                    "sell",
                    filled,
                    float(verdict.take_profit),
                    {"clientOrderId": client_order_id(client_id, verdict.symbol, "tp", "t")},
                )
                tp_id = str(tp["id"])
            except Exception as exc:
                logger.warning("take-profit was not placed for %s: %s", verdict.symbol, redact(str(exc)))
        position = Position(
            symbol=verdict.symbol,
            qty=filled,
            entry_price=fill_price,
            entry_fee=entry_fee,
            stop_loss=stop_price,
            take_profit=verdict.take_profit,
            stop_order_id=str(stop["id"]),
            tp_order_id=tp_id,
            opened_at=now,
            entry_order_id=str(order["id"]),
        )
        self.positions[verdict.symbol] = position
        self.store.save_position(position)
        self.store.save_order(client_id, order)
        return ExecutionResult(
            symbol=verdict.symbol,
            status="filled",
            action=Action.BUY,
            protected=True,
            order_id=str(order["id"]),
            stop_order_id=str(stop["id"]),
            tp_order_id=tp_id,
            qty=filled,
            price=fill_price,
            fee=entry_fee,
            reasons=verdict.reasons,
        )

    async def _unwind(self, verdict: RiskVerdict, qty: float, reason: str, now: datetime, entry: float, entry_fee: float) -> ExecutionResult:
        try:
            order = await self.exchange.create_order(
                verdict.symbol,
                "market",
                "sell",
                qty,
                None,
                {"clientOrderId": client_order_id(verdict.symbol, "unwind", reason[:12], "u")},
            )
        except Exception as exc:
            self.halt_requested = True
            self.halt_reason = "unprotected_position"
            logger.critical("failed to flatten %s after stop placement failed", verdict.symbol)
            position = Position(
                symbol=verdict.symbol,
                qty=qty,
                entry_price=entry,
                entry_fee=entry_fee,
                stop_loss=None,
                take_profit=None,
                opened_at=now,
            )
            self.positions[verdict.symbol] = position
            self.store.save_position(position)
            return ExecutionResult(
                symbol=verdict.symbol,
                status="error",
                action=Action.BUY,
                protected=False,
                qty=qty,
                error=f"stop failed ({reason}); unwind failed ({redact(str(exc))})",
                reasons=verdict.reasons,
            )
        exit_fee = float((order.get("fee") or {}).get("cost") or 0)
        exit_price = float(order["average"])
        pnl = (exit_price - entry) * qty - exit_fee - entry_fee
        self.realized_pnl += pnl
        self.store.record_closed_trade(
            ClosedTrade(
                symbol=verdict.symbol,
                qty=qty,
                entry_price=entry,
                exit_price=exit_price,
                realized_pnl=pnl,
                opened_at=now,
                closed_at=now,
                reason="unwound",
            )
        )
        return ExecutionResult(
            symbol=verdict.symbol,
            status="unwound",
            action=Action.BUY,
            protected=True,
            qty=qty,
            price=exit_price,
            fee=entry_fee + exit_fee,
            realized_pnl=pnl,
            error=f"stop failed and the entry was closed: {reason}",
            reasons=verdict.reasons,
        )

    async def _close(self, verdict: RiskVerdict, client_id: str, now: datetime, reason: str) -> ExecutionResult:
        position = self.positions.get(verdict.symbol)
        if position is None:
            return ExecutionResult(symbol=verdict.symbol, status="skipped", action=Action.CLOSE, reasons=["no position"])
        await self._cancel(position.stop_order_id, verdict.symbol)
        await self._cancel(position.tp_order_id, verdict.symbol)
        order = await self.exchange.create_order(verdict.symbol, "market", "sell", position.qty, None, {"clientOrderId": client_id})
        pnl = self._realize(position, order, now, reason)
        self._drop_position(verdict.symbol)
        return ExecutionResult(
            symbol=verdict.symbol,
            status="closed",
            action=Action.CLOSE,
            protected=True,
            order_id=str(order["id"]),
            qty=float(order["filled"]),
            price=float(order["average"]),
            fee=float((order.get("fee") or {}).get("cost") or 0),
            realized_pnl=pnl,
            reasons=verdict.reasons,
        )

    async def _sell(self, verdict: RiskVerdict, client_id: str, equity: float, now: datetime) -> ExecutionResult:
        position = self.positions.get(verdict.symbol)
        if position is None:
            return ExecutionResult(symbol=verdict.symbol, status="rejected", action=Action.SELL, error="no long to sell", reasons=verdict.reasons)
        price = float(verdict.reference_price or position.entry_price)
        qty = min(position.qty, floor_qty(equity * verdict.size_pct / 100 / price))
        if qty <= 0 or qty >= position.qty * 0.999:
            return await self._close(verdict, client_id, now, reason="sell")
        await self._cancel(position.stop_order_id, verdict.symbol)
        await self._cancel(position.tp_order_id, verdict.symbol)
        order = await self.exchange.create_order(verdict.symbol, "market", "sell", qty, None, {"clientOrderId": client_id})
        pnl = self._realize(position, order, now, "partial_sell")
        position.qty -= float(order["filled"])
        position.entry_fee = max(0.0, position.entry_fee - position.entry_fee * (float(order["filled"]) / (position.qty + float(order["filled"]))))
        if verdict.stop_loss:
            stop = await self.exchange.create_order(
                verdict.symbol,
                "stop_loss_limit",
                "sell",
                position.qty,
                float(verdict.stop_loss) * 0.999,
                {"stopPrice": float(verdict.stop_loss), "clientOrderId": client_order_id(client_id, verdict.symbol, "stop2", "s")},
            )
            position.stop_order_id = str(stop["id"])
            position.stop_loss = float(verdict.stop_loss)
        self.store.save_position(position)
        return ExecutionResult(
            symbol=verdict.symbol,
            status="filled",
            action=Action.SELL,
            protected=position.stop_order_id is not None,
            order_id=str(order["id"]),
            stop_order_id=position.stop_order_id,
            qty=float(order["filled"]),
            price=float(order["average"]),
            fee=float((order.get("fee") or {}).get("cost") or 0),
            realized_pnl=pnl,
            reasons=verdict.reasons,
        )

    async def _protection_filled(self, position: Position) -> str | None:
        if position.stop_order_id:
            order = await self.exchange.fetch_order(position.stop_order_id, position.symbol)
            if order.get("status") == "closed":
                await self._cancel(position.tp_order_id, position.symbol)
                self._realize(position, order, self._as_of, "stop")
                self._drop_position(position.symbol)
                return f"{position.symbol} stop filled"
        current = self.positions.get(position.symbol)
        if current and current.tp_order_id:
            order = await self.exchange.fetch_order(current.tp_order_id, current.symbol)
            if order.get("status") == "closed":
                await self._cancel(current.stop_order_id, current.symbol)
                self._realize(current, order, self._as_of, "take_profit")
                self._drop_position(current.symbol)
                return f"{current.symbol} take-profit filled"
        return None

    def _realize(self, position: Position, order: dict, now: datetime, reason: str) -> float:
        qty = float(order.get("filled") or 0)
        exit_price = float(order.get("average") or 0)
        exit_fee = float((order.get("fee") or {}).get("cost") or 0)
        fraction = qty / position.qty if position.qty else 1
        allocated_fee = position.entry_fee * fraction
        pnl = (exit_price - position.entry_price) * qty - exit_fee - allocated_fee
        self.realized_pnl += pnl
        opened = position.opened_at
        self.store.record_closed_trade(
            ClosedTrade(
                symbol=position.symbol,
                qty=qty,
                entry_price=position.entry_price,
                exit_price=exit_price,
                realized_pnl=pnl,
                opened_at=opened,
                closed_at=now,
                reason=reason,
            )
        )
        return pnl

    async def _cancel(self, order_id: str | None, symbol: str) -> None:
        if not order_id:
            return
        try:
            await self.exchange.cancel_order(order_id, symbol)
        except Exception as exc:
            logger.info("cancel %s failed: %s", order_id, redact(str(exc)))

    def _drop_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.store.delete_position(symbol)
