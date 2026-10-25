"""Persistence for cycles, positions, and the risk latch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from core.models import ClosedTrade, CycleRecord, Position, TradeDecision
from core.redact import redact


class RiskState(BaseModel):
    day: str
    day_start_equity: float
    peak_equity: float
    trades_today: int = 0
    halted: bool = False
    halt_reason: str | None = None
    last_loss_at: datetime | None = None
    realized_pnl: float = 0


class Base(DeclarativeBase):
    pass


class CycleRow(Base):
    __tablename__ = "cycles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[str] = mapped_column(String(64), unique=True)
    started_at: Mapped[str] = mapped_column(String(40))
    mode: Mapped[str] = mapped_column(String(32))
    dry_run: Mapped[bool] = mapped_column(Boolean)
    halted: Mapped[bool] = mapped_column(Boolean)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[str] = mapped_column(Text)


class PositionRow(Base):
    __tablename__ = "positions"
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    payload: Mapped[str] = mapped_column(Text)


class OrderRow(Base):
    __tablename__ = "orders"
    client_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[str] = mapped_column(Text)


class ClosedTradeRow(Base):
    __tablename__ = "closed_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    closed_at: Mapped[str] = mapped_column(String(40))
    realized_pnl: Mapped[float] = mapped_column(Float)
    payload: Mapped[str] = mapped_column(Text)


class EquityRow(Base):
    __tablename__ = "equity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[str] = mapped_column(String(40))
    equity: Mapped[float] = mapped_column(Float)


class RiskRow(Base):
    __tablename__ = "risk_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[str] = mapped_column(Text)


class Store:
    def __init__(self, url: str, jsonl_path: Path | None = None):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)
        self.jsonl_path = jsonl_path

    def save_cycle(self, record: CycleRecord) -> None:
        payload = redact(record.model_dump_json())
        with self._session() as session:
            session.merge(
                CycleRow(
                    cycle_id=record.cycle_id,
                    started_at=record.started_at.isoformat(),
                    mode=record.mode,
                    dry_run=record.dry_run,
                    halted=record.halted,
                    equity=record.equity,
                    payload=payload,
                )
            )
            session.commit()
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def recent_decisions(self, limit: int = 10) -> list[TradeDecision]:
        with self._session() as session:
            rows = session.scalars(select(CycleRow).order_by(CycleRow.id.desc()).limit(limit)).all()
        decisions: list[TradeDecision] = []
        for row in reversed(rows):
            record = CycleRecord.model_validate_json(row.payload)
            decisions.extend(record.decisions)
        return decisions[-limit:]

    def save_position(self, position: Position) -> None:
        with self._session() as session:
            session.merge(PositionRow(symbol=position.symbol, payload=position.model_dump_json()))
            session.commit()

    def delete_position(self, symbol: str) -> None:
        with self._session() as session:
            row = session.get(PositionRow, symbol)
            if row:
                session.delete(row)
                session.commit()

    def get_positions(self) -> list[Position]:
        with self._session() as session:
            rows = session.scalars(select(PositionRow)).all()
            return [Position.model_validate_json(row.payload) for row in rows]

    def save_order(self, client_id: str, order: dict) -> None:
        with self._session() as session:
            session.merge(OrderRow(client_id=client_id, payload=json.dumps(order, default=str)))
            session.commit()

    def record_closed_trade(self, trade: ClosedTrade) -> None:
        with self._session() as session:
            session.add(
                ClosedTradeRow(
                    symbol=trade.symbol,
                    closed_at=trade.closed_at.isoformat(),
                    realized_pnl=trade.realized_pnl,
                    payload=trade.model_dump_json(),
                )
            )
            session.commit()

    def closed_trades(self) -> list[ClosedTrade]:
        with self._session() as session:
            rows = session.scalars(select(ClosedTradeRow).order_by(ClosedTradeRow.id)).all()
            return [ClosedTrade.model_validate_json(row.payload) for row in rows]

    def add_equity(self, ts: datetime, equity: float) -> None:
        with self._session() as session:
            session.add(EquityRow(ts=ts.isoformat(), equity=equity))
            session.commit()

    def equity_points(self) -> list[tuple[str, float]]:
        with self._session() as session:
            rows = session.scalars(select(EquityRow).order_by(EquityRow.id)).all()
            return [(row.ts, row.equity) for row in rows]

    def load_risk_state(self, equity: float, now: datetime) -> RiskState:
        with self._session() as session:
            row = session.get(RiskRow, 1)
            if row is None:
                state = RiskState(day=now.astimezone(timezone.utc).date().isoformat(), day_start_equity=equity, peak_equity=equity)
                session.add(RiskRow(id=1, payload=state.model_dump_json()))
                session.commit()
                return state
            return RiskState.model_validate_json(row.payload)

    def save_risk_state(self, state: RiskState) -> None:
        with self._session() as session:
            session.merge(RiskRow(id=1, payload=state.model_dump_json()))
            session.commit()

    def clear_halt(self) -> None:
        with self._session() as session:
            row = session.get(RiskRow, 1)
            if row is None:
                return
            state = RiskState.model_validate_json(row.payload)
            state.halted = False
            state.halt_reason = None
            row.payload = state.model_dump_json()
            session.commit()

    def stats(self) -> dict:
        trades = self.closed_trades()
        wins = [trade for trade in trades if trade.realized_pnl > 0]
        losses = [trade for trade in trades if trade.realized_pnl < 0]
        points = self.equity_points()
        cycles = self.cycle_count()
        return {
            "cycles": cycles,
            "closed_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else None,
            "realized_pnl": sum(trade.realized_pnl for trade in trades),
            "equity": points[-1][1] if points else None,
            "positions": [position.model_dump(mode="json") for position in self.get_positions()],
            "equity_curve": points,
            "trades": [trade.model_dump(mode="json") for trade in trades],
        }

    def cycle_count(self) -> int:
        with self._session() as session:
            return len(session.scalars(select(CycleRow.id)).all())

    def cycles(self) -> list[CycleRecord]:
        with self._session() as session:
            rows = session.scalars(select(CycleRow).order_by(CycleRow.id)).all()
            return [CycleRecord.model_validate_json(row.payload) for row in rows]
