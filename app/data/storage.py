"""
Persistence layer.

Uses SQLAlchemy Core so the same schema/queries work against SQLite
(local dev, or Railway fallback if DATABASE_URL isn't set) and Postgres
(Railway production, via DATABASE_URL). Tables are created automatically
on startup if missing -- no separate migration step required to get a
working deployment.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert, select, update
)
from sqlalchemy.sql import func

metadata = MetaData()

trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_id", String, unique=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("symbol", String),
    Column("entry_price", Float),
    Column("duration_minutes", Integer),
    Column("lower_barrier", Float),
    Column("upper_barrier", Float),
    Column("stake", Float),
    Column("proposal_id", String),
    Column("contract_id", String),
    Column("payout", Float),
    Column("payout_multiplier", Float),
    Column("raw_probability", Float),
    Column("calibrated_probability", Float),
    Column("implied_probability", Float),
    Column("edge", Float),
    Column("expected_value", Float),
    Column("regime", String),
    Column("regime_confidence", Float),
    Column("volatility", Float),
    Column("model_disagreement", Float),
    Column("mc_path_count", Integer),
    Column("decision_score", Float),
    Column("shadow", Integer, default=0),
    Column("result", String, default="OPEN"),
    Column("profit_loss", Float, default=0.0),
    Column("balance_before", Float),
    Column("balance_after", Float),
    Column("model_version", String),
)

rejected_signals = Table(
    "rejected_signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("symbol", String),
    Column("duration_minutes", Integer),
    Column("lower_barrier", Float),
    Column("upper_barrier", Float),
    Column("raw_probability", Float),
    Column("calibrated_probability", Float),
    Column("payout", Float),
    Column("implied_probability", Float),
    Column("edge", Float),
    Column("expected_value", Float),
    Column("regime", String),
    Column("rejection_reason", String),
)

calibration_buckets = Table(
    "calibration_buckets", metadata,
    Column("bucket", String, primary_key=True),  # e.g. "0.70-0.75"
    Column("n_predictions", Integer, default=0),
    Column("n_wins", Integer, default=0),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now()),
)

system_events = Table(
    "system_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("level", String),
    Column("message", String),
    Column("context", JSON),
)


class Storage:
    def __init__(self, database_url: str, sqlite_path: str, logger):
        self.logger = logger
        if database_url:
            url = database_url
            if url.startswith("postgres://"):  # SQLAlchemy wants postgresql://
                url = url.replace("postgres://", "postgresql://", 1)
            self.engine = create_engine(url, pool_pre_ping=True)
            self.logger.info("Storage: connected to external database via DATABASE_URL")
        else:
            self.engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
            self.logger.info(f"Storage: using local SQLite at {sqlite_path}")

    def init_schema(self) -> None:
        metadata.create_all(self.engine)

    # ------------------------------------------------------------- trades
    def record_trade(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(trades).values(**row))

    def settle_trade(self, trade_id: str, result: str, profit_loss: float, balance_after: Optional[float]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(trades)
                .where(trades.c.trade_id == trade_id)
                .values(result=result, profit_loss=profit_loss, balance_after=balance_after)
            )

    def recent_trades(self, limit: int = 50) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(trades).order_by(trades.c.id.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def consecutive_losses(self, symbol: Optional[str] = None) -> int:
        with self.engine.begin() as conn:
            q = select(trades.c.result).where(trades.c.result != "OPEN").order_by(trades.c.id.desc()).limit(20)
            if symbol:
                q = q.where(trades.c.symbol == symbol)
            rows = [r[0] for r in conn.execute(q).all()]
        count = 0
        for r in rows:
            if r == "LOSS":
                count += 1
            else:
                break
        return count

    # -------------------------------------------------------- rejections
    def record_rejection(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(rejected_signals).values(**row))

    # -------------------------------------------------------- calibration
    def update_calibration(self, bucket: str, won: bool) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(calibration_buckets).where(calibration_buckets.c.bucket == bucket)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(calibration_buckets)
                    .where(calibration_buckets.c.bucket == bucket)
                    .values(
                        n_predictions=existing["n_predictions"] + 1,
                        n_wins=existing["n_wins"] + (1 if won else 0),
                    )
                )
            else:
                conn.execute(
                    insert(calibration_buckets).values(
                        bucket=bucket, n_predictions=1, n_wins=1 if won else 0
                    )
                )

    def get_calibration(self) -> Dict[str, tuple]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(calibration_buckets)).mappings().all()
        return {r["bucket"]: (r["n_predictions"], r["n_wins"]) for r in rows}

    # ------------------------------------------------------------- events
    def log_event(self, level: str, message: str, context: Optional[dict] = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(system_events).values(level=level, message=message, context=context or {}))
