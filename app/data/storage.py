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
    JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func as sa_func,
    inspect, insert, select, text, update
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
    # epoch seconds -- portable across SQLite/Postgres (unlike the tz-aware
    # created_at, which SQLite returns naive). Used for daily exposure and
    # for knowing when a shadow trade has expired and can be resolved.
    Column("opened_at", Float),
    Column("expires_at", Float),
    Column("exit_spot", Float),
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
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """create_all() never alters an existing table, so a column added to
        the schema in code would make every insert fail against a database
        created by an older version (e.g. your existing Railway Postgres).
        Adds any missing nullable columns in place. Additive only -- never
        drops or changes existing columns."""
        insp = inspect(self.engine)
        existing_tables = set(insp.get_table_names())
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have or col.primary_key:
                    continue
                col_type = col.type.compile(dialect=self.engine.dialect)
                with self.engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}'))
                self.logger.info(f"Storage: migrated {table.name}.{col.name} ({col_type})")

    # ------------------------------------------------------------- trades
    def record_trade(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(trades).values(**row))

    def settle_trade(self, trade_id: str, result: str, profit_loss: float,
                     balance_after: Optional[float] = None, exit_spot: Optional[float] = None) -> bool:
        """Idempotent: only transitions a trade out of OPEN once. Returns True
        if this call did the settling, False if it was already settled.

        Deriv can deliver the final is_sold proposal_open_contract update more
        than once (initial snapshot + stream, or again after a reconnect
        resubscribe). Callers use the return value to make sure the outcome
        is fed to calibration exactly once."""
        with self.engine.begin() as conn:
            res = conn.execute(
                update(trades)
                .where(trades.c.trade_id == trade_id)
                .where(trades.c.result == "OPEN")
                .values(result=result, profit_loss=profit_loss, balance_after=balance_after, exit_spot=exit_spot)
            )
        return (res.rowcount or 0) > 0

    def open_trades(self, shadow: bool) -> List[dict]:
        """Trades still marked OPEN. Used on startup to re-attach settlement
        tracking to live contracts opened before a restart, and by the shadow
        resolver to find hypothetical trades that have expired."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades).where(trades.c.result == "OPEN").where(trades.c.shadow == (1 if shadow else 0))
            ).mappings().all()
        return [dict(r) for r in rows]

    def retire_legacy_shadow_rows(self) -> int:
        """Shadow rows written before expires_at existed can never be
        resolved; left OPEN they would inflate the open count forever. Marks
        them UNRESOLVED (kept for the record, excluded from win-rate stats)."""
        with self.engine.begin() as conn:
            res = conn.execute(
                update(trades)
                .where(trades.c.shadow == 1).where(trades.c.result == "OPEN")
                .where(trades.c.expires_at.is_(None))
                .values(result="UNRESOLVED")
            )
        return res.rowcount or 0

    def stake_since(self, since_epoch: float, shadow: bool) -> float:
        """Total stake committed since `since_epoch` -- the basis for
        MAX_DAILY_EXPOSURE."""
        with self.engine.begin() as conn:
            total = conn.execute(
                select(sa_func.coalesce(sa_func.sum(trades.c.stake), 0.0))
                .where(trades.c.opened_at >= since_epoch)
                .where(trades.c.shadow == (1 if shadow else 0))
            ).scalar()
        return float(total or 0.0)

    def performance_summary(self, shadow: bool) -> Dict[str, Any]:
        """Settled-trade scorecard for one mode. The key calibration check is
        win_rate vs avg_predicted: if the model is honest these track each
        other; a persistent gap is the model overstating its own edge."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades.c.result, trades.c.profit_loss, trades.c.stake,
                       trades.c.calibrated_probability, trades.c.implied_probability)
                .where(trades.c.shadow == (1 if shadow else 0))
                .where(trades.c.result.in_(["WIN", "LOSS"]))
            ).all()
            open_count = conn.execute(
                select(sa_func.count()).select_from(trades)
                .where(trades.c.shadow == (1 if shadow else 0)).where(trades.c.result == "OPEN")
            ).scalar() or 0
        n = len(rows)
        wins = sum(1 for r in rows if r[0] == "WIN")
        staked = sum(float(r[2] or 0) for r in rows)
        pnl = sum(float(r[1] or 0) for r in rows)
        def avg(i):
            vals = [float(r[i]) for r in rows if r[i] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        return {
            "mode": "shadow" if shadow else "executed",
            "settled": n, "open": int(open_count), "wins": wins,
            "win_rate": round(wins / n, 4) if n else None,
            "avg_predicted": avg(3), "avg_implied": avg(4),
            "total_staked": round(staked, 2), "pnl": round(pnl, 2),
            "roi": round(pnl / staked, 4) if staked else None,
        }

    def recent_trades(self, limit: int = 50) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(trades).order_by(trades.c.id.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def consecutive_losses(self, symbol: Optional[str] = None, shadow: Optional[bool] = None) -> int:
        """shadow filters to one mode so a shadow losing streak doesn't size
        the first real stakes after switching SHADOW_MODE off (and vice
        versa)."""
        with self.engine.begin() as conn:
            q = select(trades.c.result).where(trades.c.result.in_(["WIN", "LOSS"]))
            if symbol:
                q = q.where(trades.c.symbol == symbol)
            if shadow is not None:
                q = q.where(trades.c.shadow == (1 if shadow else 0))
            q = q.order_by(trades.c.id.desc()).limit(20)
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
