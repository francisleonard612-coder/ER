"""
Production entrypoint. `python run.py`

Startup sequence (section 7): connect DB -> restore state -> connect Deriv
-> resubscribe market data -> rebuild rolling state -> reattach any
contracts still open from before the restart -> resume scanning.
Every step logs clearly; no step requires manual intervention to recover
from an ordinary Railway restart.
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np

from app.config import load_config
from app.data.loader import load_csv_if_present
from app.data.storage import Storage
from app.deriv.client import BuyAmbiguousError, DerivClient
from app.models.calibration import CalibrationTracker
from app.monitoring.logging_utils import setup_logging
from app.state_machine import State, StateMachine
from app.strategy.engine import run_scan_cycle
from app.strategy.risk import exposure_allows, utc_day_start
from app.strategy.staking import build_staking_engine

CANDLE_GRANULARITY_SECONDS = 60
CANDLE_HISTORY_COUNT = 300
SHADOW_RESOLVE_INTERVAL_SECONDS = 10
SHADOW_RESOLVE_GRACE_SECONDS = 3      # let the expiry tick land before asking for it
STATS_LOG_INTERVAL_SECONDS = 15 * 60
HEALTH_STALE_AFTER_SECONDS = 120


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.logger = setup_logging(self.cfg.log_level)
        self.sm = StateMachine(self.logger)
        self.storage = Storage(self.cfg.database_url, self.cfg.sqlite_path, self.logger)
        self.calibration = CalibrationTracker(self.storage)
        self.staking = build_staking_engine(self.cfg.staking)
        self.client: DerivClient | None = None
        self.closes_by_symbol: dict[str, deque] = {}
        # trade_id -> symbol for every position occupying a slot. Shadow
        # trades occupy slots too, until resolved: otherwise shadow mode
        # opens a new overlapping trade every scan, which live mode never
        # would, and the shadow track record stops predicting live results.
        # Keyed by trade_id so releasing a slot is naturally idempotent.
        self.open_positions: dict[str, str] = {}
        self._background: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self.started_at = time.time()

    # ------------------------------------------------------------- helpers
    def _spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _open_count(self, symbol: str | None = None) -> int:
        if symbol is None:
            return len(self.open_positions)
        return sum(1 for s in self.open_positions.values() if s == symbol)

    # ------------------------------------------------------------- startup
    async def start(self):
        self.sm.transition(State.INITIALIZING)
        problems = self.cfg.validate()
        for p in problems:
            self.logger.error(f"CONFIG PROBLEM: {p}")
        if self.cfg.is_live() and problems:
            self.logger.error("Refusing to start in LIVE mode with unresolved config problems. Falling back to DEMO behaviour (shadow-only).")
            self.cfg.shadow_mode = True

        self.logger.info(f"ACCOUNT MODE: {self.cfg.account_mode}  |  SHADOW MODE: {self.cfg.shadow_mode}")

        self.storage.init_schema()
        retired = self.storage.retire_legacy_shadow_rows()
        if retired:
            self.logger.info(f"Marked {retired} pre-upgrade shadow rows as UNRESOLVED (no expiry recorded, cannot be settled)")
        self.sm.transition(State.SYNCING_DATA)
        self._load_historical_seed()

        self.sm.transition(State.CONNECTING)
        self.client = DerivClient(
            endpoint=self.cfg.deriv_endpoint,
            app_id=self.cfg.deriv_app_id,
            api_token=self.cfg.deriv_api_token,
            logger=self.logger,
            on_reconnect=self._on_reconnect,
        )
        await self.client.connect()

        self.sm.transition(State.WARMING_UP)
        await self._prime_candles()
        await self._restore_open_positions()

        self._spawn(self._shadow_resolver_loop(), "shadow-resolver")
        self._spawn(self._stats_loop(), "stats-logger")

        self.sm.transition(State.READY)
        self.logger.info("BOT READY")

    def _load_historical_seed(self):
        # Optional seed data -- see data/candles_rows.csv, data/rejected_signals_rows.csv.
        load_csv_if_present("data/candles_rows.csv", self.logger)
        load_csv_if_present("data/rejected_signals_rows.csv", self.logger)

    async def _prime_candles(self):
        for symbol in self.cfg.symbols:
            await self._refresh_candles(symbol)
            self.logger.info(f"{symbol}: primed with {len(self.closes_by_symbol.get(symbol, ()))} candles")

    async def _refresh_candles(self, symbol: str) -> None:
        """Replaces the closes buffer with a fresh, correctly-spaced fetch
        from get_candle_history.

        NOT built on the candle *subscription* stream on purpose: Deriv pushes
        an `ohlc` update on every tick that touches the still-forming candle,
        and appending each push as a new 1-minute close crushed the computed
        volatility and produced implausible edges in production. Refetching
        finalized history every scan avoids that whole class of bug.
        """
        try:
            candles = await self.client.get_candle_history(symbol, CANDLE_GRANULARITY_SECONDS, CANDLE_HISTORY_COUNT)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"{symbol}: candle refresh failed (non-fatal, using previous history): {exc!r}")
            return
        if not candles:
            return
        buf = self.closes_by_symbol.setdefault(symbol, deque(maxlen=CANDLE_HISTORY_COUNT))
        buf.clear()
        for c in candles:
            buf.append(float(c["close"]))

    async def _restore_open_positions(self):
        """Previously, open-contract tracking lived only in memory: after a
        restart, contracts bought before it were never settled in the DB,
        never fed calibration, didn't count toward concurrency limits, and
        were invisible to the consecutive-loss logic. Reattach them."""
        for t in self.storage.open_trades(shadow=True):
            if t.get("expires_at"):
                self.open_positions[t["trade_id"]] = t["symbol"]
        live = [t for t in self.storage.open_trades(shadow=False) if t.get("contract_id")]
        for t in live:
            self.open_positions[t["trade_id"]] = t["symbol"]
            await self._track_contract(t["trade_id"], t["symbol"], t.get("raw_probability"), t["contract_id"])
        if self.open_positions:
            self.logger.info(f"Restored {len(self.open_positions)} open position(s) from the journal "
                             f"({len(live)} live contract(s) re-subscribed)")

    async def _on_reconnect(self):
        self.logger.info("Reconnect handler: subscriptions re-established, resuming normal operation")

    # --------------------------------------------------------------- loop
    async def run_forever(self):
        while not self._stop.is_set():
            self.sm.reassess_cautions()
            self.sm.transition(State.SCANNING)
            for symbol in self.cfg.symbols:
                if self._stop.is_set():
                    break
                await self._scan_symbol(symbol)
            self.sm.transition(State.READY)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.scan_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _scan_symbol(self, symbol: str):
        # capacity checks first -- no point fetching candles or simulating
        # for a symbol that can't trade this cycle anyway
        if self._open_count(symbol) >= self.cfg.max_concurrent_per_symbol:
            return  # no overlapping same-symbol trades; clears automatically on settlement
        if self._open_count() >= self.cfg.max_concurrent_contracts:
            return

        await self._refresh_candles(symbol)
        closes = np.array(self.closes_by_symbol.get(symbol, ()), dtype=float)
        if len(closes) == 0:
            return
        current_price = closes[-1]

        shadow = self.cfg.shadow_mode
        consecutive_losses = self.storage.consecutive_losses(symbol, shadow=shadow)

        self.sm.transition(State.SIMULATING, symbol)
        try:
            outcome = await run_scan_cycle(
                symbol=symbol, closes=closes, current_price=current_price, client=self.client,
                calibration=self.calibration, staking=self.staking, cfg=self.cfg,
                stake_multiplier=self.sm.stake_multiplier(),
                extra_edge_requirement=self.sm.extra_edge_requirement(),
                logger=self.logger,
                consecutive_losses=consecutive_losses,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: scan cycle failed (non-fatal): {exc!r}")
            self.sm.enter_caution(symbol, f"scan error: {exc!r}", self.cfg.caution_cooldown_seconds)
            return

        for rej in outcome.rejections:
            self.storage.record_rejection({
                "symbol": rej["symbol"], "duration_minutes": rej["duration_minutes"],
                "lower_barrier": rej["lower_barrier"], "upper_barrier": rej["upper_barrier"],
                "raw_probability": rej["raw_probability"], "calibrated_probability": rej["calibrated_probability"],
                "payout": rej["payout"], "implied_probability": rej["implied_probability"],
                "edge": rej["edge"], "expected_value": rej["expected_value"], "regime": rej["regime"],
                "rejection_reason": rej["rejection_reason"],
            })

        # consecutive-loss caution (temporary, auto-recovering -- section 8/29)
        if consecutive_losses >= self.cfg.consecutive_loss_caution_threshold and not self.sm.is_symbol_cautioned(symbol):
            self.sm.enter_caution(
                symbol, f"{consecutive_losses} consecutive losses", self.cfg.caution_cooldown_seconds,
                edge_penalty=0.03, stake_multiplier=0.5,
            )

        if not outcome.traded or outcome.trade_row is None:
            return

        row = outcome.trade_row

        # MAX_DAILY_EXPOSURE existed in config but was never enforced.
        # Resets at 00:00 UTC -- temporary by design, like every other
        # restriction in this bot. Applied in shadow too, so shadow mirrors live.
        staked_today = self.storage.stake_since(utc_day_start(), shadow=shadow)
        if not exposure_allows(staked_today, row["stake"], self.cfg.max_daily_exposure):
            self.logger.warning(
                f"{symbol}: DAILY EXPOSURE CAP -- {staked_today:.2f} staked today + {row['stake']:.2f} "
                f"would exceed {self.cfg.max_daily_exposure:.2f}; skipping until 00:00 UTC"
            )
            return

        now = time.time()
        row = {**row, "opened_at": now, "expires_at": now + row["duration_minutes"] * 60}

        # SHADOW_MODE alone controls execution vs logging-only. ACCOUNT_MODE
        # (DEMO/LIVE) only controls which Deriv account the token authorizes.
        if shadow:
            self.storage.record_trade({**row, "shadow": 1, "contract_id": None, "balance_before": None})
            self.open_positions[row["trade_id"]] = symbol
            self.logger.info(f"{symbol} SHADOW TRADE (not executed): ev={row['expected_value']:+.4f} "
                             f"stake={row['stake']} resolves in {row['duration_minutes']}m")
            return

        self.sm.transition(State.EXECUTING, symbol)
        # No balance() round-trip before buying any more: every extra request
        # here ages the proposal we're about to buy. Deriv's buy response
        # carries balance_after, from which balance_before follows.
        buy_started = time.time()
        try:
            buy_resp = await self.client.buy_contract(row["proposal_id"], row["stake"])
        except BuyAmbiguousError as exc:
            self.logger.error(f"{symbol}: {exc} -- reconciling against portfolio")
            buy_resp = await self._reconcile_ambiguous_buy(symbol, since=buy_started - 5,
                                                         stake=row["stake"], payout=row["payout"])
            if buy_resp is None:
                self.sm.enter_caution(symbol, "ambiguous buy, no contract found", self.cfg.caution_cooldown_seconds)
                return
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: buy failed (non-fatal): {exc!r}")
            self.sm.enter_caution(symbol, f"buy error: {exc!r}", self.cfg.caution_cooldown_seconds)
            return

        contract_id = str(buy_resp.get("contract_id"))
        buy_price = _float_or_none(buy_resp.get("buy_price"))
        balance_after = _float_or_none(buy_resp.get("balance_after"))
        balance_before = balance_after + buy_price if (balance_after is not None and buy_price is not None) else None
        if buy_price is not None:
            row["stake"] = buy_price

        self.storage.record_trade({**row, "shadow": 0, "contract_id": contract_id, "balance_before": balance_before})
        self.open_positions[row["trade_id"]] = symbol
        self.sm.transition(State.POSITION_OPEN, symbol)
        self.logger.info(f"{symbol} TRADE EXECUTED: contract_id={contract_id} stake={row['stake']} ev={row['expected_value']:+.4f}")
        await self._track_contract(row["trade_id"], symbol, row["raw_probability"], contract_id)

    # --------------------------------------------------------- settlement
    async def _track_contract(self, trade_id: str, symbol: str, raw_probability, contract_id: str):
        key = f"contract:{contract_id}"

        def _on_update(msg):
            poc = msg.get("proposal_open_contract") or {}
            if not poc.get("is_sold"):
                return
            profit = float(poc.get("profit", 0.0) or 0.0)
            status = str(poc.get("status") or "").lower()
            won = status == "won" if status in ("won", "lost") else profit > 0
            self._settle(trade_id, symbol, raw_probability, won, profit,
                         exit_spot=_float_or_none(poc.get("exit_tick") or poc.get("exit_spot")),
                         source="live")
            self._spawn(self.client.forget_subscription(key), f"forget-{contract_id}")

        try:
            await self.client.subscribe_contract(contract_id, _on_update)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: contract subscription failed for {contract_id} (non-fatal, "
                              f"will retry on next restart): {exc!r}")

    def _settle(self, trade_id, symbol, raw_probability, won: bool, profit: float, *, exit_spot=None, source=""):
        """Single settlement path for live and shadow. The DB update is
        conditional on the trade still being OPEN, so a duplicate is_sold
        message (initial snapshot + stream, or a resubscribe after reconnect)
        cannot double-count an outcome into calibration."""
        result = "WIN" if won else "LOSS"
        first_time = self.storage.settle_trade(trade_id, result, profit, exit_spot=exit_spot)
        self.open_positions.pop(trade_id, None)
        if not first_time:
            return
        if raw_probability is not None:
            self.calibration.record_outcome(float(raw_probability), won)
        self.logger.info(f"{symbol} SETTLED ({source}): {result} pnl={profit:+.4f}"
                         + (f" exit={exit_spot}" if exit_spot is not None else ""))

    async def _reconcile_ambiguous_buy(self, symbol: str, since: float,
                                       stake: float | None = None, payout: float | None = None) -> dict | None:
        """A timed-out buy may still have gone through. Never retry (that can
        open a second contract); look for it in the portfolio instead.

        When stake/payout are given, a contract must also match them (to the
        cent / within 1%). That stops this bot adopting a contract opened at
        the same moment by ANOTHER bot sharing the same Deriv account."""
        known = {t.get("contract_id") for t in self.storage.open_trades(shadow=False)}
        for attempt in range(3):
            try:
                contracts = await self.client.portfolio()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"{symbol}: portfolio check failed ({exc!r}), attempt {attempt + 1}/3")
                await asyncio.sleep(2)
                continue
            for c in contracts:
                cid = str(c.get("contract_id"))
                sym = c.get("symbol") or c.get("underlying_symbol")
                if (cid not in known and sym == symbol
                        and str(c.get("contract_type", "")).upper() == "EXPIRYRANGE"
                        and float(c.get("purchase_time") or 0) >= since
                        and _matches_order(c, stake, payout)):
                    self.logger.warning(f"{symbol}: ambiguous buy DID go through -> contract {cid}")
                    return {"contract_id": cid, "buy_price": c.get("buy_price")}
            return None
        return None

    async def _shadow_resolver_loop(self):
        """Shadow trades used to be logged and then left OPEN forever, so
        shadow mode -- the recommended way to validate the model -- never
        actually produced a win rate or P&L to validate against. This settles
        each one against the real tick at expiry, the same way Deriv decides
        an Ends Between contract: exit spot strictly inside the barriers."""
        while not self._stop.is_set():
            try:
                now = time.time()
                for t in self.storage.open_trades(shadow=True):
                    expires_at = t.get("expires_at")
                    if not expires_at or now < expires_at + SHADOW_RESOLVE_GRACE_SECONDS:
                        continue
                    spot = await self.client.get_spot_at(t["symbol"], int(expires_at))
                    if spot is None:
                        continue  # retry next pass
                    exit_price, _ = spot
                    won = float(t["lower_barrier"]) < exit_price < float(t["upper_barrier"])
                    stake = float(t["stake"] or 0)
                    profit = (float(t["payout"] or 0) - stake) if won else -stake
                    self._settle(t["trade_id"], t["symbol"], t.get("raw_probability"), won, profit,
                                 exit_spot=exit_price, source="shadow")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"shadow resolver pass failed (non-fatal): {exc!r}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=SHADOW_RESOLVE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    # -------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {
            "account_mode": self.cfg.account_mode,
            "shadow_mode": self.cfg.shadow_mode,
            "state": self.sm.state.value,
            "uptime_seconds": int(time.time() - self.started_at),
            "open_positions": self._open_count(),
            "active_cautions": {k: c.reason for k, c in self.sm.active_cautions().items()},
            "staked_today": round(self.storage.stake_since(utc_day_start(), shadow=self.cfg.shadow_mode), 2),
            "max_daily_exposure": self.cfg.max_daily_exposure,
            "shadow": self.storage.performance_summary(shadow=True),
            "executed": self.storage.performance_summary(shadow=False),
        }

    async def _stats_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=STATS_LOG_INTERVAL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            try:
                s = self.stats()
                for mode in ("shadow", "executed"):
                    p = s[mode]
                    if p["settled"]:
                        self.logger.info(
                            f"PERFORMANCE[{mode}] settled={p['settled']} win_rate={p['win_rate']} "
                            f"avg_predicted={p['avg_predicted']} avg_implied={p['avg_implied']} "
                            f"pnl={p['pnl']:+.2f} roi={p['roi']}"
                        )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"stats logging failed (non-fatal): {exc!r}")

    def is_healthy(self) -> tuple[bool, str]:
        if self.client is None:
            return False, "not connected"
        age = time.time() - (self.client.last_message_at or 0)
        if age > HEALTH_STALE_AFTER_SECONDS:
            return False, f"no Deriv traffic for {int(age)}s"
        return True, "ok"

    async def shutdown(self):
        self.logger.info("Shutdown requested -- closing cleanly")
        self._stop.set()
        for task in list(self._background):
            task.cancel()
        if self.client:
            await self.client.close()


def _matches_order(contract: dict, stake, payout) -> bool:
    """Same stake (to the cent) and same payout (within 1%) as the order we
    placed. Missing values on either side are not used to reject."""
    bp, po = _float_or_none(contract.get("buy_price")), _float_or_none(contract.get("payout"))
    if stake is not None and bp is not None and abs(bp - float(stake)) > 0.005:
        return False
    if payout is not None and po is not None and po > 0 and abs(po - float(payout)) / po > 0.01:
        return False
    return True


def _float_or_none(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _start_health_server(port: int, bot: Bot):
    """`/` -> 200 while Deriv traffic is flowing, 503 once it goes stale
    (previously always 200, even with a dead connection). `/stats` -> JSON
    scorecard: shadow vs executed win rate, predicted vs realized, P&L."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/stats":
                try:
                    body = json.dumps(bot.stats(), indent=2, default=str).encode()
                    code = 200
                except Exception as exc:  # noqa: BLE001
                    body, code = json.dumps({"error": repr(exc)}).encode(), 500
                self._reply(code, "application/json", body)
                return
            ok, reason = bot.is_healthy()
            self._reply(200 if ok else 503, "text/plain", reason.encode())

        def _reply(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence default access logs
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bot.logger.info(f"Health endpoint listening on :{port} (/ and /stats)")
    return server


async def main():
    bot = Bot()
    await bot.start()
    _start_health_server(bot.cfg.health_port, bot)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))
        except NotImplementedError:
            pass  # Windows dev fallback; Railway runs Linux

    await bot.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
