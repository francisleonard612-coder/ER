"""
Production entrypoint. `python run.py`

Startup sequence (section 7): connect DB -> restore state -> connect Deriv
-> resubscribe market data -> rebuild rolling state -> resume scanning.
Every step logs clearly; no step requires manual intervention to recover
from an ordinary Railway restart.
"""
from __future__ import annotations

import asyncio
import signal
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np

from app.config import load_config
from app.data.loader import load_csv_if_present
from app.data.storage import Storage
from app.deriv.client import DerivClient
from app.models.calibration import CalibrationTracker
from app.monitoring.logging_utils import setup_logging
from app.state_machine import State, StateMachine
from app.strategy.engine import run_scan_cycle
from app.strategy.staking import build_staking_engine

CANDLE_GRANULARITY_SECONDS = 60
CANDLE_HISTORY_COUNT = 300


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.logger = setup_logging(self.cfg.log_level)
        self.sm = StateMachine(self.logger)
        self.storage = Storage(self.cfg.database_url, self.cfg.sqlite_path, self.logger)
        self.calibration = CalibrationTracker(self.storage)
        self.staking = build_staking_engine(self.cfg.staking)
        self.client: DerivClient | None = None
        self.closes_by_symbol: dict[str, deque] = defaultdict(lambda: deque(maxlen=CANDLE_HISTORY_COUNT))
        self.open_contracts_by_symbol: dict[str, int] = defaultdict(int)
        self._stop = asyncio.Event()

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

        self.sm.transition(State.READY)
        self.logger.info("BOT READY")

    def _load_historical_seed(self):
        # Optional seed data -- see data/candles_rows.csv, data/rejected_signals_rows.csv.
        # Not present in this deployment yet; loader logs a warning and the bot
        # proceeds on live/online data only, per section 26 (no walk-forward prerequisite).
        load_csv_if_present("data/candles_rows.csv", self.logger)
        load_csv_if_present("data/rejected_signals_rows.csv", self.logger)

    async def _prime_candles(self):
        for symbol in self.cfg.symbols:
            try:
                candles = await self.client.get_candle_history(symbol, CANDLE_GRANULARITY_SECONDS, CANDLE_HISTORY_COUNT)
                for c in candles:
                    self.closes_by_symbol[symbol].append(float(c["close"]))
                self.logger.info(f"{symbol}: primed with {len(candles)} candles")
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"{symbol}: failed to prime candles ({exc!r})")

            def _make_cb(sym):
                def _cb(msg):
                    candle = msg.get("candles") or msg.get("ohlc")
                    if candle is None:
                        return
                    if isinstance(candle, list):
                        for c in candle:
                            self.closes_by_symbol[sym].append(float(c["close"]))
                    else:
                        close = candle.get("close")
                        if close is not None:
                            self.closes_by_symbol[sym].append(float(close))
                return _cb

            try:
                await self.client.subscribe_candles(symbol, CANDLE_GRANULARITY_SECONDS, CANDLE_HISTORY_COUNT, _make_cb(symbol))
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"{symbol}: candle subscription failed ({exc!r})")

    async def _on_reconnect(self):
        # rolling in-memory state (tick buffers etc.) survives in this process;
        # subscriptions are already re-established by the client itself.
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
        closes = np.array(self.closes_by_symbol[symbol], dtype=float)
        if len(closes) == 0:
            return
        current_price = closes[-1]

        if self.open_contracts_by_symbol[symbol] >= self.cfg.max_concurrent_per_symbol:
            return  # no overlapping same-symbol trades; clears automatically on settlement
        if sum(self.open_contracts_by_symbol.values()) >= self.cfg.max_concurrent_contracts:
            return

        self.sm.transition(State.SIMULATING, symbol)
        try:
            outcome = await run_scan_cycle(
                symbol=symbol, closes=closes, current_price=current_price, client=self.client,
                calibration=self.calibration, staking=self.staking, cfg=self.cfg,
                stake_multiplier=self.sm.stake_multiplier(),
                extra_edge_requirement=self.sm.extra_edge_requirement(),
                logger=self.logger,
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
        losses = self.storage.consecutive_losses(symbol)
        if losses >= self.cfg.consecutive_loss_caution_threshold and not self.sm.is_symbol_cautioned(symbol):
            self.sm.enter_caution(
                symbol, f"{losses} consecutive losses", self.cfg.caution_cooldown_seconds,
                edge_penalty=0.03, stake_multiplier=0.5,
            )

        if not outcome.traded or outcome.trade_row is None:
            return

        row = outcome.trade_row
        # SHADOW_MODE alone controls execution vs logging-only. ACCOUNT_MODE
        # (DEMO/LIVE) only controls which Deriv account the token authorizes
        # against -- DEMO + shadow_mode=false places real orders against your
        # demo (virtual-money) balance; LIVE + shadow_mode=false places real
        # orders against real money. Config.validate() already refuses to
        # start in LIVE without a token; it does not (and should not) block
        # DEMO from executing, since demo trades risk nothing real.
        shadow = self.cfg.shadow_mode

        if shadow:
            self.logger.info(f"{symbol} SHADOW TRADE (not executed): ev={row['expected_value']:+.4f} stake={row['stake']}")
            self.storage.record_trade({**row, "shadow": 1, "contract_id": None, "balance_before": None})
            return

        self.sm.transition(State.PROPOSAL_PENDING, symbol)
        try:
            balance_before = (await self.client.balance()).get("balance")
        except Exception:
            balance_before = None

        self.sm.transition(State.EXECUTING, symbol)
        try:
            buy_resp = await self.client.buy_contract(row["proposal_id"], row["stake"])
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: buy failed (non-fatal): {exc!r}")
            self.sm.enter_caution(symbol, f"buy error: {exc!r}", self.cfg.caution_cooldown_seconds)
            return

        contract_id = buy_resp.get("contract_id")
        self.storage.record_trade({**row, "shadow": 0, "contract_id": str(contract_id), "balance_before": balance_before})
        self.open_contracts_by_symbol[symbol] += 1
        self.sm.transition(State.POSITION_OPEN, symbol)
        self.logger.info(f"{symbol} TRADE EXECUTED: contract_id={contract_id} stake={row['stake']} ev={row['expected_value']:+.4f}")

        def _on_settle(msg):
            poc = msg.get("proposal_open_contract", {})
            if not poc.get("is_sold"):
                return
            profit = float(poc.get("profit", 0.0))
            result = "WIN" if profit > 0 else "LOSS"
            self.storage.settle_trade(row["trade_id"], result, profit, poc.get("sell_price"))
            self.calibration.record_outcome(row["calibrated_probability"], result == "WIN")
            self.open_contracts_by_symbol[symbol] = max(0, self.open_contracts_by_symbol[symbol] - 1)
            self.logger.info(f"{symbol} SETTLED: {result} pnl={profit:+.4f}")

        try:
            await self.client.subscribe_contract(str(contract_id), _on_settle)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: contract subscription failed (non-fatal): {exc!r}")

    async def shutdown(self):
        self.logger.info("Shutdown requested -- closing cleanly")
        self._stop.set()
        if self.client:
            await self.client.close()


def _start_health_server(port: int, logger):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # silence default access logs
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health endpoint listening on :{port}")
    return server


async def main():
    bot = Bot()
    await bot.start()
    _start_health_server(bot.cfg.health_port, bot.logger)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))
        except NotImplementedError:
            pass  # Windows dev fallback; Railway runs Linux

    await bot.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
