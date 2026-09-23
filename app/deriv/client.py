"""
Deriv WebSocket connection layer.

Uses the current production endpoint (`wss://ws.derivws.com/websockets/v3`).
The older `wss://ws.binaryws.com/...` host still resolves for some accounts
but has been superseded — see api.deriv.com. This client is written against
the plain `websockets` library rather than the `@deriv/deriv-api` JS SDK or
the `python_deriv_api` wrapper, so reconnect/backoff/resubscribe behaviour
is fully explicit and controllable for a long-running Railway service.

Responsibilities:
- connect / authorize / reconnect with exponential backoff
- heartbeat (ping) so idle connections aren't dropped by the server or LB
- request/response correlation via req_id
- tick + candle subscriptions that auto-resubscribe after a reconnect
- proposal and buy calls
- never raise out of the read loop on a single bad frame — log and continue
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed


class DerivAPIError(Exception):
    def __init__(self, code: str, message: str, raw: dict):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.raw = raw


@dataclass
class Subscription:
    request: Dict[str, Any]
    callback: Callable[[dict], None]
    subscription_id: Optional[str] = None


class DerivClient:
    def __init__(self, endpoint: str, app_id: str, api_token: str, logger,
                 on_reconnect: Optional[Callable[[], "asyncio.Future"]] = None):
        self.base_endpoint = endpoint
        self.app_id = app_id
        self.api_token = api_token
        self.logger = logger
        self.on_reconnect = on_reconnect

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id_counter = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}
        self._subscriptions: Dict[str, Subscription] = {}  # keyed by our local key
        self._connected = asyncio.Event()
        self._closing = False
        self._read_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0
        self.authorized = False
        self.account_info: Dict[str, Any] = {}

    @property
    def url(self) -> str:
        return f"{self.base_endpoint}?app_id={self.app_id}"

    # ---------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        self._closing = False
        await self._connect_once()
        self._read_task = asyncio.create_task(self._read_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

    async def _connect_once(self) -> None:
        attempt = 0
        while not self._closing:
            try:
                self.logger.info(f"Connecting to Deriv: {self.base_endpoint} (app_id={self.app_id})")
                self._ws = await websockets.connect(self.url, ping_interval=None, close_timeout=5)
                self._connected.set()
                self._reconnect_delay = 1.0
                self.logger.info("Deriv WebSocket connected")
                if self.api_token:
                    await self.authorize(self.api_token)
                return
            except Exception as exc:  # noqa: BLE001 - must never crash the process
                attempt += 1
                self.logger.error(f"Deriv connect failed (attempt {attempt}): {exc!r}")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def close(self) -> None:
        self._closing = True
        for task in (self._read_task, self._ping_task):
            if task:
                task.cancel()
        if self._ws is not None:
            await self._ws.close()
        self.logger.info("Deriv WebSocket closed cleanly")

    # ------------------------------------------------------------------ reads
    async def _read_loop(self) -> None:
        while not self._closing:
            try:
                if self._ws is None:
                    await asyncio.sleep(0.5)
                    continue
                raw = await self._ws.recv()
                msg = json.loads(raw)
            except ConnectionClosed as exc:
                self.logger.warning(f"Deriv connection closed: {exc!r} — reconnecting")
                await self._handle_disconnect()
                continue
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"Deriv read error (non-fatal): {exc!r}")
                await asyncio.sleep(0.5)
                continue

            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        if msg.get("error"):
            err = msg["error"]
            req_id = msg.get("req_id")
            if req_id in self._pending:
                self._pending[req_id].set_exception(
                    DerivAPIError(err.get("code", "UNKNOWN"), err.get("message", ""), msg)
                )
                del self._pending[req_id]
            else:
                self.logger.error(f"Deriv API error (unsolicited): {err}")
            return

        req_id = msg.get("req_id")
        if req_id in self._pending:
            self._pending[req_id].set_result(msg)
            del self._pending[req_id]
            return

        # subscription push (tick, ohlc, proposal_open_contract stream, etc.)
        sub_id = (msg.get("subscription") or {}).get("id")
        for sub in self._subscriptions.values():
            if sub.subscription_id == sub_id:
                try:
                    sub.callback(msg)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Subscription callback error (non-fatal): {exc!r}")
                return

    async def _handle_disconnect(self) -> None:
        self._connected.clear()
        self.authorized = False
        self._ws = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionClosed(None, None))
        self._pending.clear()
        await self._connect_once()
        # resubscribe everything that was active before the drop
        for key, sub in list(self._subscriptions.items()):
            try:
                sub.subscription_id = await self._resubscribe(sub)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"Resubscribe failed for {key}: {exc!r}")
        if self.on_reconnect:
            await self.on_reconnect()

    async def _resubscribe(self, sub: Subscription) -> Optional[str]:
        resp = await self.send(sub.request)
        return (resp.get("subscription") or {}).get("id")

    async def _ping_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(20)
            if self._ws is not None:
                try:
                    await self.send({"ping": 1})
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"Ping failed (non-fatal): {exc!r}")

    # ------------------------------------------------------------------ send
    async def send(self, payload: dict, timeout: float = 15.0) -> dict:
        await self._connected.wait()
        req_id = next(self._req_id_counter)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
        except Exception:
            self._pending.pop(req_id, None)
            raise
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"Deriv request timed out: {payload.get('msg_type', list(payload.keys()))}")

    async def authorize(self, token: str) -> dict:
        resp = await self.send({"authorize": token})
        self.authorized = True
        self.account_info = resp.get("authorize", {})
        loginid = self.account_info.get("loginid", "?")
        is_virtual = self.account_info.get("is_virtual", 1)
        self.logger.info(f"Authorized as {loginid} (virtual={bool(is_virtual)})")
        return resp

    # ------------------------------------------------------------- high level
    async def subscribe_ticks(self, symbol: str, callback: Callable[[dict], None]) -> str:
        key = f"ticks:{symbol}"
        request = {"ticks": symbol, "subscribe": 1}
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        return key

    async def subscribe_candles(self, symbol: str, granularity_seconds: int,
                                 count: int, callback: Callable[[dict], None]) -> str:
        key = f"candles:{symbol}:{granularity_seconds}"
        request = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity_seconds,
            "count": count,
            "subscribe": 1,
            "end": "latest",
        }
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        return key

    async def get_candle_history(self, symbol: str, granularity_seconds: int, count: int) -> List[dict]:
        resp = await self.send({
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity_seconds,
            "count": count,
            "end": "latest",
        })
        return resp.get("candles", [])

    async def request_proposal(self, symbol: str, duration_minutes: int, stake: float,
                                lower_barrier: float, upper_barrier: float,
                                currency: str = "USD") -> dict:
        """EXPIRYRANGE == Deriv's 'ENDSINOUT' / 'Ends Between' contract type."""
        resp = await self.send({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": "EXPIRYRANGE",
            "currency": currency,
            "duration": duration_minutes,
            "duration_unit": "m",
            "symbol": symbol,
            "barrier": f"+{lower_barrier}" if lower_barrier >= 0 else str(lower_barrier),
            "barrier2": f"+{upper_barrier}" if upper_barrier >= 0 else str(upper_barrier),
        })
        return resp.get("proposal", {})

    async def buy_contract(self, proposal_id: str, price: float) -> dict:
        resp = await self.send({"buy": proposal_id, "price": price})
        return resp.get("buy", {})

    async def subscribe_contract(self, contract_id: str, callback: Callable[[dict], None]) -> str:
        key = f"contract:{contract_id}"
        request = {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        return key

    async def balance(self) -> dict:
        resp = await self.send({"balance": 1})
        return resp.get("balance", {})
