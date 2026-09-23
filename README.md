# EXPIRYRANGE Adaptive Monte Carlo Bot (Deriv)

Adaptive "Ends Between" (Deriv contract type `EXPIRYRANGE`) trading engine:
searches 2–10 minute expiries and volatility-normalized barrier candidates,
prices each with a live Deriv proposal, and only trades when the calibrated
model probability beats the market-implied probability by a required edge,
at a payout multiplier of at least 1.40x, starting from a 0.35 stake.

**Read "Scope of this build" below before deploying with real money.**

## Project structure

```
expiryrange_bot/
├── run.py                     # production entrypoint
├── app/
│   ├── config.py               # env-var driven configuration
│   ├── state_machine.py        # explicit states + self-expiring caution states
│   ├── deriv/client.py         # WebSocket layer: connect/auth/reconnect/proposal/buy
│   ├── data/storage.py         # SQLite/Postgres persistence (trades, rejections, calibration)
│   ├── data/loader.py          # schema-agnostic CSV loader for historical seed data
│   ├── features/stats.py       # returns, volatility, skew/kurtosis, regime detection
│   ├── models/monte_carlo.py   # empirical + block bootstrap ensemble
│   ├── models/calibration.py   # online, bucketed probability calibration
│   ├── optimizer/candidate.py  # duration x barrier grid search
│   ├── strategy/filters.py     # payout/edge/EV/uncertainty decision engine
│   ├── strategy/staking.py     # fixed staking (0.35 default), pluggable
│   ├── strategy/engine.py      # per-symbol scan cycle, ties everything together
│   └── monitoring/logging_utils.py
├── tests/                      # pytest suite for the core math
├── Dockerfile / railway.toml / requirements.txt / .env.example
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: leave SHADOW_MODE=true and ACCOUNT_MODE=DEMO for your first run
python run.py
```

## Deploying to Railway

1. Push this repo to GitHub, create a new Railway project from it.
2. Railway will build via the included `Dockerfile`.
3. Add a Postgres plugin (optional but recommended) — Railway will inject
   `DATABASE_URL` automatically; the bot picks it up with no code changes.
   Without it, the bot uses a local SQLite file, which is fine for testing
   but is lost on redeploy since the container filesystem is ephemeral.
4. Set environment variables (see `.env.example` for the full list):
   - `DERIV_API_TOKEN`, `DERIV_APP_ID`
   - `DERIV_ACCOUNT_MODE=DEMO` (keep as DEMO until you've reviewed behavior)
   - `SHADOW_MODE=true` (keep true until you trust the shadow-mode logs)
5. Deploy. `railway.toml` sets the start command and a health check against
   the root path, which the bot serves via a lightweight built-in HTTP server
   on `$PORT`.

## Demo mode vs shadow mode — these are two independent switches

| ACCOUNT_MODE | SHADOW_MODE | What happens |
|---|---|---|
| DEMO | true (default) | Full pipeline runs, every decision (accept + reject) is logged with reasoning, **nothing is ever bought**. |
| DEMO | false | Real orders are placed and settled — but against your Deriv **demo** account, so no real money is at risk. Good next step once shadow logs look right. |
| LIVE | true | Full pipeline runs against live market data/pricing, decisions are logged, **nothing is ever bought**. Useful for validating the model on live proposals without risk. |
| LIVE | false | Real orders placed against your **real-money** account. Requires `DERIV_API_TOKEN` for a live account; the bot refuses to start in LIVE mode without one (see `Config.validate()`) and falls back to shadow behavior rather than crash-looping. |

`ACCOUNT_MODE` only determines which account your `DERIV_API_TOKEN`
authorizes against (demo vs. real balance); `SHADOW_MODE` is what actually
gates whether `buy_contract` is called. Recommended path: DEMO+shadow=true
first, then DEMO+shadow=false to watch it trade for real against virtual
money, then LIVE only once you're comfortable.

## Scope of this build — what's here and what isn't

This is a real, working core, not a stub-filled skeleton — every module
listed above executes actual logic (Monte Carlo simulation, live Deriv
pricing, edge/EV filtering, reconnect handling, persistence). But the
original spec describes a very large system, and I've scoped this first
pass deliberately rather than claim a fully verified, production-hardened
version of all of it:

- **No historical datasets were supplied in this conversation.** The spec
  references `candles_rows.csv` / `rejected_signals_rows.csv`; the loader
  in `app/data/loader.py` will pick them up from `data/` if you add them,
  but I haven't run the historical research/backtest step because there's
  no data to run it against yet. Send the files and I'll wire up the
  research report (win rate, drawdown, profit factor, per-regime/per-duration
  breakdowns).
- **Regime detection is a heuristic classifier**, not a learned model —
  explainable and auditable, but simpler than an ML ensemble. Swappable
  behind the same interface later.
- **No separate "shadow vs live" strategy fork** — by design, both paths run
  `run_scan_cycle` identically; only the final buy call is skipped in
  shadow mode, per the spec's requirement that shadow mode use the same
  decision engine.
- **Barrier asymmetry search is a fixed 1.3x skew factor**, not a full
  continuous search over asymmetric widths — a reasonable first pass, easily
  widened into a finer grid.
- **Test suite covers the core math** (Monte Carlo bounds, edge/EV/payout
  logic, calibration shrinkage, staking) rather than every item in the
  spec's 40+ point checklist (e.g. no live-WebSocket integration test against
  a real Deriv sandbox, no Railway-restart integration test). I ran what I
  could in this sandbox — 13 tests pass offline; `test_calibration.py`
  needs SQLAlchemy installed to run (`pip install -r requirements.txt`
  first).
- **No walk-forward backtest report** is generated yet for the same reason
  as the data point above.

None of this is guesswork dressed up as done — it's a working, testable
core with the more open-ended research/ML pieces left as clearly-marked
next steps rather than faked.

## Important

This places real trades on Deriv when `SHADOW_MODE=false` and
`DERIV_ACCOUNT_MODE=LIVE`. Nothing here is financial advice, and there is
no guarantee of profitability — a positive edge in the model's own
probability estimate does not guarantee positive real-world results,
especially before the calibration layer has accumulated enough outcomes
to matter. Start in DEMO + SHADOW mode and read the logs before risking
real funds, and keep the stake/payout/edge thresholds conservative until
you've validated behavior over time.
