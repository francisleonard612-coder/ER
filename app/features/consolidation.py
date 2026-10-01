"""
Consolidation gate: six classic range detectors on 1-minute candles.

  ER       Kaufman efficiency ratio (n bars): |net move| / sum|bar moves|.
           0 = pure back-and-forth, 1 = straight line. Low = choppy.
  SQUEEZE  Volatility compressed: Bollinger width (20, 2sd) OR ATR(14)/ATR(100)
           in the bottom X% of its own recent history.
  ADX      ADX(14) below a level = no trend strength either way.
  EMA      |EMA9 - EMA21| / ATR14 small AND both EMA slopes (over 5 bars,
           in ATR units) near zero = averages tangled together.
  MACD     |MACD histogram| / ATR14 small AND the MACD line crossed zero at
           least N times in the last 30 bars = see-saw, no momentum.
  HURST    Hurst exponent of recent returns (aggregated-variance method)
           below a level = mean-reverting. Noisy on short windows.

Every value is computed from closed bars only (no lookahead) and returned
whether or not the gate blocks, so it can be stored with each trade and the
gate's real effect measured from the trade journal.

The research test (Reversal-System tools/indicator_consolidation_test.py)
found none of these predicted a higher Ends Between win rate on
RDBULL/RDBEAR over 60 days. This module exists so that claim can be
re-checked on live/shadow trades -- compare win rate vs implied probability
for trades with cons_pass = 1 against those logged in "log" mode.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

ALL_CHECKS = ("ER", "SQUEEZE", "ADX", "EMA", "MACD", "HURST")


def _env_list(name: str, default: str) -> List[str]:
    return [x.strip().upper() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class ConsolidationConfig:
    # off  = not computed at all
    # log  = computed and stored with every trade, never blocks (use to measure)
    # on   = a trade is only considered while the rule below is met
    mode: str = field(default_factory=lambda: os.getenv("CONSOLIDATION_GATE", "on").lower())
    # checks that MUST pass ...
    required: List[str] = field(default_factory=lambda: _env_list("CONSOLIDATION_REQUIRED", "ER,ADX,SQUEEZE"))
    # ... plus at least this many of the remaining checks
    min_optional: int = int(os.getenv("CONSOLIDATION_MIN_OPTIONAL", "0"))

    er_window: int = int(os.getenv("CONS_ER_WINDOW", "20"))
    er_max: float = float(os.getenv("CONS_ER_MAX", "0.25"))
    squeeze_pct_max: float = float(os.getenv("CONS_SQUEEZE_PCT_MAX", "0.20"))
    squeeze_lookback: int = int(os.getenv("CONS_SQUEEZE_LOOKBACK", "300"))
    adx_max: float = float(os.getenv("CONS_ADX_MAX", "20"))
    ema_gap_atr_max: float = float(os.getenv("CONS_EMA_GAP_ATR_MAX", "0.5"))
    ema_slope_atr_max: float = float(os.getenv("CONS_EMA_SLOPE_ATR_MAX", "0.5"))
    macd_hist_atr_max: float = float(os.getenv("CONS_MACD_HIST_ATR_MAX", "0.15"))
    macd_min_crossings: int = int(os.getenv("CONS_MACD_MIN_CROSSINGS", "3"))
    hurst_max: float = float(os.getenv("CONS_HURST_MAX", "0.5"))
    hurst_window: int = int(os.getenv("CONS_HURST_WINDOW", "128"))

    def validate(self) -> List[str]:
        p = []
        if self.mode not in ("off", "log", "on"):
            p.append(f"CONSOLIDATION_GATE must be off, log or on (got {self.mode!r}).")
        bad = [c for c in self.required if c not in ALL_CHECKS]
        if bad:
            p.append(f"CONSOLIDATION_REQUIRED has unknown checks {bad}; valid: {', '.join(ALL_CHECKS)}.")
        return p


@dataclass
class ConsolidationReading:
    er: Optional[float] = None
    bbw_pct: Optional[float] = None
    atr_ratio_pct: Optional[float] = None
    adx: Optional[float] = None
    ema_gap_atr: Optional[float] = None
    ema_slope_atr: Optional[float] = None
    macd_hist_atr: Optional[float] = None
    macd_crossings: Optional[int] = None
    hurst: Optional[float] = None
    checks: Dict[str, bool] = field(default_factory=dict)
    passed: bool = False
    reason: str = ""

    @property
    def score(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    def as_row(self) -> dict:
        """Columns stored on every trade / rejection (prefixed cons_)."""
        d = asdict(self)
        d.pop("checks")
        d.pop("reason")
        out = {f"cons_{k}": (None if v is None else (int(v) if isinstance(v, bool) else v)) for k, v in d.items()}
        out["cons_score"] = self.score
        out["cons_checks"] = ",".join(k for k, v in self.checks.items() if v)
        return out

    def summary(self) -> str:
        def f(x, nd=2):
            return "na" if x is None else f"{x:.{nd}f}"
        flags = " ".join(f"{k}{'+' if v else '-'}" for k, v in self.checks.items())
        return (f"ER={f(self.er)} BBW%={f(self.bbw_pct)} ATR%={f(self.atr_ratio_pct)} ADX={f(self.adx, 1)} "
                f"EMAgap={f(self.ema_gap_atr)} MACDh={f(self.macd_hist_atr)} X={self.macd_crossings} "
                f"H={f(self.hurst)} [{flags}]")


# --------------------------------------------------------------- primitives
def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = float(np.mean(x[:n]))
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def true_range(high, low, close) -> np.ndarray:
    prev = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def efficiency_ratio(close: np.ndarray, n: int) -> Optional[float]:
    if len(close) <= n:
        return None
    path = float(np.sum(np.abs(np.diff(close[-n - 1:]))))
    return abs(float(close[-1] - close[-n - 1])) / path if path > 0 else 0.0


def adx(high, low, close, n: int = 14) -> Optional[float]:
    if len(close) < 3 * n:
        return None
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = wilder(true_range(high, low, close), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * wilder(pdm, n) / atr
        mdi = 100 * wilder(mdm, n) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.nan_to_num(dx[n - 1:])
    a = wilder(dx, n)
    v = a[-1]
    return None if not np.isfinite(v) else float(v)


def pct_rank_last(series: np.ndarray, lookback: int) -> Optional[float]:
    s = series[np.isfinite(series)][-lookback:]
    if len(s) < 50:
        return None
    return float(np.mean(s <= s[-1]))


def hurst_aggvar(returns: np.ndarray) -> Optional[float]:
    """Aggregated-variance Hurst estimate: Var(sum of m returns) ~ m^(2H).
    0.5 = random walk, < 0.5 = mean-reverting, > 0.5 = trending."""
    r = returns[np.isfinite(returns)]
    if len(r) < 64:
        return None
    ms, vs = [], []
    for m in (1, 2, 4, 8, 16):
        k = len(r) // m
        if k < 4:
            break
        agg = r[: k * m].reshape(k, m).sum(axis=1)
        v = float(np.var(agg, ddof=1))
        if v > 0:
            ms.append(m)
            vs.append(v)
    if len(ms) < 3:
        return None
    slope = np.polyfit(np.log(ms), np.log(vs), 1)[0]
    return float(slope / 2.0)


# ----------------------------------------------------------------- reading
def assess(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           cfg: ConsolidationConfig) -> ConsolidationReading:
    high, low, close = (np.asarray(a, dtype=float) for a in (high, low, close))
    rd = ConsolidationReading()
    if len(close) < 120:
        rd.reason = f"warming up ({len(close)} candles, need 120)"
        return rd

    tr = true_range(high, low, close)
    atr14 = wilder(tr, 14)
    atr100 = wilder(tr, 100)
    a14 = float(atr14[-1]) if np.isfinite(atr14[-1]) and atr14[-1] > 0 else None

    # 1. efficiency ratio
    rd.er = efficiency_ratio(close, cfg.er_window)

    # 2. squeeze: Bollinger width and ATR ratio, each as a percentile of their own history
    n = 20
    if len(close) >= n + 50:
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(close, n)
        bbw = 4 * w.std(axis=1) / w.mean(axis=1)
        rd.bbw_pct = pct_rank_last(bbw, cfg.squeeze_lookback)
    with np.errstate(invalid="ignore", divide="ignore"):
        rd.atr_ratio_pct = pct_rank_last(atr14 / atr100, cfg.squeeze_lookback)

    # 3. ADX
    rd.adx = adx(high, low, close, 14)

    # 4. EMA compression + 5. MACD (both in ATR units)
    if a14:
        e9, e21 = ema(close, 9), ema(close, 21)
        rd.ema_gap_atr = abs(e9[-1] - e21[-1]) / a14
        rd.ema_slope_atr = max(abs(e9[-1] - e9[-6]), abs(e21[-1] - e21[-6])) / a14
        macd = ema(close, 12) - ema(close, 26)
        hist = macd - ema(macd, 9)
        rd.macd_hist_atr = abs(float(hist[-1])) / a14
        tail = macd[-31:]
        rd.macd_crossings = int(np.sum(np.sign(tail[1:]) != np.sign(tail[:-1])))

    # 6. Hurst on recent 1-minute log returns
    rets = np.diff(np.log(close[-(cfg.hurst_window + 1):]))
    rd.hurst = hurst_aggvar(rets)

    def ok(v, test):
        return v is not None and test(v)

    rd.checks = {
        "ER": ok(rd.er, lambda v: v <= cfg.er_max),
        "SQUEEZE": ok(rd.bbw_pct, lambda v: v <= cfg.squeeze_pct_max)
                   or ok(rd.atr_ratio_pct, lambda v: v <= cfg.squeeze_pct_max),
        "ADX": ok(rd.adx, lambda v: v < cfg.adx_max),
        "EMA": ok(rd.ema_gap_atr, lambda v: v <= cfg.ema_gap_atr_max)
               and ok(rd.ema_slope_atr, lambda v: v <= cfg.ema_slope_atr_max),
        "MACD": ok(rd.macd_hist_atr, lambda v: v <= cfg.macd_hist_atr_max)
                and ok(rd.macd_crossings, lambda v: v >= cfg.macd_min_crossings),
        "HURST": ok(rd.hurst, lambda v: v < cfg.hurst_max),
    }
    missing = [c for c in cfg.required if not rd.checks.get(c, False)]
    optional = [c for c in ALL_CHECKS if c not in cfg.required]
    n_opt = sum(1 for c in optional if rd.checks[c])
    rd.passed = not missing and n_opt >= cfg.min_optional
    if missing:
        rd.reason = "not consolidating: " + ",".join(missing) + " failed"
    elif not rd.passed:
        rd.reason = f"only {n_opt}/{cfg.min_optional} optional checks passed"
    else:
        rd.reason = "consolidating"
    return rd
