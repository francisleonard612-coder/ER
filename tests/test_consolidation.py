"""Consolidation gate: indicators read ranges vs trends correctly, the gate
blocks before any Deriv request, and readings are stored with trades."""
import asyncio
import logging

import numpy as np
import pytest

from app.features.consolidation import (ConsolidationConfig, assess, efficiency_ratio,
                                        hurst_aggvar)


def _candles(close, spread=0.0002, seed=0):
    rng = np.random.default_rng(seed)
    wig = np.abs(rng.normal(0, spread, len(close))) * close
    prev = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(prev, close) + wig, np.minimum(prev, close) - wig, close


def _range(n=600, seed=1):
    rng = np.random.default_rng(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = 0.6 * x + rng.normal(0, 1.0)           # strongly mean-reverting around 1000
        out.append(1000 + x * 0.5)
    return np.array(out)


def _trend(n=600, seed=2):
    rng = np.random.default_rng(seed)
    return 1000 + np.cumsum(0.8 + rng.normal(0, 0.3, n))


def _cfg(**kw):
    c = ConsolidationConfig()
    c.mode, c.required, c.min_optional = "on", ["ER", "ADX", "SQUEEZE"], 0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_efficiency_ratio_extremes():
    assert efficiency_ratio(np.arange(50.0), 20) == pytest.approx(1.0)
    zigzag = np.array([0.0, 1.0] * 30)
    assert efficiency_ratio(zigzag, 20) < 0.1


def test_hurst_separates_mean_reversion_from_trend_persistence():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 4000)
    mean_rev = e[1:] - 0.9 * e[:-1]                 # strongly negatively autocorrelated returns
    persistent = np.convolve(e, np.ones(8) / 8, mode="valid")  # positively autocorrelated
    assert hurst_aggvar(mean_rev[-128:]) < 0.45
    assert hurst_aggvar(persistent[-128:]) > 0.55


def test_range_reads_as_consolidation_and_trend_does_not():
    r = assess(*_candles(_range()), _cfg(required=["ER", "ADX"]))
    t = assess(*_candles(_trend()), _cfg(required=["ER", "ADX"]))
    assert r.checks["ER"] and r.checks["ADX"], r.summary()
    assert not t.checks["ER"] and not t.checks["ADX"], t.summary()
    assert r.passed and not t.passed
    assert t.reason.startswith("not consolidating")


def test_squeeze_detects_recent_volatility_compression():
    rng = np.random.default_rng(4)
    wild = np.cumsum(rng.normal(0, 2.0, 500))
    calm = wild[-1] + np.cumsum(rng.normal(0, 0.2, 100))
    rd = assess(*_candles(1000 + np.concatenate([wild, calm])), _cfg())
    assert rd.checks["SQUEEZE"], rd.summary()


def test_warming_up_never_passes():
    rd = assess(*_candles(_range(80)), _cfg())
    assert not rd.passed and "warming up" in rd.reason


def test_min_optional_counts_non_required_checks():
    ohlc = _candles(_range())
    base = assess(*ohlc, _cfg(required=["ER"]))
    n_opt = sum(1 for k, v in base.checks.items() if k != "ER" and v)
    assert assess(*ohlc, _cfg(required=["ER"], min_optional=n_opt)).passed
    assert not assess(*ohlc, _cfg(required=["ER"], min_optional=n_opt + 1)).passed


def test_row_columns_match_storage_schema():
    from app.data.storage import rejected_signals, trades
    row = assess(*_candles(_range()), _cfg()).as_row()
    for table in (trades, rejected_signals):
        assert set(row) <= set(table.c.keys()), set(row) - set(table.c.keys())


def test_invalid_mode_is_reported():
    assert ConsolidationConfig(mode="maybe").validate()
    assert ConsolidationConfig(required=["NOPE"]).validate()


class _NoCallClient:
    async def request_proposal(self, **kw):  # pragma: no cover - must never be called
        raise AssertionError("gate should have blocked before any proposal request")


def test_gate_blocks_before_any_deriv_request():
    from app.config import Config
    from app.strategy.engine import run_scan_cycle

    cfg = Config()
    cfg.consolidation = _cfg(required=["ER", "ADX"])
    trend = _trend()
    out = asyncio.run(run_scan_cycle(
        symbol="TEST", closes=trend[-300:], current_price=float(trend[-1]), client=_NoCallClient(),
        calibration=None, staking=None, cfg=cfg, stake_multiplier=1.0, extra_edge_requirement=0.0,
        logger=logging.getLogger("t"), ohlc=_candles(trend)))
    assert not out.traded and out.trade_row is None and out.rejections == []
