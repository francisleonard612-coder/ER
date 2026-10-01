"""Duration priority: 5+ minute contracts are priced and chosen first;
shorter ones are only a fallback."""
import asyncio
import logging
from types import SimpleNamespace

import numpy as np

import app.strategy.engine as engine
from app.config import Config


def _cand(d):
    mc = SimpleNamespace(probability=0.95, probability_uncertainty=0.01, model_disagreement=0.0, path_count=1000)
    return SimpleNamespace(duration_minutes=d, lower_barrier=990.0, upper_barrier=1010.0, mc=mc)


class _Client:
    def __init__(self, payout_by_duration):
        self.payout_by_duration, self.priced = payout_by_duration, []

    async def request_proposal(self, symbol, duration_minutes, stake, lower_barrier, upper_barrier):
        self.priced.append(duration_minutes)
        return {"payout": self.payout_by_duration(duration_minutes), "id": f"p{duration_minutes}", "spot": 1000.0}


class _Cal:
    def snapshot(self):
        return {}

    def calibrate(self, p, buckets=None):
        return p


class _Stake:
    def stake_for(self, **kw):
        return 0.35


def _run(monkeypatch, durations, payout, **cfg_over):
    monkeypatch.setattr(engine, "build_candidates", lambda *a, **k: [_cand(d) for d in durations])
    cfg = Config()
    cfg.min_regime_confidence_to_trade = 0.0
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    client = _Client(payout)
    closes = 1000 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 300))
    out = asyncio.run(engine.run_scan_cycle(
        symbol="R_10", closes=closes, current_price=1000.0, client=client, calibration=_Cal(),
        staking=_Stake(), cfg=cfg, stake_multiplier=1.0, extra_edge_requirement=0.0,
        logger=logging.getLogger("t")))
    return out, client


def test_longer_contract_chosen_and_short_ones_not_priced(monkeypatch):
    # every duration would pass; the short ones even have the better payout
    out, client = _run(monkeypatch, [2, 3, 5, 7], lambda d: 0.90 if d < 5 else 0.70,
                       preferred_min_duration_minutes=5)
    assert out.traded and out.trade_row["duration_minutes"] >= 5
    assert all(d >= 5 for d in client.priced)


def test_short_contracts_are_the_fallback(monkeypatch):
    # long contracts pay too little to pass the edge check -> fall back to short
    out, client = _run(monkeypatch, [2, 3, 5, 7], lambda d: 0.36 if d >= 5 else 0.70,
                       preferred_min_duration_minutes=5)
    assert out.traded and out.trade_row["duration_minutes"] < 5
    assert client.priced[:2] == [5, 7]                      # long ones were tried first


def test_fallback_can_be_disabled(monkeypatch):
    out, client = _run(monkeypatch, [2, 3, 5, 7], lambda d: 0.36 if d >= 5 else 0.70,
                       preferred_min_duration_minutes=5, short_fallback_candidates=0)
    assert not out.traded and all(d >= 5 for d in client.priced)
