import numpy as np

from app.optimizer.candidate import build_candidates


class _FakeMC:
    minimum_paths, default_paths, maximum_paths, near_threshold_band = 300, 600, 1200, 0.03


def _pools():
    rng = np.random.default_rng(0)
    pool = rng.normal(0, 0.0001, size=200)
    return {d: pool for d in range(2, 11)}


def test_non_calm_regime_caps_duration():
    cands = build_candidates(
        1000.0, _pools(), current_volatility=0.0001,
        durations_minutes=list(range(2, 11)), vol_multiples=[1.0],
        trade_threshold_probability=0.71, mc_config=_FakeMC(),
        regime_name="HIGH_VOLATILITY", regime_confidence=0.9,
        non_calm_max_duration_minutes=4,
    )
    assert cands, "expected some candidates"
    assert max(c.duration_minutes for c in cands) <= 4


def test_calm_confident_regime_allows_full_duration_range():
    cands = build_candidates(
        1000.0, _pools(), current_volatility=0.0001,
        durations_minutes=list(range(2, 11)), vol_multiples=[1.0],
        trade_threshold_probability=0.71, mc_config=_FakeMC(),
        regime_name="LOW_VOLATILITY_RANGE", regime_confidence=0.8,
        calm_regimes=("LOW_VOLATILITY_RANGE", "VOLATILITY_CONTRACTION"),
        calm_regime_confidence_floor=0.6, non_calm_max_duration_minutes=4,
    )
    assert max(c.duration_minutes for c in cands) == 10


def test_calm_regime_but_low_confidence_still_capped():
    # calm regime name alone isn't enough -- confidence must clear the floor too
    cands = build_candidates(
        1000.0, _pools(), current_volatility=0.0001,
        durations_minutes=list(range(2, 11)), vol_multiples=[1.0],
        trade_threshold_probability=0.71, mc_config=_FakeMC(),
        regime_name="LOW_VOLATILITY_RANGE", regime_confidence=0.4,
        calm_regimes=("LOW_VOLATILITY_RANGE", "VOLATILITY_CONTRACTION"),
        calm_regime_confidence_floor=0.6, non_calm_max_duration_minutes=4,
    )
    assert max(c.duration_minutes for c in cands) <= 4
