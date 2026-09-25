from app.config import StakingConfig
from app.strategy.staking import build_staking_engine


def test_fixed_stake_matches_config_default():
    engine = build_staking_engine(StakingConfig())
    stake = engine.stake_for(edge=0.1, decision_score=0.8)
    assert stake == 0.35


def test_stake_multiplier_reduces_but_respects_minimum():
    engine = build_staking_engine(StakingConfig())
    stake = engine.stake_for(edge=0.1, decision_score=0.8, stake_multiplier=0.1)
    assert stake == 0.35  # floored at minimum_stake


def test_martingale_escalates_with_consecutive_losses():
    cfg = StakingConfig(mode="martingale", minimum_stake=0.35, initial_stake=0.35,
                         maximum_stake=15.0, martingale_factor=2.5, martingale_steps=4)
    engine = build_staking_engine(cfg)
    stakes = [engine.stake_for(edge=0.1, decision_score=0.8, consecutive_losses=n) for n in range(6)]
    # 0.35, 0.875, 2.1875, 5.46875, 13.671875, then capped (not escalating further)
    assert stakes[0] == 0.35
    assert abs(stakes[1] - 0.875) < 1e-9
    assert abs(stakes[2] - 2.1875) < 1e-9
    assert abs(stakes[3] - 5.46875) < 1e-9
    assert abs(stakes[4] - 13.671875) < 1e-9
    assert stakes[5] == stakes[4]  # holds at the step-4 ceiling, does not escalate to step 5


def test_martingale_resets_to_base_after_a_win():
    cfg = StakingConfig(mode="martingale", minimum_stake=0.35, initial_stake=0.35,
                         maximum_stake=15.0, martingale_factor=2.5, martingale_steps=4)
    engine = build_staking_engine(cfg)
    after_losses = engine.stake_for(edge=0.1, decision_score=0.8, consecutive_losses=3)
    after_win = engine.stake_for(edge=0.1, decision_score=0.8, consecutive_losses=0)
    assert after_losses > 0.35
    assert after_win == 0.35


def test_martingale_respects_maximum_stake_ceiling():
    # if maximum_stake isn't raised to cover the ladder, it silently clamps
    # every rung back down -- this is exactly that failure mode, asserted
    # so it can't regress silently.
    cfg = StakingConfig(mode="martingale", minimum_stake=0.35, initial_stake=0.35,
                         maximum_stake=0.35, martingale_factor=2.5, martingale_steps=4)
    engine = build_staking_engine(cfg)
    stake = engine.stake_for(edge=0.1, decision_score=0.8, consecutive_losses=3)
    assert stake == 0.35
