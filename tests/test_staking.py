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
