"""Regression: Deriv rejects stakes with >2 decimals ('Stake can not have
more than 2 decimal places'), which blocked every proposal after losses."""
from decimal import Decimal

from app.config import StakingConfig
from app.strategy.staking import FixedStaking, MartingaleStaking, to_cents


def _two_dp(x: float) -> bool:
    return Decimal(str(x)).as_tuple().exponent >= -2


def test_to_cents_rounds_down_and_respects_minimum():
    assert to_cents(0.39375) == 0.39
    assert to_cents(1.0849999) == 1.08
    assert to_cents(0.175, minimum=0.35) == 0.35


def test_every_martingale_rung_and_caution_level_is_whole_cents():
    cfg = StakingConfig()
    cfg.mode, cfg.initial_stake, cfg.martingale_factor, cfg.martingale_steps = "martingale", 0.35, 3.1, 3
    cfg.minimum_stake, cfg.maximum_stake = 0.35, 4.0
    for engine in (MartingaleStaking(cfg), FixedStaking(cfg)):
        for losses in range(6):
            for mult in (1.0, 0.5, 0.25, 0.33):
                s = engine.stake_for(0.0, 0.6, stake_multiplier=mult, consecutive_losses=losses)
                assert _two_dp(s), s
                assert cfg.minimum_stake <= s <= cfg.maximum_stake
