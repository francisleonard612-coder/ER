"""
Modular staking. Ships with fixed staking at the mandated 0.35 default,
plus an opt-in martingale mode (factor 2.5, 4 steps off a 0.35 base by
default -- see StakingConfig in app/config.py for the exact env vars).
The interface (stake_for) is deliberately narrow so further strategies
can be added without touching the decision engine.
"""
from __future__ import annotations

from app.config import StakingConfig


class FixedStaking:
    def __init__(self, cfg: StakingConfig):
        self.cfg = cfg

    def stake_for(self, edge: float, decision_score: float, stake_multiplier: float = 1.0,
                   consecutive_losses: int = 0) -> float:
        stake = self.cfg.initial_stake * stake_multiplier
        return max(self.cfg.minimum_stake, min(stake, self.cfg.maximum_stake))


class MartingaleStaking:
    """stake = initial_stake * factor ** min(consecutive_losses, steps).

    consecutive_losses is passed in by the caller (engine.py, sourced from
    Storage.consecutive_losses(symbol) in run.py) rather than tracked here,
    so the ladder position is derived from the durable trade journal, not
    in-memory state -- it survives a Railway restart correctly without any
    extra persistence code of its own.

    CAPS RATHER THAN RESETS AT THE STEP LIMIT: once consecutive_losses
    reaches `steps`, the stake holds at that top rung rather than either
    escalating further (unbounded exposure) or dropping back to the base
    stake (which would mean the ladder "gives up" recovering right as it
    reaches its own intended ceiling). This is a judgment call, not the
    only reasonable reading of "N steps" -- if you want a reset-to-base
    behavior instead once the cap is hit, that's a one-line change here
    (return self.cfg.initial_stake instead of holding at the capped value).

    stake_multiplier (from the existing consecutive-loss CAUTION mechanism
    in state_machine.py) is applied AFTER the martingale escalation and
    BEFORE the min/max clamp, same order FixedStaking uses. That caution
    mechanism triggers on the same consecutive-loss count this ladder
    escalates on by default (CONSECUTIVE_LOSS_CAUTION_THRESHOLD=4), so by
    the 4th loss both mechanisms are active at once: the escalated stake
    gets halved rather than held at full size. Raise
    CONSECUTIVE_LOSS_CAUTION_THRESHOLD above martingale_steps if you want
    the ladder to run independently of that safety mechanism.
    """

    def __init__(self, cfg: StakingConfig):
        self.cfg = cfg

    def stake_for(self, edge: float, decision_score: float, stake_multiplier: float = 1.0,
                   consecutive_losses: int = 0) -> float:
        step = min(max(consecutive_losses, 0), self.cfg.martingale_steps)
        stake = self.cfg.initial_stake * (self.cfg.martingale_factor ** step) * stake_multiplier
        return max(self.cfg.minimum_stake, min(stake, self.cfg.maximum_stake))


def build_staking_engine(cfg: StakingConfig):
    if cfg.mode == "fixed":
        return FixedStaking(cfg)
    if cfg.mode == "martingale":
        return MartingaleStaking(cfg)
    raise ValueError(f"Unknown staking mode: {cfg.mode}")
