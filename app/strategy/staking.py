"""
Modular staking. Ships with fixed staking at the mandated 0.35 default;
the interface is deliberately narrow so adaptive staking (Kelly-fraction,
confidence-scaled, etc.) can be swapped in later without touching the
decision engine.
"""
from __future__ import annotations

from app.config import StakingConfig


class FixedStaking:
    def __init__(self, cfg: StakingConfig):
        self.cfg = cfg

    def stake_for(self, edge: float, decision_score: float, stake_multiplier: float = 1.0) -> float:
        stake = self.cfg.initial_stake * stake_multiplier
        return max(self.cfg.minimum_stake, min(stake, self.cfg.maximum_stake))


def build_staking_engine(cfg: StakingConfig):
    if cfg.mode == "fixed":
        return FixedStaking(cfg)
    raise ValueError(f"Unknown staking mode: {cfg.mode}")
