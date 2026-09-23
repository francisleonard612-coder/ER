"""
The mispricing / decision engine (spec sections 11, 12, 23).

Given a candidate's calibrated probability and a *live* Deriv proposal, this
computes payout multiplier, implied probability, edge and EV, and decides
accept/reject with an explicit, logged reason. Nothing here executes trades
or talks to the network -- pure decision logic so it's independently
testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Decision:
    accept: bool
    reason: str
    payout: float
    payout_multiplier: float
    implied_probability: float
    edge: float
    expected_value: float


def payout_multiplier(payout: float, stake: float) -> float:
    if stake <= 0:
        return 0.0
    return payout / stake


def implied_probability(payout_multiplier_value: float) -> float:
    if payout_multiplier_value <= 0:
        return 1.0
    return 1.0 / payout_multiplier_value


def expected_value(calibrated_probability: float, payout_multiplier_value: float) -> float:
    return calibrated_probability * payout_multiplier_value - 1.0


def evaluate(
    calibrated_probability: float,
    probability_uncertainty: float,
    payout: float,
    stake: float,
    min_payout_multiplier: float,
    min_edge: float,
    min_ev: float,
    max_probability_uncertainty: float,
    extra_edge_requirement: float = 0.0,
) -> Decision:
    pm = payout_multiplier(payout, stake)
    implied = implied_probability(pm)
    edge = calibrated_probability - implied
    ev = expected_value(calibrated_probability, pm)

    required_edge = min_edge + extra_edge_requirement

    if pm < min_payout_multiplier:
        return Decision(False, f"PAYOUT BELOW {min_payout_multiplier:.2f}", payout, pm, implied, edge, ev)
    if probability_uncertainty > max_probability_uncertainty:
        return Decision(False, "HIGH UNCERTAINTY", payout, pm, implied, edge, ev)
    if edge < required_edge:
        return Decision(False, "INSUFFICIENT EDGE", payout, pm, implied, edge, ev)
    if ev < min_ev:
        return Decision(False, "NEGATIVE OR INSUFFICIENT EV", payout, pm, implied, edge, ev)

    return Decision(True, "ACCEPTED", payout, pm, implied, edge, ev)
