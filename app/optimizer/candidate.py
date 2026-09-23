"""
Searches the 2-10 minute duration grid x volatility-normalized barrier
grid x symmetric/asymmetric barrier placement, scoring each candidate with
the Monte Carlo engine. Returns candidates sorted by expected value so the
strategy layer can apply payout/edge/uncertainty filters on top.

This module does NOT talk to Deriv -- it produces barrier/duration
candidates and raw probabilities. Actual payout/implied-probability/edge/EV
requires a live proposal, which the strategy layer fetches per candidate
(section 22: never execute on theoretical payout).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.models.monte_carlo import MCResult, adaptive_path_count, estimate_probability_in_range


@dataclass
class BarrierCandidate:
    duration_minutes: int
    lower_barrier: float
    upper_barrier: float
    symmetric: bool
    mc: MCResult


def build_candidates(
    current_price: float,
    returns_pool_by_step: dict,   # {duration_minutes: np.ndarray of per-step returns}
    current_volatility: float,
    durations_minutes: List[int],
    vol_multiples: List[float],
    trade_threshold_probability: float,
    mc_config,
) -> List[BarrierCandidate]:
    candidates: List[BarrierCandidate] = []

    for duration in durations_minutes:
        pool = returns_pool_by_step.get(duration)
        if pool is None or len(pool) < 10:
            continue
        steps = duration  # returns_pool is pre-aggregated to per-minute steps by the caller

        for mult in vol_multiples:
            distance = mult * current_volatility * current_price
            if distance <= 0:
                continue

            # symmetric
            lower = current_price - distance
            upper = current_price + distance
            candidates.append(_score_candidate(
                current_price, pool, steps, lower, upper, True, duration,
                current_volatility, trade_threshold_probability, mc_config,
            ))

            # asymmetric: skew barrier width toward the side with more room,
            # using a modest 30% asymmetry factor rather than an unconstrained search
            skew_factor = 1.3
            candidates.append(_score_candidate(
                current_price, pool, steps,
                current_price - distance, current_price + distance * skew_factor,
                False, duration, current_volatility, trade_threshold_probability, mc_config,
            ))
            candidates.append(_score_candidate(
                current_price, pool, steps,
                current_price - distance * skew_factor, current_price + distance,
                False, duration, current_volatility, trade_threshold_probability, mc_config,
            ))

    candidates.sort(key=lambda c: c.mc.probability, reverse=True)
    return candidates


def _score_candidate(current_price, pool, steps, lower, upper, symmetric, duration,
                      current_volatility, trade_threshold_probability, mc_config) -> BarrierCandidate:
    # cheap preliminary pass to decide how many paths are worth spending
    prelim = estimate_probability_in_range(
        current_price, pool, steps, lower, upper, current_volatility,
        n_paths=mc_config.minimum_paths, seed=1,
    )
    n_paths = adaptive_path_count(
        prelim.probability, trade_threshold_probability,
        mc_config.minimum_paths, mc_config.default_paths, mc_config.maximum_paths,
        mc_config.near_threshold_band,
    )
    if n_paths == mc_config.minimum_paths:
        mc = prelim
    else:
        mc = estimate_probability_in_range(
            current_price, pool, steps, lower, upper, current_volatility, n_paths=n_paths, seed=2,
        )
    return BarrierCandidate(duration, lower, upper, symmetric, mc)
