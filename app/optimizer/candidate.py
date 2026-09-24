"""
Searches the 2-10 minute duration grid x volatility-normalized barrier
grid x symmetric/asymmetric barrier placement, scoring each candidate with
the Monte Carlo engine. Returns candidates sorted by expected value so the
strategy layer can apply payout/edge/uncertainty filters on top.

This module does NOT talk to Deriv -- it produces barrier/duration
candidates and raw probabilities. Actual payout/implied-probability/edge/EV
requires a live proposal, which the strategy layer fetches per candidate
(section 22: never execute on theoretical payout).

BARRIER WIDTH USES POOL VOLATILITY, NOT THE CALLER'S current_volatility --
this is deliberate, not an oversight, and fixes a real bug found live:
`current_volatility` (a noisy ~20-bar sample estimate, computed by the
caller) used to size BOTH the barrier distance here AND the Monte Carlo
rescale target in monte_carlo.py's estimate_probability_in_range. Driving
both from the same noisy short-window number is circular: when a quiet
20-bar stretch makes that estimate read anomalously low, the barrier
shrinks and the simulated path dispersion shrinks by the same factor,
so our own probability estimate comes out high almost by construction --
regardless of whether the 20-bar reading reflects real conditions. Deriv's
own pricing isn't fooled by that (it prices off real volatility), so it
correctly quotes a low implied probability for what is actually a very
tight barrier -- which our model then misreads as a huge "edge" that is
really just its own circular reasoning. This produced ~0.6 edges and
~2.9x payouts on accepted R_10 candidates in production, which is not a
plausible real market mispricing.

The fix: barrier width is sized from `pool`'s own (stable, full-history)
volatility -- the same denominator the Monte Carlo engine already computes
internally as `pool_vol` -- while `current_volatility` continues to flow
through to the Monte Carlo engine ONLY, where rescaling simulated
dispersion to recent/regime conditions is a legitimate, intentional
design choice. Decoupling the two means a barrier is now judged against a
stable yardstick, so a genuine gap between our probability estimate and
Deriv's pricing reflects something real rather than the same noisy number
compared against itself.
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
    current_volatility: float,    # regime-adaptive estimate; flows to Monte Carlo rescale ONLY
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

        # Stable yardstick for barrier WIDTH -- full-history pool volatility,
        # not the noisy short-window current_volatility. See module docstring.
        barrier_vol = float(np.std(pool, ddof=1)) if len(pool) > 1 else current_volatility
        if barrier_vol <= 0:
            barrier_vol = current_volatility

        for mult in vol_multiples:
            distance = mult * barrier_vol * current_price
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
