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
    *,
    regime_name: str = "",
    regime_confidence: float = 0.0,
    calm_regimes: tuple = ("LOW_VOLATILITY_RANGE", "VOLATILITY_CONTRACTION"),
    calm_regime_confidence_floor: float = 0.6,
    non_calm_max_duration_minutes: int = 4,
) -> List[BarrierCandidate]:
    """
    Duration ceiling tied to regime: the full duration grid (up to its max,
    e.g. 10 minutes) is only offered when the market is in a calm regime
    (regime_name in calm_regimes) AND the regime read is itself confident
    enough (regime_confidence >= calm_regime_confidence_floor). Otherwise
    duration is capped at non_calm_max_duration_minutes.

    Why: real price dispersion grows with sqrt(duration) (see the distance
    formula below), so a longer contract is a genuinely higher-variance bet
    -- taking it isn't free edge, it's a different, riskier point on a fair
    curve. The one legitimate source of edge here is that volatility
    clusters: a currently-calm market tends to stay calmer than its
    blended historical average for a while, which is exactly what our
    Monte Carlo pool (built from the FULL history) doesn't know on its own.
    So longer, riskier durations are only offered when the regime detector
    is actually telling us "calm," and telling us so with real confidence
    -- not just whenever the duration grid happens to reach that far.
    """
    candidates: List[BarrierCandidate] = []

    is_calm_and_confident = (
        regime_name in calm_regimes and regime_confidence >= calm_regime_confidence_floor
    )
    effective_duration_ceiling = max(durations_minutes) if is_calm_and_confident else non_calm_max_duration_minutes

    for duration in durations_minutes:
        if duration > effective_duration_ceiling:
            continue
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
            # sqrt(duration) scaling: real price dispersion over N minutes
            # grows roughly with sqrt(N) for a random-walk-like process --
            # which is exactly what our own Monte Carlo engine produces,
            # since it sums N per-minute return draws. Without this, barrier
            # width was sized purely off a per-MINUTE volatility with no
            # regard for how many minutes the contract actually runs, so a
            # fixed width applied to a short 2-3 minute contract came out
            # enormously wide relative to real short-horizon dispersion --
            # a near-guaranteed win, which is exactly what Deriv's
            # "This contract offers no return" rejection means. That
            # rejection went from an occasional, expected edge case to
            # consuming ~99% of proposal requests in production once the
            # volatility estimate itself was fixed to be accurate (see
            # run.py/candidate.py history) -- an accurate per-minute vol
            # made the missing duration scaling far more visible, not less.
            distance = mult * barrier_vol * np.sqrt(duration) * current_price
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

    # SORT BY CLOSENESS TO A PRICING SWEET SPOT, NOT RAW PROBABILITY
    # DESCENDING. Sorting purely by descending probability always prefers
    # the widest, most "boringly certain" candidates across every duration --
    # exactly the ones Deriv refuses to price at all ("no return"), since a
    # near-guaranteed outcome has no payout worth quoting. What actually
    # matters is candidates sitting comfortably above the ~1/1.40 = 0.714
    # break-even implied by the payout floor, with enough margin for a real
    # edge but not so much margin that the contract is worthless to price.
    # 0.80 is that target -- moderately above break-even, still far from the
    # near-certain range where payouts vanish.
    PRICING_SWEET_SPOT = 0.80
    candidates.sort(key=lambda c: abs(c.mc.probability - PRICING_SWEET_SPOT))
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
