"""
Per-symbol decision cycle: candles -> features/regime -> candidate search ->
live proposals for the top candidates -> accept/reject -> (if shadow) log
only, (if live) return the winning candidate for execution.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.deriv.client import DerivClient
from app.features.stats import detect_regime, log_returns, realized_volatility
from app.models.calibration import CalibrationTracker
from app.optimizer.candidate import BarrierCandidate, build_candidates
from app.strategy.filters import evaluate


@dataclass
class ScanOutcome:
    symbol: str
    traded: bool
    trade_row: Optional[dict]
    rejections: List[dict]


TRADE_THRESHOLD_PROBABILITY = 0.71  # ~ break-even prob at 1.40x, used to steer adaptive path counts

MAX_CANDIDATES_TO_PRICE = 6  # cap live proposal requests per scan to bound Deriv round-trips


def build_return_pools(closes: np.ndarray, durations_minutes: List[int]) -> dict:
    """closes: 1-minute close series. Returns per-duration return pools built
    by summing consecutive 1-minute log returns into duration-length blocks,
    which is what feeds the Monte Carlo step simulation for that horizon."""
    rets = log_returns(closes)
    pools = {}
    for d in durations_minutes:
        if len(rets) < d + 5:
            continue
        pools[d] = rets  # per-step (1-minute) pool; MC engine simulates `d` steps forward
    return pools


async def run_scan_cycle(
    symbol: str,
    closes: np.ndarray,
    current_price: float,
    client: DerivClient,
    calibration: CalibrationTracker,
    staking,
    cfg,
    stake_multiplier: float,
    extra_edge_requirement: float,
    logger,
) -> ScanOutcome:
    if len(closes) < 70:
        logger.info(f"{symbol}: warming up, insufficient history ({len(closes)} candles)")
        return ScanOutcome(symbol, False, None, [])

    rets = log_returns(closes)
    current_vol = realized_volatility(rets[-20:]) if len(rets) >= 20 else realized_volatility(rets)
    regime = detect_regime(closes)

    pools = build_return_pools(closes, cfg.durations_minutes)
    if not pools:
        return ScanOutcome(symbol, False, None, [])

    candidates = build_candidates(
        current_price, pools, current_vol, list(pools.keys()), cfg.barrier_vol_multiples,
        TRADE_THRESHOLD_PROBABILITY, cfg.monte_carlo,
        regime_name=regime.regime.value, regime_confidence=regime.confidence,
        calm_regimes=tuple(cfg.calm_regimes), calm_regime_confidence_floor=cfg.calm_regime_confidence_floor,
        non_calm_max_duration_minutes=cfg.non_calm_max_duration_minutes,
    )
    if not candidates:
        return ScanOutcome(symbol, False, None, [])

    top = candidates[:MAX_CANDIDATES_TO_PRICE]

    rejections = []
    best_accept = None
    best_row = None

    for cand in top:
        calibrated = calibration.calibrate(cand.mc.probability)
        stake = staking.stake_for(edge=0.0, decision_score=calibrated, stake_multiplier=stake_multiplier)

        # Deriv rejects EXPIRYRANGE barrier offsets past a symbol-specific
        # decimal limit (ContractBuyValidationError) -- confirmed different
        # per symbol in production: R_10 accepted 3 places, 1HZ10V only 2.
        # Rounding to 2 satisfies both observed limits; if a future symbol
        # needs even less precision this will need per-symbol pip_size
        # discovery via active_symbols rather than a single constant.
        lower_offset = round(cand.lower_barrier - current_price, 2)
        upper_offset = round(cand.upper_barrier - current_price, 2)

        # A tight candidate (small volatility multiple on a low-volatility
        # symbol) can round to a degenerate range at 3-decimal precision --
        # zero width, or even inverted -- which is what Deriv's "This
        # contract offers no return" rejection was: not a flaky error, but a
        # real candidate that stopped being a valid range once rounded to
        # the precision Deriv actually accepts. Skip it before spending a
        # rate-limited request on something that cannot price.
        if upper_offset <= lower_offset or lower_offset >= 0 or upper_offset <= 0:
            continue

        try:
            proposal = await client.request_proposal(
                symbol=symbol, duration_minutes=cand.duration_minutes, stake=stake,
                lower_barrier=lower_offset,
                upper_barrier=upper_offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{symbol}: proposal request failed for candidate ({exc!r}) -- skipping candidate")
            continue

        payout = float(proposal.get("payout", 0.0))
        proposal_id = proposal.get("id")
        if payout <= 0 or not proposal_id:
            continue

        decision = evaluate(
            calibrated_probability=calibrated,
            probability_uncertainty=cand.mc.probability_uncertainty,
            payout=payout,
            stake=stake,
            min_payout_multiplier=cfg.min_payout_multiplier,
            min_edge=cfg.min_edge,
            min_ev=cfg.min_ev,
            max_probability_uncertainty=cfg.max_probability_uncertainty,
            extra_edge_requirement=extra_edge_requirement,
            model_disagreement=cand.mc.model_disagreement,
            max_model_disagreement=cfg.max_model_disagreement,
            regime_confidence=regime.confidence,
            min_regime_confidence=cfg.min_regime_confidence_to_trade,
            duration_minutes=cand.duration_minutes,
            edge_duration_scaling=cfg.edge_duration_scaling,
        )

        row = {
            "trade_id": str(uuid.uuid4()),
            "symbol": symbol,
            "entry_price": current_price,
            "duration_minutes": cand.duration_minutes,
            "lower_barrier": cand.lower_barrier,
            "upper_barrier": cand.upper_barrier,
            "stake": stake,
            "proposal_id": proposal_id,
            "payout": payout,
            "payout_multiplier": decision.payout_multiplier,
            "raw_probability": cand.mc.probability,
            "calibrated_probability": calibrated,
            "implied_probability": decision.implied_probability,
            "edge": decision.edge,
            "expected_value": decision.expected_value,
            "regime": regime.regime.value,
            "regime_confidence": regime.confidence,
            "volatility": current_vol,
            "model_disagreement": cand.mc.model_disagreement,
            "mc_path_count": cand.mc.path_count,
            "decision_score": decision.expected_value,
            "model_version": "v1",
        }

        logger.info(
            f"{symbol} | dur={cand.duration_minutes}m | barriers=[{cand.lower_barrier:.4f},{cand.upper_barrier:.4f}] "
            f"| regime={regime.regime.value}({regime.confidence:.2f}) | vol={current_vol:.5f} | "
            f"raw_p={cand.mc.probability:.3f} cal_p={calibrated:.3f} implied_p={decision.implied_probability:.3f} "
            f"edge={decision.edge:+.3f} payout_x={decision.payout_multiplier:.3f} ev={decision.expected_value:+.4f} "
            f"-> {decision.reason}"
        )

        if decision.accept and (best_accept is None or decision.expected_value > best_accept.expected_value):
            best_accept = decision
            best_row = row
        elif not decision.accept:
            rejections.append({**row, "rejection_reason": decision.reason})

    if best_accept is not None:
        return ScanOutcome(symbol, True, best_row, rejections)

    return ScanOutcome(symbol, False, None, rejections)
