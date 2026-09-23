"""
Online probability calibration (spec section 24).

Tracks predicted-probability vs actual-outcome by bucket (e.g. "0.70-0.75")
and nudges raw Monte Carlo probabilities toward the empirically observed
win rate for that bucket. Deliberately conservative: with few observations
in a bucket, the calibrated probability stays close to the raw estimate
(shrinkage toward the model, not toward an arbitrary prior) and only pulls
harder toward the observed rate as evidence accumulates.
"""
from __future__ import annotations

from typing import Dict, Tuple


def bucket_for(probability: float, width: float = 0.05) -> str:
    probability = min(max(probability, 0.0), 0.999999)
    lo = int(probability / width) * width
    hi = lo + width
    return f"{lo:.2f}-{hi:.2f}"


class CalibrationTracker:
    def __init__(self, storage, min_observations_for_full_trust: int = 40):
        self.storage = storage
        self.min_observations_for_full_trust = min_observations_for_full_trust

    def calibrate(self, raw_probability: float) -> float:
        buckets = self.storage.get_calibration()
        key = bucket_for(raw_probability)
        n, wins = buckets.get(key, (0, 0))
        if n == 0:
            return raw_probability
        observed_rate = wins / n
        weight = min(1.0, n / self.min_observations_for_full_trust)
        return (1 - weight) * raw_probability + weight * observed_rate

    def record_outcome(self, raw_probability: float, won: bool) -> None:
        self.storage.update_calibration(bucket_for(raw_probability), won)
