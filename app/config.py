"""
Central configuration. Everything secret or environment-specific comes from
env vars so the same image runs unmodified on Railway.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [s.strip() for s in val.split(",") if s.strip()]


@dataclass
class StakingConfig:
    mode: str = "fixed"
    minimum_stake: float = 0.35
    initial_stake: float = 0.35
    maximum_stake: float = 0.35


@dataclass
class MonteCarloConfig:
    minimum_paths: int = 10_000
    default_paths: int = 25_000
    maximum_paths: int = 100_000
    # widen simulation count as a candidate's EV sits closer to the trade
    # threshold; cheap candidates get the minimum.
    near_threshold_band: float = 0.03


@dataclass
class Config:
    # --- Deriv connection ---
    deriv_api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    deriv_app_id: str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    deriv_endpoint: str = field(
        default_factory=lambda: os.getenv("DERIV_WS_URL", "wss://ws.derivws.com/websockets/v3")
    )
    # DEMO is the only default that is safe; LIVE must be requested explicitly.
    account_mode: str = field(default_factory=lambda: os.getenv("DERIV_ACCOUNT_MODE", "DEMO").upper())

    # --- persistence ---
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "data/expiryrange.db"))

    # --- logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- trading universe ---
    symbols: List[str] = field(
        default_factory=lambda: _env_list(
            "SYMBOLS",
            ["R_10"],
        )
    )

    # --- strategy thresholds ---
    min_payout_multiplier: float = float(os.getenv("MIN_PAYOUT_MULTIPLIER", "1.40"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.05"))
    min_ev: float = float(os.getenv("MIN_EV", "0.0"))
    max_probability_uncertainty: float = float(os.getenv("MAX_PROB_UNCERTAINTY", "0.06"))

    durations_minutes: List[int] = field(default_factory=lambda: list(range(2, 11)))
    barrier_vol_multiples: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    )

    # --- risk ---
    max_concurrent_contracts: int = int(os.getenv("MAX_CONCURRENT_CONTRACTS", "3"))
    max_concurrent_per_symbol: int = 1
    max_daily_exposure: float = float(os.getenv("MAX_DAILY_EXPOSURE", "50.0"))
    caution_cooldown_seconds: int = int(os.getenv("CAUTION_COOLDOWN_SECONDS", "120"))
    consecutive_loss_caution_threshold: int = int(os.getenv("CONSECUTIVE_LOSS_CAUTION_THRESHOLD", "4"))

    # --- runtime ---
    shadow_mode: bool = field(default_factory=lambda: _env_bool("SHADOW_MODE", True))
    scan_interval_seconds: float = float(os.getenv("SCAN_INTERVAL_SECONDS", "15"))
    health_port: int = int(os.getenv("PORT", "8080"))

    staking: StakingConfig = field(default_factory=StakingConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)

    def is_live(self) -> bool:
        return self.account_mode == "LIVE"

    def validate(self) -> List[str]:
        problems = []
        if self.is_live() and not self.deriv_api_token:
            problems.append("LIVE mode requested but DERIV_API_TOKEN is not set.")
        if self.staking.initial_stake < self.staking.minimum_stake:
            problems.append("initial_stake below minimum_stake.")
        if self.min_payout_multiplier < 1.0:
            problems.append("min_payout_multiplier must be >= 1.0.")
        return problems


def load_config() -> Config:
    return Config()
