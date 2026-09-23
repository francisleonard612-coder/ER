import numpy as np

from app.models.monte_carlo import estimate_probability_in_range, adaptive_path_count


def test_probability_bounded_zero_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.002, size=500)
    result = estimate_probability_in_range(
        current_price=100.0, returns_pool=returns, steps=5,
        lower_barrier=95.0, upper_barrier=105.0, current_volatility=0.002,
        n_paths=2000, seed=42,
    )
    assert 0.0 <= result.probability <= 1.0
    assert result.path_count > 0


def test_wider_barriers_increase_probability():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.002, size=500)
    narrow = estimate_probability_in_range(
        100.0, returns, 5, 99.5, 100.5, 0.002, n_paths=4000, seed=1,
    )
    wide = estimate_probability_in_range(
        100.0, returns, 5, 90.0, 110.0, 0.002, n_paths=4000, seed=1,
    )
    assert wide.probability >= narrow.probability


def test_insufficient_history_returns_uncertain_default():
    result = estimate_probability_in_range(100.0, np.array([]), 5, 95, 105, 0.01, n_paths=1000)
    assert result.probability == 0.5
    assert result.probability_uncertainty == 0.5


def test_adaptive_path_count_scales_with_distance_to_threshold():
    near = adaptive_path_count(0.705, 0.71, minimum=1000, default=5000, maximum=20000, band=0.03)
    far = adaptive_path_count(0.20, 0.71, minimum=1000, default=5000, maximum=20000, band=0.03)
    assert near >= far
    assert far == 1000
