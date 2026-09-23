from app.strategy.filters import evaluate, expected_value, implied_probability, payout_multiplier


def test_payout_multiplier():
    assert payout_multiplier(0.49, 0.35) - 1.4 < 1e-9


def test_implied_probability_matches_break_even_math():
    p = implied_probability(1.40)
    assert abs(p - (1 / 1.40)) < 1e-9


def test_ev_positive_when_model_beats_implied():
    ev = expected_value(calibrated_probability=0.80, payout_multiplier_value=1.40)
    assert ev > 0


def test_rejects_low_payout():
    d = evaluate(
        calibrated_probability=0.9, probability_uncertainty=0.01, payout=0.40, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert not d.accept
    assert "PAYOUT" in d.reason


def test_rejects_insufficient_edge():
    # payout implies ~71.4% breakeven; model only slightly above -> reject
    d = evaluate(
        calibrated_probability=0.72, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert not d.accept
    assert "EDGE" in d.reason


def test_accepts_strong_edge_and_payout():
    d = evaluate(
        calibrated_probability=0.85, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert d.accept


def test_caution_penalty_raises_required_edge():
    kwargs = dict(
        calibrated_probability=0.78, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    normal = evaluate(**kwargs, extra_edge_requirement=0.0)
    cautious = evaluate(**kwargs, extra_edge_requirement=0.10)
    assert normal.accept
    assert not cautious.accept
