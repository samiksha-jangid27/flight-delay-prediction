"""Synthetic contracts for the offline rescheduling what-if analysis."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.evaluation.rescheduling import (
    CostModel,
    cost_ratio_sweep,
    net_benefit_curve,
    no_action_cost,
    oracle_cost,
    policy_outcome,
    rescheduling_analysis,
)


def test_cost_model_validates_inputs():
    with pytest.raises(ValueError):
        CostModel(cost_delay=0.0)
    with pytest.raises(ValueError):
        CostModel(recovery=1.5)
    with pytest.raises(ValueError):
        CostModel(cost_action=-1.0)


def test_optimal_threshold_formula():
    # tau* = C_a / (recovery * C_d)
    assert CostModel(cost_delay=1.0, cost_action=0.25, recovery=0.8).optimal_threshold == 0.3125
    # Acting can never help when recovery is zero.
    assert CostModel(cost_action=0.1, recovery=0.0).optimal_threshold is None
    # Boundary above one means acting is never worthwhile.
    assert CostModel(cost_delay=1.0, cost_action=0.9, recovery=0.5).optimal_threshold is None


def test_net_savings_closed_form():
    # Two true delays acted on, one false alarm, one true delay missed.
    y = [1, 1, 0, 1, 0]
    p = [0.9, 0.8, 0.7, 0.1, 0.2]  # threshold 0.5 -> act on first three
    cost = CostModel(cost_delay=10.0, cost_action=2.0, recovery=0.75)
    out = policy_outcome(y, p, 0.5, cost)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 1)
    # tp * (r*C_d - C_a) - fp * C_a = 2*(7.5-2) - 1*2 = 9.0
    assert out["net_savings"] == pytest.approx(9.0)
    # Equivalent to status-quo cost minus policy cost.
    assert out["net_savings"] == pytest.approx(out["no_action_cost"] - out["policy_cost"])


def test_oracle_bounds_every_policy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    cost = CostModel(cost_delay=1.0, cost_action=0.2, recovery=0.9)
    best = no_action_cost(int(y.sum()), cost) - oracle_cost(int(y.sum()), cost)
    for tau in np.linspace(0, 1, 21):
        assert policy_outcome(y, p, tau, cost)["net_savings"] <= best + 1e-9


def test_no_action_and_full_action_endpoints():
    y = [1, 0, 1, 0]
    p = [0.3, 0.6, 0.9, 0.1]
    cost = CostModel()
    # Threshold above every score -> act on none -> zero savings, no cost beyond delays.
    high = policy_outcome(y, p, 1.0, cost)
    assert high["acted"] == 0 and high["net_savings"] == pytest.approx(0.0)
    assert high["policy_cost"] == pytest.approx(high["no_action_cost"])
    # Threshold zero -> act on all.
    low = policy_outcome(y, p, 0.0, cost)
    assert low["acted"] == 4


def test_net_benefit_curve_matches_pointwise():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    cost = CostModel(cost_delay=4.0, cost_action=1.0, recovery=0.8)
    curve = net_benefit_curve(y, p, cost)
    for tau in [0.1, 0.5, 0.9]:
        row = curve.iloc[(curve.threshold - tau).abs().argmin()]
        point = policy_outcome(y, p, float(row.threshold), cost)
        assert point["net_savings_per_flight"] == pytest.approx(row.net_savings_per_flight)


def test_cost_ratio_sweep_threshold_tracks_ratio():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    sweep = cost_ratio_sweep(y, p, recovery=0.5, ratios=[0.1, 0.2, 0.3])
    assert list(sweep.cost_ratio) == [0.1, 0.2, 0.3]
    # optimal threshold = ratio / recovery
    assert sweep.optimal_threshold.tolist() == pytest.approx([0.2, 0.4, 0.6])


def test_rescheduling_analysis_report_shape():
    frame = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 1, 0],
            "y_probability": [0.1, 0.8, 0.4, 0.6, 0.9, 0.3],
        }
    )
    report = rescheduling_analysis(frame, frozen_threshold=0.5)
    assert set(report) >= {
        "assumptions",
        "population",
        "baselines_per_flight",
        "policy_at_optimal_threshold",
        "cost_ratio_sweep",
        "policy_at_frozen_threshold",
    }
    assert report["population"]["n_delayed"] == 3
    assert report["policy_at_frozen_threshold"]["threshold"] == 0.5


def test_rescheduling_analysis_rejects_bad_frame():
    with pytest.raises(ValueError):
        rescheduling_analysis(pd.DataFrame({"y_true": [0, 1]}))
    with pytest.raises(ValueError):
        rescheduling_analysis(
            pd.DataFrame({"y_true": [], "y_probability": []})
        )
