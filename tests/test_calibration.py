import numpy as np

from churn_risk_service.calibration import COST_FN, COST_FP, expected_cost, run


def test_expected_cost_matches_hand_calc():
    y_true = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    # threshold 0.5 -> pred [1,1,0,0]: tp=1, fp=1, fn=1
    cost = expected_cost(y_true, proba, 0.5)
    assert cost == COST_FP * 2 + COST_FN * 1


def test_cost_threshold_beats_default_and_favors_recall():
    s = run(write=False)
    # a cost-aware threshold is no worse than the arbitrary 0.5
    assert s["cost_at_best"] <= s["cost_at_0.5"]
    # 10x FN cost pushes the threshold below 0.5 and recall high
    assert s["best_threshold"] < 0.5
    assert s["recall_at_best"] > 0.85
