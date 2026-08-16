from src.train import select_best


def test_select_best_uses_cross_validation_score():
    results = {
        "better_test_score": {"cv_r2": 0.80, "test_r2": 0.95},
        "better_cv_score": {"cv_r2": 0.90, "test_r2": 0.85},
    }
    assert select_best(results) == "better_cv_score"


def test_stacking_must_earn_its_complexity():
    results = {
        "catboost": {"cv_r2": 0.8830},
        "mlp": {"cv_r2": 0.8700},
        "stacking": {"cv_r2": 0.8835},
    }

    assert select_best(results, stacking_min_r2_gain=0.002) == "catboost"


def test_stacking_wins_when_gain_is_material():
    results = {
        "catboost": {"cv_r2": 0.8830},
        "stacking": {"cv_r2": 0.8860},
    }

    assert select_best(results, stacking_min_r2_gain=0.002) == "stacking"
