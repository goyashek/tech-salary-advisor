from src.train import select_best


def test_select_best_uses_cross_validation_score():
    results = {
        "better_test_score": {"cv_r2": 0.80, "test_r2": 0.95},
        "better_cv_score": {"cv_r2": 0.90, "test_r2": 0.85},
    }
    assert select_best(results) == "better_cv_score"
