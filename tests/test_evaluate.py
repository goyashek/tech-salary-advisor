import numpy as np

from src.evaluate import conformal_interval, conformal_quantile, interval_report


def test_split_conformal_interval_reports_overall_and_segments():
    actual = np.array([100, 200, 300, 400])
    predictions = np.array([90, 190, 280, 430])
    width = conformal_quantile(actual, predictions, level=0.75)
    lower, upper = conformal_interval(predictions, width)
    report = interval_report(
        actual,
        lower,
        upper,
        {"experience": np.array(["junior", "junior", "senior", "senior"])},
    )

    assert width == 30
    assert report["coverage"] == 1.0
    assert report["mean_width"] == 60.0
    assert report["segments"]["experience"]["junior"]["rows"] == 2


def test_conformal_quantile_rejects_invalid_calibration():
    try:
        conformal_quantile([], [], level=0.9)
    except ValueError as exc:
        assert "finite and non-empty" in str(exc)
    else:
        raise AssertionError("empty calibration data should be rejected")
