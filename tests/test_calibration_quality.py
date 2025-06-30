from src.us_congressional_districts.calibrate import calibrate
import pandas as pd


def test_calibration_quality() -> None:
    """
    Test the quality of the calibration process by checking the calibration log.
    """
    calibration_log = pd.read_csv("calibration_log.csv")
    latest_epoch = calibration_log[
        calibration_log["epoch"] == calibration_log["epoch"].max()
    ]

    excellent = (latest_epoch["rel_abs_error"] < 0.05).sum()
    good = (
        (latest_epoch["rel_abs_error"] >= 0.05)
        & (latest_epoch["rel_abs_error"] < 0.20)
    ).sum()
    quality_score = (excellent * 100 + good * 75) / len(latest_epoch)

    assert (
        quality_score >= 60
    ), f"Calibration quality score is {quality_score:.2f}, expected at least 60.0"
