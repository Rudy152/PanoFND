from __future__ import annotations

import unittest

from panoagent.calibration import apply_calibration, fit_calibration


def row(identifier, gold, predicted, confidence, modality="text"):
    return {
        "id": identifier,
        "gold": gold,
        "prediction": predicted,
        "modality_key": modality,
        "stages": {
            "judge": {"data": {"binary_label": predicted, "confidence": confidence}}
        },
    }


class CalibrationTests(unittest.TestCase):
    def test_fit_and_apply_uses_validation_confidence_only(self) -> None:
        validation = [
            row("1", "REAL", "REAL", 0.9),
            row("2", "FAKE", "REAL", 0.55),
            row("3", "FAKE", "FAKE", 0.8),
            row("4", "REAL", "REAL", 0.7),
        ]
        calibration = fit_calibration(validation, min_fake_recall=1.0)
        self.assertLessEqual(calibration["global"]["threshold"], 0.45)
        calibrated = apply_calibration(validation, calibration)
        self.assertEqual(calibrated[1]["prediction"], "FAKE")
        self.assertEqual(calibrated[1]["calibration"]["uncalibrated_prediction"], "REAL")

    def test_rows_without_confidence_are_left_unchanged(self) -> None:
        calibration = {"global": {"threshold": 0.4}, "by_modality": {}}
        rows = [{"id": "1", "gold": "REAL", "prediction": "REAL"}]
        self.assertEqual(apply_calibration(rows, calibration), rows)


if __name__ == "__main__":
    unittest.main()
