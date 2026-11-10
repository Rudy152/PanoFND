from __future__ import annotations

import unittest

from panoagent.evaluation import evaluate_records


class EvaluationTests(unittest.TestCase):
    def test_metrics(self) -> None:
        records = [
            {"gold": "REAL", "prediction": "REAL", "modality_key": "text"},
            {"gold": "FAKE", "prediction": "FAKE", "modality_key": "text"},
            {"gold": "FAKE", "prediction": "REAL", "modality_key": "image"},
        ]
        metrics = evaluate_records(records)
        self.assertAlmostEqual(metrics["overall"]["accuracy"], 2 / 3)
        self.assertEqual(metrics["overall"]["evaluated"], 3)
        confusion = metrics["overall"]["confusion_matrix"]
        self.assertEqual(confusion["fake_as_real"], 1)
        self.assertEqual(confusion["real_as_fake"], 0)
        self.assertEqual(metrics["overall"]["majority_baseline"]["label"], "FAKE")
        self.assertAlmostEqual(
            metrics["overall"]["majority_baseline"]["accuracy"], 2 / 3
        )

    def test_latest_retry_replaces_error_record(self) -> None:
        records = [
            {"id": "1", "gold": "REAL", "prediction": "ERROR", "modality_key": "text", "error": "temporary"},
            {"id": "1", "gold": "REAL", "prediction": "REAL", "modality_key": "text", "error": None},
        ]
        metrics = evaluate_records(records)
        self.assertEqual(metrics["overall"]["evaluated"], 1)
        self.assertEqual(metrics["overall"]["parse_or_runtime_errors"], 0)
        self.assertEqual(metrics["overall"]["accuracy"], 1.0)

    def test_stage_errors_are_reported_separately(self) -> None:
        records = [{
            "id": "1", "gold": "REAL", "prediction": "REAL", "modality_key": "text",
            "stages": {
                "evidence_organizer": {"error": "bad schema"},
                "judge": {"error": None},
            },
        }]
        metrics = evaluate_records(records)
        self.assertEqual(metrics["overall"]["parse_or_runtime_errors"], 0)
        self.assertEqual(metrics["overall"]["records_with_stage_errors"], 1)
        self.assertEqual(metrics["overall"]["stage_complete_records"], 0)
        self.assertEqual(metrics["overall"]["stage_error_counts"], {"evidence_organizer": 1})
        self.assertEqual(
            metrics["overall"]["strict_end_to_end"]["accuracy_errors_as_wrong"],
            0.0,
        )
        self.assertEqual(metrics["overall"]["stage_complete_subset"]["evaluated"], 0)

    def test_invalid_prediction_reduces_coverage_and_end_to_end_accuracy(self) -> None:
        records = [
            {"id": "1", "gold": "REAL", "prediction": "REAL", "modality_key": "text"},
            {"id": "2", "gold": "FAKE", "prediction": "ERROR", "modality_key": "text"},
        ]
        metrics = evaluate_records(records)["overall"]
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["end_to_end_accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["fake_errors"], 1)
        self.assertEqual(metrics["majority_baseline"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
