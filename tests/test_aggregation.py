from __future__ import annotations

import unittest

from panoagent.aggregation import deterministic_verdict


def report(**updates):
    signals = {
        "stage_valid": True,
        "claim_supported": True,
        "claim_contradicted": False,
        "identity_mismatch": False,
        "attribution_mismatch": False,
        "temporal_mismatch": False,
        "jurisdiction_mismatch": False,
        "location_mismatch": False,
        "numeric_mismatch": False,
        "manipulated_overlay": False,
        "materiality": "none",
        "confidence": 0.8,
    }
    signals.update(updates)
    return {"signals": signals}


class AggregationTests(unittest.TestCase):
    def test_central_identity_mismatch_is_fake(self):
        verdict = deterministic_verdict(
            report(identity_mismatch=True, materiality="central", confidence=0.9)
        )
        self.assertEqual(verdict["binary_label"], "FAKE")

    def test_supported_without_positive_contradiction_is_real(self):
        self.assertEqual(deterministic_verdict(report())["binary_label"], "REAL")

    def test_central_numeric_mismatch_is_fake(self):
        verdict = deterministic_verdict(
            report(numeric_mismatch=True, materiality="central", confidence=0.9)
        )
        self.assertEqual(verdict["binary_label"], "FAKE")

    def test_ambiguous_signals_fail_closed(self):
        verdict = deterministic_verdict(
            report(claim_supported="unknown", identity_mismatch="unknown", confidence=0.4)
        )
        self.assertEqual(verdict["binary_label"], "ERROR")


if __name__ == "__main__":
    unittest.main()
