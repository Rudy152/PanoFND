import unittest

from panoagent.routing import apply_hybrid_routing, select_hybrid_routing


def row(sample_id, profile, gold, prediction, modality, confidence, tokens):
    return {
        "id": sample_id,
        "gold": gold,
        "prediction": prediction,
        "modality_key": modality,
        "stages": {"judge": {"data": {"binary_label": prediction, "confidence": confidence}}},
        "model_metadata": {"usage": {"total_tokens": tokens}},
        "profile": profile,
    }


class RoutingTests(unittest.TestCase):
    def test_apply_routes_audio_to_full(self):
        fused = [row("1", "fused", "REAL", "FAKE", "audio", 0.9, 10)]
        full = [row("1", "full", "REAL", "REAL", "audio", 0.8, 100)]
        rule = {"kind": "contains_any", "modalities": ["audio"], "fallback_on_error": True}
        routed = apply_hybrid_routing(fused, full, rule)
        self.assertEqual(routed[0]["prediction"], "REAL")
        self.assertEqual(routed[0]["routing"]["profile"], "full")
        self.assertEqual(routed[0]["routing"]["executed_profiles"], ["full"])
        self.assertEqual(routed[0]["model_metadata"]["usage"]["total_tokens"], 100)

    def test_confidence_fallback_counts_both_profile_costs(self):
        fused = [row("1", "fused", "REAL", "FAKE", "text", 0.2, 10)]
        full = [row("1", "full", "REAL", "REAL", "text", 0.8, 100)]
        rule = {
            "kind": "low_confidence", "confidence_threshold": 0.5,
            "fallback_on_error": True,
        }
        routed = apply_hybrid_routing(fused, full, rule)
        self.assertEqual(routed[0]["routing"]["executed_profiles"], ["fused", "full"])
        self.assertEqual(routed[0]["model_metadata"]["usage"]["total_tokens"], 110)

    def test_selection_respects_token_reduction_constraint(self):
        fused = [
            row("1", "fused", "REAL", "FAKE", "audio", 0.9, 10),
            row("2", "fused", "FAKE", "FAKE", "text", 0.9, 10),
        ]
        full = [
            row("1", "full", "REAL", "REAL", "audio", 0.8, 100),
            row("2", "full", "FAKE", "FAKE", "text", 0.8, 100),
        ]
        result = select_hybrid_routing(fused, full, min_token_reduction=0.4)
        selected = result["selected"]
        self.assertGreaterEqual(selected["token_reduction_vs_full"], 0.4)
        self.assertEqual(selected["metrics"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
