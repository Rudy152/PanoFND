from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from panoagent.schemas import NewsSample
from panoagent_v2.agents import _fingerprint, _normalize_pair, p_fake_from_logprobs
from panoagent_v2.client import build_content
from panoagent_v2.features import FEATURES, build_features, localise
from panoagent_v2.media import Attachment, ContentHasher, PreparedItem
from panoagent_v2.scorer import binary_metrics, cross_validated
import panoagent_v2.prompts as prompts


def sample(**kw) -> NewsSample:
    base = dict(id="9001", modalities=("text", "audio"), title="T", text="Body", image_paths=(),
                video_paths=(), audio_paths=(), label="fake", source="https://x.test/a",
                publish_time="2025/01/01", description="LEAK description")
    base.update(kw)
    return NewsSample(**base)


class V2Tests(unittest.TestCase):
    def test_p_fake_uses_true_false_mass(self):
        top = [{"token": "TRUE", "logprob": np.log(0.2)}, {"token": "FALSE", "logprob": np.log(0.6)},
               {"token": "true", "logprob": np.log(0.2)}]
        self.assertAlmostEqual(p_fake_from_logprobs(top, "FALSE"), 0.6 / 1.0, places=6)
        self.assertEqual(p_fake_from_logprobs([], "TRUE"), 0.0)
        self.assertIsNone(p_fake_from_logprobs([], "maybe"))

    def test_pair_normalisation_has_no_optimistic_defaults(self):
        out = _normalize_pair({"consistency": "looks fine", "a_integrity": "ok"})
        self.assertEqual(out["consistency"], "insufficient")
        self.assertEqual(out["a_integrity"], "uncertain")
        self.assertEqual(out["b_integrity"], "uncertain")

    def test_prompts_have_no_prefilled_values_or_sample_patches(self):
        text = prompts.PAIR_USER + prompts.UNI_USER + prompts.SINGLE_USER
        for forbidden in ("true,", "false,", "199%", "192%", "Boelter", "Khamenei", "license"):
            self.assertNotIn(forbidden, text)

    def test_fingerprint_follows_file_content_not_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.jpg"
            path.write_bytes(b"one")
            hasher = ContentHasher(Path(directory) / "index.json")
            first = hasher.digest(path)
            item = PreparedItem(sample(), "T", ("text", "image"),
                                {"image": [Attachment("[NEWS IMAGE 1]", "image", path, first)]})
            fp1 = _fingerprint("m", "s", "u", item, ("text", "image"), {})
            import os, time
            time.sleep(0.01)
            path.write_bytes(b"two-replaced-in-place")
            os.utime(path, None)
            second = hasher.digest(path)
            self.assertNotEqual(first, second)
            item.attachments["image"][0].digest = second
            self.assertNotEqual(fp1, _fingerprint("m", "s", "u", item, ("text", "image"), {}))

    def test_description_never_reaches_model_text(self):
        from panoagent_v2.media import MediaPreparer

        with tempfile.TemporaryDirectory() as directory:
            item = MediaPreparer(directory).prepare(sample())
        self.assertNotIn("LEAK", item.text)
        self.assertNotIn("x.test", item.text)

    def test_labels_precede_each_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image

            path = Path(directory) / "f.jpg"
            Image.new("RGB", (8, 8)).save(path)
            content = build_content("P", [("VIDEO:", [Attachment("[VIDEO FRAME 1 @ 00:05]", "image", path, "d")])],
                                    [("[NEWS TEXT]", "hello")])
        kinds = [c["type"] for c in content]
        self.assertEqual(kinds, ["text", "text", "text", "text", "image_url"])
        self.assertEqual(content[3]["text"], "[VIDEO FRAME 1 @ 00:05]")

    def test_features_and_localisation(self):
        calls = [{"stage": "pair_text_audio", "modalities": ["text", "audio"], "result": {
            "consistency": "conflict", "a_integrity": "intact", "b_integrity": "manipulated",
            "slot_checks": [{"field": "time", "a_value": "July 4", "b_value": "Dec 25", "status": "mismatch"}],
            "cues": [], "summary": ""}}]
        f = build_features(("text", "audio"), {"p_fake": 0.7}, calls, {"image": [], "video": []})
        self.assertEqual(set(f), set(FEATURES))
        self.assertEqual(f["any_conflict"], 1.0)
        self.assertEqual(f["slot_mismatch"], 1.0)
        self.assertEqual(f["frac_manipulated"], 0.5)
        loc = localise(calls)
        self.assertEqual(loc["manipulated_modalities"], ["audio"])
        self.assertEqual(loc["conflicting_pairs"], ["text-audio"])

    def test_cross_validation_is_grouped_and_out_of_fold(self):
        rng = np.random.default_rng(0)
        rows, groups = [], []
        for g in range(40):
            for _ in range(3):
                fake = rng.random() < 0.5
                features = {name: 0.0 for name in FEATURES}
                features["single_p_fake"] = float(np.clip(rng.normal(0.65 if fake else 0.35, 0.15), 0, 1))
                rows.append({"gold": "FAKE" if fake else "REAL", "features": features})
                groups.append(f"g{g}")
        result = cross_validated(rows, FEATURES, groups)
        self.assertEqual(len(result["predictions"]), len(rows))
        metrics = binary_metrics([r["gold"] for r in rows], result["predictions"])
        self.assertGreater(metrics["balanced_accuracy"], 0.7)


    def test_frozen_scorer_reproduces_and_guards_code(self):
        from panoagent_v2.evaluation import apply_frozen, freeze, mcnemar, verify_frozen

        rng = np.random.default_rng(1)
        rows, groups = [], []
        for g in range(60):
            fake = g % 2 == 0
            features = {name: float(rng.random()) for name in FEATURES}
            features["single_p_fake"] = float(np.clip(rng.normal(0.7 if fake else 0.3, 0.1), 0, 1))
            rows.append({"gold": "FAKE" if fake else "REAL", "features": features})
            groups.append(f"g{g}")
        frozen = freeze(rows, FEATURES, groups)
        prob, pred = apply_frozen(frozen, rows)
        self.assertEqual(len(pred), len(rows))
        self.assertGreater(binary_metrics([r["gold"] for r in rows], pred)["balanced_accuracy"], 0.8)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in prob))
        verify_frozen(frozen)
        frozen["code_hashes"]["prompts.py"] = "0" * 64
        with self.assertRaises(SystemExit):
            verify_frozen(frozen)
        self.assertEqual(mcnemar(["FAKE"] * 3, ["FAKE"] * 3, ["REAL"] * 3)["only_a_correct"], 3)


if __name__ == "__main__":
    unittest.main()
