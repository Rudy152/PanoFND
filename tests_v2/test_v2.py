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


    def test_ablation_refuses_holdout_and_pairs_correctly(self):
        from panoagent.dataset import load_split
        from panoagent_v2.ablation import ROOT, Evaluation, check_tuning_only

        check_tuning_only(load_split(ROOT / "splits" / "half_eval_seed42.json", "train")[:5])
        with self.assertRaises(SystemExit):
            check_tuning_only(load_split(ROOT / "splits" / "final_holdout_clean_765_20261031.json", "test")[:1])
        gold = ["FAKE", "FAKE", "REAL", "REAL"]
        ev = Evaluation(gold, ["a", "b", "c", "d"], n_boot=50)
        perfect = (np.array([0.9, 0.8, 0.1, 0.2]), np.array(gold))
        worse = (np.array([0.9, 0.1, 0.2, 0.3]), np.array(["FAKE", "REAL", "REAL", "REAL"]))
        self.assertAlmostEqual(ev.paired([perfect], [worse])["ba_diff"], 0.25)
        self.assertAlmostEqual(ev.paired([perfect], [perfect])["ba_diff"], 0.0)


    def test_joint_answer_maps_to_the_three_pairs_or_fails(self):
        from panoagent_v2.ablation_d1 import JOINT_USER, joint_to_calls

        pair = lambda name, c: {"pair": name, "consistency": c, "slot_checks": [], "cues": [], "summary": ""}
        data = {"pairs": [pair("A-B", "consistent"), pair("A-C", "conflict"), pair("B-C", "unrelated")],
                "integrity": {"A": "intact", "B": "uncertain", "C": "manipulated"}}
        mods = ("text", "image", "audio")
        calls = joint_to_calls(data, mods)
        self.assertEqual([c["stage"] for c in calls], ["pair_text_image", "pair_text_audio", "pair_image_audio"])
        f = build_features(mods, {"p_fake": 0.5}, calls, {})
        self.assertAlmostEqual(f["frac_conflict"], 1 / 3)
        self.assertAlmostEqual(f["frac_manipulated"], 1 / 3)  # each source counted twice, as with pair calls
        for bad in ({}, {**data, "pairs": data["pairs"][:2]}, {**data, "integrity": {"A": "intact"}},
                    {"field": "time", "a_value": "x", "b_value": "y", "status": "match"}):
            with self.assertRaises(ValueError):
                joint_to_calls(bad, mods)
        for line in ("compatible facts;", "voice-synthesized", "resolve it by deciding from outside knowledge"):
            self.assertIn(line, JOINT_USER)
            self.assertIn(line, prompts.PAIR_USER)


    def test_table_e_variants_change_only_their_factor(self):
        from panoagent_v2 import ablation_e as e
        from panoagent_v2.agents import _sections

        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image

            img = Path(directory) / "n.jpg"
            Image.new("RGB", (2000, 100)).save(img)
            mp3 = Path(directory) / "a.mp3"
            mp3.write_bytes(b"x")
            s = sample(modalities=("text", "image", "audio"), transcript="word " * 3000)
            item = PreparedItem(s, "TITLE: T\nARTICLE: Body\nPUBLICATION DATE: 2025/01/01", ("text", "image", "audio"), {
                "image": [Attachment("[NEWS IMAGE 1]", "image", img, "i")],
                "audio": [Attachment("[NEWS AUDIO PART 1/1 | 00:00-00:01]", "audio", mp3, "a")]})
            b = e.Builder.__new__(e.Builder)
            b.prep = b.prep16 = type("P", (), {"prepare": staticmethod(lambda _: item)})()
            build = lambda v: b.build(v, s)
            tb, se = _sections(item, item.modalities)
            self.assertEqual(e.to_content(build("R")), build_content(prompts.SINGLE_USER.format(modalities="text+image+audio"), se, tb))
            text = lambda v: "\n".join(p[1] for p in build(v) if p[0] == "text")
            self.assertNotIn("[NEWS IMAGE 1]", text("L").replace("[NEWS TEXT]", ""))
            self.assertEqual([p for p in build("L") if p[0] != "text"], [p for p in build("R") if p[0] != "text"])
            self.assertFalse(any(p[0] == "audio" for p in build("A1")))
            self.assertIn("TRUNCATED", text("A1"))
            self.assertNotIn("TRUNCATED", text("A2"))
            self.assertIn("2025/01/01", text("R"))
            self.assertNotIn("2025/01/01", text("D"))
            self.assertTrue(any(p[0] == "image" and p[2] == "raw" for p in build("I")))
            self.assertIn("DIRECT_DETECTION", text("P"))
            for v in ("R", "L", "A1", "A2", "I", "D", "P"):
                self.assertNotIn("LEAK", text(v))
                self.assertNotIn("x.test", text(v))

    def test_rerun_band_grows_with_noise_and_is_zero_without_it(self):
        from panoagent_v2.ablation_e import rerun_band

        rng = np.random.default_rng(0)
        y = np.array([0, 1] * 60)
        p = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0.01, 0.99)
        keys = ["a"] * 120
        band = lambda sd: rerun_band(y, p, keys, {"a": rng.normal(0, sd, 200)}, draws=200)
        self.assertEqual(rerun_band(y, p, keys, {"a": np.zeros(5)}, draws=5), 0.0)
        self.assertLess(band(0.2), band(2.0))


if __name__ == "__main__":
    unittest.main()
