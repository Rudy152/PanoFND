"""Regression tests for the changes made after the 2026-11-16 methodology review."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from panoagent import prompts
from panoagent.agents.base import JsonAgent
from panoagent.calibration import fake_score
from panoagent.direct import run_direct_sample
from panoagent.media import visual_manifest
from panoagent.model_clients import MockBackend
from panoagent.pipeline import PanoAgentPipeline
from panoagent.retrieval import FACT_CHECK_DOMAINS, TavilySearchProvider, make_search_provider
from panoagent.schemas import ModelRequest, NewsSample
from panoagent.slot_diff import numeric_diffs


class Capturing(MockBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


def item(**overrides) -> NewsSample:
    base = NewsSample(
        id="9100", modalities=("text", "image", "video"), title="Storm hits coast",
        text="A storm hit the coast on Monday.", image_paths=(), video_paths=(), audio_paths=(),
        label="true", source="https://example.com/a", publish_time="2024/05/06",
        description="annotation that must never be shown",
    )
    return replace(base, **overrides)


class JsonResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


class VisualManifestTests(unittest.TestCase):
    def test_manifest_separates_news_images_from_video_frames(self) -> None:
        text = visual_manifest(2, 8, True)
        self.assertIn("images 1-2: NEWS IMAGES", text)
        self.assertIn("images 3-10: VIDEO FRAMES", text)
        self.assertIn("could not be decoded", visual_manifest(1, 0, True))
        self.assertEqual(visual_manifest(0, 0, False), "")

    def test_agents_see_manifest_but_direct_prompt_is_unchanged(self) -> None:
        sample = item(visual_manifest=visual_manifest(1, 4, True))
        backend = Capturing()
        with tempfile.TemporaryDirectory() as directory:
            PanoAgentPipeline(backend, cache_root=directory, profile="fused").run(sample)
        fused = next(r for r in backend.requests if r.task_name == "fused_investigator")
        self.assertIn("VISUAL INPUT ORDER", fused.user_prompt)
        direct_backend = Capturing()
        run_direct_sample(sample, direct_backend, max_new_tokens=4)
        self.assertNotIn("VISUAL INPUT ORDER", direct_backend.requests[0].user_prompt)


class DecisionRuleTests(unittest.TestCase):
    def test_external_evidence_cannot_vouch_for_non_text_modalities(self) -> None:
        for template in (prompts.JUDGE_USER, prompts.JUDGE_STRUCTURED_USER):
            self.assertIn("NOT evidence that an image, audio, or video is", template)
            self.assertNotIn("support REAL", template)

    def test_fused_template_has_no_reassuring_prefill(self) -> None:
        self.assertNotIn('"claim_supported": true', prompts.FUSED_USER)
        self.assertNotIn('"temporal_mismatch": false', prompts.FUSED_USER)

    def test_no_sample_specific_patches_in_agent_prompts(self) -> None:
        agent_prompts = "".join(
            value for name, value in vars(prompts).items()
            if isinstance(value, str) and not name.startswith("DIRECT")
        )
        for patch in ("Belter", "Khamenei", "Wilders", "192%", "199%", "license revoked"):
            self.assertNotIn(patch, agent_prompts)

    def test_image_policy_reaches_fused_judge(self) -> None:
        backend = Capturing()
        with tempfile.TemporaryDirectory() as directory:
            PanoAgentPipeline(backend, cache_root=directory, profile="fused").run(
                item(modalities=("image",), title="", text="")
            )
        judge = next(r for r in backend.requests if r.task_name == "judge")
        self.assertIn("IMAGE_ONLY_POLICY", judge.user_prompt)


class FusedNormalizationTests(unittest.TestCase):
    def test_free_form_signal_becomes_unknown_instead_of_error(self) -> None:
        class FreeForm(MockBackend):
            def generate(self, request):
                raw = super().generate(request)
                if request.task_name == "fused_investigator":
                    data = json.loads(raw)
                    data["signals"]["numeric_mismatch"] = "partially"
                    return json.dumps(data)
                return raw

        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(FreeForm(), cache_root=directory, profile="fused").run(item())
        fused = record.stages["fused_investigator"]
        self.assertIsNone(fused["error"])
        self.assertEqual(fused["data"]["signals"]["numeric_mismatch"], "unknown")


class RetrievalProtocolTests(unittest.TestCase):
    def test_search_is_bounded_by_publication_date(self) -> None:
        bodies = []

        def fake_urlopen(request, timeout):
            bodies.append(json.loads(request.data.decode("utf-8")))
            return JsonResponse({"results": [
                {"url": "https://a.example/before", "content": "x" * 900, "published_date": "2024-05-05"},
                {"url": "https://a.example/after", "content": "y" * 900, "published_date": "2024-06-01"},
                {"url": "https://a.example/undated", "content": "z" * 900},
            ]})

        provider = TavilySearchProvider(api_key="tvly-test", urlopen_fn=fake_urlopen, fetch_fn=None)
        report = provider.search_with_report(["q"], end_date="2024-05-07")
        self.assertEqual(bodies[0]["end_date"], "2024-05-07")
        self.assertEqual(
            [doc.url for doc in report.documents],
            ["https://a.example/before", "https://a.example/undated"],
        )
        self.assertEqual(report.dropped_after_end_date, 1)
        self.assertEqual(report.undated_documents, 1)

    def test_fact_check_sites_are_excluded_by_default(self) -> None:
        provider = TavilySearchProvider(api_key="tvly-test", urlopen_fn=lambda r, timeout: None)
        self.assertIn("snopes.com", provider.exclude_domains)
        os.environ.setdefault("TAVILY_API_KEY", "tvly-test")
        self.assertTrue(set(FACT_CHECK_DOMAINS) <= set(make_search_provider("tavily").exclude_domains))


class GuardAndCalibrationTests(unittest.TestCase):
    def test_slot_guard_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(MockBackend(), cache_root=directory)
        self.assertFalse(pipeline.external_slot_guard)

    def test_calibration_score_follows_final_prediction(self) -> None:
        overridden = {
            "prediction": "FAKE",
            "stages": {
                "judge": {"data": {"binary_label": "REAL", "confidence": 0.9}},
                "external_slot_guard": {"data": {"final_label": "FAKE"}},
            },
        }
        self.assertEqual(fake_score(overridden), 1.0)
        with_probability = {
            "prediction": "REAL",
            "stages": {"judge": {"data": {"binary_label": "REAL", "confidence": 0.9, "fake_probability": 0.37}}},
        }
        self.assertAlmostEqual(fake_score(with_probability), 0.37)
        self.assertIsNone(fake_score({"prediction": "ERROR", "stages": {}}))

    def test_numeric_diff_handles_unpunctuated_text(self) -> None:
        words = ("the regional office said that " * 6).split()
        news = " ".join(words + "unemployment in the northern district rose by 112 percent last year".split() + words)
        evidence = [{"evidence_id": "E1", "retrieved_text": (
            "Unemployment in the northern district rose by 11.2 percent last year."
        )}]
        diffs = numeric_diffs(news, evidence, min_similarity=0.3, min_shared_words=4)
        self.assertTrue(diffs)


class CacheFingerprintTests(unittest.TestCase):
    def test_in_place_media_change_invalidates_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image_0001_1.jpg"
            image.write_bytes(b"old image bytes")
            request = ModelRequest(task_name="t", system_prompt="s", user_prompt="u", image_paths=(image,))
            before = JsonAgent._request_fingerprint(request)
            time.sleep(0.01)
            image.write_bytes(b"new image bytes!")
            self.assertNotEqual(before, JsonAgent._request_fingerprint(request))


class OutputFormatRobustnessTests(unittest.TestCase):
    """qwen3.8-omni-flash sometimes answers `{}`/`[]` or nests top-level keys one level deep."""

    def _agent(self, directory, replies):
        from panoagent.agents.relation_investigator import RelationInvestigator
        from panoagent.cache import StageCache

        class Scripted(MockBackend):
            def __init__(self):
                self.calls = 0

            def generate(self, request):
                self.calls += 1
                return replies.pop(0) if replies else '{"relations": []}'

        backend = Scripted()
        return RelationInvestigator(backend, StageCache(directory, "mock")), backend

    def test_empty_answer_is_resent_not_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, backend = self._agent(directory, ["{}", '{"relations": [{"pair": "text-image"}]}'])
            request = ModelRequest(task_name="relation_investigator", system_prompt="s", user_prompt="u")
            result = agent.run_request(item(), request)
        self.assertIsNone(result.error)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(result.data["relations"], [{"pair": "text-image"}])

    def test_cached_empty_answer_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, backend = self._agent(directory, ["[]", "[]"])
            request = ModelRequest(task_name="relation_investigator", system_prompt="s", user_prompt="u")
            self.assertIsNotNone(agent.run_request(item(), request).error)
            second = agent.run_request(item(), request)
        self.assertIsNone(second.error)
        self.assertFalse(second.cached)

    def test_keys_nested_one_level_are_hoisted_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, _ = self._agent(directory, [])
        data = agent._parse_and_validate('{"wrapper": {"relations": [{"pair": "x"}]}}')
        self.assertEqual(data["relations"], [{"pair": "x"}])
        self.assertNotIn("relations", data["wrapper"])


class JudgeMediaTests(unittest.TestCase):
    def test_judge_media_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image_9100_1.jpg"
            image.write_bytes(b"jpeg")
            sample = item(image_paths=(image,), visual_manifest=visual_manifest(1, 0, False))
            for enabled in (False, True):
                backend = Capturing()
                PanoAgentPipeline(
                    backend, cache_root=Path(directory) / str(enabled), profile="fused",
                    judge_media=enabled,
                ).run(sample)
                judge = next(r for r in backend.requests if r.task_name == "judge")
                self.assertEqual(bool(judge.image_paths), enabled)
                self.assertEqual("ATTACHED MEDIA" in judge.user_prompt, enabled)


if __name__ == "__main__":
    unittest.main()
