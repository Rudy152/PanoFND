from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from panoagent.direct import run_direct_sample
from panoagent.decision_policy import IMAGE_ONLY_POLICY, decision_policy_for
from panoagent.agents.relation_investigator import RelationInvestigator
from panoagent.model_clients import MockBackend
from panoagent.media import prepare_sample_media
from panoagent.pipeline import PanoAgentPipeline
from panoagent.retrieval import EvidenceDocument, SearchProvider
from panoagent.schemas import NewsSample


class TruncatedRelationBackend(MockBackend):
    def generate(self, request):
        if request.task_name == "relation_investigator":
            return '{"relations": [{"status": "unknown"}'
        return super().generate(request)


class RepairableRelationBackend(MockBackend):
    def __init__(self):
        self.relation_calls = 0

    def generate(self, request):
        if request.task_name == "relation_investigator":
            self.relation_calls += 1
            if self.relation_calls == 1:
                return '{"relations": [{"status": "unknown"}'
            return '{"relations": {"text-image": {"status": "unknown"}}}'
        return super().generate(request)


class CountingContextBackend(MockBackend):
    def __init__(self):
        self.context_calls = 0

    def generate(self, request):
        if request.task_name == "context_verifier":
            self.context_calls += 1
        return super().generate(request)


class ArrayOrganizerBackend(MockBackend):
    def generate(self, request):
        if request.task_name == "evidence_organizer":
            return '[{"entities": ["A"]}, {"text_claims": ["Claim B"]}]'
        return super().generate(request)


class RecordingBackend(MockBackend):
    def __init__(self):
        self.last_request = None

    def generate(self, request):
        self.last_request = request
        return super().generate(request)


class CapturingBackend(MockBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class UsageBackend(MockBackend):
    def request_metadata(self):
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class SparseFusedBackend(MockBackend):
    def generate(self, request):
        if request.task_name == "fused_investigator":
            return """{
              "central_claim": "An event happened.",
              "case_file": {}, "modalities": {}, "relations": {},
              "signals": {"claim_supported": true, "materiality": "none", "confidence": 0.5},
              "retrieval": {"needed": false, "claims": [], "queries": []}
            }"""
        return super().generate(request)


class NestedFusedBackend(MockBackend):
    def generate(self, request):
        if request.task_name == "fused_investigator":
            return """{
              "central_claim": "An event happened.",
              "case_file": {}, "modalities": {}, "relations": {},
              "signals": {
                "claim_supported": {"value": true, "materiality": "central", "confidence": 0.8, "evidence_ids": ["E1"]},
                "claim_contradicted": {"value": "false", "materiality": "none", "confidence": 0.9, "evidence_ids": []}
              },
              "retrieval": {"needed": false, "claims": [], "queries": []}
            }"""
        return super().generate(request)


class SuspiciousSignalBackend(MockBackend):
    def generate(self, request):
        if request.task_name == "fused_investigator":
            return """{
              "central_claim": "An event happened.",
              "case_file": {}, "modalities": {}, "relations": {},
              "signals": {
                "claim_supported": true, "claim_contradicted": false,
                "identity_mismatch": "suspicious", "materiality": "minor",
                "confidence": 0.7
              },
              "retrieval": {"needed": false, "claims": [], "queries": []}
            }"""
        return super().generate(request)


class CountingSearchProvider(SearchProvider):
    def __init__(self):
        self.calls = 0

    def search(self, queries, max_results_per_query=3):
        self.calls += 1
        return [EvidenceDocument(
            evidence_id="E001",
            query="test query",
            title="Test source",
            url="https://example.test/source",
            publisher="example.test",
            publication_date="2026-01-01",
            retrieved_text="A fixed test document.",
            retrieval_timestamp="2026-01-01T00:00:00+00:00",
        )]


def sample() -> NewsSample:
    return NewsSample(
        id="0001", modalities=("text",), title="Ordinary report", text="An event happened.",
        image_paths=(), video_paths=(), audio_paths=(), label="true", source="", publish_time="",
        description="Do not expose this annotation.",
    )


class PipelineTests(unittest.TestCase):
    def test_decision_policy_routes_image_only_samples(self) -> None:
        image_sample = replace(sample(), modalities=("image",))
        multimodal_sample = replace(sample(), modalities=("text", "image"))
        self.assertEqual(decision_policy_for(image_sample), IMAGE_ONLY_POLICY)
        self.assertIsNone(decision_policy_for(multimodal_sample))

    def test_headline_counts_as_text_evidence_for_relation_pairs(self) -> None:
        audio_sample = replace(sample(), modalities=("audio",), text="")
        self.assertEqual(
            RelationInvestigator._available_pairs(audio_sample),
            {"text-audio"},
        )

    def test_image_policy_reaches_final_judge_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            image_sample = replace(sample(), modalities=("image",), text="")
            PanoAgentPipeline(backend, cache_root=directory).run(image_sample)
            judge = next(request for request in backend.requests if request.task_name == "judge")
            self.assertIn("IMAGE_ONLY_POLICY", judge.user_prompt)
            self.assertIn("numbers and units", judge.user_prompt)
            self.assertNotIn("collapse", judge.user_prompt.casefold())
            self.assertNotIn("teetering", judge.user_prompt.casefold())

    def test_full_profile_audits_exact_atomic_slots_without_real_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            multimodal = replace(sample(), modalities=("text", "image"))
            PanoAgentPipeline(backend, cache_root=directory).run(multimodal)
            requests = {request.task_name: request for request in backend.requests}
            self.assertIn("numeric_claims", requests["evidence_organizer"].user_prompt)
            self.assertIn("192% and 199%", requests["relation_investigator"].user_prompt)
            self.assertIn("Treat REAL and FAKE as equally plausible", requests["judge"].user_prompt)
            self.assertNotIn("Return FAKE only when", requests["judge"].user_prompt)

    def test_context_reuses_cached_retrieval_before_network_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = CountingSearchProvider()
            pipeline = PanoAgentPipeline(
                MockBackend(), cache_root=directory, search_provider=search
            )
            pipeline.run(sample())
            second = pipeline.run(sample())
            self.assertEqual(search.calls, 1)
            self.assertTrue(second.stages["context_verifier"]["cached"])

            pipeline.context.max_new_tokens += 1
            third = pipeline.run(sample())
            self.assertEqual(search.calls, 1)
            self.assertFalse(third.stages["context_verifier"]["cached"])

    def test_context_salvages_complete_claims_when_sources_are_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(MockBackend(), cache_root=directory)
            raw = (
                '{"claims":[{"claim":"C","status":"supported",'
                '"evidence_ids":["E001"]}],"status":"mixed","sources":["E001",'
            )
            parsed = pipeline.context._parse_and_validate(raw)
            self.assertEqual(parsed["status"], "mixed")
            self.assertEqual(parsed["sources"], ["E001"])

    def test_cached_stage_usage_is_not_counted_as_new_api_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(UsageBackend(), cache_root=directory)
            first = pipeline.run(sample())
            second = pipeline.run(sample())
            self.assertGreater(first.model_metadata["usage"]["total_tokens"], 0)
            self.assertEqual(second.model_metadata["usage"]["total_tokens"], 0)
            self.assertGreaterEqual(
                first.stages["judge"]["model_metadata"]["latency_seconds"], 0
            )

    def test_publication_date_is_explicit_in_verification_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            dated = replace(sample(), publish_time="2025/07/05")
            PanoAgentPipeline(backend, cache_root=directory).run(dated)
            sufficiency = next(
                request for request in backend.requests
                if request.task_name == "evidence_sufficiency"
            )
            judge = next(request for request in backend.requests if request.task_name == "judge")
            self.assertIn("SOURCE PUBLICATION DATE (metadata): 2025/07/05", sufficiency.user_prompt)
            self.assertIn("ORIGINAL NEWS CONTENT:", sufficiency.user_prompt)
            self.assertIn("An event happened.", sufficiency.user_prompt)
            self.assertIn("SOURCE PUBLICATION DATE (metadata): 2025/07/05", judge.user_prompt)

    def test_transcript_media_mode_drops_raw_audio_video(self) -> None:
        original = sample()
        prepared = prepare_sample_media(original, "transcript")
        self.assertEqual(prepared.audio_paths, ())
        self.assertEqual(prepared.video_paths, ())

    def test_local_context_budget_is_deterministic(self) -> None:
        original = replace(sample(), text="a" * 200, transcript="b" * 200)
        prepared = prepare_sample_media(
            original,
            "transcript",
            max_text_chars=100,
            max_transcript_chars=120,
        )
        self.assertEqual(len(prepared.text), 100)
        self.assertEqual(len(prepared.transcript), 120)
        self.assertIn("TRUNCATED FOR LOCAL CONTEXT BUDGET", prepared.text)

    def test_direct_mock(self) -> None:
        record = run_direct_sample(sample(), MockBackend())
        self.assertEqual(record.prediction, "REAL")

    def test_debiased_direct_prompt_is_opt_in(self) -> None:
        backend = RecordingBackend()
        run_direct_sample(sample(), backend, prompt_profile="debiased")
        self.assertIn("equally likely", backend.last_request.system_prompt)
        self.assertIn("Do not default to REAL", backend.last_request.user_prompt)

    def test_organizer_merges_partial_object_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(ArrayOrganizerBackend(), cache_root=directory).run(sample())
            organizer = record.stages["evidence_organizer"]
            self.assertIsNone(organizer["error"])
            self.assertEqual(organizer["data"]["entities"], ["A"])
            self.assertEqual(organizer["data"]["text_claims"], ["Claim B"])
            self.assertEqual(organizer["data"]["audio_claims"], [])

    def test_agent_mock_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(MockBackend(), cache_root=directory)
            first = pipeline.run(sample())
            second = pipeline.run(sample())
            self.assertEqual(first.prediction, "REAL")
            self.assertTrue(second.stages["judge"]["cached"])

            changed = PanoAgentPipeline(
                MockBackend(), cache_root=directory, max_new_tokens=128
            ).run(sample())
            self.assertFalse(changed.stages["judge"]["cached"])

    def test_truncated_outer_json_cannot_pass_as_inner_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(TruncatedRelationBackend(), cache_root=directory)
            record = pipeline.run(replace(sample(), modalities=("text", "image")))
            error = record.stages["relation_investigator"]["error"]
            self.assertIn("missing top-level keys: relations", error)

    def test_invalid_json_gets_one_schema_repair_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = RepairableRelationBackend()
            pipeline = PanoAgentPipeline(backend, cache_root=directory)
            record = pipeline.run(replace(sample(), modalities=("text", "image")))
            relation = record.stages["relation_investigator"]
            self.assertIsNone(relation["error"])
            self.assertEqual(backend.relation_calls, 2)
            self.assertIn("[JSON_REPAIR]", relation["raw_output"])

    def test_no_retrieval_evidence_cannot_be_replaced_by_model_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CountingContextBackend()
            pipeline = PanoAgentPipeline(backend, cache_root=directory)
            record = pipeline.run(sample())
            context = record.stages["context_verifier"]
            self.assertEqual(backend.context_calls, 0)
            self.assertEqual(context["data"]["status"], "insufficient")
            self.assertEqual(context["data"]["sources"], [])

    def test_fused_profile_uses_two_calls_and_one_media_evidence_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            record = PanoAgentPipeline(
                backend, cache_root=directory, profile="fused"
            ).run(sample())
            self.assertEqual(record.prediction, "REAL")
            self.assertEqual(
                [request.task_name for request in backend.requests],
                ["fused_investigator", "judge"],
            )
            self.assertEqual(
                set(record.stages), {"fused_investigator", "judge"}
            )
            self.assertIn("claim_contradicted", backend.requests[1].user_prompt)

    def test_fused_missing_signals_become_unknown_not_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(
                SparseFusedBackend(), cache_root=directory, profile="fused"
            ).run(sample())
            signals = record.stages["fused_investigator"]["data"]["signals"]
            self.assertEqual(signals["claim_contradicted"], "unknown")
            self.assertIn("claim_contradicted", signals["normalized_missing_signals"])
            self.assertTrue(signals["stage_valid"])

    def test_fused_nested_signal_objects_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(
                NestedFusedBackend(), cache_root=directory, profile="fused-rule"
            ).run(sample())
            signals = record.stages["fused_investigator"]["data"]["signals"]
            self.assertIs(signals["claim_supported"], True)
            self.assertIs(signals["claim_contradicted"], False)
            self.assertEqual(signals["materiality"], "central")
            self.assertEqual(signals["confidence"], 0.9)
            self.assertEqual(signals["evidence_ids"], ["E1"])

    def test_fused_epistemic_signal_status_becomes_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(
                SuspiciousSignalBackend(), cache_root=directory, profile="fused-rule"
            ).run(sample())
            signals = record.stages["fused_investigator"]["data"]["signals"]
            self.assertEqual(signals["identity_mismatch"], "unknown")
            self.assertIn("identity_mismatch", signals["normalized_missing_signals"])

    def test_fused_rule_profile_uses_one_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            record = PanoAgentPipeline(
                backend, cache_root=directory, profile="fused-rule"
            ).run(sample())
            self.assertEqual(
                [request.task_name for request in backend.requests],
                ["fused_investigator"],
            )
            self.assertEqual(record.prediction, "ERROR")
            self.assertIn("deterministic_aggregator", record.stages)

    def test_description_never_enters_full_or_fused_prompts(self) -> None:
        secret = "GOLD_LABEL_LEAK_SENTINEL"
        leaking_sample = replace(sample(), description=secret)
        for profile in ("full", "fused"):
            with tempfile.TemporaryDirectory() as directory:
                backend = CapturingBackend()
                PanoAgentPipeline(
                    backend, cache_root=directory, profile=profile
                ).run(leaking_sample)
                self.assertTrue(backend.requests)
                for request in backend.requests:
                    self.assertNotIn(secret, request.system_prompt)
                    self.assertNotIn(secret, request.user_prompt)

    def test_bad_video_does_not_abort_frame_preparation(self) -> None:
        class BrokenExtractor:
            def extract(self, sample_id, video_path):
                raise RuntimeError("broken video")

        video_sample = replace(
            sample(), modalities=("text", "video"),
            video_paths=(Path("broken.mp4"),),
        )
        with self.assertWarns(RuntimeWarning):
            prepared = prepare_sample_media(
                video_sample, "frames", frame_extractor=BrokenExtractor()
            )
        self.assertEqual(prepared.video_paths, ())
        self.assertEqual(prepared.audio_paths, ())

    def test_strict_stage_errors_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(
                TruncatedRelationBackend(),
                cache_root=directory,
                strict_stage_errors=True,
            )
            record = pipeline.run(replace(sample(), modalities=("text", "image")))
            self.assertEqual(record.prediction, "ERROR")
            self.assertIn("relation_investigator", record.error)


if __name__ == "__main__":
    unittest.main()
