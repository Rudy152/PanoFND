from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import replace

from panoagent.aggregation import deterministic_verdict
from panoagent.model_clients import MockBackend
from panoagent.pipeline import PanoAgentPipeline, _fallback_query, compact_context_for_judge
from panoagent.retrieval import (
    EvidenceDocument,
    SearchProvider,
    TavilySearchProvider,
    normalize_url,
    select_passages,
)
from panoagent.schemas import NewsSample


class JsonResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def news(**overrides) -> NewsSample:
    base = NewsSample(
        id="9001",
        modalities=("text",),
        title="Europe housing crisis deepens",
        text="Property prices rose 199% between 2010 and 2022.",
        image_paths=(),
        video_paths=(),
        audio_paths=(),
        label="fake",
        source="https://www.example.com/news/housing-crisis/",
        publish_time="2024/05/06",
        description="LEAK: the number was changed",
    )
    return replace(base, **overrides)


class StaticSearch(SearchProvider):
    name = "static"

    def __init__(self, urls=("https://example.com/news/housing-crisis",)):
        self.calls: list[list[str]] = []
        self.urls = urls

    def search(self, queries, max_results_per_query=3):
        self.calls.append(list(queries))
        return [
            EvidenceDocument(
                evidence_id=f"E{index:03d}", query=queries[0] if queries else "", title="T",
                url=url, publisher="example.com", publication_date="2024-05-06",
                retrieved_text="Property prices rose 192% between 2010 and 2022.",
                retrieval_timestamp="2026-01-01T00:00:00+00:00",
            )
            for index, url in enumerate(self.urls, 1)
        ]


class FailingSearch(SearchProvider):
    name = "failing"

    def search(self, queries, max_results_per_query=3):
        raise RuntimeError("network down")


class CapturingBackend(MockBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class PassageSelectionTests(unittest.TestCase):
    def test_keeps_middle_passage_with_claim_numbers(self) -> None:
        filler = "\n".join(f"Unrelated paragraph number {i} about weather." for i in range(80))
        key = "Between 2010 and 2022, property prices across the bloc rose 192%."
        text = "Lead paragraph about Europe housing.\n" + filler + "\n" + key + "\n" + filler
        selected = select_passages(text, "property prices rose 199% between 2010 and 2022", 400)
        self.assertIn("192%", selected)
        self.assertTrue(selected.startswith("Lead paragraph"))
        self.assertLessEqual(len(selected), 400)

    def test_returns_empty_when_nothing_overlaps(self) -> None:
        self.assertEqual(select_passages("x" * 500, "property prices", 100), "")


class TavilyProviderTests(unittest.TestCase):
    def test_retries_transient_errors_and_caches_queries(self) -> None:
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(json.loads(request.data.decode("utf-8")))
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, io.BytesIO(b""))
            return JsonResponse({"results": [
                {"url": "https://news.example/a", "title": "A", "raw_content": "Story"},
                {"url": "https://huggingface.co/datasets/x", "title": "Leak", "raw_content": "x"},
            ]})

        with tempfile.TemporaryDirectory() as directory:
            provider = TavilySearchProvider(
                api_key="tvly-test", urlopen_fn=fake_urlopen, cache_dir=directory,
                sleep_fn=lambda seconds: None, extract_short_results=0,
            )
            first = provider.search_with_report(["housing crisis 2024"])
            second = provider.search_with_report(["housing crisis 2024"])
        self.assertEqual(len(calls), 2)  # one retry, then served from the query cache
        self.assertEqual([doc.url for doc in first.documents], ["https://news.example/a"])
        self.assertEqual([doc.url for doc in second.documents], ["https://news.example/a"])
        self.assertEqual(calls[0]["topic"], "general")
        self.assertIn("huggingface.co", calls[0]["exclude_domains"])
        self.assertNotIn("tvly-test", json.dumps(calls[0]))

    def test_snippet_only_hits_are_expanded_with_full_text(self) -> None:
        endpoints = []
        article = "Lead.\n" + "Filler sentence.\n" * 50 + "Prices rose 192% from 2010 to 2022.\n"

        def fake_urlopen(request, timeout):
            endpoints.append(request.full_url.rsplit("/", 1)[-1])
            if request.full_url.endswith("/extract"):
                return JsonResponse({"results": [
                    {"url": "https://www.news.example/a/", "raw_content": article},
                ]})
            return JsonResponse({"results": [
                {"url": "https://news.example/a", "title": "A", "content": "Short snippet."},
            ]})

        provider = TavilySearchProvider(
            api_key="tvly-test", urlopen_fn=fake_urlopen, max_content_chars=300,
        )
        report = provider.search_with_report(["q"], focus_text="prices rose 199% 2010 2022")
        self.assertEqual(endpoints, ["search", "extract"])
        self.assertIn("192%", report.documents[0].retrieved_text)
        self.assertLessEqual(len(report.documents[0].retrieved_text), 300)

    def test_direct_fetch_fallback_when_extractor_is_blocked(self) -> None:
        fetched = []

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/extract"):
                return JsonResponse({"results": [], "failed_results": [{"url": "x"}]})
            return JsonResponse({"results": [
                {"url": "https://paper.example/a", "title": "A", "content": "Snippet."},
            ]})

        def fake_fetch(url):
            fetched.append(url)
            return "Estonia recorded a 192% rise between 2010 and 2022."

        with tempfile.TemporaryDirectory() as directory:
            provider = TavilySearchProvider(
                api_key="tvly-test", urlopen_fn=fake_urlopen, fetch_fn=fake_fetch,
                cache_dir=directory,
            )
            first = provider.search_with_report(["q"])
            provider.search_with_report(["q"])
        self.assertEqual(fetched, ["https://paper.example/a"])  # second run uses page cache
        self.assertIn("192%", first.documents[0].retrieved_text)

    def test_article_parser_skips_scripts(self) -> None:
        from panoagent.retrieval import _ArticleTextParser

        parser = _ArticleTextParser()
        parser.feed(
            "<html><script>var x='Ignore this script content entirely please';</script>"
            "<p>Between 2010 and 2022, Estonia recorded a <b>192%</b> rise in prices.</p>"
            "<nav><p>Menu item that should be skipped from the article text</p></nav></html>"
        )
        self.assertEqual(
            parser.blocks,
            ["Between 2010 and 2022, Estonia recorded a 192% rise in prices."],
        )

    def test_query_failure_is_reported_not_raised(self) -> None:
        def broken(request, timeout):
            raise urllib.error.URLError("offline")

        provider = TavilySearchProvider(
            api_key="tvly-test", urlopen_fn=broken, sleep_fn=lambda seconds: None
        )
        report = provider.search_with_report(["q"])
        self.assertEqual(report.documents, [])
        self.assertTrue(report.errors)

    def test_normalize_url(self) -> None:
        self.assertEqual(
            normalize_url("https://www.Example.com/a/b/?x=1"),
            normalize_url("http://example.com/a/b"),
        )


class PipelineRetrievalTests(unittest.TestCase):
    def test_fallback_query_uses_title_and_year_only(self) -> None:
        query = _fallback_query(news())
        self.assertEqual(query, "Europe housing crisis deepens 2024")
        self.assertNotIn("LEAK", query)

    def test_text_policy_forces_search_in_fused_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = StaticSearch()
            backend = CapturingBackend()
            record = PanoAgentPipeline(
                backend, cache_root=directory, search_provider=search,
                profile="fused", retrieval_policy="text",
            ).run(news())
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(search.calls[0][0], "Europe housing crisis deepens 2024")
        audit = record.model_metadata["retrieval"]
        self.assertTrue(audit["source_url_retrieved"])
        self.assertEqual(audit["documents"], 1)
        context_request = next(r for r in backend.requests if r.task_name == "context_verifier")
        self.assertIn("NEWS ITEM", context_request.user_prompt)
        self.assertIn("199%", context_request.user_prompt)
        self.assertNotIn("LEAK", context_request.user_prompt)
        self.assertNotIn("example.com/news/housing-crisis", context_request.system_prompt)
        judge = next(r for r in backend.requests if r.task_name == "judge")
        self.assertNotIn("retrieved_text", judge.user_prompt)

    def test_model_policy_keeps_previous_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = StaticSearch()
            PanoAgentPipeline(
                MockBackend(), cache_root=directory, search_provider=search,
                profile="fused", retrieval_policy="model",
            ).run(news())
        self.assertEqual(search.calls, [])

    def test_exclude_source_url_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = StaticSearch(urls=(
                "https://example.com/news/housing-crisis", "https://other.example/report",
            ))
            pipeline = PanoAgentPipeline(
                MockBackend(), cache_root=directory, search_provider=search,
                profile="fused", retrieval_policy="text", exclude_source_url=True,
            )
            record = pipeline.run(news())
        evidence = record.stages["context_verifier"]["data"]["retrieved_evidence"]
        self.assertEqual([item["url"] for item in evidence], ["https://other.example/report"])
        self.assertEqual(evidence[0]["evidence_id"], "E001")
        self.assertFalse(record.model_metadata["retrieval"]["source_url_retrieved"])

    def test_search_failure_does_not_crash_or_poison_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PanoAgentPipeline(
                MockBackend(), cache_root=directory, search_provider=FailingSearch(),
                profile="fused", retrieval_policy="text",
            )
            record = pipeline.run(news())
            context = record.stages["context_verifier"]
            self.assertEqual(context["data"]["status"], "insufficient")
            self.assertTrue(context["data"]["retrieval_errors"])
            self.assertIsNone(pipeline.context.cache.load("9001", "context_verifier"))

    def test_verifier_normalizes_status_and_salvages_slot_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = PanoAgentPipeline(MockBackend(), cache_root=directory).context
            parsed = verifier._parse_and_validate(
                '{"same_report": {"found": true, "evidence_ids": ["E001"]}, "slot_checks": [],'
                ' "claims": [{"claim": "C", "status": "mostly supported"}],'
                ' "status": "mostly supported", "sources": []}'
            )
            self.assertEqual(parsed["status"], "mixed")
            self.assertEqual(parsed["claims"][0]["status"], "mixed")
            truncated = verifier._parse_and_validate(
                '{"same_report": {"found": true, "evidence_ids": ["E001"]},'
                ' "slot_checks": [{"field": "number", "status": "mismatch", "evidence_id": "E001"}],'
                ' "claims": [{"claim": "C", "status": "refuted", "evidence_ids": ["E001"]}],'
                ' "status": "refuted", "sources": ["E0'
            )
            self.assertEqual(truncated["status"], "refuted")
            self.assertEqual(truncated["same_report"]["evidence_ids"], ["E001"])
            self.assertEqual(truncated["slot_checks"][0]["status"], "mismatch")

    def test_slot_guard_overrides_judge_on_original_report_mismatch(self) -> None:
        class SlotBackend(MockBackend):
            def generate(self, request):
                if request.task_name == "context_verifier":
                    return json.dumps({
                        "same_report": {"found": True, "evidence_ids": ["E001"]},
                        "slot_checks": [{
                            "field": "number", "news_value": "199%", "evidence_value": "192%",
                            "status": "mismatch", "materiality": "central", "evidence_id": "E001",
                        }],
                        "claims": [], "status": "mixed", "sources": ["E001"],
                    })
                return super().generate(request)

        for guard, expected in ((True, "FAKE"), (False, "REAL")):
            with tempfile.TemporaryDirectory() as directory:
                record = PanoAgentPipeline(
                    SlotBackend(), cache_root=directory, search_provider=StaticSearch(),
                    profile="fused", retrieval_policy="text", external_slot_guard=guard,
                ).run(news())
            self.assertEqual(record.prediction, expected)
            self.assertEqual(record.stages["judge"]["data"]["binary_label"], "REAL")
            self.assertEqual("external_slot_guard" in record.stages, guard)

    def test_disabled_search_never_reuses_cached_web_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            PanoAgentPipeline(
                MockBackend(), cache_root=directory, search_provider=StaticSearch(),
                profile="fused", retrieval_policy="always",
            ).run(news())
            backend = CapturingBackend()
            verifier = PanoAgentPipeline(backend, cache_root=directory, profile="fused").context
            result = verifier.run(news(), {
                "claims_to_verify": ["Property prices rose 199%"],
                "search_queries": ["Europe housing crisis deepens 2024"],
            })
        self.assertEqual(result.data["retrieved_evidence"], [])
        self.assertEqual(result.data["status"], "insufficient")
        self.assertFalse(any(r.task_name == "context_verifier" for r in backend.requests))

    def test_compact_context_drops_full_texts(self) -> None:
        compact = compact_context_for_judge({
            "status": "refuted", "claims": [], "sources": ["E001"],
            "retrieved_evidence": [{"evidence_id": "E001", "title": "T", "retrieved_text": "long"}],
        })
        self.assertNotIn("retrieved_evidence", compact)
        self.assertEqual(compact["evidence_index"][0]["evidence_id"], "E001")


class SpeechRecognitionToleranceTests(unittest.TestCase):
    """Cases observed on real REAL audio items in the tuning943 run (2026-11-07)."""

    def _mismatch(self, field, news_value, evidence_value):
        return {
            "field": field, "news_value": news_value, "evidence_value": evidence_value,
            "status": "mismatch", "materiality": "central",
        }

    def test_asr_name_variants_do_not_trigger_guard(self) -> None:
        from panoagent.aggregation import is_hard_slot_mismatch

        for news_value, evidence_value in (
            ("Vance Belter", "Vance Boelter"),
            ("Ayatollah Ali Harmony", "Ayatollah Ali Khamenei"),
            ("hilders", "Geert Wilders"),
            ("Lauren Sanchez", "Lauren Sánchez"),
            ("Aaron Patterson", "Erin Patterson"),
        ):
            self.assertFalse(
                is_hard_slot_mismatch(self._mismatch("person", news_value, evidence_value)),
                news_value,
            )

    def test_wording_dates_and_adhoc_fields_do_not_trigger_guard(self) -> None:
        from panoagent.aggregation import is_hard_slot_mismatch

        self.assertFalse(is_hard_slot_mismatch(self._mismatch(
            "quote", "weighing whether negotiations are in its interest",
            "no understanding about renewing talks",
        )))
        self.assertFalse(is_hard_slot_mismatch(self._mismatch("time", "June 24, 2025", "June 23, 2025")))
        self.assertFalse(is_hard_slot_mismatch(self._mismatch(
            "data_source", "SEER and CDC data", "CISNET models",
        )))
        self.assertFalse(is_hard_slot_mismatch(self._mismatch("number", "47%", "47% (2010-2022)")))

    def test_real_edits_still_trigger_guard(self) -> None:
        from panoagent.aggregation import is_hard_slot_mismatch

        self.assertTrue(is_hard_slot_mismatch(self._mismatch("quote", "199%", "192%")))
        self.assertTrue(is_hard_slot_mismatch(self._mismatch("number", "800,000", "400,000")))
        self.assertTrue(is_hard_slot_mismatch(self._mismatch("location", "Morwell", "Leongatha")))
        self.assertTrue(is_hard_slot_mismatch(self._mismatch("person", "Joe Biden", "Kamala Harris")))

    def test_prompts_carry_asr_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = CapturingBackend()
            PanoAgentPipeline(
                backend, cache_root=directory, search_provider=StaticSearch(),
                profile="fused", retrieval_policy="text",
            ).run(news(transcript="Speaker says prices rose."))
        for task in ("fused_investigator", "context_verifier", "judge"):
            prompt = next(r for r in backend.requests if r.task_name == task).user_prompt
            self.assertIn("automatic speech recognition", prompt, task)


class NumericDiffTests(unittest.TestCase):
    def test_flags_substituted_number_in_matching_sentence(self) -> None:
        from panoagent.slot_diff import numeric_diffs

        news = ("Between 2010 and 2022, property prices across the bloc surged by 47%. "
                "In some countries they almost trebled: Estonia recorded a 199% rise.")
        evidence = [{"evidence_id": "E002", "retrieved_text": (
            "Between 2010 and 2022, property prices across the 27-member bloc surged by 47%. "
            "In some countries they almost trebled: Estonia recorded a 192% rise."
        )}]
        diffs = numeric_diffs(news, evidence, min_similarity=0.6, min_shared_words=4)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["news_numbers"], ["199"])
        self.assertEqual(diffs[0]["evidence_numbers"], ["192"])

    def test_identical_or_unrelated_sentences_are_not_flagged(self) -> None:
        from panoagent.slot_diff import numeric_diffs

        text = "In some countries they almost trebled: Estonia recorded a 192% rise in prices."
        same = [{"evidence_id": "E1", "retrieved_text": text}]
        other = [{"evidence_id": "E2", "retrieved_text": "Rents in Lisbon climbed 35% last year alone."}]
        self.assertEqual(numeric_diffs(text, same, min_shared_words=4), [])
        self.assertEqual(numeric_diffs(text, other, min_shared_words=4), [])

    def test_pipeline_guard_uses_numeric_diff_and_relabels_variants(self) -> None:
        class Search(StaticSearch):
            def search(self, queries, max_results_per_query=3):
                docs = super().search(queries)
                docs[0].retrieved_text = (
                    "Analysts said property prices across Europe rose 192% between 2010 and 2022 overall."
                )
                return docs

        class VariantBackend(MockBackend):
            def generate(self, request):
                if request.task_name == "context_verifier":
                    return json.dumps({
                        "same_report": {"found": True, "evidence_ids": ["E001"]},
                        "slot_checks": [{
                            "field": "person", "news_value": "Vance Belter",
                            "evidence_value": "Vance Boelter", "status": "mismatch",
                            "materiality": "central", "evidence_id": "E001",
                        }],
                        "claims": [], "status": "mixed", "sources": ["E001"],
                    })
                return super().generate(request)

        item = news(text="Analysts said property prices across Europe rose 199% between 2010 and 2022 overall.")
        with tempfile.TemporaryDirectory() as directory:
            record = PanoAgentPipeline(
                VariantBackend(), cache_root=directory, search_provider=Search(),
                profile="fused", retrieval_policy="text",
            ).run(item)
        self.assertEqual(record.prediction, "FAKE")
        guard = record.stages["external_slot_guard"]["data"]
        self.assertEqual(guard["external_slot_mismatches"], [])  # Belter/Boelter is a variant
        self.assertEqual(guard["numeric_diffs"][0]["news_numbers"], ["199"])
        compact = compact_context_for_judge(
            {"slot_checks": [{"field": "person", "news_value": "Vance Belter",
                              "evidence_value": "Vance Boelter", "status": "mismatch"}]}
        )
        self.assertEqual(compact["slot_checks"][0]["status"], "variant")


class AggregatorExternalEvidenceTests(unittest.TestCase):
    def _report(self, **signals):
        base = {
            "claim_supported": "unknown", "claim_contradicted": "unknown",
            "materiality": "unknown", "confidence": 0.6, "stage_valid": True,
        }
        base.update(signals)
        return {"signals": base}

    def test_aligned_slot_mismatch_yields_fake(self) -> None:
        verdict = deterministic_verdict(self._report(), {
            "status": "mixed", "claims": [],
            "same_report": {"found": True},
            "slot_checks": [{
                "field": "number", "news_value": "199%", "evidence_value": "192%",
                "status": "mismatch", "materiality": "central",
            }],
        })
        self.assertEqual(verdict["binary_label"], "FAKE")

    def test_other_source_estimate_is_not_a_refutation(self) -> None:
        verdict = deterministic_verdict(self._report(), {
            "status": "mixed", "claims": [],
            "same_report": {"found": True, "evidence_ids": ["E002"]},
            "slot_checks": [{
                "field": "number", "news_value": "800,000", "evidence_value": "400,000",
                "status": "mismatch", "materiality": "central", "evidence_id": "E008",
            }],
        })
        self.assertEqual(verdict["binary_label"], "ERROR")

    def test_topic_hit_without_same_report_does_not_decide(self) -> None:
        verdict = deterministic_verdict(self._report(), {
            "status": "mixed", "claims": [],
            "same_report": {"found": False},
            "slot_checks": [{"field": "number", "status": "mismatch", "materiality": "central"}],
        })
        self.assertEqual(verdict["binary_label"], "ERROR")

    def test_aligned_report_matching_all_slots_yields_real(self) -> None:
        verdict = deterministic_verdict(self._report(), {
            "status": "supported", "claims": [],
            "same_report": {"found": True},
            "slot_checks": [{"field": "number", "status": "match", "materiality": "central"}],
        })
        self.assertEqual(verdict["binary_label"], "REAL")


if __name__ == "__main__":
    unittest.main()
