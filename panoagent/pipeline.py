from __future__ import annotations

import re
import time

from .aggregation import deterministic_verdict, external_slot_mismatches, is_hard_slot_mismatch
from .slot_diff import numeric_diffs

from .agents import (
    ContextVerifier,
    EvidenceOrganizer,
    EvidenceSufficiencyAgent,
    FusedInvestigator,
    JudgeAgent,
    RelationInvestigator,
    UnimodalInvestigator,
)
from .cache import StageCache
from .model_clients import ModelBackend
from .retrieval import NullSearchProvider, SearchProvider, normalize_url
from .schemas import NewsSample, PredictionRecord, StageResult


RETRIEVAL_POLICIES = ("model", "text", "always")


def _fallback_query(sample: NewsSample, max_words: int = 18) -> str:
    """Deterministic query aimed at the original report of the event.

    It uses only model-visible fields (title, else the first sentence of the article or
    transcript) plus the publication year; never the label, description, or source URL.
    """

    seed = sample.title.strip()
    if not seed:
        body = (sample.text or sample.transcript or "").strip()
        seed = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", body, maxsplit=1)[0] if body else ""
    words = seed.split()
    if not words:
        return ""
    query = " ".join(words[:max_words])
    year = re.search(r"(19|20)\d{2}", sample.publish_time or "")
    if year and year.group(0) not in query:
        query = f"{query} {year.group(0)}"
    return query


def compact_context_for_judge(context: dict, numeric_diff_report: list | None = None) -> dict:
    """Drop full retrieved texts before the Judge call.

    The Judge only needs the verifier's structured findings plus an index of sources;
    passing every retrieved article again multiplied Judge tokens without new evidence.
    Verifier "mismatches" that are only spelling/transcription/wording variants are
    relabelled `variant`, because qwen3-vl-plus ignores that instruction in its prompt.
    """

    compact = {key: value for key, value in context.items() if key != "retrieved_evidence"}
    if isinstance(compact.get("slot_checks"), list):
        relabelled = []
        for item in compact["slot_checks"]:
            if (
                isinstance(item, dict)
                and item.get("status") == "mismatch"
                and not is_hard_slot_mismatch({**item, "materiality": "central"})
            ):
                item = {**item, "status": "variant"}
            relabelled.append(item)
        compact["slot_checks"] = relabelled
    if numeric_diff_report:
        compact["numeric_diffs"] = numeric_diff_report
    evidence = context.get("retrieved_evidence") or []
    if evidence:
        compact["evidence_index"] = [
            {
                "evidence_id": item.get("evidence_id"),
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "publication_date": item.get("publication_date"),
            }
            for item in evidence if isinstance(item, dict)
        ]
    return compact


class PanoAgentPipeline:
    def __init__(
        self,
        backend: ModelBackend,
        cache_root: str,
        search_provider: SearchProvider | None = None,
        max_new_tokens: int = 768,
        profile: str = "full",
        strict_stage_errors: bool = False,
        retrieval_policy: str = "model",
        exclude_source_url: bool = False,
        external_slot_guard: bool = True,
    ):
        if profile not in {"full", "fused", "fused-rule"}:
            raise ValueError("profile must be full, fused, or fused-rule")
        if retrieval_policy not in RETRIEVAL_POLICIES:
            raise ValueError(f"retrieval_policy must be one of {RETRIEVAL_POLICIES}")
        self.retrieval_policy = retrieval_policy
        self.external_slot_guard = external_slot_guard
        self.backend = backend
        self.profile = profile
        self.strict_stage_errors = strict_stage_errors
        cache = StageCache(cache_root, backend.name)
        common = {"backend": backend, "cache": cache, "max_new_tokens": max_new_tokens}
        self.organizer = EvidenceOrganizer(**common)
        self.unimodal = UnimodalInvestigator(**common)
        self.relation = RelationInvestigator(**common)
        self.sufficiency = EvidenceSufficiencyAgent(**common)
        self.search_provider = search_provider or NullSearchProvider()
        self.context = ContextVerifier(
            backend=backend,
            cache=cache,
            # Slot checks + per-claim alignment outgrew 768 tokens on real evidence sets.
            max_new_tokens=max(max_new_tokens, 1500) if max_new_tokens >= 768 else max_new_tokens,
            search_provider=self.search_provider,
        )
        self.context.exclude_source_url = exclude_source_url
        self.fused = FusedInvestigator(
            backend=backend,
            cache=cache,
            # 768 truncated real qwen3-vl-plus reports once atomic checks and up to three
            # retrieval queries were requested; truncation fails closed as ERROR.
            max_new_tokens=1200 if max_new_tokens >= 768 else max_new_tokens,
        )
        self.judge = JudgeAgent(
            backend=backend,
            cache=cache,
            max_new_tokens=min(max_new_tokens, 384) if profile == "fused" else max_new_tokens,
        )

    def _force_retrieval(self, sample: NewsSample) -> bool:
        if not self.search_provider.enabled or self.retrieval_policy == "model":
            return False
        if self.retrieval_policy == "always":
            return True
        return bool(sample.title.strip() or sample.text.strip() or sample.transcript.strip())

    @staticmethod
    def _retrieval_request(
        sample: NewsSample, claims: list, queries: list, extra_claim: str = ""
    ) -> dict:
        claim_list = [str(item) for item in claims if str(item).strip()]
        if extra_claim.strip() and extra_claim.strip() not in claim_list:
            claim_list.insert(0, extra_claim.strip())
        query_list = [str(item) for item in queries if str(item).strip()]
        fallback = _fallback_query(sample)
        if fallback and fallback.casefold() not in {item.casefold() for item in query_list}:
            # First, so the document cap cannot crowd out the search for the original report.
            query_list.insert(0, fallback)
        return {"claims_to_verify": claim_list[:4], "search_queries": query_list[:4]}

    def _apply_slot_guard(self, stages: dict[str, StageResult], prediction: str) -> str:
        """Deterministic FAKE when the original report contradicts a central exact slot.

        LLM judges tend to dismiss a single changed number as "peripheral" once the broad
        event is confirmed. When the verifier has located the same report and a central
        slot differs from it, the published item is not that report. The Judge's own
        label is preserved in the stage record for offline ablation.
        """

        context = stages.get("context_verifier")
        if not self.external_slot_guard or context is None or context.error:
            return prediction
        mismatches = external_slot_mismatches(context.data)
        diffs = (stages.get("numeric_diff").data.get("numeric_diffs") or []) if stages.get("numeric_diff") else []
        if (not mismatches and not diffs) or prediction == "FAKE":
            return prediction
        verdict = {
            "judge_label": prediction,
            "final_label": "FAKE",
            "reason": "retrieved report contradicts an exact slot of the news item",
            "external_slot_mismatches": mismatches,
            "numeric_diffs": diffs,
        }
        stages["external_slot_guard"] = StageResult(
            stage="external_slot_guard", data=verdict, raw_output=str(verdict)
        )
        return "FAKE"

    def _numeric_diff_stage(self, sample: NewsSample, stages: dict[str, StageResult]) -> list:
        """Local, zero-cost number diff of title/article against retrieved reports."""
        context = stages.get("context_verifier")
        if context is None or context.error:
            return []
        evidence = [item for item in context.data.get("retrieved_evidence") or [] if isinstance(item, dict)]
        same = context.data.get("same_report") or {}
        report_ids = {str(value) for value in same.get("evidence_ids") or []}
        if same.get("found") is True and report_ids:
            evidence = [item for item in evidence if str(item.get("evidence_id")) in report_ids]
            threshold = 0.6
        else:
            threshold = 0.7  # no confirmed same report: require a near-verbatim sentence
        news_text = "\n".join(part for part in (sample.title, sample.text) if part.strip())
        diffs = numeric_diffs(news_text, evidence, min_similarity=threshold)
        stages["numeric_diff"] = StageResult(
            stage="numeric_diff", data={"numeric_diffs": diffs}, raw_output=str(diffs)
        )
        return diffs

    def _retrieval_audit(self, sample: NewsSample, stages: dict[str, StageResult]) -> dict | None:
        context = stages.get("context_verifier")
        if context is None:
            return None
        evidence = context.data.get("retrieved_evidence") or []
        urls = [normalize_url(item.get("url", "")) for item in evidence if isinstance(item, dict)]
        source_key = normalize_url(sample.source) if sample.source else ""
        source_host = source_key.split("/", 1)[0] if source_key else ""
        # Audit only (never shown to a model): how often search reached the source article.
        return {
            "provider": getattr(self.search_provider, "name", "custom"),
            "policy": self.retrieval_policy,
            "documents": len(evidence),
            "source_url_retrieved": bool(source_key) and source_key in urls,
            "source_domain_retrieved": bool(source_host)
            and any(url.split("/", 1)[0] == source_host for url in urls),
            "status": context.data.get("status"),
            "same_report_found": (context.data.get("same_report") or {}).get("found"),
            "errors": context.data.get("retrieval_errors") or [],
        }

    def _record(
        self,
        sample: NewsSample,
        stages: dict[str, StageResult],
        start: float,
        prediction: str,
        raw_output: str,
    ) -> PredictionRecord:
        errors = [f"{name}: {result.error}" for name, result in stages.items() if result.error]
        if self.strict_stage_errors and errors:
            prediction = "ERROR"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stage_metadata: dict[str, dict] = {}
        for name, result in stages.items():
            stage_metadata[name] = result.model_metadata
            if result.cached:
                continue
            stage_usage = result.model_metadata.get("usage") or {}
            for key in usage:
                value = stage_usage.get(key)
                if isinstance(value, (int, float)):
                    usage[key] += value
        metadata = {"usage": usage, "stages": stage_metadata, "profile": self.profile}
        retrieval_audit = self._retrieval_audit(sample, stages)
        if retrieval_audit is not None:
            metadata["retrieval"] = retrieval_audit
        return PredictionRecord(
            id=sample.id,
            gold=sample.binary_label,
            prediction=prediction,
            modality_key=sample.modality_key,
            raw_output=raw_output,
            backend=self.backend.name,
            latency_seconds=time.perf_counter() - start,
            error="; ".join(errors) or None,
            model_metadata=metadata,
            stages={name: result.to_dict() for name, result in stages.items()},
        )

    def _run_fused(self, sample: NewsSample, refresh: bool, start: float) -> PredictionRecord:
        stages: dict[str, StageResult] = {}
        stages["fused_investigator"] = self.fused.run(sample, refresh=refresh)
        fused = stages["fused_investigator"]
        if fused.error:
            return self._record(sample, stages, start, "ERROR", fused.raw_output)

        retrieval = fused.data.get("retrieval") or {}
        model_requested = retrieval.get("needed") is True
        if model_requested or self._force_retrieval(sample):
            if self.search_provider.enabled:
                sufficiency = self._retrieval_request(
                    sample,
                    retrieval.get("claims") or [],
                    retrieval.get("queries") or [],
                    extra_claim=str(fused.data.get("central_claim") or ""),
                )
            else:
                sufficiency = {
                    "claims_to_verify": retrieval.get("claims") or [],
                    "search_queries": retrieval.get("queries") or [],
                }
            stages["context_verifier"] = self.context.run(
                sample, sufficiency, refresh=refresh
            )
            context = stages["context_verifier"].data
        else:
            context = {"status": "not_requested", "claims": [], "sources": []}

        if self.profile == "fused-rule":
            verdict = deterministic_verdict(
                fused.data,
                {**context, "numeric_diffs": self._numeric_diff_stage(sample, stages)},
            )
            stages["deterministic_aggregator"] = StageResult(
                stage="deterministic_aggregator",
                data=verdict,
                raw_output=str(verdict),
            )
            prediction = str(verdict.get("binary_label", "ERROR")).upper()
            return self._record(
                sample,
                stages,
                start,
                prediction if prediction in {"REAL", "FAKE"} else "ERROR",
                stages["deterministic_aggregator"].raw_output,
            )

        stages["judge"] = self.judge.run_structured(
            sample, fused.data,
            compact_context_for_judge(context, self._numeric_diff_stage(sample, stages)),
            refresh=refresh,
        )
        judge = stages["judge"]
        prediction = str(judge.data.get("binary_label", "ERROR")).upper()
        if prediction not in {"REAL", "FAKE"}:
            prediction = "ERROR"
        prediction = self._apply_slot_guard(stages, prediction)
        return self._record(sample, stages, start, prediction, judge.raw_output)

    def run(self, sample: NewsSample, refresh: bool = False) -> PredictionRecord:
        start = time.perf_counter()
        if self.profile in {"fused", "fused-rule"}:
            return self._run_fused(sample, refresh, start)
        stages: dict[str, StageResult] = {}

        stages["evidence_organizer"] = self.organizer.run(sample, refresh=refresh)
        case_file = stages["evidence_organizer"].data
        stages["unimodal_investigator"] = self.unimodal.run(sample, case_file, refresh=refresh)
        unimodal = stages["unimodal_investigator"].data
        stages["relation_investigator"] = self.relation.run(
            sample, case_file, unimodal, refresh=refresh
        )
        relation = stages["relation_investigator"].data
        stages["evidence_sufficiency"] = self.sufficiency.run(
            sample, case_file, unimodal, relation, refresh=refresh
        )
        sufficiency = stages["evidence_sufficiency"].data

        if sufficiency.get("sufficient") is False or self._force_retrieval(sample):
            if self.search_provider.enabled:
                sufficiency = self._retrieval_request(
                    sample,
                    sufficiency.get("claims_to_verify") or [],
                    sufficiency.get("search_queries") or [],
                )
            stages["context_verifier"] = self.context.run(sample, sufficiency, refresh=refresh)
            context = stages["context_verifier"].data
        else:
            context = {"status": "not_requested", "claims": [], "sources": []}

        stages["judge"] = self.judge.run(
            sample, case_file, unimodal, relation,
            compact_context_for_judge(context, self._numeric_diff_stage(sample, stages)),
            refresh=refresh,
        )
        judge = stages["judge"]
        prediction = str(judge.data.get("binary_label", "ERROR")).upper()
        if prediction not in {"REAL", "FAKE"}:
            prediction = "ERROR"
        prediction = self._apply_slot_guard(stages, prediction)
        return self._record(sample, stages, start, prediction, judge.raw_output)
