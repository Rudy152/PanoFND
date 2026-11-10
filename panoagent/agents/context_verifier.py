from __future__ import annotations

import json
import re

from .. import prompts
from ..retrieval import SearchProvider, normalize_url
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class ContextVerifier(JsonAgent):
    stage = "context_verifier"
    required_keys = frozenset({"claims", "status", "sources"})

    def __init__(self, *args, search_provider: SearchProvider, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_provider = search_provider
        # Ablation switch: drop the dataset's own source article from retrieved evidence
        # so results can be reported with and without access to the original report.
        self.exclude_source_url = False

    @staticmethod
    def _complete_array(text: str, key: str) -> tuple[list, int]:
        """Extract one fully closed JSON array even if a later field was truncated."""
        marker = f'"{key}"'
        key_index = text.find(marker)
        if key_index < 0:
            raise ValueError(f"missing {key}")
        start = text.find("[", key_index + len(marker))
        if start < 0:
            raise ValueError(f"missing {key} array")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth == 0:
                    value = json.loads(text[start:index + 1])
                    if not isinstance(value, list):
                        raise ValueError(f"{key} is not an array")
                    return value, index + 1
        raise ValueError(f"incomplete {key} array")

    _STATUSES = ("supported", "refuted", "mixed", "insufficient")

    @classmethod
    def _normalize_status(cls, value: object) -> str:
        """Map free-form statuses ("mostly supported", "verified") onto the fixed enum."""
        text = str(value or "").strip().casefold().replace("_", " ")
        if text in cls._STATUSES:
            return text
        if text in {"verified", "all supported", "confirmed"}:
            return "supported"
        if "refut" in text or "contradict" in text or "false" == text:
            return "refuted" if not ("partial" in text or "mostly" in text) else "mixed"
        if "support" in text or "mixed" in text or "partial" in text:
            return "mixed"
        return "insufficient"

    def _parse_and_validate(self, raw: str) -> dict:
        data = self._parse_raw(raw)
        data["status"] = self._normalize_status(data.get("status"))
        for claim in data.get("claims") or []:
            if isinstance(claim, dict):
                claim["status"] = self._normalize_status(claim.get("status"))
        if not isinstance(data.get("same_report"), dict):
            data["same_report"] = {"found": False, "evidence_ids": [], "note": "not reported"}
        if not isinstance(data.get("slot_checks"), list):
            data["slot_checks"] = []
        return data

    def _parse_raw(self, raw: str) -> dict:
        try:
            return super()._parse_and_validate(raw)
        except ValueError as original_error:
            # Context responses can finish every claim and then truncate the redundant
            # top-level sources list. Recover only when the claims array and overall
            # status are complete; sources are deterministically derived from evidence_ids.
            try:
                claims, claims_end = self._complete_array(raw, "claims")
                status_match = re.search(
                    r'"status"\s*:\s*"(supported|refuted|mixed|insufficient|verified|all_supported)"',
                    raw[claims_end:],
                )
                if status_match is None or not all(isinstance(item, dict) for item in claims):
                    raise ValueError("incomplete context status")
                sources: list[str] = []
                for claim in claims:
                    for evidence_id in claim.get("evidence_ids", []):
                        value = str(evidence_id)
                        if value not in sources:
                            sources.append(value)
                salvaged = {
                    "claims": claims,
                    "status": status_match.group(1),
                    "sources": sources,
                }
                # same_report/slot_checks precede claims, so they are complete here.
                try:
                    slot_checks, _ = self._complete_array(raw[:claims_end], "slot_checks")
                    salvaged["slot_checks"] = slot_checks
                    same = re.search(r'"same_report"\s*:\s*(\{.*?\})\s*,\s*"slot_checks"', raw, re.S)
                    if same:
                        salvaged["same_report"] = json.loads(same.group(1))
                except (ValueError, json.JSONDecodeError):
                    pass
                return salvaged
            except (TypeError, ValueError, json.JSONDecodeError):
                raise original_error

    def _request(self, claims: list[str], evidence: list[dict], news_excerpt: str = "") -> ModelRequest:
        return ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.CONTEXT_SYSTEM,
            user_prompt=prompts.CONTEXT_USER.format(
                asr_tolerance=prompts.ASR_TOLERANCE,
                claims=json.dumps(claims, ensure_ascii=False),
                news_excerpt=news_excerpt or "(not supplied)",
                evidence=json.dumps(evidence, ensure_ascii=False),
            ),
            max_new_tokens=self.max_new_tokens,
        )

    @staticmethod
    def _news_excerpt(sample: NewsSample, max_chars: int = 5000) -> str:
        """Bounded copy of the news item so exact slots can be compared with sources."""
        text = sample.detector_text()
        if len(text) <= max_chars:
            return text
        marker = "\n...[NEWS EXCERPT TRUNCATED]...\n"
        head = int((max_chars - len(marker)) * 0.75)
        tail = max_chars - len(marker) - head
        return text[:head] + marker + text[-tail:]

    def _filter_excluded(self, evidence: list[dict], sample: NewsSample) -> list[dict]:
        source_key = normalize_url(sample.source) if sample.source else ""
        if not self.exclude_source_url or not source_key:
            return evidence
        kept = [item for item in evidence if normalize_url(item.get("url", "")) != source_key]
        for index, item in enumerate(kept, 1):
            item["evidence_id"] = f"E{index:03d}"
        return kept

    def run(self, sample: NewsSample, sufficiency: dict, refresh: bool = False) -> StageResult:
        queries = [str(value) for value in sufficiency.get("search_queries", []) if str(value).strip()]
        claims = [str(value) for value in sufficiency.get("claims_to_verify", []) if str(value).strip()]
        news_excerpt = self._news_excerpt(sample)

        # Retrieval timestamps make fresh search results fingerprint differently on every retry.
        # Reconstruct the original request from cached evidence before touching the network.
        cached_evidence: list[dict] | None = None
        # With search disabled, evidence cached by an earlier search-enabled run must never
        # leak in: a shared cache made the "no-search" arm verify against web evidence.
        if not refresh and self.search_provider.enabled:
            cached = self.cache.load(sample.id, self.stage)
            cached_data = dict(cached.get("data", {})) if cached is not None else {}
            if "retrieved_evidence" in cached_data:
                candidate_evidence = cached_data.get("retrieved_evidence")
                if isinstance(candidate_evidence, list):
                    cached_evidence = candidate_evidence
                    cached_request = self._request(claims, cached_evidence, news_excerpt)
                    if cached.get("request_fingerprint") == self._request_fingerprint(cached_request):
                        cached_error = cached.get("error")
                        if cached_error and cached.get("raw_output"):
                            try:
                                repaired_data = self._parse_and_validate(str(cached["raw_output"]))
                                repaired_data["retrieved_evidence"] = cached_evidence
                                repaired = StageResult(
                                    stage=self.stage,
                                    data=repaired_data,
                                    raw_output=str(cached["raw_output"]),
                                    cached=True,
                                    model_metadata=dict(cached.get("model_metadata") or {}),
                                )
                                payload = repaired.to_dict()
                                payload["request_fingerprint"] = self._request_fingerprint(cached_request)
                                self.cache.save(sample.id, self.stage, payload)
                                return repaired
                            except ValueError:
                                pass
                        return StageResult(
                            stage=self.stage,
                            data=cached_data,
                            raw_output=str(cached.get("raw_output", "")),
                            cached=True,
                            error=cached.get("error"),
                            model_metadata=dict(cached.get("model_metadata") or {}),
                        )

        # A prompt/claim revision can re-evaluate the already retrieved corpus. This avoids
        # timestamp-driven result drift and network cost; missing coverage is reported as
        # insufficient rather than silently fetching a different evidence set.
        evidence = cached_evidence
        retrieval_errors: list[str] = []
        if evidence is None or refresh:
            report = self.search_provider.search_with_report(
                queries, focus_text="\n".join(claims)
            )
            retrieval_errors = list(report.errors)
            evidence = self._filter_excluded([item.to_dict() for item in report.documents], sample)
        request = self._request(claims, evidence, news_excerpt)
        if not evidence:
            data = {
                "claims": [
                    {"claim": claim, "status": "insufficient", "evidence_ids": []}
                    for claim in claims
                ],
                "status": "insufficient",
                "sources": [],
                "retrieved_evidence": [],
            }
            if retrieval_errors:
                data["retrieval_errors"] = retrieval_errors
            result = StageResult(
                stage=self.stage,
                data=data,
                raw_output=json.dumps(data, ensure_ascii=False),
            )
            payload = result.to_dict()
            if retrieval_errors:
                # A transient network failure must not be frozen into the cache.
                return result
            payload["request_fingerprint"] = self._request_fingerprint(request)
            self.cache.save(sample.id, self.stage, payload)
            return result
        result = self.run_request(sample, request, refresh=refresh)
        if evidence and not result.cached:
            result.data["retrieved_evidence"] = evidence
            if retrieval_errors:
                result.data["retrieval_errors"] = retrieval_errors
            payload = result.to_dict()
            payload["request_fingerprint"] = self._request_fingerprint(request)
            self.cache.save(sample.id, self.stage, payload)
        return result
