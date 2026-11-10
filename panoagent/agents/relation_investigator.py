from __future__ import annotations

import json

from .. import prompts
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class RelationInvestigator(JsonAgent):
    stage = "relation_investigator"
    required_keys = frozenset({"relations"})

    @staticmethod
    def _available_pairs(sample: NewsSample) -> set[str]:
        available = set(sample.modalities)
        # Dataset modality labels describe the primary payload, but every sample may also
        # carry a model-visible headline or article body. Treat that as text evidence for
        # relation analysis without changing the benchmark's modality grouping.
        if sample.title.strip() or sample.text.strip():
            available.add("text")
        pairs = (
            ("text", "image"), ("text", "audio"), ("text", "video"),
            ("image", "audio"), ("image", "video"), ("audio", "video"),
        )
        return {f"{left}-{right}" for left, right in pairs if {left, right} <= available}

    def run(self, sample: NewsSample, case_file: dict, unimodal: dict, refresh: bool = False) -> StageResult:
        allowed_pairs = self._available_pairs(sample)
        if not allowed_pairs:
            result = StageResult(
                stage=self.stage,
                data={"relations": {}},
                raw_output='{"relations": {}}',
                cached=False,
            )
            self.cache.save(sample.id, self.stage, result.to_dict())
            return result
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.RELATION_SYSTEM,
            user_prompt=prompts.RELATION_USER.format(
                asr_tolerance=prompts.ASR_TOLERANCE,
                modalities=", ".join(sorted({part for pair in allowed_pairs for part in pair.split("-")})),
                case_file=json.dumps(case_file, ensure_ascii=False),
                unimodal=json.dumps(unimodal, ensure_ascii=False),
                content=sample.detector_text(),
            ),
            max_new_tokens=self.max_new_tokens,
            **self.media_kwargs(sample),
        )
        result = self.run_request(sample, request, refresh=refresh)
        relations = result.data.get("relations")
        if result.error is None and not isinstance(relations, dict):
            result.data = {}
            result.error = "ValueError: relation_investigator `relations` must be an object"
        elif isinstance(relations, dict):
            filtered = {key: value for key, value in relations.items() if key in allowed_pairs}
            if filtered != relations:
                result.data["relations"] = filtered
        payload = result.to_dict()
        payload["request_fingerprint"] = self._request_fingerprint(request)
        self.cache.save(sample.id, self.stage, payload)
        return result
