from __future__ import annotations

import json

from .. import prompts
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class EvidenceSufficiencyAgent(JsonAgent):
    stage = "evidence_sufficiency"
    required_keys = frozenset({"sufficient", "reason", "claims_to_verify", "search_queries"})

    def run(self, sample: NewsSample, case_file: dict, unimodal: dict, relation: dict, refresh: bool = False) -> StageResult:
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.SUFFICIENCY_SYSTEM,
            user_prompt=prompts.SUFFICIENCY_USER.format(
                publish_time=sample.publish_time or "unknown",
                content=sample.detector_text(),
                case_file=json.dumps(case_file, ensure_ascii=False),
                unimodal=json.dumps(unimodal, ensure_ascii=False),
                relation=json.dumps(relation, ensure_ascii=False),
            ),
            max_new_tokens=self.max_new_tokens,
        )
        return self.run_request(sample, request, refresh=refresh)
