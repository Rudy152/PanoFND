from __future__ import annotations

import json

from .. import prompts
from ..decision_policy import decision_policy_for
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class JudgeAgent(JsonAgent):
    stage = "judge"
    required_keys = frozenset({
        "central_claim", "central_claim_status", "binary_label",
        "misinformation_type", "affected_modalities",
        "affected_relations", "confidence", "evidence_ids",
    })

    def run(
        self,
        sample: NewsSample,
        case_file: dict,
        unimodal: dict,
        relation: dict,
        context: dict,
        refresh: bool = False,
    ) -> StageResult:
        decision_policy = decision_policy_for(sample)
        user_template = (
            prompts.JUDGE_IMAGE_USER if decision_policy is not None else prompts.JUDGE_USER
        )
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.JUDGE_SYSTEM,
            user_prompt=user_template.format(
                asr_tolerance=prompts.ASR_TOLERANCE,
                decision_policy=decision_policy or "",
                publish_time=sample.publish_time or "unknown",
                case_file=json.dumps(case_file, ensure_ascii=False),
                unimodal=json.dumps(unimodal, ensure_ascii=False),
                relation=json.dumps(relation, ensure_ascii=False),
                context=json.dumps(context, ensure_ascii=False),
            ),
            max_new_tokens=self.max_new_tokens,
        )
        return self.run_request(sample, request, refresh=refresh)

    def run_structured(
        self,
        sample: NewsSample,
        fused_report: dict,
        context: dict,
        refresh: bool = False,
    ) -> StageResult:
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.JUDGE_SYSTEM,
            user_prompt=prompts.JUDGE_STRUCTURED_USER.format(
                asr_tolerance=prompts.ASR_TOLERANCE,
                publish_time=sample.publish_time or "unknown",
                fused_report=json.dumps(fused_report, ensure_ascii=False),
                context=json.dumps(context, ensure_ascii=False),
            ),
            max_new_tokens=self.max_new_tokens,
        )
        return self.run_request(sample, request, refresh=refresh)
