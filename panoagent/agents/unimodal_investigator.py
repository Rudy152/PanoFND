from __future__ import annotations

import json

from .. import prompts
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class UnimodalInvestigator(JsonAgent):
    stage = "unimodal_investigator"
    required_keys = frozenset({"modalities"})

    def run(self, sample: NewsSample, case_file: dict, refresh: bool = False) -> StageResult:
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.UNIMODAL_SYSTEM,
            user_prompt=prompts.UNIMODAL_USER.format(
                case_file=json.dumps(case_file, ensure_ascii=False),
                content=sample.detector_text(),
            ),
            max_new_tokens=self.max_new_tokens,
            **self.media_kwargs(sample),
        )
        return self.run_request(sample, request, refresh=refresh)
