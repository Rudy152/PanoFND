from __future__ import annotations

from .. import prompts
from ..parsing import parse_json_document
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class EvidenceOrganizer(JsonAgent):
    stage = "evidence_organizer"
    field_names = (
        "entities", "persons", "locations", "times", "events", "text_claims",
        "numeric_claims", "image_observations", "audio_claims", "video_observations",
    )
    required_keys = frozenset(field_names)

    def _parse_and_validate(self, raw: str) -> dict[str, object]:
        """Normalize a common small-model failure without inventing evidence.

        Some checkpoints return a JSON array containing several partial objects.
        Merging their list values and filling absent categories with empty arrays
        is deterministic and preserves every extracted item.
        """
        value = parse_json_document(raw)
        if isinstance(value, dict):
            objects = [value]
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            objects = value
        else:
            raise ValueError("evidence_organizer must return an object or an array of objects")

        merged: dict[str, list[object]] = {key: [] for key in self.field_names}
        recognized = False
        for item in objects:
            for key in self.field_names:
                if key not in item:
                    continue
                recognized = True
                field = item[key]
                if field is None:
                    continue
                if isinstance(field, list):
                    merged[key].extend(field)
                else:
                    merged[key].append(field)
        if not recognized:
            raise ValueError("evidence_organizer response contains none of the required keys")
        return merged

    def run(self, sample: NewsSample, refresh: bool = False) -> StageResult:
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.ORGANIZER_SYSTEM,
            user_prompt=prompts.ORGANIZER_USER.format(
                modalities=sample.modality_key,
                content=sample.detector_text(),
            ),
            max_new_tokens=self.max_new_tokens,
            **self.media_kwargs(sample),
        )
        return self.run_request(sample, request, refresh=refresh)
