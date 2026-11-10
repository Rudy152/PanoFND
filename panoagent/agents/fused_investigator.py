from __future__ import annotations

from .. import prompts
from ..schemas import ModelRequest, NewsSample, StageResult
from .base import JsonAgent


class FusedInvestigator(JsonAgent):
    """One-pass multimodal evidence extraction for the fast ablation profile.

    The full profile remains the reproducibility baseline. This stage collapses
    Organizer, Unimodal, Relation, and Sufficiency into one schema-constrained
    call so media is uploaded and encoded once per sample.
    """

    stage = "fused_investigator"
    # This stage is the speed path. A second model call to repair a long,
    # truncated report costs more than failing closed and obscures call counts.
    enable_json_repair = False
    required_keys = frozenset({
        "central_claim", "case_file", "modalities", "relations", "signals", "retrieval",
    })

    _signal_flags = (
        "claim_supported", "claim_contradicted", "identity_mismatch",
        "attribution_mismatch", "temporal_mismatch", "jurisdiction_mismatch",
        "location_mismatch", "numeric_mismatch", "manipulated_overlay",
    )

    def _parse_and_validate(self, raw: str) -> dict:
        data = super()._parse_and_validate(raw)
        for key in ("case_file", "modalities", "relations", "signals", "retrieval"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"fused_investigator `{key}` must be an object")
        atomic_checks = data.setdefault("atomic_checks", [])
        if not isinstance(atomic_checks, list):
            raise ValueError("fused_investigator `atomic_checks` must be an array")
        signals = data["signals"]
        normalized_missing: list[str] = []
        # Accept the more verbose per-signal object shape emitted by some
        # schema-following models, but normalize it to the flat contract used
        # by both deterministic and model-based judges.
        nested_materialities: list[str] = []
        nested_confidences: list[float] = []
        nested_evidence_ids: list[str] = []
        for key in self._signal_flags:
            if key not in signals:
                signals[key] = "unknown"
                normalized_missing.append(key)
            value = signals.get(key)
            if isinstance(value, dict):
                materiality = value.get("materiality")
                if materiality in {"none", "minor", "central", "unknown"}:
                    nested_materialities.append(materiality)
                confidence = value.get("confidence")
                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                    nested_confidences.append(float(confidence))
                for evidence_id in value.get("evidence_ids") or []:
                    evidence_id = str(evidence_id)
                    if evidence_id not in nested_evidence_ids:
                        nested_evidence_ids.append(evidence_id)
                value = value.get("value", "unknown")
                signals[key] = value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                signals[key] = value.lower() == "true"
            elif isinstance(value, str) and value.lower() in {
                "suspicious", "uncertain", "insufficient", "undetermined", "n/a",
            }:
                # These are epistemic statuses, not positive boolean evidence.
                # Preserve fail-closed semantics by treating them as unknown.
                signals[key] = "unknown"
                normalized_missing.append(key)
            if signals.get(key) not in {True, False, "unknown"}:
                raise ValueError(
                    f"fused_investigator signal `{key}` must be true|false|unknown"
                )
        if "materiality" not in signals:
            if "central" in nested_materialities:
                signals["materiality"] = "central"
            elif "minor" in nested_materialities:
                signals["materiality"] = "minor"
            elif nested_materialities and all(item == "none" for item in nested_materialities):
                signals["materiality"] = "none"
            else:
                signals["materiality"] = "unknown"
                normalized_missing.append("materiality")
        if signals.get("materiality") not in {"none", "minor", "central", "unknown"}:
            raise ValueError(
                "fused_investigator materiality must be none|minor|central|unknown"
            )
        if "confidence" not in signals and nested_confidences:
            signals["confidence"] = max(nested_confidences)
        confidence = signals.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("fused_investigator confidence must be between 0 and 1")
        # Schema validity is determined locally rather than trusted to the model.
        signals["stage_valid"] = True
        if "evidence_ids" not in signals:
            signals["evidence_ids"] = nested_evidence_ids
        signals["normalized_missing_signals"] = normalized_missing
        return data

    def run(self, sample: NewsSample, refresh: bool = False) -> StageResult:
        request = ModelRequest(
            task_name=self.stage,
            system_prompt=prompts.FUSED_SYSTEM,
            user_prompt=prompts.FUSED_USER.format(
                asr_tolerance=prompts.ASR_TOLERANCE,
                modalities=sample.modality_key,
                publish_time=sample.publish_time or "unknown",
                content=sample.detector_text(),
            ),
            max_new_tokens=self.max_new_tokens,
            **self.media_kwargs(sample),
        )
        result = self.run_request(sample, request, refresh=refresh)
        relations = result.data.get("relations")
        if isinstance(relations, dict):
            available = set(sample.modalities)
            if sample.title.strip() or sample.text.strip():
                available.add("text")
            ordered_pairs = (
                ("text", "image"), ("text", "audio"), ("text", "video"),
                ("image", "audio"), ("image", "video"), ("audio", "video"),
            )
            allowed = {
                f"{left}-{right}" for left, right in ordered_pairs
                if {left, right} <= available
            }
            result.data["relations"] = {
                key: value for key, value in relations.items() if key in allowed
            }
            payload = result.to_dict()
            payload["request_fingerprint"] = self._request_fingerprint(request)
            self.cache.save(sample.id, self.stage, payload)
        return result
