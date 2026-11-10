from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any
import unicodedata


MISMATCH_SIGNALS = (
    "claim_contradicted",
    "identity_mismatch",
    "attribution_mismatch",
    "temporal_mismatch",
    "jurisdiction_mismatch",
    "location_mismatch",
    "numeric_mismatch",
    "manipulated_overlay",
)


def same_report_slot_checks(context: dict[str, Any] | None) -> tuple[bool, list[dict[str, Any]]]:
    """Slot checks made against the aligned original report only.

    External slot evidence counts only when the verifier found a report of the same
    specific event; topic-level hits and other sources' estimates never decide the label.
    """

    context = context or {}
    info = context.get("same_report") or {}
    found = info.get("found") is True
    report_ids = {str(value) for value in info.get("evidence_ids") or []}
    checks = [
        item for item in context.get("slot_checks") or []
        if isinstance(item, dict)
        and (not report_ids or str(item.get("evidence_id", "")) in report_ids)
    ]
    return found, checks


_NAME_FIELDS = {"person", "organization", "location", "attribution"}
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(value: Any) -> list[str]:
    return [item.replace(",", "") for item in _NUMBER_RE.findall(str(value or ""))]


def _name_tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    return [token for token in re.findall(r"[a-z一-鿿]+", text) if len(token) >= 3]


def names_similar(left: Any, right: Any, threshold: float = 0.75) -> bool:
    """True for sound-alike or partial spellings (speech-recognition variants)."""
    left_tokens, right_tokens = _name_tokens(left), _name_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return any(
        a == b or (len(a) >= 4 and len(b) >= 4 and SequenceMatcher(None, a, b).ratio() >= threshold)
        for a in left_tokens for b in right_tokens
    )


def is_hard_slot_mismatch(item: dict[str, Any]) -> bool:
    """A central mismatch that cannot be explained by transcription or paraphrase.

    Numbers must differ (any field label, since models often file a statistic under
    "quote"); names must be genuinely different rather than sound-alike variants.
    Dates, quotation wording, and ad-hoc fields never trigger the guard.
    """

    if item.get("status") != "mismatch" or item.get("materiality") != "central":
        return False
    field = str(item.get("field") or "").casefold()
    news, evidence = item.get("news_value"), item.get("evidence_value")
    if field not in {"time", "date"}:
        news_numbers, evidence_numbers = _numbers(news), _numbers(evidence)
        # A news number absent from the source value ("199%" vs "192%"); extra context
        # numbers in the source ("47% (2010-2022)") are not a mismatch.
        if news_numbers and evidence_numbers and not set(news_numbers) <= set(evidence_numbers):
            return True
    if field in _NAME_FIELDS:
        return bool(_name_tokens(news)) and bool(_name_tokens(evidence)) and not names_similar(
            news, evidence
        )
    return False


def external_slot_mismatches(context: dict[str, Any] | None) -> list[str]:
    found, checks = same_report_slot_checks(context)
    if not found:
        return []
    return [
        f"{item.get('field')}: {item.get('news_value')} vs {item.get('evidence_value')}"
        for item in checks
        if is_hard_slot_mismatch(item)
    ]


def deterministic_verdict(
    fused_report: dict[str, Any],
    context: dict[str, Any] | None = None,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate fixed evidence signals without another model call.

    The policy is intentionally conservative: central, confident contradictory
    evidence yields FAKE; supported evidence with no positive contradiction
    yields REAL; otherwise it abstains as ERROR. Validation may tune only the
    confidence threshold, never the rule using held-out test labels.
    """

    signals = fused_report.get("signals") or {}
    if signals.get("stage_valid") is not True:
        return {"binary_label": "ERROR", "reason": "invalid fused evidence stage"}

    confidence = signals.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    positive_mismatches = [key for key in MISMATCH_SIGNALS if signals.get(key) is True]
    materiality = signals.get("materiality", "unknown")

    aligned_refutations = []
    for claim in (context or {}).get("claims", []):
        if not isinstance(claim, dict) or claim.get("status") != "refuted":
            continue
        alignment = [
            claim.get("subject_match"), claim.get("predicate_match"),
            claim.get("time_match"), claim.get("jurisdiction_match"),
        ]
        if all(value is not False for value in alignment):
            aligned_refutations.append(str(claim.get("claim", "")))

    context = context or {}
    same_report, slot_checks = same_report_slot_checks(context)
    external_central_mismatches = external_slot_mismatches(context) + [
        f"number: {','.join(item.get('news_numbers') or [])} vs {','.join(item.get('evidence_numbers') or [])}"
        for item in context.get("numeric_diffs") or [] if isinstance(item, dict)
    ]
    if aligned_refutations or external_central_mismatches:
        return {
            "binary_label": "FAKE",
            "reason": "aligned external evidence refutes a material slot",
            "positive_signals": positive_mismatches,
            "aligned_refutations": aligned_refutations,
            "external_slot_mismatches": external_central_mismatches,
            "confidence": confidence,
        }

    if (
        confidence >= confidence_threshold
        and materiality == "central"
        and positive_mismatches
    ):
        return {
            "binary_label": "FAKE",
            "reason": "central structured contradiction",
            "positive_signals": positive_mismatches,
            "aligned_refutations": aligned_refutations,
            "confidence": confidence,
        }

    contradictions = [signals.get(key) for key in MISMATCH_SIGNALS]
    externally_confirmed = (
        same_report
        and context.get("status") == "supported"
        and bool(slot_checks)
        and all(item.get("status") in {"match", "not_found"} for item in slot_checks)
        and any(item.get("status") == "match" for item in slot_checks)
    )
    if (
        (signals.get("claim_supported") is True or externally_confirmed)
        and all(value is not True for value in contradictions)
        and (externally_confirmed or all(value is False for value in contradictions))
        and confidence >= confidence_threshold
    ):
        return {
            "binary_label": "REAL",
            "reason": "supported with no positive material contradiction",
            "positive_signals": [],
            "aligned_refutations": [],
            "confidence": confidence,
        }

    return {
        "binary_label": "ERROR",
        "reason": "insufficient structured evidence for a fail-closed verdict",
        "positive_signals": positive_mismatches,
        "aligned_refutations": aligned_refutations,
        "confidence": confidence,
    }
