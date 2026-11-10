from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any

from .calibration import fake_score
from .evaluation import binary_metrics


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def _confidence(row: dict[str, Any]) -> float | None:
    judge = ((row.get("stages") or {}).get("judge") or {}).get("data") or {}
    value = judge.get("confidence")
    return float(value) if isinstance(value, (int, float)) else None


def _tokens(row: dict[str, Any]) -> float:
    value = ((row.get("model_metadata") or {}).get("usage") or {}).get("total_tokens")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _combined_usage(*rows: dict[str, Any]) -> dict[str, float]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(
            float((((row.get("model_metadata") or {}).get("usage") or {}).get(key) or 0))
            for row in rows
        )
        for key in keys
    }


def route_to_full(fused_row: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, str]:
    kind = str(rule.get("kind"))
    prediction = fused_row.get("prediction")
    if rule.get("fallback_on_error") and prediction not in {"REAL", "FAKE"}:
        return True, "fused_error"
    if kind == "all_full":
        return True, "all_full"
    if kind == "all_fused":
        return False, "all_fused"

    parts = set(str(fused_row.get("modality_key", "")).split("+"))
    requested = set(rule.get("modalities") or [])
    modality_match = bool(parts & requested)
    if kind == "contains_any":
        return modality_match, "modality_match" if modality_match else "fused_default"
    if kind == "single_modality":
        matched = len(parts) == 1
        return matched, "single_modality" if matched else "fused_default"
    if kind == "multimodal":
        matched = len(parts) > 1
        return matched, "multimodal" if matched else "fused_default"

    confidence = _confidence(fused_row)
    threshold = float(rule.get("confidence_threshold", 0.0))
    low_confidence = confidence is None or confidence < threshold
    if kind == "low_confidence":
        return low_confidence, "low_confidence" if low_confidence else "fused_default"
    if kind == "contains_any_or_low_confidence":
        matched = modality_match or low_confidence
        if modality_match:
            reason = "modality_match"
        elif low_confidence:
            reason = "low_confidence"
        else:
            reason = "fused_default"
        return matched, reason
    raise ValueError(f"Unknown routing rule kind: {kind}")


def apply_hybrid_routing(
    fused_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    fused = _latest_by_id(fused_rows)
    full = _latest_by_id(full_rows)
    if set(fused) != set(full):
        missing_full = sorted(set(fused) - set(full))
        missing_fused = sorted(set(full) - set(fused))
        raise ValueError(
            f"Prediction ID mismatch; missing full={missing_full[:5]}, "
            f"missing fused={missing_fused[:5]}"
        )
    output: list[dict[str, Any]] = []
    for sample_id, fused_row in fused.items():
        full_row = full[sample_id]
        if fused_row.get("gold") != full_row.get("gold"):
            raise ValueError(f"Gold label mismatch for sample {sample_id}")
        use_full, reason = route_to_full(fused_row, rule)
        pre_routed_full = use_full and reason in {
            "all_full", "modality_match", "single_modality", "multimodal",
        }
        full_valid = full_row.get("prediction") in {"REAL", "FAKE"} and not full_row.get("error")
        if use_full and full_valid:
            selected_profile = "full"
            selected = deepcopy(full_row)
        else:
            selected_profile = "fused"
            selected = deepcopy(fused_row)
            if use_full:
                reason = "full_error_fallback_to_fused"
        executed_rows = []
        executed_profiles = []
        if not pre_routed_full or (use_full and not full_valid):
            executed_rows.append(fused_row)
            executed_profiles.append("fused")
        if use_full:
            executed_rows.append(full_row)
            executed_profiles.append("full")
        selected.setdefault("model_metadata", {})["usage"] = _combined_usage(*executed_rows)
        selected["routing"] = {
            "profile": selected_profile,
            "attempted_full": use_full,
            "executed_profiles": executed_profiles,
            "reason": reason,
            "rule": deepcopy(rule),
            "fused_prediction": fused_row.get("prediction"),
            "full_prediction": full_row.get("prediction"),
        }
        output.append(selected)
    return output


def candidate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {"name": "all_fused", "kind": "all_fused", "fallback_on_error": False},
        {"name": "all_full", "kind": "all_full", "fallback_on_error": True},
        {"name": "single_modality_to_full", "kind": "single_modality", "fallback_on_error": True},
        {"name": "multimodal_to_full", "kind": "multimodal", "fallback_on_error": True},
    ]
    modalities = ("audio", "image", "video")
    modality_sets = [combo for size in (1, 2) for combo in combinations(modalities, size)]
    for combo in modality_sets:
        label = "_or_".join(combo)
        rules.append({
            "name": f"{label}_to_full", "kind": "contains_any",
            "modalities": list(combo), "fallback_on_error": True,
        })
    thresholds = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    for threshold in thresholds:
        suffix = str(threshold).replace(".", "_")
        rules.append({
            "name": f"confidence_below_{suffix}_to_full", "kind": "low_confidence",
            "confidence_threshold": threshold, "fallback_on_error": True,
        })
        for combo in modality_sets:
            label = "_or_".join(combo)
            rules.append({
                "name": f"{label}_or_confidence_below_{suffix}_to_full",
                "kind": "contains_any_or_low_confidence",
                "modalities": list(combo), "confidence_threshold": threshold,
                "fallback_on_error": True,
            })
    return rules


def select_hybrid_routing(
    fused_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    min_token_reduction: float = 0.50,
) -> dict[str, Any]:
    if not 0 <= min_token_reduction < 1:
        raise ValueError("min_token_reduction must be in [0, 1)")
    full_total = sum(_tokens(row) for row in _latest_by_id(full_rows).values())
    leaderboard: list[dict[str, Any]] = []
    for rule in candidate_rules():
        routed = apply_hybrid_routing(fused_rows, full_rows, rule)
        metrics = binary_metrics(routed)
        routed_full = sum(1 for row in routed if row["routing"]["attempted_full"])
        total_tokens = sum(_tokens(row) for row in routed)
        reduction = 1.0 - total_tokens / full_total if full_total else 0.0
        leaderboard.append({
            "rule": rule,
            "metrics": metrics,
            "routed_full": routed_full,
            "routed_fused": len(routed) - routed_full,
            "full_fraction": routed_full / len(routed) if routed else 0.0,
            "total_tokens": total_tokens,
            "token_reduction_vs_full": reduction,
        })
    feasible = [
        row for row in leaderboard
        if row["token_reduction_vs_full"] >= min_token_reduction
    ]
    if not feasible:
        raise ValueError("No candidate satisfies the minimum token-reduction constraint")
    best = max(
        feasible,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["end_to_end_accuracy"],
            row["metrics"]["coverage"],
            -row["full_fraction"],
        ),
    )
    ranked = sorted(
        leaderboard,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["balanced_accuracy"],
            row["token_reduction_vs_full"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "1.0",
        "method": "predeclared_validation_routing_candidates",
        "selection_partition_requirement": "source-grouped validation only",
        "objective": "maximize macro_f1 subject to minimum token reduction",
        "min_token_reduction": min_token_reduction,
        "selected": best,
        "leaderboard": ranked,
    }
