from __future__ import annotations

from copy import deepcopy
from typing import Any

from .evaluation import binary_metrics


def fake_score(row: dict[str, Any]) -> float | None:
    judge = ((row.get("stages") or {}).get("judge") or {}).get("data") or {}
    label = str(judge.get("binary_label") or row.get("prediction") or "").upper()
    confidence = judge.get("confidence")
    if label not in {"REAL", "FAKE"} or not isinstance(confidence, (int, float)):
        return None
    confidence = min(1.0, max(0.0, float(confidence)))
    return confidence if label == "FAKE" else 1.0 - confidence


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = fake_score(row)
        if score is None:
            continue
        item = dict(row)
        item["prediction"] = "FAKE" if score >= threshold else "REAL"
        scored.append(item)
    return binary_metrics(scored)


def fit_threshold(
    rows: list[dict[str, Any]],
    min_fake_recall: float = 0.5,
) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("gold") in {"REAL", "FAKE"} and fake_score(row) is not None
    ]
    if not usable:
        raise ValueError("No rows contain a valid Judge label and confidence")
    scores = sorted({fake_score(row) for row in usable if fake_score(row) is not None})
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    ranked: list[tuple[tuple[float, float, float], float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = _threshold_metrics(usable, threshold)
        fake_recall = metrics["per_class"]["FAKE"]["recall"]
        if fake_recall < min_fake_recall:
            continue
        rank = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(threshold - 0.5),
        )
        ranked.append((rank, threshold, metrics))
    if not ranked:
        raise ValueError("No threshold satisfies the requested FAKE-recall constraint")
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    return {
        "threshold": threshold,
        "samples": len(usable),
        "min_fake_recall": min_fake_recall,
        "validation_metrics": metrics,
    }


def fit_calibration(
    rows: list[dict[str, Any]],
    min_fake_recall: float = 0.5,
    per_modality: bool = False,
    min_group_size: int = 30,
) -> dict[str, Any]:
    global_fit = fit_threshold(rows, min_fake_recall=min_fake_recall)
    by_modality: dict[str, Any] = {}
    if per_modality:
        modalities = sorted({str(row.get("modality_key", "unknown")) for row in rows})
        for modality in modalities:
            group = [row for row in rows if str(row.get("modality_key", "unknown")) == modality]
            labels = {row.get("gold") for row in group}
            if len(group) < min_group_size or not {"REAL", "FAKE"} <= labels:
                continue
            try:
                by_modality[modality] = fit_threshold(
                    group, min_fake_recall=min_fake_recall
                )
            except ValueError:
                continue
    return {
        "schema_version": "1.0",
        "method": "judge_confidence_threshold",
        "fit_partition_requirement": "source-grouped validation only",
        "global": global_fit,
        "by_modality": by_modality,
    }


def apply_calibration(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    global_threshold = float(calibration["global"]["threshold"])
    by_modality = calibration.get("by_modality") or {}
    output: list[dict[str, Any]] = []
    for source in rows:
        row = deepcopy(source)
        score = fake_score(row)
        if score is None:
            output.append(row)
            continue
        modality = str(row.get("modality_key", "unknown"))
        threshold = float((by_modality.get(modality) or {}).get("threshold", global_threshold))
        original = row.get("prediction")
        row["prediction"] = "FAKE" if score >= threshold else "REAL"
        row["calibration"] = {
            "method": "judge_confidence_threshold",
            "fake_score": score,
            "threshold": threshold,
            "uncalibrated_prediction": original,
        }
        output.append(row)
    return output
