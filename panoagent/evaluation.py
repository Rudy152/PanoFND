from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _confusion_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        gold: {prediction: 0 for prediction in ("REAL", "FAKE", "ERROR")}
        for gold in ("REAL", "FAKE")
    }
    for row in records:
        gold = row.get("gold")
        if gold not in counts:
            continue
        prediction = row.get("prediction")
        prediction = prediction if prediction in {"REAL", "FAKE"} else "ERROR"
        counts[gold][prediction] += 1
    return {
        "rows": "gold",
        "columns": ["REAL", "FAKE", "ERROR"],
        "counts": counts,
        "real_as_fake": counts["REAL"]["FAKE"],
        "fake_as_real": counts["FAKE"]["REAL"],
        "real_errors": counts["REAL"]["ERROR"],
        "fake_errors": counts["FAKE"]["ERROR"],
    }


def _majority_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        label: sum(1 for row in records if row.get("gold") == label)
        for label in ("REAL", "FAKE")
    }
    total = counts["REAL"] + counts["FAKE"]
    label = "FAKE" if counts["FAKE"] >= counts["REAL"] else "REAL"
    accuracy = _safe_div(counts[label], total)
    majority_precision = accuracy
    majority_f1 = _safe_div(2 * majority_precision, majority_precision + 1.0)
    per_class = {
        current: {
            "precision": majority_precision if current == label else 0.0,
            "recall": 1.0 if current == label and counts[current] else 0.0,
            "f1": majority_f1 if current == label else 0.0,
            "support": counts[current],
        }
        for current in ("REAL", "FAKE")
    }
    return {
        "label": label,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 if total else 0.0,
        "macro_f1": majority_f1 / 2.0,
        "per_class": per_class,
        "class_counts": counts,
    }


def binary_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    eligible = [row for row in records if row.get("gold") in {"REAL", "FAKE"}]
    usable = [row for row in records if row.get("gold") in {"REAL", "FAKE"} and row.get("prediction") in {"REAL", "FAKE"}]
    total = len(usable)
    errors = sum(1 for row in eligible if row.get("prediction") not in {"REAL", "FAKE"})
    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in ("REAL", "FAKE"):
        tp = sum(1 for row in usable if row["gold"] == label and row["prediction"] == label)
        fp = sum(1 for row in usable if row["gold"] != label and row["prediction"] == label)
        fn = sum(1 for row in usable if row["gold"] == label and row["prediction"] != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        recalls.append(recall)
        f1s.append(f1)
    correct = sum(1 for row in usable if row["gold"] == row["prediction"])
    majority = _majority_baseline(eligible)
    end_to_end_accuracy = _safe_div(correct, len(eligible))
    return {
        "evaluated": total,
        "eligible": len(eligible),
        "coverage": _safe_div(total, len(eligible)),
        "parse_or_runtime_errors": errors,
        "accuracy": _safe_div(correct, total),
        "end_to_end_accuracy": end_to_end_accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "confusion_matrix": _confusion_matrix(eligible),
        "majority_baseline": majority,
        "accuracy_above_majority_baseline": end_to_end_accuracy - majority["accuracy"],
    }


def _row_has_stage_error(row: dict[str, Any]) -> bool:
    stages = row.get("stages") or {}
    return isinstance(stages, dict) and any(
        isinstance(result, dict) and bool(result.get("error"))
        for result in stages.values()
    )


def _strict_end_to_end(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in records if row.get("gold") in {"REAL", "FAKE"}]
    complete = [
        row for row in eligible
        if row.get("prediction") in {"REAL", "FAKE"}
        and not row.get("error")
        and not _row_has_stage_error(row)
    ]
    correct = sum(1 for row in complete if row["gold"] == row["prediction"])
    return {
        "eligible": len(eligible),
        "stage_complete_predictions": len(complete),
        "coverage": _safe_div(len(complete), len(eligible)),
        "correct": correct,
        "accuracy_errors_as_wrong": _safe_div(correct, len(eligible)),
    }


def _slice_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        row for row in records
        if row.get("prediction") in {"REAL", "FAKE"}
        and not row.get("error")
        and not _row_has_stage_error(row)
    ]
    result = binary_metrics(records) | _stage_error_metrics(records)
    result["stage_complete_subset"] = binary_metrics(complete)
    result["strict_end_to_end"] = _strict_end_to_end(records)
    return result


def _stage_error_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    counts: dict[str, int] = defaultdict(int)
    records_with_errors = 0
    for row in records:
        row_has_error = False
        stages = row.get("stages") or {}
        if isinstance(stages, dict):
            for name, result in stages.items():
                if isinstance(result, dict) and result.get("error"):
                    counts[str(name)] += 1
                    row_has_error = True
        if row_has_error:
            records_with_errors += 1
    return {
        "records_with_stage_errors": records_with_errors,
        "stage_complete_records": len(records) - records_with_errors,
        "stage_error_counts": dict(sorted(counts.items())),
    }


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    # A resumed run may append a successful retry after an earlier error.
    # Evaluate the latest record per sample so retries do not inflate N.
    latest_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(records):
        key = str(row.get("id") or f"__row_{index}")
        latest_by_id[key] = row
    records = list(latest_by_id.values())

    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        slices[str(row.get("modality_key", "unknown"))].append(row)
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in records:
        usage = (row.get("model_metadata") or {}).get("usage") or {}
        for key in token_usage:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                token_usage[key] += value
    overall = _slice_metrics(records)
    return {
        "overall": overall,
        "api_usage": token_usage,
        "by_modality_combination": {
            key: _slice_metrics(values)
            for key, values in sorted(slices.items())
        },
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records
