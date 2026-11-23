"""Freeze the L4 scorer on the tuning set and apply it, unchanged, to the held-out test set."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .scorer import _best_threshold, _inner_select, _fit_model

# Files whose content determines the evidence and features. If any changes after the
# freeze, the holdout no longer measures the frozen system, so extraction refuses to run.
FROZEN_CODE = ("media.py", "media_worker.py", "detectors.py", "client.py", "prompts.py",
               "agents.py", "features.py", "scorer.py")
PRIMARY_SYSTEM = "v2 scorer (all features)"


def code_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN_CODE}


def verify_frozen(frozen: dict) -> None:
    current = code_hashes()
    changed = [name for name, digest in frozen["code_hashes"].items() if current.get(name) != digest]
    if changed:
        raise SystemExit(f"v2 code changed since the freeze ({', '.join(changed)}); refusing to run the holdout")


def freeze(rows: list[dict], feature_names: tuple[str, ...], groups: list[str], seed: int = 42) -> dict[str, Any]:
    """Fit the final scorer on all tuning rows (C and threshold from grouped inner CV)."""
    X = np.array([[row["features"][name] for name in feature_names] for row in rows], dtype=float)
    y = np.array([1 if row["gold"] == "FAKE" else 0 for row in rows])
    C, threshold = _inner_select(X, y, np.array(groups), seed + 99)
    model = _fit_model(X, y, C)
    single = np.array([row["features"]["single_p_fake"] for row in rows], dtype=float)
    return {
        "primary_system": PRIMARY_SYSTEM,
        "features": list(feature_names), "C": C, "threshold": threshold,
        "scaler_mean": model[0].mean_.tolist(), "scaler_scale": model[0].scale_.tolist(),
        "coef": model[-1].coef_[0].tolist(), "intercept": float(model[-1].intercept_[0]),
        "single_call_threshold": _best_threshold(y, single),
        "n_train": int(len(y)), "code_hashes": code_hashes(),
    }


def apply_frozen(frozen: dict, rows: list[dict]) -> tuple[list[float], list[str]]:
    X = np.array([[row["features"][name] for name in frozen["features"]] for row in rows], dtype=float)
    z = (X - np.array(frozen["scaler_mean"])) / np.array(frozen["scaler_scale"])
    logit = z @ np.array(frozen["coef"]) + frozen["intercept"]
    prob = 1 / (1 + np.exp(-logit))
    return prob.tolist(), ["FAKE" if p >= frozen["threshold"] else "REAL" for p in prob]


def bootstrap_ci(gold: list[str], pred: list[str], metric, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    gold_a, pred_a = np.array(gold, dtype=object), np.array(pred, dtype=object)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(gold), len(gold))
        values.append(metric(gold_a[idx].tolist(), pred_a[idx].tolist()))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def mcnemar(gold: list[str], a: list[str], b: list[str]) -> dict[str, Any]:
    only_a = sum(1 for g, x, y in zip(gold, a, b) if x == g and y != g)
    only_b = sum(1 for g, x, y in zip(gold, a, b) if y == g and x != g)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_exact": p}
