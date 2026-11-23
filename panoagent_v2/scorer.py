"""L4: small trainable scorer with nested, source-grouped cross-validation.

Every reported prediction is out-of-fold: the model and threshold that score an item
were fitted without that item or any item from its source group.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def binary_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    cm = Counter(zip(gold, pred))
    out: dict[str, Any] = {"n": len(gold)}
    f1s = []
    for label in ("REAL", "FAKE"):
        support = sum(v for (g, _), v in cm.items() if g == label)
        tp = cm[(label, label)]
        predicted = sum(v for (_, p), v in cm.items() if p == label)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        out[f"{label.lower()}_recall"] = recall
    out["balanced_accuracy"] = (out["real_recall"] + out["fake_recall"]) / 2
    out["macro_f1"] = sum(f1s) / 2
    out["accuracy"] = (cm[("REAL", "REAL")] + cm[("FAKE", "FAKE")]) / len(gold) if gold else 0.0
    out["fake_as_real"] = cm[("FAKE", "REAL")]
    out["real_as_fake"] = cm[("REAL", "FAKE")]
    out["errors"] = sum(v for (_, p), v in cm.items() if p not in {"REAL", "FAKE"})
    return out


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best, best_score = 0.5, -1.0
    for t in np.unique(np.concatenate([p, [0.5]])):
        pred = p >= t
        rec_fake = pred[y == 1].mean() if (y == 1).any() else 0.0
        rec_real = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
        score = (rec_fake + rec_real) / 2
        if score > best_score + 1e-12:
            best, best_score = float(t), score
    return best


def _splitter(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _fit_model(X: np.ndarray, y: np.ndarray, C: float):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    model.fit(X, y)
    return model


def _inner_select(X, y, groups, seed) -> tuple[float, float]:
    """Choose C and the threshold on inner out-of-fold probabilities only."""
    folds = list(_splitter(4, seed + 1).split(X, y, groups))
    best = None
    for C in C_GRID:
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = _fit_model(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
        threshold = _best_threshold(y, oof)
        pred = oof >= threshold
        score = ((pred[y == 1]).mean() + (~pred[y == 0]).mean()) / 2
        if best is None or score > best[0] + 1e-9:
            best = (score, C, threshold)
    return best[1], best[2]


def cross_validated(
    rows: list[dict],
    feature_names: tuple[str, ...],
    groups: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    X = np.array([[row["features"][name] for name in feature_names] for row in rows], dtype=float)
    y = np.array([1 if row["gold"] == "FAKE" else 0 for row in rows])
    groups_arr = np.array(groups)
    prob = np.zeros(len(y))
    pred = np.empty(len(y), dtype=object)
    fold_info = []
    for fold, (tr, te) in enumerate(_splitter(n_splits, seed).split(X, y, groups_arr)):
        C, threshold = _inner_select(X[tr], y[tr], groups_arr[tr], seed + fold)
        model = _fit_model(X[tr], y[tr], C)
        prob[te] = model.predict_proba(X[te])[:, 1]
        pred[te] = np.where(prob[te] >= threshold, "FAKE", "REAL")
        fold_info.append({"fold": fold, "C": C, "threshold": round(threshold, 4), "test_n": int(len(te))})
    final_C, final_threshold = _inner_select(X, y, groups_arr, seed + 99)
    final = _fit_model(X, y, final_C)
    coef = final[-1].coef_[0]
    return {
        "probabilities": prob.tolist(),
        "predictions": pred.tolist(),
        "folds": fold_info,
        "final_model": {
            "features": list(feature_names), "C": final_C, "threshold": final_threshold,
            "scaler_mean": final[0].mean_.tolist(), "scaler_scale": final[0].scale_.tolist(),
            "coef_standardized": coef.tolist(), "intercept": float(final[-1].intercept_[0]),
        },
    }


def cross_validated_threshold(rows: list[dict], score: Callable[[dict], float], groups: list[str],
                              n_splits: int = 5, seed: int = 42) -> list[str]:
    """One-feature baseline: the threshold is chosen on training folds only."""
    s = np.array([score(row) for row in rows], dtype=float)
    y = np.array([1 if row["gold"] == "FAKE" else 0 for row in rows])
    pred = np.empty(len(y), dtype=object)
    for tr, te in _splitter(n_splits, seed).split(s.reshape(-1, 1), y, np.array(groups)):
        t = _best_threshold(y[tr], s[tr])
        pred[te] = np.where(s[te] >= t, "FAKE", "REAL")
    return pred.tolist()
