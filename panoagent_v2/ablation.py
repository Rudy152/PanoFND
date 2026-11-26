"""Offline module ablations of v2 on the 943 tuning items (tables A, B, C of artifacts/ABLATION_PLAN_20261124.md).

    python -m panoagent_v2.ablation --dataset fakenewsDataSet \
        --features runs/v2_tuning943_features_20261121.jsonl --output-dir runs/v2_ablation_abc_20261124

No API calls. Refuses any row that is not a tuning item. The frozen v2 modules are only imported.
Every prediction is out-of-fold under nested, source-grouped CV, repeated over several split seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from panoagent.dataset import PanoFNDDataset, load_split

from .cli import _load_rows
from .features import AGENT_FEATURES, DETECTOR_FEATURES, FEATURES, build_features
from .scorer import _best_threshold, _fit_model, _inner_select, _splitter, binary_metrics, cross_validated

ROOT = Path(__file__).resolve().parents[1]
TUNING_SPLIT, TUNING_PARTITION = ROOT / "splits" / "half_eval_seed42.json", "train"
SEEDS = tuple(range(42, 52))  # 42 is the split used in the published report
HAS = tuple(f for f in FEATURES if f.startswith("has_"))
GROUPS_B = {
    "B1 consistency": ("frac_conflict", "frac_unrelated", "any_conflict"),
    "B2 slots": ("slot_mismatch", "slot_match"),
    "B3 integrity": ("frac_manipulated", "any_manipulated", "frac_integrity_uncertain"),
    "B4 unimodal review": ("uni_contradictory",),
}
PAIR_STAGES = ("pair_text_image", "pair_text_audio", "pair_image_audio", "pair_text_video", "pair_image_video")
GBDT_GRID = ((50, 2), (100, 2), (100, 3))  # (trees, depth)


def check_tuning_only(ids: list[str]) -> None:
    outside = set(ids) - set(load_split(TUNING_SPLIT, TUNING_PARTITION))
    if outside:
        raise SystemExit(f"{len(outside)} rows are not tuning items (e.g. {sorted(outside)[:3]}); refusing to run")


def add_onehot(rows: list[dict]) -> tuple[str, ...]:
    keys = sorted({r["modality_key"] for r in rows})
    for r in rows:
        for k in keys:
            r["features"]["mk_" + k] = float(r["modality_key"] == k)
    return tuple("mk_" + k for k in keys)


def rebuild_without(row: dict, stage: str | None) -> dict[str, float]:
    calls = [c for c in row["calls"] if c["stage"] != stage]
    return build_features(tuple(row["modalities"]), row["single_call"], calls, row["detector_scores"])


def rule(row: dict) -> bool:
    f = row["features"]
    return bool(f["any_conflict"] or f["any_manipulated"] or f["uni_contradictory"])


# --- predictors: (rows, groups, seed) -> (probabilities, predictions) -------------------------------------------

def lr(names: tuple[str, ...]):
    def run(rows, groups, seed):
        r = cross_validated(rows, names, groups, seed=seed)
        return np.array(r["probabilities"]), np.array(r["predictions"])
    return run


def _xy(rows, names):
    X = np.array([[r["features"][n] for n in names] for r in rows], dtype=float)
    return X, np.array([1 if r["gold"] == "FAKE" else 0 for r in rows])


def lr_fixed_half(names):
    """Same model and C selection as v2, but the decision threshold stays at 0.5."""
    def run(rows, groups, seed):
        X, y = _xy(rows, names)
        g, prob = np.array(groups), np.zeros(len(y))
        for fold, (tr, te) in enumerate(_splitter(5, seed).split(X, y, g)):
            C, _ = _inner_select(X[tr], y[tr], g[tr], seed + fold)
            prob[te] = _fit_model(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
        return prob, np.where(prob >= 0.5, "FAKE", "REAL")
    return run


def single_threshold(per_modality: bool):
    """Single-call score with one threshold, or one threshold per modality combination."""
    def run(rows, groups, seed):
        X, y = _xy(rows, ("single_p_fake",))
        s, keys = X[:, 0], np.array([r["modality_key"] for r in rows])
        pred = np.empty(len(y), dtype=object)
        # scores are shifted by their threshold so that pooled AUROC reflects the calibration used
        shifted = np.zeros(len(y))
        for tr, te in _splitter(5, seed).split(X, y, np.array(groups)):
            for key in (sorted(set(keys)) if per_modality else [None]):
                a, b = (tr, te) if key is None else (tr[keys[tr] == key], te[keys[te] == key])
                if len(b) == 0:
                    continue
                t = _best_threshold(y[a], s[a])
                pred[b] = np.where(s[b] >= t, "FAKE", "REAL")
                shifted[b] = s[b] - t if per_modality else s[b]
        return shifted, pred
    return run


def _gbdt(Xt, yt, trees, depth, Xv):
    w = np.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    model = GradientBoostingClassifier(n_estimators=trees, max_depth=depth, learning_rate=0.05,
                                       subsample=0.8, random_state=0)
    return model.fit(Xt, yt, sample_weight=w).predict_proba(Xv)[:, 1]


def gbdt(names):
    def run(rows, groups, seed):
        X, y = _xy(rows, names)
        g, prob, pred = np.array(groups), np.zeros(len(y)), np.empty(len(y), dtype=object)
        for fold, (tr, te) in enumerate(_splitter(5, seed).split(X, y, g)):
            inner = list(_splitter(4, seed + fold + 1).split(X[tr], y[tr], g[tr]))
            best = None
            for trees, depth in GBDT_GRID:
                oof = np.zeros(len(tr))
                for a, b in inner:
                    oof[b] = _gbdt(X[tr][a], y[tr][a], trees, depth, X[tr][b])
                t = _best_threshold(y[tr], oof)
                yy, pp = y[tr], oof >= t
                score = (pp[yy == 1].mean() + (~pp[yy == 0]).mean()) / 2
                if best is None or score > best[0] + 1e-9:
                    best = (score, trees, depth, t)
            prob[te] = _gbdt(X[tr], y[tr], best[1], best[2], X[te])
            pred[te] = np.where(prob[te] >= best[3], "FAKE", "REAL")
        return prob, pred
    return run


def fixed_rule():
    def run(rows, groups, seed):
        p = np.array([float(rule(r)) for r in rows])
        return p, np.where(p > 0, "FAKE", "REAL")
    return run


# --- evaluation -----------------------------------------------------------------------------------------------

class Evaluation:
    """Per-item correctness and score, averaged over split seeds, with source-clustered paired bootstraps."""

    def __init__(self, gold, groups, n_boot=2000, auc_boot=500):
        self.auc_boot = auc_boot
        self.gold = np.array(gold)
        self.y = (self.gold == "FAKE").astype(int)
        names = sorted(set(groups))
        code = {n: i for i, n in enumerate(names)}
        self.cluster = np.array([code[x] for x in groups])
        self.k = len(names)
        rng = np.random.default_rng(0)
        self.weights = np.stack([np.bincount(rng.integers(0, self.k, self.k), minlength=self.k) for _ in range(n_boot)])

    def summarise(self, runs):
        m = [binary_metrics(self.gold.tolist(), list(pred)) for _, pred in runs]
        auc = [roc_auc_score(self.y, p) for p, _ in runs]
        out = {}
        for key in ("balanced_accuracy", "macro_f1", "real_recall", "fake_recall"):
            v = [x[key] for x in m]
            out[key] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "seed42": float(v[0])}
        out["auroc"] = {"mean": float(np.mean(auc)), "min": float(min(auc)), "max": float(max(auc)), "seed42": float(auc[0])}
        return out

    def _ba_parts(self, correct):
        num = np.stack([np.bincount(self.cluster, weights=correct * (self.y == c), minlength=self.k) for c in (0, 1)])
        den = np.stack([np.bincount(self.cluster, weights=(self.y == c).astype(float), minlength=self.k) for c in (0, 1)])
        return num, den

    def paired(self, runs_a, runs_b):
        ca = np.mean([np.array(p) == self.gold for _, p in runs_a], axis=0)
        cb = np.mean([np.array(p) == self.gold for _, p in runs_b], axis=0)
        na, den = self._ba_parts(ca)
        nb, _ = self._ba_parts(cb)
        ba = lambda w: ((w @ (na - nb).T) / (w @ den.T)).mean(axis=-1)
        boot = ba(self.weights.astype(float))
        sa = np.mean([self._rank(p) for p, _ in runs_a], axis=0)
        sb = np.mean([self._rank(p) for p, _ in runs_b], axis=0)
        auc = []
        for w in self.weights[: self.auc_boot]:
            sw = w[self.cluster].astype(float)
            if not (sw[self.y == 0].sum() and sw[self.y == 1].sum()):
                continue  # a resample without one class has no AUROC
            auc.append(roc_auc_score(self.y, sa, sample_weight=sw) - roc_auc_score(self.y, sb, sample_weight=sw))
        ci = lambda v: [float(np.percentile(v, q)) for q in (2.5, 97.5)]
        boot_p = lambda v: float(min(1.0, 2 * (1 + min((np.asarray(v) <= 0).sum(), (np.asarray(v) >= 0).sum())) / (len(v) + 1)))
        return {"ba_diff": float(ba(np.ones((1, self.k)))[0]), "ba_ci95": ci(boot),
                "auc_diff": float(roc_auc_score(self.y, sa) - roc_auc_score(self.y, sb)), "auc_ci95": ci(auc), "auc_p_boot": boot_p(auc), "ba_p_boot": boot_p(boot)}

    @staticmethod
    def _rank(p):
        return np.argsort(np.argsort(p, kind="stable"), kind="stable") / len(p)


def systems_for(rows: list[dict], mk: tuple[str, ...]) -> dict[str, tuple]:
    """name -> (predictor, rows). Column 'plain' is v2's own features, 'onehot' adds the modality combination."""
    without = lambda drop: tuple(f for f in FEATURES if f not in drop)
    agents_has = tuple(f for f in FEATURES if f in AGENT_FEATURES) + HAS
    base = {
        "A0 full v2": FEATURES,
        "A1 -L1 detectors": without(DETECTOR_FEATURES),
        "A2 -L2a single call": without(("single_p_fake",)),
        "A3 -L2b agents": without(AGENT_FEATURES),
        "A5 only L2b agents": agents_has,
        "A6 only L1 detectors": DETECTOR_FEATURES + HAS,
        "A7 only modality": HAS,
    }
    out: dict[str, tuple] = {}
    for name, feats in base.items():
        out[f"{name} | plain"] = (lr(feats), rows)
        out[f"{name} | onehot"] = (lr(mk if name.startswith("A7") else feats + mk), rows)
    out["A4 only L2a single call | plain"] = (single_threshold(False), rows)
    out["A4 only L2a single call | onehot"] = (lr(("single_p_fake",) + mk), rows)
    for name in ("A0 full v2", "A1 -L1 detectors", "A2 -L2a single call", "A3 -L2b agents"):
        f = tuple(x for x in base[name] if x != "failed_calls")
        out[f"{name}, -failed_calls | plain"] = (lr(f), rows)
        out[f"{name}, -failed_calls | onehot"] = (lr(f + mk), rows)
    for label, drop in GROUPS_B.items():
        for tag, feats in (("A0", FEATURES), ("A5", agents_has)):
            f = tuple(x for x in feats if x not in drop)
            out[f"{label} removed from {tag} | plain"] = (lr(f), rows)
            out[f"{label} removed from {tag} | onehot"] = (lr(f + mk), rows)
    for stage in PAIR_STAGES:
        changed = [{**r, "features": {**r["features"], **rebuild_without(r, stage)}} for r in rows]
        for tag, feats in (("A0", FEATURES), ("A5", agents_has)):
            out[f"{stage} removed from {tag} | plain"] = (lr(feats), changed)
            out[f"{stage} removed from {tag} | onehot"] = (lr(feats + mk), changed)
    out["C1 fixed rule | plain"] = (fixed_rule(), rows)
    out["C2 single call, threshold per modality | plain"] = (single_threshold(True), rows)
    out["C3 gradient boosting | plain"] = (gbdt(FEATURES), rows)
    out["C3 gradient boosting | onehot"] = (gbdt(FEATURES + mk), rows)
    out["C4 logistic regression, threshold 0.5 | plain"] = (lr_fixed_half(FEATURES), rows)
    out["C4 logistic regression, threshold 0.5 | onehot"] = (lr_fixed_half(FEATURES + mk), rows)
    return out


def _job(name, predictor, rows, groups, seed):
    return name, seed, predictor(rows, groups, seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="panoagent_v2.ablation")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--jobs", type=int, default=-2)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args(argv)
    rows = _load_rows(args.features)
    check_tuning_only([r["id"] for r in rows])
    if any(r.get("fatal_error") for r in rows):
        raise SystemExit("fatal rows present; complete the extraction first")
    for r in rows:  # cached calls must reproduce the stored features before any are dropped
        if rebuild_without(r, None) != r["features"]:
            raise SystemExit(f"stored features of {r['id']} are not reproducible from its cached calls")
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    groups = [PanoFNDDataset.group_key(samples[r["id"]]) for r in rows]
    mk = add_onehot(rows)
    systems = systems_for(rows, mk)
    seeds = SEEDS[: args.seeds]
    print(f"{len(rows)} tuning rows, {len(set(groups))} source groups, {len(systems)} systems x {len(seeds)} seeds", flush=True)
    done = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(_job)(name, fn, data, groups, seed) for name, (fn, data) in systems.items() for seed in seeds)
    runs: dict[str, list] = {name: [] for name in systems}
    for name, seed, result in done:
        runs[name].append((seed, result))
    runs = {name: [r for _, r in sorted(v, key=lambda t: t[0])] for name, v in runs.items()}
    ev = Evaluation([r["gold"] for r in rows], groups)
    report: dict = {"n": len(rows), "source_groups": ev.k, "seeds": list(seeds), "systems": {}, "paired": {}}
    for name, v in runs.items():
        report["systems"][name] = ev.summarise(v)
    refs = []
    for name in systems:
        body, col = name.rsplit(" | ", 1)
        if body in ("A0 full v2",):
            continue
        if "removed from A5" in body:
            refs.append((f"A5 only L2b agents | {col}", name))
        else:
            refs.append((f"A0 full v2 | {col}", name))
    refs += [("A0 full v2 | onehot", "A0 full v2 | plain"),
             ("A0 full v2 | plain", "A4 only L2a single call | onehot"),
             ("A0 full v2 | plain", "C2 single call, threshold per modality | plain"),
             ("A0 full v2 | onehot", "C2 single call, threshold per modality | plain"),
             ("A5 only L2b agents | onehot", "A4 only L2a single call | onehot"),
             ("A4 only L2a single call | onehot", "A4 only L2a single call | plain")]
    paired = Parallel(n_jobs=args.jobs)(delayed(ev.paired)(runs[a], runs[b]) for a, b in refs)
    for (a, b), value in zip(refs, paired):
        report["paired"][f"{a}  -  {b}"] = value
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_abc.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, s in report["systems"].items():
        b, a = s["balanced_accuracy"], s["auroc"]
        print(f"{name:62} BA {b['mean']:.1%} [{b['min']:.1%},{b['max']:.1%}] s42 {b['seed42']:.1%} F1 {s['macro_f1']['mean']:.1%} "
              f"AUC {a['mean']:.3f} R {s['real_recall']['mean']:.1%} F {s['fake_recall']['mean']:.1%}")
    for key, v in report["paired"].items():
        print(f"{key}: BA {v['ba_diff']:+.1%} [{v['ba_ci95'][0]:+.1%},{v['ba_ci95'][1]:+.1%}] "
              f"AUC {v['auc_diff']:+.3f} [{v['auc_ci95'][0]:+.3f},{v['auc_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
