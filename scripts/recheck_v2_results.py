"""Offline recheck of the v2 reports (no API calls, frozen scorer unchanged).

    .venv\\Scripts\\python.exe scripts\\recheck_v2_results.py

Everything here is post-hoc analysis of results that were already final; none of it selects a model.
Writes artifacts/v2_results_20261122/recheck_20261124.json.
"""

from __future__ import annotations

from collections import Counter
import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from panoagent.dataset import PanoFNDDataset
from panoagent_v2.cli import _baseline, _scored_rows
from panoagent_v2.evaluation import apply_frozen, freeze, mcnemar
from panoagent_v2.features import AGENT_FEATURES, FEATURES
from panoagent_v2.scorer import binary_metrics, cross_validated, cross_validated_threshold

DS = str(ROOT / "fakenewsDataSet")
RUNS = ROOT / "runs"
FILES = {"tuning180": "v2_tuning180_features_20261121.jsonl", "tuning943": "v2_tuning943_features_20261121.jsonl",
         "holdout765": "v2_holdout765_features_20261121.jsonl"}
frozen = json.loads((RUNS / "v2_frozen_20261121/frozen.json").read_text(encoding="utf-8"))
v1 = _baseline(str(RUNS / "qwen38_omni_flash_full_1950_20261107.jsonl"))
data = {name: _scored_rows(DS, str(RUNS / f)) for name, f in FILES.items()}
samples = {s.id: s for s in PanoFNDDataset(DS)}
ids = {"train": json.loads((ROOT / "splits/half_eval_seed42.json").read_text(encoding="utf-8"))["partitions"]["train"],
       "test1007": json.loads((ROOT / "splits/half_eval_seed42.json").read_text(encoding="utf-8"))["partitions"]["test"],
       "holdout": json.loads((ROOT / "splits/final_holdout_clean_765_20261031.json").read_text(encoding="utf-8"))["partitions"]["test"]}
ids = {k: [str(i) for i in v] for k, v in ids.items()}
group = PanoFNDDataset.group_key
out: dict = {}


def short(gold, pred):
    m = binary_metrics(gold, pred)
    return {k: round(m[k], 4) if isinstance(m[k], float) else m[k]
            for k in ("n", "balanced_accuracy", "macro_f1", "accuracy", "real_recall", "fake_recall", "errors")}


def failure_kind(error):
    for pattern, kind in (("data_inspection_failed", "content moderation"), ("JSON", "invalid JSON"),
                          ("stream ended", "stream ended early"), ("getaddrinfo", "dns"),
                          ("timed out|10054|_ssl", "upload timeout or connection reset")):
        if re.search(pattern, str(error)):
            return kind
    return "other"


def single_variants(rows):
    p = [r["features"]["single_p_fake"] for r in rows]
    return {
        "v1 hard output": [v1.get(r["id"], "ERROR") for r in rows],
        "v2 inputs, raw TRUE/FALSE": ["FAKE" if "FALSE" in str(r["single_call"].get("raw")).upper() else "REAL" for r in rows],
        "v2 inputs, threshold 0.5": ["FAKE" if x >= 0.5 else "REAL" for x in p],
        "v2 inputs, frozen threshold": ["FAKE" if x >= frozen["single_call_threshold"] else "REAL" for x in p],
    }


def paired_ba(gold, a, b, n=2000):
    rng = np.random.default_rng(0)
    g = np.array(gold)
    ca, cb = np.array(a) == g, np.array(b) == g
    R, F = np.where(g == "REAL")[0], np.where(g == "FAKE")[0]
    diff = lambda r, f: ((ca[r].mean() + ca[f].mean()) - (cb[r].mean() + cb[f].mean())) / 2
    d = [diff(rng.choice(R, len(R)), rng.choice(F, len(F))) for _ in range(n)]
    return {"diff": round(float(diff(R, F)), 4), "ci95": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]}


# 1. How much of the single-call gain is the threshold?
for name in ("tuning180", "tuning943", "holdout765"):
    _, rows, _ = data[name]
    gold = [r["gold"] for r in rows]
    sv = single_variants(rows)
    out[name] = {
        "single_call": {k: short(gold, v) for k, v in sv.items()},
        "calls_per_item": dict(Counter(1 + len(r["calls"]) for r in rows)),
        "single_call_failed": sum(r["single_call"].get("p_fake") is None for r in rows),
        "rows_with_failed_call": sum(bool(r["single_call"].get("error") or any(c.get("error") for c in r["calls"])) for r in rows),
        "failure_kinds": dict(Counter(failure_kind(c["error"]) for r in rows
                                      for c in [r["single_call"], *r["calls"]] if c.get("error"))),
        "recorded_tokens_uncached": sum(int((r.get("usage") or {}).get("total_tokens") or 0) for r in rows),
    }

# 2. Holdout: paired balanced-accuracy differences and the per-modality picture
_, hold, _ = data["holdout765"]
_, tune, tgroups = data["tuning943"]
gold = [r["gold"] for r in hold]
sv = single_variants(hold)
sv["v2 scorer"] = apply_frozen(frozen, hold)[1]
agent_names = tuple(f for f in FEATURES if f in AGENT_FEATURES or f.startswith("has_"))
sv["agents only"] = apply_frozen(freeze(tune, agent_names, tgroups), hold)[1]
pairs = [("v2 inputs, raw TRUE/FALSE", "v1 hard output"), ("v2 inputs, threshold 0.5", "v1 hard output"),
         ("v2 inputs, frozen threshold", "v2 inputs, threshold 0.5"), ("v2 inputs, frozen threshold", "v2 inputs, raw TRUE/FALSE"),
         ("v2 scorer", "v2 inputs, frozen threshold"), ("v2 scorer", "v1 hard output"), ("agents only", "v2 inputs, frozen threshold")]
out["holdout765"]["paired_balanced_accuracy"] = {f"{a} - {b}": paired_ba(gold, sv[a], sv[b]) for a, b in pairs}
by_mod = {}
for mod in sorted({r["modality_key"] for r in hold}):
    ix = [i for i, r in enumerate(hold) if r["modality_key"] == mod]
    by_mod[mod] = {"n": len(ix), "gold": dict(Counter(gold[i] for i in ix)),
                   **{k: round(binary_metrics([gold[i] for i in ix], [v[i] for i in ix])["balanced_accuracy"], 4) for k, v in sv.items()}}
out["holdout765"]["by_modality"] = by_mod
ix = [i for i, r in enumerate(hold) if r["modality_key"] == "image"]
sub = lambda k: [sv[k][i] for i in ix]
out["holdout765"]["image_only_tests"] = {
    f"agents only vs {k}": {"mcnemar": mcnemar([gold[i] for i in ix], sub("agents only"), sub(k)),
                            "paired_ba": paired_ba([gold[i] for i in ix], sub("agents only"), sub(k))}
    for k in ("v2 inputs, frozen threshold", "v2 inputs, raw TRUE/FALSE")}

# 3. Overlap between tuning and holdout beyond the source groups
title = lambda i: samples[i].title.strip().lower()
tt = {title(i) for i in ids["train"] if title(i)}
shared = [r["id"] for r in hold if title(r["id"]) in tt]
split_groups = {k: {group(samples[i]) for i in v} for k, v in ids.items()}
out["overlap"] = {
    "source_groups_tuning_vs_holdout": len(split_groups["train"] & split_groups["holdout"]),
    "source_groups_tuning_vs_test1007": len(split_groups["train"] & split_groups["test1007"]),
    "orphan_items": {k: sum(group(samples[i]).startswith("orphan") for i in v) for k, v in ids.items()},
    "holdout_items_sharing_title_with_tuning": len(shared), "distinct_titles": len({title(i) for i in shared}),
    "shared_by_modality": dict(Counter(r["modality_key"] for r in hold if r["id"] in set(shared))),
    "tuning_source_is_orphan": sum(group(samples[i]).startswith("orphan") for i in ids["train"] if title(i) in {title(j) for j in shared}),
}
s180 = [r["id"] for r in data["tuning180"][1]]
st = {title(i) for i in s180 if title(i)}
out["overlap"]["holdout_items_sharing_title_with_tuning180"] = sum(title(r["id"]) in st for r in hold)
keep = [i for i, r in enumerate(hold) if r["id"] not in set(shared)]
out["overlap"]["holdout_without_shared_titles"] = {k: short([gold[i] for i in keep], [v[i] for i in keep]) for k, v in sv.items()}

# 4. Tuning set: is the agent gain explained by the failed single calls? (cross-validated)
for label, rows_idx in (("all", range(len(tune))), ("without items whose single call failed",
                                                   [i for i, r in enumerate(tune) if r["single_call"].get("p_fake") is not None])):
    rows = [tune[i] for i in rows_idx]
    g = [tgroups[i] for i in rows_idx]
    gd = [r["gold"] for r in rows]
    a = cross_validated(rows, FEATURES, g)["predictions"]
    b = cross_validated_threshold(rows, lambda r: r["features"]["single_p_fake"], g)
    out["tuning943"][f"cv {label}"] = {"v2 scorer": short(gd, a), "single call": short(gd, b),
                                      "mcnemar": mcnemar(gd, a, b), "paired_ba": paired_ba(gd, a, b)}

# 5. Metadata-only shortcut probe (tuning set, grouped CV)
y = np.array([1 if r["gold"] == "FAKE" else 0 for r in tune])
S = [samples[r["id"]] for r in tune]


def day(s):
    m = re.match(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", s.publish_time or "")
    return datetime.date(*map(int, m.groups())).toordinal() if m else -1


probes = {"date": [[day(s)] for s in S], "text and title length": [[len(s.text or ""), len(s.title or "")] for s in S],
          "modality and media counts": [[len(s.image_paths), len(s.video_paths), len(s.audio_paths),
                                         *[m in s.modalities for m in ("text", "image", "audio", "video")]] for s in S]}
probes["all"] = [a + b + c for a, b, c in zip(*probes.values())]
out["metadata_probe_tuning943"] = {}
for name, values in probes.items():
    X, p = np.array(values, float), np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, tgroups):
        w = np.where(y[tr] == 1, 1 / y[tr].mean(), 1 / (1 - y[tr].mean()))
        p[te] = GradientBoostingClassifier(random_state=0, max_depth=2, n_estimators=80).fit(X[tr], y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
    out["metadata_probe_tuning943"][name] = round(binary_metrics(["FAKE" if v else "REAL" for v in y],
                                                                 ["FAKE" if q >= 0.5 else "REAL" for q in p])["balanced_accuracy"], 4)

target = ROOT / "artifacts/v2_results_20261122/recheck_20261124.json"
target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
