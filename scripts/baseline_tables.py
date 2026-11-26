"""Baseline tables on the 765-item holdout from saved predictions only (no API or GPU calls).

    .venv\\Scripts\\python.exe scripts\\baseline_tables.py

Base models use the old v1 protocol exactly as run on 2026-11-07: transcripts + 16 frames, the paper
prompt, raw TRUE/FALSE output, no threshold. Every refusal/ERROR counts as wrong for every system.
Fact-R1 accepts video items only, so it is compared on the video subset. Items without a source URL
are FakeTT-style clips (all FAKE); Fact-R1's released training data includes FakeTT, so results are
also reported without them. Writes runs/baseline_tables_20261124.json.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panoagent.dataset import PanoFNDDataset
from panoagent_v2.evaluation import bootstrap_ci, mcnemar
from panoagent_v2.scorer import binary_metrics

RUNS = ROOT / "runs"
samples = {s.id: s for s in PanoFNDDataset(str(ROOT / "fakenewsDataSet"))}
split = lambda p, k: [str(i) for i in json.loads((ROOT / p).read_text(encoding="utf-8"))["partitions"][k]]
holdout = split("splits/final_holdout_clean_765_20261031.json", "test")
tuning = split("splits/half_eval_seed42.json", "train")


def load(path: str, first: bool = False) -> dict[str, str]:
    """Last row per id wins, as in every earlier report; `first` is the sensitivity check."""
    out: dict[str, str] = {}
    for line in (RUNS / path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not (first and row["id"] in out):
            out[row["id"]] = row["prediction"] if row["prediction"] in {"REAL", "FAKE"} else "ERROR"
    return out


Q35, Q38 = "qwen35_27b_full_1950_20261107.jsonl", "qwen38_omni_flash_full_1950_20261107.jsonl"
systems = {"qwen3.5-27b (v1 protocol)": load(Q35), "qwen3.8-omni-flash (v1 protocol)": load(Q38)}
fact_path = RUNS / "fact_r1_video525_20261124.jsonl"
if fact_path.is_file():
    systems["Fact-R1 (4-bit, 8 frames + title)"] = load(fact_path.name)
v2_path = RUNS / "v2_holdout765_eval_20261121/predictions.jsonl"
systems["v2 scorer (reference)"] = {r["id"]: r["prediction"] for r in map(json.loads, v2_path.read_text(encoding="utf-8").splitlines())}
# v2 scored 4 refused items as p=0.5; count them as errors here so all systems follow one rule.
refused = [r["id"] for r in map(json.loads, (RUNS / "v2_holdout765_features_20261121.jsonl").read_text(encoding="utf-8").splitlines())
           if r["single_call"].get("p_fake") is None]
systems["v2 scorer (refusals as errors)"] = {i: ("ERROR" if i in refused else p) for i, p in systems["v2 scorer (reference)"].items()}

has_source = lambda i: samples[i].source.strip() not in {"", "[]"}
subsets = {
    "holdout765": holdout,
    "holdout video": [i for i in holdout if samples[i].video_paths],
    "holdout video, with source URL (no FakeTT-style)": [i for i in holdout if samples[i].video_paths and has_source(i)],
    "holdout video, text+video": [i for i in holdout if samples[i].modality_key == "text+video"],
    "holdout video, text+image+video": [i for i in holdout if samples[i].modality_key == "text+image+video"],
    "tuning video": [i for i in tuning if samples[i].video_paths],
    "all 525 video": [i for i, s in samples.items() if s.video_paths],
}
report: dict = {}
for name, ids in subsets.items():
    gold = [samples[i].binary_label for i in ids]
    print(f"\n== {name}: n={len(ids)} {dict(Counter(gold))}")
    report[name] = {"n": len(ids), "gold": dict(Counter(gold)), "systems": {}}
    preds = {}
    for system, values in systems.items():
        if any(i not in values for i in ids):
            print(f"{system:40} not run on all items ({sum(i in values for i in ids)}/{len(ids)})")
            continue
        pred = [values[i] for i in ids]
        preds[system] = pred
        m = binary_metrics(gold, pred)
        m["ci95"] = bootstrap_ci(gold, pred, lambda g, p: binary_metrics(g, p)["balanced_accuracy"])
        report[name]["systems"][system] = m
        print(f"{system:40} BA {m['balanced_accuracy']:.1%} [{m['ci95'][0]:.1%}, {m['ci95'][1]:.1%}] F1 {m['macro_f1']:.1%} "
              f"acc {m['accuracy']:.1%} REAL {m['real_recall']:.1%} FAKE {m['fake_recall']:.1%} err {m['errors']}")
    names = list(preds)
    report[name]["mcnemar"] = {f"{a} || {b}": mcnemar(gold, preds[a], preds[b]) for i, a in enumerate(names) for b in names[i + 1:]}

for label, path in (("qwen3.5-27b", Q35), ("qwen3.8-omni-flash", Q38)):
    a, b = load(path), load(path, first=True)
    gold = [samples[i].binary_label for i in holdout]
    report[f"duplicate-row sensitivity {label}"] = {
        "holdout ids that differ": [i for i in holdout if a[i] != b[i]],
        "last row BA": binary_metrics(gold, [a[i] for i in holdout])["balanced_accuracy"],
        "first row BA": binary_metrics(gold, [b[i] for i in holdout])["balanced_accuracy"]}
    print(label, report[f"duplicate-row sensitivity {label}"])
(RUNS / "baseline_tables_20261124.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
