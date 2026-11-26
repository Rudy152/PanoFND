"""Base-model baseline with a tuned threshold: the complete v1 request (table E's S0) with P(fake) output.

    freeze    fit one threshold on the 943 tuning items (table E's S0 results) and record it with code hashes
    run       one pass over the 765 holdout items with that frozen threshold (resumable after a crash only)
    evaluate  metrics on the holdout against v1's hard labels, the v2 single call and the v2 scorer

Added 2026-11-25 at the supervisor's request, so that the base model gets the same threshold tuning as the v2
single call. The threshold is frozen before any holdout call; nothing here selects a model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from panoagent.dataset import PanoFNDDataset

from . import prompts
from .ablation_e import Builder, _cost, to_content
from .agents import StageCache, p_fake_from_logprobs
from .client import OmniClient
from .scorer import _best_threshold, binary_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v2_s0_holdout_20261125"
FROZEN = OUT / "frozen_s0.json"
TUNING_S0 = ROOT / "runs/v2_ablation_e_20261124/S0.jsonl"
CODE = ("panoagent_v2/ablation_e.py", "panoagent_v2/baseline_s0.py", "panoagent/prompts.py", "panoagent/media.py",
        "panoagent_v2/prompts.py", "panoagent_v2/client.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(split: str, part: str) -> list[str]:
    return [str(i) for i in json.loads((ROOT / "splits" / split).read_text(encoding="utf-8"))["partitions"][part]]


def tuning_ids() -> list[str]:
    return _ids("half_eval_seed42.json", "train")


def holdout_ids() -> list[str]:
    return _ids("final_holdout_clean_765_20261031.json", "test")


def _rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.is_file() else {}


def command_freeze(args) -> None:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists; the threshold is frozen once")
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    ids = tuning_ids()
    if set(ids) & set(holdout_ids()):
        raise SystemExit("tuning and holdout overlap")
    got = _rows(TUNING_S0)
    if set(got) != set(ids):
        raise SystemExit("S0 tuning results do not cover exactly the 943 tuning items")
    ok = [i for i in ids if got[i].get("p_fake") is not None and not got[i].get("error")]
    y = np.array([samples[i].label == "fake" for i in ok], dtype=int)
    p = np.array([got[i]["p_fake"] for i in ok])
    t = _best_threshold(y, p)
    pred = lambda thr: ["FAKE" if got[i]["p_fake"] >= thr else "REAL" for i in ok]
    gold = ["FAKE" if v else "REAL" for v in y]
    frozen = {"threshold": t, "n_tuning_ok": len(ok), "n_tuning_failed": len(ids) - len(ok),
              "tuning_in_sample_ba": {"frozen": binary_metrics(gold, pred(t))["balanced_accuracy"],
                                      "0.5": binary_metrics(gold, pred(0.5))["balanced_accuracy"]},
              "rule": "scorer._best_threshold on all successful tuning items, as for the v2 single-call threshold",
              "tuning_results_sha256": _sha(TUNING_S0), "code_sha256": {c: _sha(ROOT / c) for c in CODE},
              "model_id": args.model_id, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))


def _verify() -> dict:
    if not FROZEN.is_file():
        raise SystemExit("freeze the threshold first")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = [c for c, h in frozen["code_sha256"].items() if _sha(ROOT / c) != h]
    if changed or _sha(TUNING_S0) != frozen["tuning_results_sha256"]:
        raise SystemExit(f"code or tuning results changed since the freeze: {changed}")
    return frozen


def command_run(args) -> None:
    frozen = _verify()
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    ids = holdout_ids()[: args.limit] if args.limit else holdout_ids()
    builder, client, lock = Builder(), OmniClient(frozen["model_id"]), threading.Lock()
    path = OUT / "s0_holdout.jsonl"
    have = _rows(path)
    spent = [0.0]
    todo = [i for i in ids if i not in have or have[i].get("error") and "data_inspection_failed" not in str(have[i]["error"])]
    print(f"{len(ids)} holdout items, {len(todo)} to run", flush=True)

    def call(i: str) -> dict:
        s = samples[i]
        row = {"id": i, "modality_key": s.modality_key}
        started = time.perf_counter()
        try:
            parts = builder.build("S0", s)
            fp = builder.fingerprint("S0", frozen["model_id"], prompts.SINGLE_SYSTEM, parts)
            cache = StageCache(ROOT / "cache/v2_s0_holdout" / frozen["model_id"])
            rec = cache.load(i, "single_call", fp)
            if spent[0] > args.max_cost:
                return {**row, "error": "skipped: cost limit"}
            if rec is None or rec.get("error"):
                rec = {"fingerprint": fp}
                try:
                    res = client.chat(prompts.SINGLE_SYSTEM, to_content(parts), 4, logprobs=True)
                    rec.update(raw=res.text, usage=res.usage, model=res.model, request_id=res.request_id, error=None,
                               p_fake=p_fake_from_logprobs(res.first_token_logprobs or [], res.text))
                    with lock:
                        spent[0] += _cost(res.usage)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
                cache.save(i, "single_call", rec)
            row.update(p_fake=rec.get("p_fake"), raw=rec.get("raw"), usage=rec.get("usage"), error=rec.get("error"))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:600]
        row["latency_seconds"] = round(time.perf_counter() - started, 2)
        with lock, path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(call, i) for i in todo]), 1):
            r = fut.result()
            if n % 25 == 0 or n == len(todo) or r.get("error"):
                print(f"{n}/{len(todo)} {r['id']} {r['modality_key']:17} p={r.get('p_fake')} spent={spent[0]:.2f} "
                      f"err={str(r.get('error'))[:80]}", flush=True)
    have = _rows(path)
    print(f"done: {sum(1 for i in ids if i in have and not have[i].get('error'))}/{len(ids)} ok, spent {spent[0]:.2f} CNY")


def _paired_ba(gold, a, b, n=2000):
    """Label-stratified paired bootstrap of the Balanced Acc. difference, as in scripts/recheck_v2_results.py."""
    rng = np.random.default_rng(0)
    g = np.array(gold)
    ca, cb = np.array(a) == g, np.array(b) == g
    R, F = np.where(g == "REAL")[0], np.where(g == "FAKE")[0]
    diff = lambda r, f: ((ca[r].mean() + ca[f].mean()) - (cb[r].mean() + cb[f].mean())) / 2
    d = [diff(rng.choice(R, len(R)), rng.choice(F, len(F))) for _ in range(n)]
    return {"diff": float(diff(R, F)), "ci95": [float(np.percentile(d, q)) for q in (2.5, 97.5)]}


def _ba_ci(gold, pred, n=2000):
    rng = np.random.default_rng(0)
    g = np.array(gold)
    c = np.array(pred) == g
    R, F = np.where(g == "REAL")[0], np.where(g == "FAKE")[0]
    d = [(c[rng.choice(R, len(R))].mean() + c[rng.choice(F, len(F))].mean()) / 2 for _ in range(n)]
    return [float(np.percentile(d, q)) for q in (2.5, 97.5)]


def command_evaluate(args) -> None:
    from .cli import _baseline, _scored_rows
    from .evaluation import apply_frozen, verify_frozen

    frozen = _verify()
    v2f = json.loads((ROOT / "runs/v2_frozen_20261121/frozen.json").read_text(encoding="utf-8"))
    verify_frozen(v2f)
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    ids = holdout_ids()
    got = _rows(OUT / "s0_holdout.jsonl")
    v1 = _baseline(str(ROOT / "runs/qwen38_omni_flash_full_1950_20261107.jsonl"))
    _, rows, _ = _scored_rows(args.dataset, str(ROOT / "runs/v2_holdout765_features_20261121.jsonl"))
    by = {r["id"]: r for r in rows}
    _, v2pred = apply_frozen(v2f, rows)
    v2pred = dict(zip([r["id"] for r in rows], v2pred))
    gold = {i: "FAKE" if samples[i].label == "fake" else "REAL" for i in ids}
    p = lambda i: (got.get(i) or {}).get("p_fake")
    systems = {
        "v1 hard output": {i: v1.get(i, "ERROR") for i in ids},
        "S0 raw TRUE/FALSE": {i: ("FAKE" if "FALSE" in str(got[i].get("raw")).upper() else "REAL") if p(i) is not None else "ERROR" for i in ids},
        "S0 threshold 0.5": {i: ("FAKE" if p(i) >= 0.5 else "REAL") if p(i) is not None else "ERROR" for i in ids},
        "S0 frozen threshold": {i: ("FAKE" if p(i) >= frozen["threshold"] else "REAL") if p(i) is not None else "ERROR" for i in ids},
        "v2 inputs, raw TRUE/FALSE": {i: "FAKE" if "FALSE" in str(by[i]["single_call"].get("raw")).upper() else "REAL" for i in ids},
        "v2 inputs, frozen threshold": {i: "FAKE" if by[i]["features"]["single_p_fake"] >= v2f["single_call_threshold"] else "REAL" for i in ids},
        "v2 scorer": {i: v2pred.get(i, "ERROR") for i in ids},
    }
    title = lambda i: samples[i].title.strip().lower()
    tt = {title(i) for i in tuning_ids() if title(i)}
    subsets = {"all 765": ids, "without titles shared with tuning": [i for i in ids if title(i) not in tt],
               "S0 succeeded": [i for i in ids if p(i) is not None]}
    out = {"frozen_s0": frozen, "n_failed_s0": sum(p(i) is None for i in ids),
           "s0_failures": {i: str((got.get(i) or {}).get("error"))[:120] for i in ids if p(i) is None}, "subsets": {}}
    pairs = [("S0 frozen threshold", "v1 hard output"), ("S0 frozen threshold", "S0 threshold 0.5"),
             ("S0 frozen threshold", "S0 raw TRUE/FALSE"), ("S0 raw TRUE/FALSE", "v1 hard output"),
             ("v2 inputs, frozen threshold", "S0 frozen threshold"), ("v2 inputs, raw TRUE/FALSE", "S0 raw TRUE/FALSE"),
             ("v2 scorer", "S0 frozen threshold")]
    for name, sub in subsets.items():
        g = [gold[i] for i in sub]
        e = {"n": len(sub), "systems": {}, "paired_ba": {}}
        for k, v in systems.items():
            m = binary_metrics(g, [v[i] for i in sub])
            e["systems"][k] = {**{x: m[x] for x in ("balanced_accuracy", "macro_f1", "real_recall", "fake_recall", "errors")},
                               "ba_ci95": _ba_ci(g, [v[i] for i in sub])}
        for a, b in pairs:
            e["paired_ba"][f"{a} - {b}"] = _paired_ba(g, [systems[a][i] for i in sub], [systems[b][i] for i in sub])
        out["subsets"][name] = e
    ok = [i for i in ids if p(i) is not None and v1.get(i) in ("REAL", "FAKE")]
    out["S0_raw_vs_v1_same_label"] = float(np.mean([systems["S0 raw TRUE/FALSE"][i] == v1[i] for i in ok]))
    out["usage"] = {k: sum((r.get("usage") or {}).get(k, 0) for r in got.values()) for k in ("prompt_tokens", "completion_tokens")}
    (OUT / "s0_holdout_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, e in out["subsets"].items():
        print(f"== {name} (n={e['n']})")
        for k, v in e["systems"].items():
            print(f"  {k:30} BA {v['balanced_accuracy']:.3f} [{v['ba_ci95'][0]:.3f},{v['ba_ci95'][1]:.3f}] F1 {v['macro_f1']:.3f} "
                  f"R {v['real_recall']:.3f} F {v['fake_recall']:.3f} err {v['errors']}")
        for k, v in e["paired_ba"].items():
            print(f"  {k:55} {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
    print("S0 raw = v1 label:", out["S0_raw_vs_v1_same_label"], "failed:", out["n_failed_s0"], out["usage"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="panoagent_v2.baseline_s0")
    ap.add_argument("command", choices=("freeze", "run", "evaluate"))
    ap.add_argument("--dataset", default=str(ROOT / "fakenewsDataSet"))
    ap.add_argument("--model-id", default="qwen3.8-omni-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    {"freeze": command_freeze, "run": command_run, "evaluate": command_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
