"""D1: pairwise agent calls versus one joint call over all three modalities (artifacts/ABLATION_PLAN_20261124.md).

    extract  one joint call per three-modality tuning item, same model, media, enums and definitions as the pair prompt
    score    put the joint findings through the same features and scorer, and compare with the cached pairwise ones

Only the 943 tuning items are accepted. The frozen v2 modules are imported, never changed, and the joint
calls use their own cache. Primary contrast (fixed before the run): AUROC of the full 17-feature scorer, plain
features, on the three-modality items where both variants succeeded.
"""

from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from joblib import Parallel, delayed

from panoagent.dataset import PanoFNDDataset

from . import prompts
from .ablation import HAS, SEEDS, Evaluation, add_onehot, check_tuning_only, fixed_rule, lr
from .agents import CONSISTENCY, INTEGRITY, PAIR_ORDER, EvidenceAgents, StageCache, _normalize_pair
from .cli import _load_rows
from .client import OmniClient
from .features import AGENT_FEATURES, FEATURES, build_features
from .media import MediaPreparer

PRICE_IN, PRICE_OUT = 0.8e-6, 2.7e-6  # CNY per token, checked 2026-11-24
MAX_TOKENS = 1500
JOINT_SYSTEM = """You are a multimodal forensic analyst. You compare three evidence sources taken from
one news item and report structured findings. Return one JSON object only, with no markdown."""

# Definitions and rules are copied from prompts.PAIR_USER; only the number of sources and the layout differ.
JOINT_USER = """TASK: Compare SOURCE A ({a_name}), SOURCE B ({b_name}) and SOURCE C ({c_name}) from the same news item.
Each attachment is preceded by its label, e.g. [NEWS IMAGE 1], [VIDEO FRAME 3 @ 00:42],
[NEWS AUDIO PART 1/2 | 00:00-20:00], [VIDEO SOUNDTRACK PART 1/1].

Return a JSON object with exactly these keys:
- "pairs": array of exactly 3 objects, one for each of the pairs "A-B", "A-C" and "B-C". Each object has the keys:
  - "pair": "A-B", "A-C" or "B-C".
  - "consistency": one of "consistent", "conflict", "unrelated", "insufficient".
     consistent = both sources present the same event with compatible facts;
     conflict = they state or show incompatible facts about the same event;
     unrelated = they concern different events or topics;
     insufficient = one source carries too little information to compare.
  - "slot_checks": array of at most 6 objects, each with keys "field" (one of "number", "person",
     "organization", "location", "time", "quote", "event"), "a_value", "b_value" (the values in the first and
     second source of the pair), and "status" (one of "match", "mismatch", "unverifiable"). Include only facts
     that both sources state or show.
  - "cues": array of at most 3 short strings naming the observations behind your answer.
  - "summary": one sentence.
- "integrity": object with the keys "A", "B" and "C", each one of "intact", "manipulated", "uncertain": whether that
   source itself appears generated, edited, spliced, voice-synthesized, or captioned in a misleading way.

Rules: judge only what the sources contain. A conflict between the sources is evidence in itself;
do not resolve it by deciding from outside knowledge which source is right. Treat spelling variants,
transliterations, and paraphrases as matches. Use "uncertain" or "insufficient" instead of guessing."""


def joint_to_calls(data: dict, mods: tuple[str, ...]) -> list[dict]:
    """Turn one joint answer into the three pair records that build_features expects. Incomplete answers raise."""
    letters = dict(zip("ABC", mods))
    pairs = {str(p.get("pair", "")).strip().upper(): p for p in data.get("pairs") or [] if isinstance(p, dict)}
    integrity = data.get("integrity")
    if set(pairs) != {"A-B", "A-C", "B-C"} or not isinstance(integrity, dict) or not set("ABC") <= set(integrity):
        raise ValueError("joint answer lacks one of the three pairs or the integrity of a source")
    calls = []
    for a, b in itertools.combinations("ABC", 2):
        p = pairs[f"{a}-{b}"]
        if p.get("consistency") not in CONSISTENCY:
            raise ValueError(f"joint answer has no valid consistency for {a}-{b}")
        result = _normalize_pair({**p, "a_integrity": integrity[a], "b_integrity": integrity[b]})
        calls.append({"stage": f"pair_{letters[a]}_{letters[b]}", "modalities": [letters[a], letters[b]],
                      "result": result, "error": None})
    if not all(integrity[x] in INTEGRITY for x in "ABC"):
        raise ValueError("joint answer has an invalid integrity value")
    return calls


def joint_call(agents: EvidenceAgents, item) -> dict:
    mods = tuple(m for m in PAIR_ORDER if m in item.modalities)
    names = prompts.MODALITY_NAMES
    user = JOINT_USER.format(a_name=names[mods[0]], b_name=names[mods[1]], c_name=names[mods[2]])
    tags = {m: f"SOURCE {x} " for x, m in zip("ABC", mods)}
    record = agents._run(item, "joint", JOINT_SYSTEM, user, mods, tags, json_mode=True, logprobs=False,
                         max_tokens=MAX_TOKENS)
    if not record.get("error"):
        try:
            record["calls"] = joint_to_calls(record.get("data") or {}, mods)
        except ValueError as exc:
            record["error"] = f"ValueError: {exc}"
    return record


def command_extract(args) -> None:
    if sys.platform == "win32":  # no idle sleep while this process runs; no setting is changed
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    reference = _load_rows(args.features)
    check_tuning_only([r["id"] for r in reference])
    wanted = [r["id"] for r in reference if len(r["modalities"]) == 3]
    if args.ids:
        wanted = [i for i in wanted if i in set(args.ids)]
    samples = {s.id: s for s in PanoFNDDataset(args.dataset).iter_ids(wanted)}
    expected = {r["id"]: tuple(r["modalities"]) for r in reference}
    output = Path(args.output)
    done = set()
    if output.is_file():
        done = {r["id"] for r in _load_rows(str(output)) if not r.get("error")}
    todo = [i for i in wanted if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    preparer = MediaPreparer(args.media_cache)
    agents = EvidenceAgents(OmniClient(args.model_id), StageCache(Path(args.cache) / args.model_id))
    lock, spent = threading.Lock(), [0.0]
    print(f"{len(todo)} joint calls to run ({len(done)} already done)", flush=True)

    def run(sample_id):
        started = time.perf_counter()
        row = {"id": sample_id, "modality_key": samples[sample_id].modality_key}
        if spent[0] > args.max_cost:
            return {**row, "error": "skipped: cost limit reached", "skipped": True, "latency_seconds": 0.0}
        try:
            item = preparer.prepare(samples[sample_id])
            if tuple(item.modalities) != expected[sample_id]:
                raise RuntimeError(f"modalities {item.modalities} differ from the v2 run {expected[sample_id]}")
            rec = joint_call(agents, item)
            usage = rec.get("usage") or {}
            row.update({"modalities": list(item.modalities), "error": rec.get("error"), "calls": rec.get("calls"),
                        "raw": rec.get("raw"), "usage": usage, "cached": rec.get("cached")})
            if not rec.get("cached"):
                with lock:
                    spent[0] += usage.get("prompt_tokens", 0) * PRICE_IN + usage.get("completion_tokens", 0) * PRICE_OUT
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
            row["error"] = f"{type(exc).__name__}: {exc}"[:800]
        row["latency_seconds"] = round(time.perf_counter() - started, 2)
        with lock:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(run, i) for i in todo]), 1):
            row = future.result()
            print(f"{n}/{len(todo)} {row['id']} {row['modality_key']:17} out={(row.get('usage') or {}).get('completion_tokens')} "
                  f"cost={spent[0]:.2f} err={str(row.get('error'))[:90]} {row['latency_seconds']}s", flush=True)


def _job(name, fn, rows, groups, seed):
    return name, seed, fn(rows, groups, seed)


def command_score(args) -> None:
    rows = _load_rows(args.features)
    check_tuning_only([r["id"] for r in rows])
    joint = {r["id"]: r for r in _load_rows(args.joint)}
    check_tuning_only(list(joint))
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    groups = [PanoFNDDataset.group_key(samples[r["id"]]) for r in rows]
    mk = add_onehot(rows)
    three = np.array([len(r["modalities"]) == 3 for r in rows])
    missing = [r["id"] for r, t in zip(rows, three) if t and r["id"] not in joint]
    if missing:
        raise SystemExit(f"{len(missing)} joint calls missing, e.g. {missing[:5]}; finish extract first")
    jrows, both_ok = [], []
    for r, t in zip(rows, three):
        if not t:
            jrows.append(r)
            both_ok.append(False)
            continue
        j = joint[r["id"]]
        calls = j["calls"] if not j.get("error") else [{"stage": "joint", "error": j["error"], "result": None}]
        new = build_features(tuple(r["modalities"]), r["single_call"], calls, r["detector_scores"])
        jrows.append({**r, "features": {**r["features"], **new}})
        both_ok.append(not j.get("error") and not any(c.get("error") for c in r["calls"]))
    both_ok = np.array(both_ok)
    agents_has = tuple(f for f in FEATURES if f in AGENT_FEATURES) + HAS
    feature_sets = {"A0 plain": FEATURES, "A0 onehot": FEATURES + mk, "A5 plain": agents_has, "A5 onehot": agents_has + mk}
    systems = {}
    for name, f in feature_sets.items():
        systems[f"pairwise {name}"] = (lr(f), rows)
        systems[f"joint {name}"] = (lr(f), jrows)
    no_agents = tuple(f for f in FEATURES if f not in AGENT_FEATURES)
    systems["no agents A3 plain"] = (lr(no_agents), rows)
    systems["no agents A3 onehot"] = (lr(no_agents + mk), rows)
    systems["pairwise rule"], systems["joint rule"] = (fixed_rule(), rows), (fixed_rule(), jrows)
    done = Parallel(n_jobs=-2, batch_size=1)(
        delayed(_job)(n, fn, d, groups, s) for n, (fn, d) in systems.items() for s in SEEDS)
    runs = {n: [r for _, r in sorted([(s, r) for m, s, r in done if m == n], key=lambda t: t[0])] for n in systems}
    gold = np.array([r["gold"] for r in rows])
    keys = np.array([r["modality_key"] for r in rows])
    masks = {"three-modality, both succeeded (primary)": both_ok, "three-modality, all": three, "all 943": np.ones(len(rows), bool)}
    for k in sorted(set(keys[three])):
        masks[f"{k}, both succeeded"] = both_ok & (keys == k)
    report = {"seeds": list(SEEDS), "n_three_modality": int(three.sum()), "n_both_succeeded": int(both_ok.sum()),
              "joint_failed": int(sum(bool(joint[r["id"]].get("error")) for r, t in zip(rows, three) if t)),
              "pairwise_items_with_failed_call": int(sum(any(c.get("error") for c in r["calls"]) for r, t in zip(rows, three) if t)),
              "subsets": {}}
    for label, mask in masks.items():
        idx = np.where(mask)[0]
        ev = Evaluation(gold[idx].tolist(), [groups[i] for i in idx], auc_boot=2000)
        sub = {n: [(np.asarray(p)[idx], np.asarray(q)[idx]) for p, q in v] for n, v in runs.items()}
        entry = {"n": len(idx), "fake": int((gold[idx] == "FAKE").sum()), "source_groups": ev.k,
                 "systems": {n: ev.summarise(v) for n, v in sub.items()}, "pairwise_minus_joint": {}}
        for n in list(feature_sets) + ["rule"]:
            entry["pairwise_minus_joint"][n] = ev.paired(sub[f"pairwise {n}"], sub[f"joint {n}"])
        for n in ("plain", "onehot"):
            entry["pairwise_minus_joint"][f"joint A0 {n} minus no agents"] = ev.paired(sub[f"joint A0 {n}"], sub[f"no agents A3 {n}"])
            entry["pairwise_minus_joint"][f"pairwise A0 {n} minus no agents"] = ev.paired(sub[f"pairwise A0 {n}"], sub[f"no agents A3 {n}"])
        report["subsets"][label] = entry
    # Post hoc: the main analysis trains one scorer on a mix of joint (421) and pairwise (522) features, whose
    # distributions differ. Here both variants are trained and evaluated on three-modality items only.
    ix = np.where(three)[0]
    sub_groups = [groups[i] for i in ix]
    tri = {}
    for n, f in feature_sets.items():
        tri[f"pairwise {n}"] = (lr(f), [rows[i] for i in ix])
        tri[f"joint {n}"] = (lr(f), [jrows[i] for i in ix])
    tri["no agents A3 plain"] = (lr(no_agents), [rows[i] for i in ix])
    tri["no agents A3 onehot"] = (lr(no_agents + mk), [rows[i] for i in ix])
    done3 = Parallel(n_jobs=-2, batch_size=1)(delayed(_job)(n, fn, d, sub_groups, s) for n, (fn, d) in tri.items() for s in SEEDS)
    runs3 = {n: [r for _, r in sorted([(s, r) for m, s, r in done3 if m == n], key=lambda t: t[0])] for n in tri}
    entry = {"n": len(ix), "systems": {}, "pairwise_minus_joint": {}}
    keep = np.where(both_ok[ix])[0]
    ev = Evaluation(gold[ix][keep].tolist(), [sub_groups[i] for i in keep], auc_boot=2000)
    sub3 = {n: [(np.asarray(p)[keep], np.asarray(q)[keep]) for p, q in v] for n, v in runs3.items()}
    entry["n_evaluated"] = len(keep)
    entry["systems"] = {n: ev.summarise(v) for n, v in sub3.items()}
    for n in feature_sets:
        entry["pairwise_minus_joint"][n] = ev.paired(sub3[f"pairwise {n}"], sub3[f"joint {n}"])
    for n in ("plain", "onehot"):
        for v in ("pairwise", "joint"):
            entry["pairwise_minus_joint"][f"{v} A0 {n} minus no agents"] = ev.paired(sub3[f"{v} A0 {n}"], sub3[f"no agents A3 {n}"])
    report["posthoc_trained_on_three_modality_only"] = entry
    # how often the two formats say the same thing
    agree = {}
    for feat in ("any_conflict", "any_manipulated", "frac_unrelated", "frac_integrity_uncertain", "slot_mismatch", "slot_match"):
        a = np.array([r["features"][feat] for r in rows])[both_ok]
        b = np.array([r["features"][feat] for r in jrows])[both_ok]
        agree[feat] = {"pairwise_mean": float(a.mean()), "joint_mean": float(b.mean()), "same": float((a == b).mean())}
    report["feature_means_both_succeeded"] = agree
    used = [j for j in joint.values() if j.get("usage")]
    p = sum(j["usage"].get("prompt_tokens", 0) for j in used)
    c = sum(j["usage"].get("completion_tokens", 0) for j in used)
    report["joint_usage"] = {"calls_with_usage": len(used), "prompt_tokens": p, "completion_tokens": c,
                             "cost_cny": round(p * PRICE_IN + c * PRICE_OUT, 2),
                             "hit_token_cap": sum(j["usage"].get("completion_tokens", 0) >= MAX_TOKENS for j in used)}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_d1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "subsets"}, ensure_ascii=False))
    e = report["posthoc_trained_on_three_modality_only"]
    for label, e in [*report["subsets"].items(), ("POST HOC trained on three-modality only", {**e, "n": e["n_evaluated"], "fake": -1, "source_groups": -1})]:
        print(f"== {label}: n={e['n']} fake={e['fake']} groups={e['source_groups']}")
        for n, s in e["systems"].items():
            print(f"   {n:22} BA {s['balanced_accuracy']['mean']:.1%} F1 {s['macro_f1']['mean']:.1%} AUC {s['auroc']['mean']:.3f}")
        for n, v in e["pairwise_minus_joint"].items():
            print(f"   diff {n:38} AUC {v['auc_diff']:+.3f} [{v['auc_ci95'][0]:+.3f},{v['auc_ci95'][1]:+.3f}] "
                  f"BA {v['ba_diff']:+.1%} [{v['ba_ci95'][0]:+.1%},{v['ba_ci95'][1]:+.1%}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="panoagent_v2.ablation_d1")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in (("extract", command_extract), ("score", command_score)):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--features", required=True, help="v2 tuning feature rows (the pairwise variant)")
        p.set_defaults(func=fn)
        if name == "extract":
            p.add_argument("--output", required=True)
            p.add_argument("--model-id", default="qwen3.8-omni-flash")
            p.add_argument("--cache", default="cache/v2_ablation_d1")
            p.add_argument("--media-cache", default="cache/v2_media")
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--limit", type=int)
            p.add_argument("--ids", nargs="*")
            p.add_argument("--max-cost", type=float, default=5.0, help="stop starting calls above this many CNY")
        else:
            p.add_argument("--joint", required=True)
            p.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
