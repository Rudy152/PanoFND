"""v2 command line: `extract` (L0-L2 evidence) and `score` (L4 cross-validated scorer)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time

from panoagent.dataset import PanoFNDDataset, load_split

from .agents import EvidenceAgents, StageCache
from .client import OmniClient
from .detectors import ClipSyntheticImageDetector
from .features import AGENT_FEATURES, DETECTOR_FEATURES, FEATURES, build_features, explain, localise
from .media import MediaPreparer
from .evaluation import PRIMARY_SYSTEM, apply_frozen, bootstrap_ci, freeze, mcnemar, verify_frozen
from .scorer import binary_metrics, cross_validated, cross_validated_threshold


def _usage(records: list[dict]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        if record.get("cached"):
            continue
        for key in total:
            total[key] += int((record.get("usage") or {}).get(key) or 0)
    return total


def command_extract(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    ids = load_split(args.split, args.partition)
    if "final_holdout" in Path(args.split).name:
        if not args.frozen:
            raise SystemExit("the final holdout may only be extracted with --frozen (run `freeze` first)")
        verify_frozen(json.loads(Path(args.frozen).read_text(encoding="utf-8")))
    if args.limit:
        ids = ids[: args.limit]
    output = Path(args.output)
    done: set[str] = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("fatal_error"):
                done.add(row["id"])
    samples = [s for s in dataset.iter_ids(ids) if s.id not in done]
    preparer = MediaPreparer(args.media_cache, frame_count=args.frame_count)
    detector = ClipSyntheticImageDetector(args.clip_weights, Path(args.media_cache) / "clip_scores.json")
    agents = EvidenceAgents(OmniClient(args.model_id), StageCache(Path(args.cache) / args.model_id))
    write_lock = threading.Lock()
    print(f"{len(samples)} samples to process ({len(done)} already done)", flush=True)

    def run(sample):
        started = time.perf_counter()
        row = {"id": sample.id, "gold": sample.binary_label, "modality_key": sample.modality_key}
        try:
            item = preparer.prepare(sample)
            scores = {
                name: [detector.score(a.path, a.digest) for a in item.attachments.get(name, []) if a.kind == "image"]
                for name in ("image", "video")
            }
            single = agents.single_call(item)
            calls = []
            for step in agents.plan(item):
                calls.append(agents.unimodal(item, step[1]) if step[0] == "uni" else agents.pair(item, step[1], step[2]))
            row.update({
                "modalities": list(item.modalities),
                "media_errors": item.media_errors,
                "detector_scores": scores,
                "single_call": {k: single.get(k) for k in ("raw", "p_fake", "error", "cached")},
                "calls": [{k: c.get(k) for k in ("stage", "modalities", "result", "error", "cached", "raw")} for c in calls],
                "features": build_features(item.modalities, single, calls, scores),
                "localisation": localise(calls),
                "usage": _usage([single, *calls]),
            })
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
            row["fatal_error"] = f"{type(exc).__name__}: {exc}"[:800]
        row["latency_seconds"] = round(time.perf_counter() - started, 2)
        with write_lock:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, s) for s in samples]
        for future in as_completed(futures):
            row = future.result()
            calls_failed = sum(1 for c in row.get("calls", []) if c.get("error"))
            print(f"{row['id']} {row['modality_key']:18} gold={row['gold']} p_single="
                  f"{(row.get('single_call') or {}).get('p_fake')} failed_calls={calls_failed} "
                  f"fatal={row.get('fatal_error')} {row['latency_seconds']}s", flush=True)


def _load_rows(path: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["id"]] = row
    return [rows[key] for key in sorted(rows)]


def _baseline(path: str) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["id"]] = row["prediction"] if row["prediction"] in {"REAL", "FAKE"} else "ERROR"
    return out


def command_score(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    samples = {s.id: s for s in dataset}
    all_rows = _load_rows(args.features)
    rows = [r for r in all_rows if not r.get("fatal_error")]
    groups = [PanoFNDDataset.group_key(samples[r["id"]]) for r in rows]
    gold = [r["gold"] for r in rows]

    systems: dict[str, list[str]] = {}
    full = cross_validated(rows, FEATURES, groups)
    systems["v2 scorer (all features)"] = full["predictions"]
    systems["v2 scorer w/o single-call feature"] = cross_validated(
        rows, tuple(f for f in FEATURES if f != "single_p_fake"), groups)["predictions"]
    systems["v2 scorer w/o detector features"] = cross_validated(
        rows, tuple(f for f in FEATURES if f not in DETECTOR_FEATURES), groups)["predictions"]
    systems["v2 agents only (no single call, no detectors)"] = cross_validated(
        rows, tuple(f for f in FEATURES if f in AGENT_FEATURES or f.startswith("has_")), groups)["predictions"]
    systems["single call, same v2 inputs (CV threshold)"] = cross_validated_threshold(
        rows, lambda r: r["features"]["single_p_fake"], groups)
    systems["rule: any conflict or manipulated (no training)"] = [
        "FAKE" if (r["features"]["any_conflict"] or r["features"]["any_manipulated"]
                   or r["features"]["uni_contradictory"]) else "REAL" for r in rows]
    for name, path in (args.baseline or []):
        base = _baseline(path)
        systems[name] = [base.get(r["id"], "ERROR") for r in rows]

    report: dict = {"n_scored": len(rows), "n_fatal": len(all_rows) - len(rows), "systems": {}}
    for name, pred in systems.items():
        metrics = binary_metrics(gold, pred)
        by_mod = {}
        for mod in sorted({r["modality_key"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["modality_key"] == mod]
            by_mod[mod] = binary_metrics([gold[i] for i in idx], [pred[i] for i in idx])
        report["systems"][name] = {"overall": metrics, "by_modality": by_mod}
    report["final_model"] = full["final_model"]
    report["folds"] = full["folds"]
    report["usage"] = {k: sum(int((r.get("usage") or {}).get(k) or 0) for r in all_rows)
                       for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, r in enumerate(rows):
            label = full["predictions"][i]
            handle.write(json.dumps({
                "id": r["id"], "gold": r["gold"], "prediction": label, "modality_key": r["modality_key"],
                "p_fake": full["probabilities"][i], "localisation": r["localisation"],
                "explanation": explain(label, full["probabilities"][i], r["localisation"], r["calls"], r["single_call"]),
            }, ensure_ascii=False) + "\n")
    print(f"scored {len(rows)} items ({report['n_fatal']} fatal)")
    print(f"{'system':52} {'BalAcc':>7} {'MacroF1':>8} {'REALrec':>8} {'FAKErec':>8} {'F->R':>5} {'R->F':>5} {'ERR':>4}")
    for name, value in report["systems"].items():
        m = value["overall"]
        print(f"{name:52} {m['balanced_accuracy']:>7.1%} {m['macro_f1']:>8.1%} {m['real_recall']:>8.1%} "
              f"{m['fake_recall']:>8.1%} {m['fake_as_real']:>5} {m['real_as_fake']:>5} {m['errors']:>4}")


def _scored_rows(dataset_path: str, features_path: str) -> tuple[list[dict], list[dict], list[str]]:
    samples = {s.id: s for s in PanoFNDDataset(dataset_path)}
    all_rows = _load_rows(features_path)
    rows = [r for r in all_rows if not r.get("fatal_error")]
    return all_rows, rows, [PanoFNDDataset.group_key(samples[r["id"]]) for r in rows]


def command_freeze(args: argparse.Namespace) -> None:
    all_rows, rows, groups = _scored_rows(args.dataset, args.features)
    frozen = freeze(rows, FEATURES, groups)
    frozen["tuning_features"] = args.features
    frozen["n_tuning_rows"], frozen["n_tuning_fatal"] = len(all_rows), len(all_rows) - len(rows)
    frozen["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen scorer on {len(rows)} tuning rows: C={frozen['C']} threshold={frozen['threshold']:.4f} -> {out}")


def command_evaluate(args: argparse.Namespace) -> None:
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    verify_frozen(frozen)
    all_rows, rows, _ = _scored_rows(args.dataset, args.features)
    gold = [r["gold"] for r in rows]
    prob, primary = apply_frozen(frozen, rows)
    # Fatal rows count as errors for every system: nothing is silently dropped.
    fatal = [r for r in all_rows if r.get("fatal_error")]
    gold_all = gold + [r["gold"] for r in fatal]
    systems = {
        PRIMARY_SYSTEM: primary,
        "single call, same v2 inputs (threshold frozen on tuning)": [
            "FAKE" if r["features"]["single_p_fake"] >= frozen["single_call_threshold"] else "REAL" for r in rows],
    }
    for name, path in (args.baseline or []):
        base = _baseline(path)
        systems[name] = [base.get(r["id"], "ERROR") for r in rows]
    systems = {name: pred + ["ERROR"] * len(fatal) for name, pred in systems.items()}
    report: dict = {"n_items": len(all_rows), "n_fatal": len(fatal), "frozen": args.frozen, "systems": {},
                    "comparisons": {}}
    keys = [r["modality_key"] for r in rows] + [r["modality_key"] for r in fatal]
    for name, pred in systems.items():
        by_mod = {}
        for mod in sorted(set(keys)):
            idx = [i for i, k in enumerate(keys) if k == mod]
            by_mod[mod] = binary_metrics([gold_all[i] for i in idx], [pred[i] for i in idx])
        report["systems"][name] = {
            "overall": binary_metrics(gold_all, pred),
            "balanced_accuracy_ci95": bootstrap_ci(gold_all, pred, lambda g, p: binary_metrics(g, p)["balanced_accuracy"]),
            "by_modality": by_mod,
        }
    for name in systems:
        if name != PRIMARY_SYSTEM:
            report["comparisons"][f"{PRIMARY_SYSTEM} vs {name}"] = mcnemar(gold_all, systems[PRIMARY_SYSTEM], systems[name])
    report["usage"] = {k: sum(int((r.get("usage") or {}).get(k) or 0) for r in all_rows)
                       for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, r in enumerate(rows):
            handle.write(json.dumps({
                "id": r["id"], "gold": r["gold"], "prediction": primary[i], "modality_key": r["modality_key"],
                "p_fake": prob[i], "localisation": r["localisation"],
                "explanation": explain(primary[i], prob[i], r["localisation"], r["calls"], r["single_call"]),
            }, ensure_ascii=False) + "\n")
    print(f"holdout: {len(all_rows)} items ({len(fatal)} fatal, counted as errors)")
    for name, value in report["systems"].items():
        m, ci = value["overall"], value["balanced_accuracy_ci95"]
        print(f"{name:60} BalAcc {m['balanced_accuracy']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] MacroF1 {m['macro_f1']:.1%} "
              f"REAL {m['real_recall']:.1%} FAKE {m['fake_recall']:.1%} ERR {m['errors']}")
    for name, value in report["comparisons"].items():
        print(f"McNemar {name}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="panoagent_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--dataset", required=True)
    ex.add_argument("--split", required=True)
    ex.add_argument("--partition", required=True)
    ex.add_argument("--output", required=True)
    ex.add_argument("--model-id", default="qwen3.8-omni-flash")
    ex.add_argument("--cache", default="cache/v2_stages")
    ex.add_argument("--media-cache", default="cache/v2_media")
    ex.add_argument("--clip-weights", default="models/miragenews/openai_clip/ViT-L-14-336px.pt")
    ex.add_argument("--frame-count", type=int, default=8)
    ex.add_argument("--workers", type=int, default=4)
    ex.add_argument("--limit", type=int)
    ex.add_argument("--frozen", help="frozen scorer JSON; required for the final holdout")
    ex.set_defaults(func=command_extract)
    fr = sub.add_parser("freeze")
    fr.add_argument("--dataset", required=True)
    fr.add_argument("--features", required=True)
    fr.add_argument("--output", required=True)
    fr.set_defaults(func=command_freeze)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--features", required=True)
    ev.add_argument("--frozen", required=True)
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--baseline", nargs=2, action="append", metavar=("NAME", "PREDICTIONS_JSONL"))
    ev.set_defaults(func=command_evaluate)
    sc = sub.add_parser("score")
    sc.add_argument("--dataset", required=True)
    sc.add_argument("--features", required=True)
    sc.add_argument("--output-dir", required=True)
    sc.add_argument("--baseline", nargs=2, action="append", metavar=("NAME", "PREDICTIONS_JSONL"))
    sc.set_defaults(func=command_score)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
