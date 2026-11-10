from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .calibration import apply_calibration, fit_calibration
from .dataset import PanoFNDDataset, load_split, save_json
from .direct import run_direct_sample
from .evaluation import evaluate_records, read_jsonl
from .model_clients import make_backend
from .media import VideoFrameExtractor, prepare_sample_media
from .pipeline import PanoAgentPipeline
from .retrieval import make_search_provider
from .routing import apply_hybrid_routing, select_hybrid_routing
from .schemas import NewsSample, PredictionRecord


def _backend_from_args(args: argparse.Namespace):
    return make_backend(
        args.backend,
        args.model_id,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        timeout_seconds=args.request_timeout,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


def _search_provider(args: argparse.Namespace):
    return make_search_provider(
        args.search_provider,
        cache_dir=args.search_cache or None,
        max_results_per_query=args.search_max_results,
        max_content_chars=args.search_max_chars,
        max_documents=args.search_max_documents,
    )


def _validate_workers(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers > 1 and args.backend != "dashscope":
        raise ValueError("parallel workers are supported only for the stateless DashScope backend")


def _sample_iterator(
    dataset: PanoFNDDataset,
    split_path: str | None,
    partition: str,
    limit: int | None,
    selected_ids: list[str] | None = None,
) -> Iterable[NewsSample]:
    if selected_ids:
        iterator = dataset.iter_ids(selected_ids)
    else:
        iterator = dataset.iter_ids(load_split(split_path, partition)) if split_path else iter(dataset)
    for index, sample in enumerate(iterator):
        if limit is not None and index >= limit:
            break
        yield sample


def _append_record(path: Path, record: PredictionRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row.get("id"))
        for row in read_jsonl(path)
        if (
            (not row.get("error") and row.get("prediction") in {"REAL", "FAKE"})
            # Provider content-safety refusals are deterministic for the same
            # unchanged benchmark input. Retrying them wastes API calls and
            # can prevent a resumable full-dataset job from advancing.
            or "data_inspection_failed" in str(row.get("error") or "")
        )
    }


def command_audit(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    audit = dataset.audit()
    save_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def command_split(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    payload = dataset.create_grouped_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    save_json(args.output, payload)
    print(json.dumps(payload["counts"], indent=2))


def command_pilot_split(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    candidate_ids = load_split(args.split, args.partition)
    payload = dataset.create_stratified_subset(
        candidate_ids,
        per_label_modality=args.per_cell,
        seed=args.seed,
    )
    save_json(args.output, payload)
    print(json.dumps(payload["cell_counts"], indent=2))


def command_clean_holdout(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    candidates = load_split(args.candidate_split, args.candidate_partition)
    excluded = load_split(args.exclude_split, args.exclude_partition)
    excluded.extend(args.exclude_ids or [])
    payload = dataset.create_clean_holdout(candidates, excluded)
    save_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_direct(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    _validate_workers(args)
    output = Path(args.output)
    completed = _existing_ids(output) if args.resume else set()
    extractor = VideoFrameExtractor(args.frame_cache, args.frame_count) if args.media_mode == "frames" else None
    samples = [
        sample for sample in _sample_iterator(
            dataset, args.split, args.partition, args.limit, args.ids
        ) if sample.id not in completed
    ]

    def run_one(sample: NewsSample, backend=None) -> PredictionRecord:
        sample = prepare_sample_media(
            sample,
            args.media_mode,
            extractor,
            max_text_chars=args.max_text_chars,
            max_transcript_chars=args.max_transcript_chars,
        )
        record = run_direct_sample(
            sample,
            backend or _backend_from_args(args),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prompt_profile=args.direct_prompt,
        )
        return record

    if args.workers == 1:
        backend = _backend_from_args(args)
        records = (run_one(sample, backend) for sample in samples)
        for record in records:
            _append_record(output, record)
            print(f"{record.id}: gold={record.gold} prediction={record.prediction} error={record.error}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, sample): sample.id for sample in samples}
        for future in as_completed(futures):
            record = future.result()
            _append_record(output, record)
            print(f"{record.id}: gold={record.gold} prediction={record.prediction} error={record.error}")


def command_agent(args: argparse.Namespace) -> None:
    dataset = PanoFNDDataset(args.dataset)
    _validate_workers(args)
    output = Path(args.output)
    completed = _existing_ids(output) if args.resume else set()
    extractor = VideoFrameExtractor(args.frame_cache, args.frame_count) if args.media_mode == "frames" else None
    samples = [
        sample for sample in _sample_iterator(
            dataset, args.split, args.partition, args.limit, args.ids
        ) if sample.id not in completed
    ]

    def build_pipeline():
        return PanoAgentPipeline(
            backend=_backend_from_args(args),
            cache_root=args.cache,
            search_provider=_search_provider(args),
            max_new_tokens=args.max_new_tokens,
            profile=args.agent_profile,
            strict_stage_errors=args.strict_stage_errors,
            retrieval_policy=args.retrieval_policy,
            exclude_source_url=args.exclude_source_url,
            external_slot_guard=not args.no_external_slot_guard,
        )

    def run_one(sample: NewsSample, pipeline=None) -> PredictionRecord:
        sample = prepare_sample_media(
            sample,
            args.media_mode,
            extractor,
            max_text_chars=args.max_text_chars,
            max_transcript_chars=args.max_transcript_chars,
        )
        return (pipeline or build_pipeline()).run(sample, refresh=args.refresh_cache)

    if args.workers == 1:
        pipeline = build_pipeline()
        records = (run_one(sample, pipeline) for sample in samples)
        for record in records:
            _append_record(output, record)
            print(f"{record.id}: gold={record.gold} prediction={record.prediction} error={record.error}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, sample): sample.id for sample in samples}
        for future in as_completed(futures):
            record = future.result()
            _append_record(output, record)
            print(f"{record.id}: gold={record.gold} prediction={record.prediction} error={record.error}")


def command_search(args: argparse.Namespace) -> None:
    """Standalone check of the web-search module (no model call)."""
    provider = _search_provider(args)
    if not provider.enabled:
        raise SystemExit(
            "Web search is disabled: set TAVILY_API_KEY in the environment or pass "
            "--search-provider tavily."
        )
    report = provider.search_with_report(args.query, focus_text=args.focus or "")
    payload = {
        "provider": provider.name,
        "queries": report.queries,
        "errors": report.errors,
        "documents": [
            {**item.to_dict(), "retrieved_text": item.retrieved_text[: args.preview_chars]}
            for item in report.documents
        ],
    }
    try:  # Windows consoles default to a legacy code page; never crash on output.
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if report.errors and not report.documents:
        raise SystemExit(1)


def _add_search_arguments(parser: argparse.ArgumentParser, default_provider: str) -> None:
    parser.add_argument(
        "--search-provider", default=default_provider, choices=["none", "tavily", "auto"],
        help="web search backend; auto uses Tavily when TAVILY_API_KEY is set",
    )
    parser.add_argument(
        "--search-cache", default="cache/search",
        help="query-level cache of raw search responses (empty string disables)",
    )
    parser.add_argument("--search-max-results", type=int, default=3)
    parser.add_argument(
        "--search-max-chars", type=int, default=2000,
        help="per-document budget; claim-focused passages are kept when trimming",
    )
    parser.add_argument("--search-max-documents", type=int, default=8)


def command_evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_records(read_jsonl(args.predictions))
    save_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_fit_calibration(args: argparse.Namespace) -> None:
    calibration = fit_calibration(
        read_jsonl(args.predictions),
        min_fake_recall=args.min_fake_recall,
        per_modality=args.per_modality,
        min_group_size=args.min_group_size,
    )
    save_json(args.output, calibration)
    print(json.dumps(calibration, ensure_ascii=False, indent=2))


def command_apply_calibration(args: argparse.Namespace) -> None:
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    rows = apply_calibration(read_jsonl(args.predictions), calibration)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} calibrated records to {output}")


def _write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def command_select_routing(args: argparse.Namespace) -> None:
    fused = read_jsonl(args.fused_predictions)
    full = read_jsonl(args.full_predictions)
    selection = select_hybrid_routing(
        fused, full, min_token_reduction=args.min_token_reduction
    )
    save_json(args.output, selection)
    routed = apply_hybrid_routing(fused, full, selection["selected"]["rule"])
    _write_rows(args.hybrid_output, routed)
    print(json.dumps(selection["selected"], ensure_ascii=False, indent=2))


def command_apply_routing(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.routing).read_text(encoding="utf-8"))
    rule = (selection.get("selected") or {}).get("rule") or selection.get("rule")
    if not isinstance(rule, dict):
        raise ValueError("Routing JSON does not contain selected.rule or rule")
    routed = apply_hybrid_routing(
        read_jsonl(args.fused_predictions), read_jsonl(args.full_predictions), rule
    )
    _write_rows(args.output, routed)
    print(f"wrote {len(routed)} routed records to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="panoagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)

    split = subparsers.add_parser("split")
    split.add_argument("--dataset", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--validation-ratio", type=float, default=0.10)
    split.set_defaults(func=command_split)

    pilot = subparsers.add_parser("pilot-split")
    pilot.add_argument("--dataset", required=True)
    pilot.add_argument("--split", required=True)
    pilot.add_argument("--partition", default="test")
    pilot.add_argument("--per-cell", type=int, default=10)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--output", required=True)
    pilot.set_defaults(func=command_pilot_split)

    holdout = subparsers.add_parser("clean-holdout")
    holdout.add_argument("--dataset", required=True)
    holdout.add_argument("--candidate-split", required=True)
    holdout.add_argument("--candidate-partition", default="test")
    holdout.add_argument("--exclude-split", required=True)
    holdout.add_argument("--exclude-partition", default="validation")
    holdout.add_argument("--exclude-ids", nargs="*")
    holdout.add_argument("--output", required=True)
    holdout.set_defaults(func=command_clean_holdout)

    for name, func in (("direct", command_direct), ("agent", command_agent)):
        runner = subparsers.add_parser(name)
        runner.add_argument("--dataset", required=True)
        runner.add_argument("--split")
        runner.add_argument(
            "--partition", default="test", choices=["train", "validation", "test", "pilot"]
        )
        runner.add_argument("--backend", default="mock", choices=["mock", "qwen", "dashscope"])
        runner.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
        runner.add_argument(
            "--api-key-env",
            default="DASHSCOPE_API_KEY",
            help="Environment variable containing the API key; never pass the key itself",
        )
        runner.add_argument(
            "--base-url",
            help="Workspace-specific Model Studio base_url; defaults to DASHSCOPE_BASE_URL",
        )
        runner.add_argument("--request-timeout", type=float, default=300.0)
        runner.add_argument(
            "--min-pixels", type=int, default=16 * 28 * 28,
            help="Local Qwen minimum pixels per image/frame",
        )
        runner.add_argument(
            "--max-pixels", type=int, default=128 * 28 * 28,
            help="Local Qwen maximum pixels per image/frame (default is sized for 16 GB VRAM)",
        )
        runner.add_argument("--temperature", type=float, default=0.0)
        runner.add_argument(
            "--max-text-chars", type=int, default=12000,
            help="Deterministic head/tail character budget for article text",
        )
        runner.add_argument(
            "--max-transcript-chars", type=int, default=8000,
            help="Deterministic head/tail character budget for audio transcripts",
        )
        runner.add_argument("--limit", type=int)
        runner.add_argument(
            "--workers", type=int, default=1,
            help="parallel samples for DashScope only; lowers wall time but not token cost",
        )
        runner.add_argument("--ids", nargs="+", help="Run exact sample IDs, bypassing split selection")
        runner.add_argument(
            "--media-mode", default="transcript", choices=["native", "transcript", "frames"],
            help="native media, transcript+images, or transcript+uniform video frames",
        )
        runner.add_argument(
            "--frame-count", type=int, default=4,
            help="Uniform video frames (local 16 GB default: 4; paper/cloud protocol: 16)",
        )
        runner.add_argument("--frame-cache", default="cache/video_frames")
        runner.add_argument("--max-new-tokens", type=int, default=16 if name == "direct" else 768)
        runner.add_argument("--output", required=True)
        runner.add_argument("--resume", action="store_true")
        if name == "direct":
            runner.add_argument(
                "--direct-prompt", default="paper", choices=["paper", "debiased"],
                help="paper reproduces the original zero-shot prompt; debiased is a small-model ablation",
            )
        if name == "agent":
            runner.add_argument("--cache", default="cache")
            runner.add_argument("--refresh-cache", action="store_true")
            _add_search_arguments(runner, default_provider="auto")
            runner.add_argument(
                "--retrieval-policy", default="text", choices=["model", "text", "always"],
                help=(
                    "when search is enabled: model = only when the model asks; text = also every "
                    "item with title/article/transcript (default); always = every item"
                ),
            )
            runner.add_argument(
                "--exclude-source-url", action="store_true",
                help="ablation: drop the dataset source article from retrieved evidence",
            )
            runner.add_argument(
                "--no-external-slot-guard", action="store_true",
                help=(
                    "ablation: keep the Judge label even when the retrieved original report "
                    "contradicts a central exact slot"
                ),
            )
            runner.add_argument(
                "--agent-profile", default="full", choices=["full", "fused", "fused-rule"],
                help="full uses five/six calls; fused uses two; fused-rule uses one plus a fail-closed deterministic aggregator",
            )
            runner.add_argument(
                "--strict-stage-errors", action="store_true",
                help="mark the final prediction ERROR when any executed stage fails",
            )
        runner.set_defaults(func=func)

    search = subparsers.add_parser("search", help="query the web-search module directly")
    search.add_argument("--query", nargs="+", required=True)
    search.add_argument("--focus", help="claim text used to pick relevant passages")
    search.add_argument("--preview-chars", type=int, default=600)
    _add_search_arguments(search, default_provider="auto")
    search.set_defaults(func=command_search)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=command_evaluate)

    fit = subparsers.add_parser("fit-calibration")
    fit.add_argument("--predictions", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--min-fake-recall", type=float, default=0.5)
    fit.add_argument("--per-modality", action="store_true")
    fit.add_argument("--min-group-size", type=int, default=30)
    fit.set_defaults(func=command_fit_calibration)

    apply = subparsers.add_parser("apply-calibration")
    apply.add_argument("--predictions", required=True)
    apply.add_argument("--calibration", required=True)
    apply.add_argument("--output", required=True)
    apply.set_defaults(func=command_apply_calibration)
    select_routing = subparsers.add_parser("select-routing")
    select_routing.add_argument("--fused-predictions", required=True)
    select_routing.add_argument("--full-predictions", required=True)
    select_routing.add_argument("--min-token-reduction", type=float, default=0.50)
    select_routing.add_argument("--output", required=True)
    select_routing.add_argument("--hybrid-output", required=True)
    select_routing.set_defaults(func=command_select_routing)

    apply_routing = subparsers.add_parser("apply-routing")
    apply_routing.add_argument("--fused-predictions", required=True)
    apply_routing.add_argument("--full-predictions", required=True)
    apply_routing.add_argument("--routing", required=True)
    apply_routing.add_argument("--output", required=True)
    apply_routing.set_defaults(func=command_apply_routing)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
