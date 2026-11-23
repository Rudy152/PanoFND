from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panoagent.dataset import PanoFNDDataset, load_split, save_json
from panoagent.evaluation import evaluate_records, read_jsonl
from panoagent.fact_r1_video import (
    EXPECTED_WEIGHT_FILES,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_URL,
    OFFICIAL_REPOSITORY,
    PAPER_URL,
    FactR1VideoClassifier,
)
from panoagent.media import VideoFrameExtractor
from panoagent.schemas import PredictionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fact-R1 evaluation using its released 8-frame FakeVV protocol."
    )
    parser.add_argument("--dataset", default="fakenewsDataSet")
    parser.add_argument("--split", default="splits/pilot_180.json")
    parser.add_argument("--partition", default="pilot")
    parser.add_argument("--model", default="models/fact-r1-final-ckpt")
    parser.add_argument("--output", default="runs/fact_r1_pilot_video40.jsonl")
    parser.add_argument("--metrics", default="runs/fact_r1_pilot_video40_metrics.json")
    parser.add_argument("--frame-cache", default="cache/internvl_frames")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-transcript-chars", type=int, default=6_000)
    parser.add_argument("--quantization", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.video_frames != 8:
        raise ValueError("The released Fact-R1 protocol uses exactly 8 video frames")
    if args.max_transcript_chars < 128:
        raise ValueError("--max-transcript-chars must be at least 128")
    if args.device == "cpu" and args.quantization == "4bit":
        raise ValueError("Use --quantization bf16 for CPU inference")

    dataset = PanoFNDDataset(args.dataset)
    split_ids = load_split(args.split, args.partition)
    requested_ids = set(args.ids or split_ids)
    samples = [
        sample
        for sample in dataset.iter_ids(split_ids)
        if sample.id in requested_ids and sample.video_paths
    ]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No video samples were selected")
    if args.ids:
        missing = set(args.ids) - {sample.id for sample in samples}
        if missing:
            raise ValueError(f"Requested IDs are not video samples in this split: {sorted(missing)}")

    output_path = Path(args.output)
    prior = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    completed = {
        str(row.get("id"))
        for row in prior
        if row.get("prediction") in {"REAL", "FAKE"}
    }
    if not args.resume and output_path.exists():
        output_path.unlink()

    import torch

    model_path = Path(args.model).resolve()
    shards = [model_path / name for name in EXPECTED_WEIGHT_FILES]
    for path in shards:
        expected_size = int(EXPECTED_WEIGHT_FILES[path.name]["size"])
        if not path.is_file() or path.stat().st_size != expected_size:
            actual_size = path.stat().st_size if path.exists() else None
            raise FileNotFoundError(
                f"Invalid Fact-R1 shard {path.name}: expected {expected_size} bytes, "
                f"got {actual_size}"
            )
    weight_sha256 = {path.name: sha256_file(path) for path in shards}
    hash_mismatches = {
        name: {"expected": EXPECTED_WEIGHT_FILES[name]["sha256"], "actual": digest}
        for name, digest in weight_sha256.items()
        if digest != EXPECTED_WEIGHT_FILES[name]["sha256"]
    }
    if hash_mismatches:
        raise RuntimeError(f"Fact-R1 weight checksum mismatch: {hash_mismatches}")

    print(
        f"Loading {MODEL_ID}@{MODEL_REVISION} from {model_path} "
        f"({args.quantization})...",
        flush=True,
    )
    classifier = FactR1VideoClassifier(
        model_path,
        device=args.device,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )
    extractor = VideoFrameExtractor(
        args.frame_cache, frame_count=args.video_frames, max_edge=768
    )
    static_metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_url": MODEL_URL,
        "paper": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "weight_sha256": weight_sha256,
        "quantization": args.quantization,
        "video_sampling": "8 uniformly spaced cached frames",
        "image_min_pixels": 3_136,
        "image_max_pixels": 25_088,
        "input_policy": (
            "8 video frames + title + existing audio transcript; article text, "
            "news still images, gold label, and gold-leaking description excluded"
        ),
        "prompt_policy": "verbatim released Fact-R1 FakeVV test question and system format",
        "evaluation_protocol": "deterministic zero-shot binary classification",
    }

    processed = 0
    for sample in samples:
        if sample.id in completed:
            continue
        started = time.perf_counter()
        prediction = "ERROR"
        raw_output = ""
        error = None
        inference_metadata: dict[str, object] = {}
        try:
            frame_paths: list[Path] = []
            for video_path in sample.video_paths:
                frame_paths.extend(extractor.extract(sample.id, video_path))
            # Match the official protocol even if a row unexpectedly lists >1 video.
            frame_paths = frame_paths[:8]
            prediction, raw_output, inference_metadata = classifier.classify(
                sample,
                frame_paths,
                max_transcript_chars=args.max_transcript_chars,
            )
            if prediction == "ERROR":
                error = "Could not parse Yes/No from the model's <answer> tag"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if args.device == "cuda":
                torch.cuda.empty_cache()

        latency = time.perf_counter() - started
        record = PredictionRecord(
            id=sample.id,
            gold=sample.binary_label,
            prediction=prediction,
            modality_key=sample.modality_key,
            raw_output=raw_output,
            backend="fact-r1-local-video",
            latency_seconds=latency,
            error=error,
            model_metadata=static_metadata | inference_metadata,
        ).to_dict()
        append_jsonl(output_path, record)
        processed += 1
        print(
            f"{processed + len(completed)}/{len(samples)} id={sample.id} "
            f"gold={sample.binary_label} prediction={prediction} "
            f"latency={latency:.2f}s error={error}",
            flush=True,
        )

    records = read_jsonl(output_path)
    selected_ids = {sample.id for sample in samples}
    records = [row for row in records if str(row.get("id")) in selected_ids]
    metrics = evaluate_records(records)
    metrics["experiment"] = {
        "model": MODEL_ID,
        "samples": len({str(row.get("id")) for row in records}),
        "partition": args.partition,
        "split": str(Path(args.split).resolve()),
        "device": args.device,
        **static_metadata,
    }
    save_json(args.metrics, metrics)
    overall = metrics["overall"]
    print(
        "Finished: "
        f"accuracy={overall['accuracy']:.4f}, "
        f"balanced_accuracy={overall['balanced_accuracy']:.4f}, "
        f"macro_f1={overall['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Predictions: {output_path.resolve()}", flush=True)
    print(f"Metrics: {Path(args.metrics).resolve()}", flush=True)


if __name__ == "__main__":
    main()
