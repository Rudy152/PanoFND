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
from panoagent.internvl_video import (
    MODEL_ID,
    MODEL_RELEASE_DATE,
    OFFICIAL_REPOSITORY,
    PAPER_URL,
    InternVLVideoClassifier,
)
from panoagent.media import VideoFrameExtractor
from panoagent.schemas import PredictionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot InternVL2.5 video benchmark on a PanoFND split."
    )
    parser.add_argument("--dataset", default="fakenewsDataSet")
    parser.add_argument("--split", default="splits/pilot_180.json")
    parser.add_argument("--partition", default="pilot")
    parser.add_argument("--model", default="models/InternVL2_5-2B")
    parser.add_argument("--output", default="runs/internvl2_5_2b_pilot_180.jsonl")
    parser.add_argument(
        "--metrics", default="runs/internvl2_5_2b_pilot_180_metrics.json"
    )
    parser.add_argument("--frame-cache", default="cache/internvl_frames")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--image-max-tiles", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    if args.video_frames < 1 or args.image_max_tiles < 1:
        raise ValueError("frame and tile counts must be positive")
    if args.max_text_chars < 256:
        raise ValueError("--max-text-chars must be at least 256")

    dataset = PanoFNDDataset(args.dataset)
    samples = list(dataset.iter_ids(load_split(args.split, args.partition)))
    if args.only_video:
        samples = [sample for sample in samples if sample.video_paths]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("The selected evaluation partition is empty")

    output_path = Path(args.output)
    prior = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    completed = {
        str(row.get("id")) for row in prior if row.get("prediction") in {"REAL", "FAKE"}
    }
    if not args.resume and output_path.exists():
        output_path.unlink()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Loading {MODEL_ID} from {Path(args.model).resolve()}...", flush=True)
    classifier = InternVLVideoClassifier(
        args.model, device=args.device, max_new_tokens=args.max_new_tokens
    )
    frame_extractor = VideoFrameExtractor(
        args.frame_cache, frame_count=args.video_frames, max_edge=768
    )
    weight_path = Path(args.model) / "model.safetensors"
    weight_sha256 = sha256_file(weight_path)
    static_metadata = {
        "model_id": MODEL_ID,
        "release_date": MODEL_RELEASE_DATE,
        "paper": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "weight_sha256": weight_sha256,
        "video_sampling": f"{args.video_frames} uniformly spaced cached frames",
        "image_max_tiles": args.image_max_tiles,
        "max_text_chars": args.max_text_chars,
        "input_policy": (
            "title + article text + audio transcript + publication date + news image + "
            "uniformly sampled video frames; text uses a fixed head-tail limit; "
            "gold-leaking description excluded"
        ),
        "evaluation_protocol": "deterministic zero-shot binary classification",
    }

    processed = 0
    for sample in samples:
        if sample.id in completed:
            continue
        started = time.perf_counter()
        error = None
        raw_output = ""
        prediction = "ERROR"
        visual_metadata: dict[str, int] = {}
        try:
            frame_paths = []
            for video_path in sample.video_paths:
                frame_paths.extend(frame_extractor.extract(sample.id, video_path))
            image_paths = [path for path in sample.image_paths if path.is_file()]
            prediction, raw_output, visual_metadata = classifier.classify(
                sample,
                image_paths,
                frame_paths,
                image_max_tiles=args.image_max_tiles,
                max_text_chars=args.max_text_chars,
            )
            if prediction == "ERROR":
                error = "Could not parse a binary label from model output"
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
            backend="internvl2.5-2b-local-video",
            latency_seconds=latency,
            error=error,
            model_metadata=static_metadata | visual_metadata,
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
    records = [record for record in records if str(record.get("id")) in selected_ids]
    metrics = evaluate_records(records)
    metrics["experiment"] = {
        "model": MODEL_ID,
        "samples": len({str(record.get('id')) for record in records}),
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
