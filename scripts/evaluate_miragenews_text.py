from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panoagent.dataset import PanoFNDDataset, load_split, save_json
from panoagent.evaluation import evaluate_records
from panoagent.miragenews_text import (
    label_from_fake_probability,
    load_text_classifier,
    sha256_file,
)
from panoagent.schemas import PredictionRecord


PAPER_URL = "https://aclanthology.org/2024.findings-emnlp.959/"
OFFICIAL_REPOSITORY = "https://github.com/nosna/miragenews"
MIRAGENEWS_COMMIT = "ab3ec303467954a3a655c4f261b8778315a1ba76"
OPENAI_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the official MiRAGe-Txt linear checkpoint on a PanoFND split."
    )
    parser.add_argument("--dataset", default="fakenewsDataSet")
    parser.add_argument("--split", default="splits/pilot_180.json")
    parser.add_argument("--partition", default="pilot")
    parser.add_argument(
        "--clip-checkpoint",
        default="models/miragenews/openai_clip/ViT-L-14-336px.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        default="third_party/miragenews/checkpoints/text/best-txt-linear.pt",
    )
    parser.add_argument(
        "--output",
        default="runs/miragenews_text_linear_pilot_180.jsonl",
    )
    parser.add_argument(
        "--metrics",
        default="runs/miragenews_text_linear_pilot_180_metrics.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import clip
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    clip_path = Path(args.clip_checkpoint).resolve()
    detector_path = Path(args.detector_checkpoint).resolve()
    actual_clip_sha256 = sha256_file(clip_path)
    if actual_clip_sha256 != CLIP_SHA256:
        raise ValueError(
            f"Unexpected CLIP checkpoint SHA-256: {actual_clip_sha256}; expected {CLIP_SHA256}"
        )

    dataset = PanoFNDDataset(args.dataset)
    samples = list(dataset.iter_ids(load_split(args.split, args.partition)))
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("The selected evaluation partition is empty")

    print(f"Loading OpenAI CLIP ViT-L/14@336px on {device}...", flush=True)
    clip_model, _ = clip.load(str(clip_path), device=device, jit=False)
    clip_model.eval()
    classifier, threshold = load_text_classifier(detector_path, device)

    static_metadata = {
        "paper": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "miragenews_repository_commit": MIRAGENEWS_COMMIT,
        "openai_clip_repository_commit": OPENAI_CLIP_COMMIT,
        "component": "official MiRAGe-Txt linear checkpoint",
        "encoder": "OpenAI CLIP ViT-L/14@336px",
        "encoder_sha256": actual_clip_sha256,
        "detector_checkpoint_sha256": sha256_file(detector_path),
        "threshold": threshold,
        "input_policy": (
            "title + article text + audio transcript + publication date; "
            "gold-leaking description excluded; CLIP truncates to 77 tokens"
        ),
        "domain_note": "strict zero-shot out-of-domain transfer from AI-generated-news detection",
    }

    records: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            texts = [sample.detector_text() for sample in batch]
            tokens = clip.tokenize(texts, truncate=True).to(device)
            started = time.perf_counter()
            features = clip_model.encode_text(tokens)
            probabilities = classifier(features.float()).flatten().cpu().tolist()
            elapsed = time.perf_counter() - started

            for sample, probability in zip(batch, probabilities, strict=True):
                prediction = label_from_fake_probability(float(probability), threshold)
                record = PredictionRecord(
                    id=sample.id,
                    gold=sample.binary_label,
                    prediction=prediction,
                    modality_key=sample.modality_key,
                    raw_output=f"fake_probability={probability:.8f}",
                    backend="miragenews-text-linear",
                    latency_seconds=elapsed / len(batch),
                    model_metadata=static_metadata
                    | {"fake_probability": float(probability)},
                ).to_dict()
                records.append(record)
            print(f"Processed {len(records)}/{len(samples)}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_path)

    metrics = evaluate_records(records)
    metrics["experiment"] = {
        "model": "MiRAGe-Txt linear (official checkpoint)",
        "paper": PAPER_URL,
        "samples": len(records),
        "partition": args.partition,
        "split": str(Path(args.split).resolve()),
        "device": device,
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
