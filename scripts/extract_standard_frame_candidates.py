from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


ROOT = Path(r"C:\Users\11378\Desktop\MMLLM")
VIDEO_DIR = ROOT / "fakenewsDataSet" / "video"
DEFAULT_OUTPUT = ROOT / "artifacts" / "frame_candidates"


def video_duration_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    raise RuntimeError(f"Cannot determine video duration: {path}")


def extract_at(path: Path, target_seconds: float, output: Path) -> tuple[float, int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(
            int(target_seconds * av.time_base),
            any_frame=False,
            backward=True,
        )
        selected = None
        for frame in container.decode(stream):
            selected = frame
            frame_time = frame.time
            if frame_time is None or frame_time >= target_seconds:
                break
        if selected is None:
            raise RuntimeError(f"No frame decoded from {path} at {target_seconds:.3f}s")
        image = selected.to_image().convert("RGB")
        image.save(output, format="JPEG", quality=95, subsampling=0)
        return float(selected.time or target_seconds), image.width, image.height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record_id in ("1775", "1779"):
        path = VIDEO_DIR / f"video_{record_id}_1.mp4"
        duration = video_duration_seconds(path)
        print(f"{record_id}: duration={duration:.3f}s")
        for label, fraction in (("quarter", 0.25), ("middle", 0.50), ("three_quarter", 0.75)):
            target = max(0.0, duration * fraction)
            output = args.output_dir / f"{record_id}_{label}.jpg"
            actual, width, height = extract_at(path, target, output)
            print(f"  {label}: target={target:.3f}s actual={actual:.3f}s size={width}x{height} -> {output}")


if __name__ == "__main__":
    main()
