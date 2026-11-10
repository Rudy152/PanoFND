from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


ROOT = Path(r"C:\Users\11378\Desktop\MMLLM")
DATASET_ROOT = ROOT / "fakenewsDataSet"
DATA_PATH = DATASET_ROOT / "data.json"
IMAGE_DIR = DATASET_ROOT / "image"
CANDIDATE_DIR = ROOT / "artifacts" / "frame_candidates"
BACKUP_DIR = DATASET_ROOT / "backups" / "media_path_finalize_20261031"
DATA_BACKUP = DATASET_ROOT / "data.before_media_path_finalize_20261031.json"

SELECTED_FRAMES = {
    "1775": CANDIDATE_DIR / "1775_middle.jpg",
    "1779": CANDIDATE_DIR / "1779_middle.jpg",
}


def load_records() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("data.json must contain a JSON array")
    return payload


def write_json_atomic(payload: list[dict]) -> None:
    temp = DATA_PATH.with_name(f".{DATA_PATH.name}.media-finalize.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, DATA_PATH)


def validate_jpeg(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG: {path}, got {image.format}")
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    records = load_records()
    by_id = {str(row.get("id", "")).zfill(4): row for row in records}
    expected_ids = {f"{value:04d}" for value in range(1, 1951)}
    if set(by_id) != expected_ids:
        raise RuntimeError("Canonical data.json does not contain exactly IDs 0001-1950")

    counts = {
        "image_zero_padding": 0,
        "audio_zero_padding": 0,
        "video_prefix": 0,
        "png_reference": 0,
    }

    for value in range(676, 751):
        record_id = f"{value:04d}"
        record = by_id[record_id]
        old_image = f"image_{value}_1.jpg"
        new_image = f"image_{record_id}_1.jpg"
        old_audio = f"audio_{value}_1.mp3"
        new_audio = f"audio_{record_id}_1.mp3"
        if record.get("image_paths") == [old_image]:
            record["image_paths"] = [new_image]
            counts["image_zero_padding"] += 1
        elif record.get("image_paths") != [new_image]:
            raise ValueError(f"Unexpected image path for {record_id}: {record.get('image_paths')}")
        if record.get("audio_paths") == [old_audio]:
            record["audio_paths"] = [new_audio]
            counts["audio_zero_padding"] += 1
        elif record.get("audio_paths") != [new_audio]:
            raise ValueError(f"Unexpected audio path for {record_id}: {record.get('audio_paths')}")

    for value in range(1201, 1276):
        record_id = f"{value:04d}"
        record = by_id[record_id]
        old_video = f"vedio_{record_id}_1.mp4"
        new_video = f"video_{record_id}_1.mp4"
        if record.get("video_paths") == [old_video]:
            record["video_paths"] = [new_video]
            counts["video_prefix"] += 1
        elif record.get("video_paths") != [new_video]:
            raise ValueError(f"Unexpected video path for {record_id}: {record.get('video_paths')}")

    for record_id in ("1727", "1728"):
        record = by_id[record_id]
        old_image = f"image_{record_id}_1.jpg"
        new_image = f"image_{record_id}_1.png"
        if record.get("image_paths") == [old_image]:
            record["image_paths"] = [new_image]
            counts["png_reference"] += 1
        elif record.get("image_paths") != [new_image]:
            raise ValueError(f"Unexpected image path for {record_id}: {record.get('image_paths')}")

    for record_id, source in SELECTED_FRAMES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        width, height = validate_jpeg(source)
        print(f"FRAME {record_id}: {source.name}, {width}x{height}")

    for value in range(676, 751):
        record_id = f"{value:04d}"
        for kind, filename in (
            ("image", f"image_{record_id}_1.jpg"),
            ("audio", f"audio_{record_id}_1.mp3"),
        ):
            path = DATASET_ROOT / kind / filename
            if not path.is_file():
                raise FileNotFoundError(path)
    for value in range(1201, 1276):
        path = DATASET_ROOT / "video" / f"video_{value:04d}_1.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
    for record_id in ("1727", "1728"):
        path = IMAGE_DIR / f"image_{record_id}_1.png"
        if not path.is_file():
            raise FileNotFoundError(path)

    print(json.dumps(counts, ensure_ascii=False))
    for record_id in SELECTED_FRAMES:
        print(f"CREATE {IMAGE_DIR / f'image_{record_id}_1.jpg'}")
    for record_id in ("1727", "1728"):
        print(
            f"MOVE {IMAGE_DIR / f'image_{record_id}_1.jpg'} -> "
            f"{BACKUP_DIR / f'image_{record_id}_1.jpg'}"
        )

    if not args.apply:
        print("DRY RUN: no files changed")
        return

    if DATA_BACKUP.exists():
        raise FileExistsError(DATA_BACKUP)
    for record_id in SELECTED_FRAMES:
        target = IMAGE_DIR / f"image_{record_id}_1.jpg"
        if target.exists():
            raise FileExistsError(target)
    for record_id in ("1727", "1728"):
        target = BACKUP_DIR / f"image_{record_id}_1.jpg"
        if target.exists():
            raise FileExistsError(target)

    shutil.copy2(DATA_PATH, DATA_BACKUP)
    for record_id, source in SELECTED_FRAMES.items():
        target = IMAGE_DIR / f"image_{record_id}_1.jpg"
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    write_json_atomic(records)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for record_id in ("1727", "1728"):
        source = IMAGE_DIR / f"image_{record_id}_1.jpg"
        target = BACKUP_DIR / source.name
        shutil.move(str(source), str(target))

    print("APPLIED: data.json updated, 2 frames created, 2 superseded JPEGs archived")


if __name__ == "__main__":
    main()
