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


WORKSPACE = Path(r"C:\Users\11378\Desktop\MMLLM")
SOURCE_ROOT = Path(r"C:\Users\11378\Desktop\虚假新闻项目")
GENERATED_ROOT = Path(
    r"C:\Users\11378\.codex\generated_images\01a193e7-be76-7592-b5c4-0de40daf49be"
)

JSON_TARGETS = [
    WORKSPACE / "fakenewsDataSet" / "data.json",
    SOURCE_ROOT / "数据集收集自己" / "data.json",
    SOURCE_ROOT / "PanoFND dataset.json",
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "⽂字_视频(假文字1201-1275)"
    / "⽂字_视频(假文字1201-1275).json",
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "文字_图片（假文字1351-1425）"
    / "文字_图片（假文字1351-1425）.json",
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "文字_图片_视频（假图片1651-1725）meizuowan"
    / "文字_图片_视频（假图片1651-1725没改label）.json",
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "文字_图片_视频（假图片1651-1725）meizuowan"
    / "文字_图片_视频_改label为fake.json",
]

IMAGE_SOURCE_DIR = (
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "文字_图片_视频（假图片1651-1725）meizuowan"
    / "文字_图片_视频（假图片1651-1725）_image"
)

GENERATED_IMAGES = {
    "1657": GENERATED_ROOT / "exec-bad295fa-30c0-4884-95f5-9fb18a241741.png",
    "1663": GENERATED_ROOT / "exec-34e57823-d06c-41d7-802f-38e68cac86fe.png",
    "1674": GENERATED_ROOT / "exec-4893ee23-e78b-496c-8a0a-7369181c6766.png",
}

IMAGE_TARGETS = {
    record_id: [
        WORKSPACE / "fakenewsDataSet" / "image" / f"image_{record_id}_1.jpg",
        IMAGE_SOURCE_DIR / f"image_{record_id}_1.jpg",
    ]
    for record_id in GENERATED_IMAGES
}

DESCRIPTIONS = {
    "1234": (
        "The video reports that the unemployment rate held steady at 4.2%, "
        "but the text falsely states that it rose to 4.7%."
    ),
    "1246": (
        "The video shows Bernie Sanders calling for an immediate cease-fire in Gaza, "
        "while the title and text falsely claim that he rejected a cease-fire and "
        "supported continuing military operations."
    ),
    "1424": (
        "The image shows that Joe Biden posted the TikTok video on February 11, "
        "while the text falsely states that he posted it on February 12."
    ),
    "1657": (
        "The image is unrelated to the article and video: it shows an empty, "
        "war-damaged Middle Eastern urban street, while the report concerns a snake "
        "found in the cargo hold of an Australian domestic flight."
    ),
    "1663": (
        "The image is unrelated to the article and video: it shows a Ukrainian city "
        "during a distant nighttime attack, while the report concerns the Dalai Lama's "
        "90th birthday and Tibetan culture."
    ),
    "1674": (
        "The image is unrelated to the article and video: it shows the aftermath of a "
        "gas-station explosion near Rome, while the report concerns King Charles III's "
        "commemoration of the 7/7 London bombings."
    ),
}

NEW_TEXT = {
    "1234": (
        "The unemployment rate rose to 4.7% as the Trump administration continued its "
        "efforts to cut federal jobs and spending. WSJ's Nick Timiraos explains what "
        "the report means for the economy and the Federal Reserve. "
    ),
    "1246": (
        "In a speech at the Democratic National Convention, Vermont Sen. Bernie Sanders "
        "rejected calls for an immediate cease-fire in Gaza and argued that military "
        "operations should continue. "
    ),
}


def normalized_id(value: object) -> str:
    return str(value).zfill(4)


def records_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Unsupported JSON structure")


def update_record(record: dict) -> list[str]:
    record_id = normalized_id(record.get("id"))
    if record_id not in DESCRIPTIONS:
        return []

    changed: list[str] = []
    if record_id == "1246":
        new_title = "Bernie Sanders Rejects Immediate Cease-Fire in Gaza at DNC"
        if record.get("title") != new_title:
            record["title"] = new_title
            changed.append("title")

    if record_id in NEW_TEXT and record.get("text") != NEW_TEXT[record_id]:
        record["text"] = NEW_TEXT[record_id]
        changed.append("text")

    if record_id == "1424":
        old_text = record.get("text", "")
        new_text = old_text.replace("joined TikTok on February 11", "joined TikTok on February 12", 1)
        if new_text == old_text and "joined TikTok on February 12" not in old_text:
            raise ValueError("ID 1424 does not contain the expected February 11 text")
        if new_text != old_text:
            record["text"] = new_text
            changed.append("text")

    if record.get("description") != DESCRIPTIONS[record_id]:
        record["description"] = DESCRIPTIONS[record_id]
        changed.append("description")
    return changed


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.before_remaining_six_fix_20261031{path.suffix}")


def write_json_atomic(path: Path, payload: object) -> None:
    temp = path.with_name(f".{path.name}.remaining-six.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def write_jpeg_atomic(source: Path, target: Path) -> None:
    temp = target.with_name(f".{target.stem}.remaining-six.tmp.jpg")
    with Image.open(source) as image:
        image.convert("RGB").save(temp, format="JPEG", quality=95, subsampling=0)
    os.replace(temp, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for path in JSON_TARGETS:
        if not path.is_file():
            raise FileNotFoundError(path)
    for source in GENERATED_IMAGES.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    for targets in IMAGE_TARGETS.values():
        for target in targets:
            if not target.is_file():
                raise FileNotFoundError(target)

    staged_json: dict[Path, object] = {}
    found_ids: dict[str, int] = {record_id: 0 for record_id in DESCRIPTIONS}
    for path in JSON_TARGETS:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        file_changes: list[str] = []
        for record in records_from_payload(payload):
            record_id = normalized_id(record.get("id"))
            fields = update_record(record)
            if fields:
                found_ids[record_id] += 1
                file_changes.append(f"{record_id}({','.join(fields)})")
        if file_changes:
            staged_json[path] = payload
            print(f"JSON {path}: {', '.join(file_changes)}")

    missing = [record_id for record_id, count in found_ids.items() if count == 0]
    if missing:
        raise RuntimeError(f"IDs not found in any JSON target: {missing}")

    for record_id, targets in IMAGE_TARGETS.items():
        source = GENERATED_IMAGES[record_id]
        for target in targets:
            print(f"IMAGE {source} -> {target}")

    if not args.apply:
        print("DRY RUN: no files changed")
        return

    all_targets = list(staged_json)
    all_targets.extend(target for targets in IMAGE_TARGETS.values() for target in targets)
    for target in all_targets:
        backup = backup_path(target)
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
    for target in all_targets:
        shutil.copy2(target, backup_path(target))

    for path, payload in staged_json.items():
        write_json_atomic(path, payload)
    for record_id, targets in IMAGE_TARGETS.items():
        for target in targets:
            write_jpeg_atomic(GENERATED_IMAGES[record_id], target)

    print(f"APPLIED: {len(staged_json)} JSON files and 6 image files updated")


if __name__ == "__main__":
    main()
