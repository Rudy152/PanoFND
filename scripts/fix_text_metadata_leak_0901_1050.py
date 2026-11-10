from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


WORKSPACE = Path(r"C:\Users\11378\Desktop\MMLLM")
SOURCE_ROOT = Path(r"C:\Users\11378\Desktop\虚假新闻项目")

JSON_TARGETS = [
    WORKSPACE / "fakenewsDataSet" / "data.json",
    SOURCE_ROOT / "数据集收集自己" / "data.json",
    SOURCE_ROOT / "PanoFND dataset.json",
    SOURCE_ROOT / "数据集收集自己" / "文字（真新闻901-975）" / "文字.json",
    SOURCE_ROOT
    / "数据集收集自己"
    / "假新闻"
    / "文字(假文字976-1050)"
    / "文字(假文字976-1050).json",
]

TARGET_IDS = {f"{value:04d}" for value in range(901, 1051)}
METADATA_FIELDS = ("title", "source", "publish_time")
BACKUP_MARKER = "before_text_metadata_normalization_20261031"


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


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.{BACKUP_MARKER}{path.suffix}")


def write_json_atomic(path: Path, payload: object) -> None:
    temp = path.with_name(f".{path.name}.text-metadata.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    staged: dict[Path, object] = {}
    coverage: dict[Path, set[str]] = {}

    for path in JSON_TARGETS:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)

        found: set[str] = set()
        changed_ids: list[str] = []
        changed_fields = 0
        for record in records_from_payload(payload):
            record_id = normalized_id(record.get("id"))
            if record_id not in TARGET_IDS:
                continue
            found.add(record_id)
            record_changed = False
            for field in METADATA_FIELDS:
                if record.get(field, "") != "":
                    record[field] = ""
                    changed_fields += 1
                    record_changed = True
            if record_changed:
                changed_ids.append(record_id)

        coverage[path] = found
        if changed_ids:
            staged[path] = payload
        print(
            f"{path}: found={len(found)}, changed_records={len(changed_ids)}, "
            f"changed_fields={changed_fields}"
        )

    primary_targets = JSON_TARGETS[:3]
    for path in primary_targets:
        missing = TARGET_IDS - coverage[path]
        if missing:
            raise RuntimeError(f"{path} is missing target IDs: {sorted(missing)}")

    true_source_ids = {f"{value:04d}" for value in range(901, 976)}
    fake_source_ids = {f"{value:04d}" for value in range(976, 1051)}
    if coverage[JSON_TARGETS[3]] != true_source_ids:
        raise RuntimeError("The true-news source file does not contain exactly IDs 0901-0975")
    if coverage[JSON_TARGETS[4]] != fake_source_ids:
        raise RuntimeError("The fake-news source file does not contain exactly IDs 0976-1050")

    if not args.apply:
        print("DRY RUN: no files changed")
        return

    for path in staged:
        backup = backup_path(path)
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
    for path in staged:
        shutil.copy2(path, backup_path(path))
    for path, payload in staged.items():
        write_json_atomic(path, payload)

    print(f"APPLIED: {len(staged)} JSON files updated")


if __name__ == "__main__":
    main()
