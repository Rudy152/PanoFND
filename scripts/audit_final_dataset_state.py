from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from panoagent.dataset import PanoFNDDataset


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


WORKSPACE = Path(r"C:\Users\11378\Desktop\MMLLM")
ROOT = WORKSPACE / "fakenewsDataSet"
CANONICAL = ROOT / "data.json"
COMPARISONS = [
    Path(r"C:\Users\11378\Desktop\虚假新闻项目\数据集收集自己\data.json"),
    Path(r"C:\Users\11378\Desktop\虚假新闻项目\PanoFND dataset.json"),
]

MEDIA_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac"},
    "video": {".mp4", ".mov", ".mkv", ".webm"},
}


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array: {path}")
    return payload


def compare_records(reference: list[dict], candidate: list[dict]) -> dict:
    left = {str(row.get("id", "")): row for row in reference}
    right = {str(row.get("id", "")): row for row in candidate}
    field_counts: Counter[str] = Counter()
    differing_ids: list[str] = []
    for record_id in sorted(set(left) & set(right)):
        fields = sorted(set(left[record_id]) | set(right[record_id]))
        changed = False
        for field in fields:
            if left[record_id].get(field) != right[record_id].get(field):
                field_counts[field] += 1
                changed = True
        if changed:
            differing_ids.append(record_id)
    return {
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "missing_ids": sorted(set(left) - set(right)),
        "extra_ids": sorted(set(right) - set(left)),
        "differing_ids": differing_ids,
        "field_counts": dict(field_counts),
    }


def main() -> None:
    raw = load(CANONICAL)
    dataset = PanoFNDDataset(ROOT)
    audit = dataset.audit()
    print("DATASET_AUDIT")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    resolved_references: dict[str, list[Path]] = {kind: [] for kind in MEDIA_EXTENSIONS}
    empty_referenced: list[str] = []
    for sample in dataset:
        for kind in MEDIA_EXTENSIONS:
            for path in getattr(sample, f"{kind}_paths"):
                if path.is_file():
                    resolved_references[kind].append(path.resolve())
                    if path.stat().st_size == 0:
                        empty_referenced.append(str(path))

    print("MEDIA_INVENTORY")
    for kind, extensions in MEDIA_EXTENSIONS.items():
        actual = {
            path.resolve()
            for path in (ROOT / kind).iterdir()
            if path.is_file()
            and path.suffix.casefold() in extensions
            and ".before_" not in path.name
        }
        referenced = set(resolved_references[kind])
        duplicate_refs = len(resolved_references[kind]) - len(referenced)
        print(
            json.dumps(
                {
                    "kind": kind,
                    "actual_active_files": len(actual),
                    "resolved_unique_references": len(referenced),
                    "duplicate_resolved_references": duplicate_refs,
                    "unreferenced_active_files": sorted(path.name for path in actual - referenced),
                },
                ensure_ascii=False,
            )
        )
    print(f"empty_referenced_files={len(empty_referenced)}")

    print("JSON_COMPARISONS")
    for path in COMPARISONS:
        result = compare_records(raw, load(path))
        result["path"] = str(path)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
