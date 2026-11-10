from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


CSV_PATH = Path(r"C:\Users\11378\Desktop\虚假新闻项目\数据集收集\文字\VLPFN\train.csv")
SPLIT_PATHS = [CSV_PATH, CSV_PATH.with_name("dev.csv"), CSV_PATH.with_name("test.csv")]
DATA_PATH = Path(r"C:\Users\11378\Desktop\MMLLM\fakenewsDataSet\data.json")


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    with DATA_PATH.open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    by_id = {str(record["id"]).zfill(4): record for record in records}

    real_rows = [row for row in rows if clean(row.get("det_fake_label")) == "0"][:75]
    fake_rows = [row for row in rows if clean(row.get("det_fake_label")) == "1"][:75]

    print(f"columns={columns}")
    print(f"row_count={len(rows)}")
    print(f"label_counts={dict(Counter(clean(row.get('det_fake_label')) for row in rows))}")
    print(f"first_75_real={len(real_rows)} first_75_fake={len(fake_rows)}")
    print(f"fake_folder_nonblank={sum(bool(clean(row.get('folder'))) for row in fake_rows)}/75")
    print(f"fake_source_nonblank={sum(bool(clean(row.get('source'))) for row in fake_rows)}/75")
    print(f"fake_source_values={dict(Counter(clean(row.get('source')) for row in fake_rows))}")
    print(f"fake_folder_values={dict(Counter(clean(row.get('folder')) for row in fake_rows))}")

    expected_metadata = {"title", "publish_time", "date", "url", "link"}
    print(f"explicit_metadata_columns={sorted(expected_metadata.intersection(columns))}")
    url_values = [
        clean(row.get(column))
        for row in fake_rows
        for column in columns
        if clean(row.get(column)).startswith(("http://", "https://"))
    ]
    print(f"fake_explicit_url_values={len(url_values)}")

    real_ids = [f"{value:04d}" for value in range(901, 976)]
    fake_ids = [f"{value:04d}" for value in range(976, 1051)]
    real_exact = sum(clean(row.get("content")) == clean(by_id[id_].get("text")) for id_, row in zip(real_ids, real_rows))
    fake_exact = sum(clean(row.get("content")) == clean(by_id[id_].get("text")) for id_, row in zip(fake_ids, fake_rows))
    print(f"sequential_text_exact_real={real_exact}/75")
    print(f"sequential_text_exact_fake={fake_exact}/75")

    for label, ids in (("real", real_ids), ("fake", fake_ids)):
        print(
            f"dataset_{label}_metadata="
            f"title:{sum(bool(clean(by_id[id_].get('title'))) for id_ in ids)}/75,"
            f"source:{sum(bool(clean(by_id[id_].get('source'))) for id_ in ids)}/75,"
            f"publish_time:{sum(bool(clean(by_id[id_].get('publish_time'))) for id_ in ids)}/75"
        )

    date_pattern = re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b",
        re.IGNORECASE,
    )
    fake_with_date_in_content = [row for row in fake_rows if date_pattern.search(clean(row.get("content")))]
    print(f"fake_content_with_detectable_date={len(fake_with_date_in_content)}/75")

    duplicate_fake_contents = len(fake_rows) - len({clean(row.get("content")) for row in fake_rows})
    print(f"duplicate_fake_contents={duplicate_fake_contents}")

    all_split_rows: list[dict] = []
    for split_path in SPLIT_PATHS:
        with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
            all_split_rows.extend(csv.DictReader(handle))
    all_real_rows = [row for row in all_split_rows if clean(row.get("det_fake_label")) == "0"]
    for join_key in ("", "Unnamed: 0", "idx"):
        real_values = {clean(row.get(join_key)) for row in all_real_rows}
        matched = sum(clean(row.get(join_key)) in real_values for row in fake_rows)
        print(f"fake_to_real_match_by_{join_key or 'first_column'}={matched}/75")

    print("sample_fake_rows:")
    for dataset_id, row in list(zip(fake_ids, fake_rows))[:5]:
        content = clean(row.get("content")).replace("\n", " ")
        print(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "csv_idx": clean(row.get("idx")),
                    "folder": clean(row.get("folder")),
                    "source": clean(row.get("source")),
                    "content_prefix": content[:120],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
