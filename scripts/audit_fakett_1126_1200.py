from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


SOURCE = Path(r"C:\Users\11378\Desktop\虚假新闻项目\数据集收集\FakeTT_DATA_OPENSOURCE\data.json")
DATASET = Path(r"C:\Users\11378\Desktop\MMLLM\fakenewsDataSet\data.json")


def main() -> None:
    source_rows = []
    with SOURCE.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                source_rows.append((line_number, json.loads(line)))

    first_75_fake = [(line_number, row) for line_number, row in source_rows if row.get("annotation") == "fake"][:75]

    with DATASET.open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    by_id = {str(record["id"]).zfill(4): record for record in records}
    current = [by_id[f"{value:04d}"] for value in range(1126, 1201)]

    title_matches = 0
    text_matches = 0
    date_matches = 0
    label_matches = 0
    for record, (_, source) in zip(current, first_75_fake):
        title_matches += record.get("title") == source.get("event")
        text_matches += record.get("text") == source.get("description")
        source_date = datetime.fromtimestamp(source["publish_time"] / 1000, tz=timezone.utc).strftime("%Y/%m/%d")
        current_date = datetime.strptime(record.get("publish_time"), "%Y/%m/%d").strftime("%Y/%m/%d")
        date_matches += current_date == source_date
        label_matches += record.get("label") == "fake"

    event_counts = Counter(row["event"] for _, row in first_75_fake)
    repeated_rows = sum(count for count in event_counts.values() if count > 1)

    print(f"source_format=jsonl")
    print(f"source_rows={len(source_rows)}")
    print(f"source_fake_rows={sum(row.get('annotation') == 'fake' for _, row in source_rows)}")
    print(f"selected_first_75_fake={len(first_75_fake)}")
    print(f"dataset_range=1126-1200 count={len(current)}")
    print(f"id_1125_label={by_id['1125'].get('label')}")
    print(f"title_event_matches={title_matches}/75")
    print(f"text_description_matches={text_matches}/75")
    print(f"publish_date_matches={date_matches}/75")
    print(f"fake_label_matches={label_matches}/75")
    print(f"unique_events={len(event_counts)}")
    print(f"rows_in_repeated_event_groups={repeated_rows}/75")
    print("largest_event_groups:")
    for event, count in event_counts.most_common(10):
        print(f"{count:2d}  {event}")


if __name__ == "__main__":
    main()
