from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


class StageCache:
    def __init__(self, root: str | Path, backend_name: str):
        self.root = Path(root) / safe_component(backend_name)

    def path_for(self, sample_id: str, stage: str) -> Path:
        return self.root / safe_component(sample_id) / f"{safe_component(stage)}.json"

    def load(self, sample_id: str, stage: str) -> dict[str, Any] | None:
        path = self.path_for(sample_id, stage)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, sample_id: str, stage: str, payload: dict[str, Any]) -> None:
        path = self.path_for(sample_id, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

