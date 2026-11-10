from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schemas import NewsSample, VALID_LABELS, VALID_MODALITIES


_MEDIA_FIELDS = {
    "image": "image_paths",
    "audio": "audio_paths",
    "video": "video_paths",
}


class DatasetError(RuntimeError):
    pass


class PanoFNDDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.data_path = self.root / "data.json"
        if not self.data_path.is_file():
            raise DatasetError(f"Missing dataset manifest: {self.data_path}")
        with self.data_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise DatasetError("data.json must contain a JSON array")
        self._raw: list[dict[str, Any]] = raw
        self._name_maps: dict[str, dict[str, Path]] = {}

    def __len__(self) -> int:
        return len(self._raw)

    def ids(self) -> list[str]:
        return [str(row.get("id", "")) for row in self._raw]

    def _name_map(self, kind: str) -> dict[str, Path]:
        if kind not in self._name_maps:
            folder = self.root / kind
            self._name_maps[kind] = {
                path.name.casefold(): path.resolve()
                for path in folder.iterdir()
                if path.is_file()
            }
        return self._name_maps[kind]

    def resolve_media_path(self, kind: str, declared: str, sample_id: str) -> tuple[Path, str | None]:
        folder = self.root / kind
        direct = (folder / declared).resolve()
        if direct.is_file():
            return direct, None

        name_map = self._name_map(kind)
        declared_name = Path(declared).name
        candidates = [declared_name]

        # One contiguous block in the supplied manifest spells the video prefix
        # as "vedio" while the files on disk use "video".
        if kind == "video" and declared_name.casefold().startswith("vedio_"):
            candidates.append("video_" + declared_name[len("vedio_"):])

        match = re.match(r"^(image|audio|video)_(\d+)(_.+)$", declared_name, flags=re.I)
        if match:
            candidates.append(f"{match.group(1)}_{int(match.group(2)):04d}{match.group(3)}")

        suffix_match = re.match(r"^(image|audio|video)_\d+(_.+)$", declared_name, flags=re.I)
        if suffix_match and sample_id.isdigit():
            candidates.append(f"{suffix_match.group(1)}_{int(sample_id):04d}{suffix_match.group(2)}")

        for candidate in candidates:
            resolved = name_map.get(candidate.casefold())
            if resolved is not None:
                return resolved, f"{kind}/{declared} -> {kind}/{resolved.name}"
        return direct, None

    def _load_transcript(self, audio_paths: tuple[Path, ...]) -> str:
        chunks: list[str] = []
        for audio_path in audio_paths:
            transcript_path = audio_path.with_suffix(".txt")
            if transcript_path.is_file():
                chunks.append(transcript_path.read_text(encoding="utf-8", errors="replace").strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def sample_from_row(self, row: dict[str, Any]) -> NewsSample:
        sample_id = str(row.get("id", ""))
        repairs: list[str] = []
        resolved: dict[str, tuple[Path, ...]] = {}
        for kind, field_name in _MEDIA_FIELDS.items():
            paths: list[Path] = []
            for declared in row.get(field_name, []) or []:
                path, repair = self.resolve_media_path(kind, str(declared), sample_id)
                paths.append(path)
                if repair:
                    repairs.append(repair)
            resolved[kind] = tuple(paths)

        audio_paths = resolved["audio"]
        return NewsSample(
            id=sample_id,
            modalities=tuple(str(value) for value in row.get("modalities", [])),
            title=str(row.get("title", "") or ""),
            text=str(row.get("text", "") or ""),
            image_paths=resolved["image"],
            video_paths=resolved["video"],
            audio_paths=audio_paths,
            label=str(row.get("label", "") or ""),
            source=str(row.get("source", "") or ""),
            publish_time=str(row.get("publish_time", "") or ""),
            description=str(row.get("description", "") or ""),
            transcript=self._load_transcript(audio_paths),
            path_repairs=tuple(repairs),
        )

    def __iter__(self) -> Iterator[NewsSample]:
        for row in self._raw:
            yield self.sample_from_row(row)

    def iter_ids(self, ids: Iterable[str]) -> Iterator[NewsSample]:
        wanted = set(ids)
        for row in self._raw:
            if str(row.get("id", "")) in wanted:
                yield self.sample_from_row(row)

    @staticmethod
    def group_key(sample: NewsSample) -> str:
        source = sample.source.strip().casefold()
        return f"source:{source}" if source else f"orphan:{sample.id}"

    def create_grouped_split(
        self,
        seed: int = 42,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.10,
    ) -> dict[str, Any]:
        if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
            raise ValueError("Ratios must leave a positive test partition")
        groups: dict[str, list[NewsSample]] = defaultdict(list)
        for sample in self:
            groups[self.group_key(sample)].append(sample)

        partitions: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        group_assignment: dict[str, str] = {}
        for key in sorted(groups):
            digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
            score = int.from_bytes(digest[:8], "big") / 2**64
            if score < train_ratio:
                partition = "train"
            elif score < train_ratio + validation_ratio:
                partition = "validation"
            else:
                partition = "test"
            group_assignment[key] = partition
            partitions[partition].extend(sample.id for sample in groups[key])

        for values in partitions.values():
            values.sort()
        return {
            "strategy": "sha256_source_grouped",
            "seed": seed,
            "ratios": {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": 1 - train_ratio - validation_ratio,
            },
            "counts": {name: len(values) for name, values in partitions.items()},
            "groups": {name: sum(1 for value in group_assignment.values() if value == name) for name in partitions},
            "partitions": partitions,
        }

    def create_clean_holdout(
        self,
        candidate_ids: Iterable[str],
        excluded_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Remove every source group touched by prior model selection.

        Excluding whole groups, rather than only individual rows, prevents a
        real/fake variant or syndicated copy of a used item entering final test.
        """
        samples = {sample.id: sample for sample in self}
        candidates = {str(value) for value in candidate_ids}
        excluded = {str(value) for value in excluded_ids}
        unknown = sorted((candidates | excluded).difference(samples))
        if unknown:
            raise DatasetError(f"Unknown sample IDs: {unknown[:10]}")
        excluded_groups = {self.group_key(samples[sample_id]) for sample_id in excluded}
        clean = sorted(
            sample_id for sample_id in candidates
            if self.group_key(samples[sample_id]) not in excluded_groups
        )
        removed = sorted(candidates.difference(clean))
        return {
            "strategy": "source_group_exclusion_after_model_selection",
            "candidate_count": len(candidates),
            "excluded_id_count": len(excluded),
            "excluded_source_groups": len(excluded_groups),
            "removed_candidate_count": len(removed),
            "counts": {"test": len(clean)},
            "class_counts": {
                "REAL": sum(samples[sample_id].binary_label == "REAL" for sample_id in clean),
                "FAKE": sum(samples[sample_id].binary_label == "FAKE" for sample_id in clean),
            },
            "partitions": {"test": clean},
        }

    def create_stratified_subset(
        self,
        candidate_ids: Iterable[str],
        per_label_modality: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Select equal label counts within every modality combination."""
        if per_label_modality < 1:
            raise ValueError("per_label_modality must be positive")
        candidate_set = set(candidate_ids)
        cells: dict[tuple[str, str], list[str]] = defaultdict(list)
        for sample in self:
            if sample.id in candidate_set:
                cells[(sample.label, sample.modality_key)].append(sample.id)

        expected = {
            (label, modality)
            for label in VALID_LABELS
            for modality in sorted({sample.modality_key for sample in self})
        }
        missing = {
            f"{label}|{modality}": len(cells.get((label, modality), []))
            for label, modality in expected
            if len(cells.get((label, modality), [])) < per_label_modality
        }
        if missing:
            raise DatasetError(
                f"Not enough candidates for requested stratified subset: {missing}"
            )

        rng = random.Random(seed)
        selected: list[str] = []
        cell_counts: dict[str, int] = {}
        for cell in sorted(expected):
            values = sorted(cells[cell])
            rng.shuffle(values)
            chosen = values[:per_label_modality]
            selected.extend(chosen)
            cell_counts[f"{cell[0]}|{cell[1]}"] = len(chosen)
        selected.sort()
        return {
            "strategy": "label_modality_stratified_from_existing_partition",
            "seed": seed,
            "per_label_modality": per_label_modality,
            "counts": {"pilot": len(selected)},
            "cell_counts": cell_counts,
            "partitions": {"pilot": selected},
        }

    def audit(self) -> dict[str, Any]:
        labels: Counter[str] = Counter()
        combinations: Counter[str] = Counter()
        declared_missing: list[str] = []
        repaired: list[str] = []
        unresolved: list[str] = []
        transcript_rows = 0
        description_label_leaks = 0
        ids: list[str] = []
        sources: Counter[str] = Counter()

        leak_pattern = re.compile(r"fake news|does not match|incorrect|fabricated|mismatch|error", re.I)
        for sample in self:
            ids.append(sample.id)
            labels[sample.label] += 1
            combinations[sample.modality_key] += 1
            sources[sample.source.strip()] += 1
            if sample.transcript:
                transcript_rows += 1
            if sample.label == "fake" and leak_pattern.search(sample.description):
                description_label_leaks += 1
            repaired.extend(sample.path_repairs)
            for kind in _MEDIA_FIELDS:
                for path in getattr(sample, f"{kind}_paths"):
                    if not path.is_file():
                        unresolved.append(f"{kind}/{path.name}")

        for row in self._raw:
            for kind, field_name in _MEDIA_FIELDS.items():
                for declared in row.get(field_name, []) or []:
                    if not (self.root / kind / str(declared)).is_file():
                        declared_missing.append(f"{kind}/{declared}")

        invalid_modalities = sorted({mod for row in self._raw for mod in row.get("modalities", []) if mod not in VALID_MODALITIES})
        invalid_labels = sorted(set(labels) - VALID_LABELS)
        return {
            "dataset_root": str(self.root),
            "rows": len(self),
            "unique_ids": len(set(ids)),
            "duplicate_ids": sorted(key for key, count in Counter(ids).items() if count > 1),
            "labels": dict(sorted(labels.items())),
            "modality_combinations": dict(sorted(combinations.items())),
            "unique_nonempty_sources": len([key for key in sources if key]),
            "source_groups_with_multiple_rows": sum(1 for key, count in sources.items() if key and count > 1),
            "description_label_leak_rows": description_label_leaks,
            "transcript_rows": transcript_rows,
            "declared_missing_paths": len(declared_missing),
            "repaired_paths": len(repaired),
            "unresolved_paths": len(unresolved),
            "unresolved_examples": unresolved[:25],
            "repair_examples": repaired[:25],
            "invalid_modalities": invalid_modalities,
            "invalid_labels": invalid_labels,
            "warnings": [
                "Never pass description to a detector; it frequently discloses the gold label.",
                "Use source-grouped splits to prevent real/fake variants of one article crossing partitions.",
            ],
        }


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def load_split(path: str | Path, partition: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        values = payload["partitions"][partition]
    except (KeyError, TypeError) as exc:
        raise DatasetError(f"Split does not contain partition {partition!r}") from exc
    return [str(value) for value in values]
