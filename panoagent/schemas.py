from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_MODALITIES = {"text", "image", "audio", "video"}
VALID_LABELS = {"true", "fake"}


@dataclass(slots=True)
class NewsSample:
    id: str
    modalities: tuple[str, ...]
    title: str
    text: str
    image_paths: tuple[Path, ...]
    video_paths: tuple[Path, ...]
    audio_paths: tuple[Path, ...]
    label: str
    source: str
    publish_time: str
    description: str
    transcript: str = ""
    path_repairs: tuple[str, ...] = ()

    @property
    def binary_label(self) -> str:
        return "REAL" if self.label == "true" else "FAKE"

    @property
    def modality_key(self) -> str:
        return "+".join(self.modalities)

    def detector_text(self, include_transcript: bool = True) -> str:
        """Return model-visible text. Description is deliberately excluded."""
        parts: list[str] = []
        if self.title.strip():
            parts.append(f"TITLE:\n{self.title.strip()}")
        if self.text.strip():
            parts.append(f"ARTICLE TEXT:\n{self.text.strip()}")
        if include_transcript and self.transcript.strip():
            parts.append(f"AUDIO TRANSCRIPT:\n{self.transcript.strip()}")
        if self.publish_time.strip():
            parts.append(f"PUBLICATION DATE (metadata): {self.publish_time.strip()}")
        return "\n\n".join(parts)


@dataclass(slots=True)
class ModelRequest:
    task_name: str
    system_prompt: str
    user_prompt: str
    image_paths: tuple[Path, ...] = ()
    audio_paths: tuple[Path, ...] = ()
    video_paths: tuple[Path, ...] = ()
    max_new_tokens: int = 768
    temperature: float = 0.0


@dataclass(slots=True)
class StageResult:
    stage: str
    data: dict[str, Any]
    raw_output: str
    cached: bool = False
    error: str | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PredictionRecord:
    id: str
    gold: str
    prediction: str
    modality_key: str
    raw_output: str
    backend: str
    latency_seconds: float
    error: str | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
