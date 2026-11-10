from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


MIRAGENEWS_TEXT_DIMENSION = 768


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_from_fake_probability(probability: float, threshold: float) -> str:
    """Apply the authors' convention: label 1 is fake and uses a strict > threshold."""
    return "FAKE" if probability > threshold else "REAL"


def load_text_classifier(checkpoint_path: str | Path, device: str) -> tuple[Any, float]:
    """Load the official MiRAGe-Txt linear head without allowing pickle code execution."""
    import torch

    class TextClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(MIRAGENEWS_TEXT_DIMENSION, 1)

        def forward(self, features: Any) -> Any:
            return torch.sigmoid(self.linear(features))

    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("MiRAGe checkpoint must be a dictionary")
    state_dict = payload.get("model_state_dict")
    threshold = payload.get("best_threshold")
    if not isinstance(state_dict, dict) or not isinstance(threshold, (int, float)):
        raise ValueError("MiRAGe checkpoint is missing model_state_dict or best_threshold")

    classifier = TextClassifier()
    classifier.load_state_dict(state_dict, strict=True)
    classifier.to(device).eval()
    return classifier, float(threshold)
