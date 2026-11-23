"""L1: local forensic detectors. They emit scores used as features, never verdicts.

Available now: a CLIP ViT-L/14-336 zero-shot "synthetic/edited vs. camera photo"
score, using weights already on disk (models/miragenews/openai_clip). It is a weak,
generic detector; the trained scorer decides how much weight it deserves.

Stronger detectors (MiRAGe-Img needs BLIP-2 flan-t5-xl, ~15 GB; an audio anti-spoofing
model such as AASIST) need downloads and are deliberately not fetched automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

REAL_PROMPTS = (
    "a real photograph taken with a camera",
    "an authentic unedited news photo",
    "a genuine photo of a real scene",
)
FAKE_PROMPTS = (
    "an AI-generated image",
    "a synthetic computer-generated picture",
    "a digitally manipulated or photoshopped image",
)


class ClipSyntheticImageDetector:
    name = "clip_zero_shot_v1"

    def __init__(self, weights: str | Path, cache_path: str | Path, device: str | None = None):
        self.weights = Path(weights)
        self.cache_path = Path(cache_path)
        self.device = device
        self._model = None
        self._preprocess = None
        self._text = None
        self._lock = threading.Lock()
        self._cache: dict[str, float] = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _load(self) -> None:
        if self._model is not None:
            return
        import clip
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = clip.load(str(self.weights), device=device)
        model.eval()
        with torch.no_grad():
            tokens = clip.tokenize(list(REAL_PROMPTS + FAKE_PROMPTS)).to(device)
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)
        self._model, self._preprocess, self._text, self.device = model, preprocess, text, device

    def score(self, path: Path, digest: str) -> float:
        """Probability mass on the synthetic/edited prompts (0..1)."""
        key = f"{self.name}:{digest}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            self._load()
            import torch
            from PIL import Image

            with Image.open(path) as image:
                pixels = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self._model.encode_image(pixels)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                probs = (100.0 * feature @ self._text.T).softmax(dim=-1)[0]
            value = float(probs[len(REAL_PROMPTS):].sum().item())
            self._cache[key] = value
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self.cache_path)
            return value
