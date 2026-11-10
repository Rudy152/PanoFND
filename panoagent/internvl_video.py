from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

from .schemas import NewsSample


MODEL_ID = "OpenGVLab/InternVL2_5-2B"
MODEL_RELEASE_DATE = "2024-12-05"
PAPER_URL = "https://arxiv.org/abs/2412.05271"
OFFICIAL_REPOSITORY = "https://github.com/OpenGVLab/InternVL"

CLASSIFICATION_INSTRUCTIONS = """You are evaluating whether a news item is authentic.
Use all supplied evidence: the title, article text, audio transcript, news image, and
chronologically ordered video frames. REAL means the central factual claim is true
and the media are consistent with it. FAKE means the central claim is false,
misleading, manipulated, or the media materially contradict it. Do not decide from
writing style alone. If evidence is incomplete, choose the label best supported by
all available modalities.

Return exactly one line and no explanation:
LABEL: REAL
or
LABEL: FAKE"""


def parse_binary_label(output: str) -> str:
    """Parse the deliberately constrained InternVL classification response."""
    match = re.search(r"\bLABEL\s*:\s*(REAL|FAKE)\b", output, flags=re.I)
    if match:
        return match.group(1).upper()
    stripped = output.strip().upper()
    if stripped in {"REAL", "FAKE"}:
        return stripped
    return "ERROR"


def truncate_head_tail(value: str, max_chars: int | None) -> str:
    if max_chars is None or len(value) <= max_chars:
        return value
    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    marker = "\n...[MIDDLE TRUNCATED FOR LOCAL CONTEXT BUDGET]...\n"
    remaining = max_chars - len(marker)
    head = remaining // 2
    return value[:head] + marker + value[-(remaining - head) :]


def build_question(
    sample: NewsSample,
    visual_labels: Iterable[str],
    max_text_chars: int | None = None,
) -> str:
    prefixes = "".join(f"{label}: <image>\n" for label in visual_labels)
    evidence = truncate_head_tail(sample.detector_text(), max_text_chars)
    return f"{prefixes}{CLASSIFICATION_INSTRUCTIONS}\n\nNEWS ITEM:\n{evidence}".strip()


def _build_transform(input_size: int = 448):
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_difference = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_tiles: int = 1,
    max_tiles: int = 4,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """InternVL's official dynamic tiling, kept local for reproducible inference."""
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    ratios = sorted(
        {
            (columns, rows)
            for tile_count in range(min_tiles, max_tiles + 1)
            for columns in range(1, tile_count + 1)
            for rows in range(1, tile_count + 1)
            if min_tiles <= columns * rows <= max_tiles
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = _closest_aspect_ratio(
        width / height, ratios, width, height, image_size
    )
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_visual(path: str | Path, max_tiles: int) -> "object":
    import torch

    with Image.open(path) as handle:
        tiles = dynamic_preprocess(
            handle.convert("RGB"), max_tiles=max_tiles, use_thumbnail=True
        )
    transform = _build_transform()
    return torch.stack([transform(tile) for tile in tiles])


class InternVLVideoClassifier:
    """Thin deterministic wrapper around the official InternVL2.5 chat model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        max_new_tokens: int = 16,
    ) -> None:
        project_cache = Path(model_path).resolve().parents[1] / "cache" / "huggingface"
        os.environ.setdefault("HF_HOME", str(project_cache))
        os.environ.setdefault(
            "TRANSFORMERS_MODULES_CACHE", str(project_cache / "modules")
        )
        import torch
        from transformers import (
            AutoModel,
            AutoTokenizer,
            GenerationConfig,
            GenerationMixin,
        )

        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.max_new_tokens = max_new_tokens
        self.model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        # This 2024 remote-code model predates the Transformers 4.50 change
        # that removed GenerationMixin from PreTrainedModel. Restore the same
        # generation API without modifying the downloaded official files.
        language_model = self.model.language_model
        if not hasattr(language_model, "generate"):
            patched_class = type(
                f"{language_model.__class__.__name__}WithGeneration",
                (language_model.__class__, GenerationMixin),
                {},
            )
            language_model.__class__ = patched_class
        if getattr(language_model, "generation_config", None) is None:
            language_model.generation_config = GenerationConfig.from_model_config(
                language_model.config
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=False
        )

    def classify(
        self,
        sample: NewsSample,
        still_image_paths: Iterable[str | Path],
        video_frame_paths: Iterable[str | Path],
        image_max_tiles: int = 4,
        max_text_chars: int | None = 12_000,
    ) -> tuple[str, str, dict[str, int]]:
        import torch

        still_image_paths = tuple(still_image_paths)
        video_frame_paths = tuple(video_frame_paths)
        tensors = []
        patch_counts: list[int] = []
        visual_labels: list[str] = []

        for index, path in enumerate(still_image_paths, 1):
            pixels = load_visual(path, max_tiles=image_max_tiles)
            tensors.append(pixels)
            patch_counts.append(int(pixels.shape[0]))
            visual_labels.append(f"News Image-{index}")
        for index, path in enumerate(video_frame_paths, 1):
            pixels = load_visual(path, max_tiles=1)
            tensors.append(pixels)
            patch_counts.append(int(pixels.shape[0]))
            visual_labels.append(f"Video Frame-{index}")

        pixel_values = None
        if tensors:
            pixel_values = torch.cat(tensors).to(self.dtype).to(self.device)
        question = build_question(
            sample, visual_labels, max_text_chars=max_text_chars
        )
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        kwargs = {}
        if patch_counts:
            kwargs["num_patches_list"] = patch_counts
        with torch.inference_mode():
            output = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False,
                **kwargs,
            )
        output = str(output).strip()
        metadata = {
            "still_images": len(still_image_paths),
            "video_frames": len(video_frame_paths),
            "visual_tiles": sum(patch_counts),
        }
        return parse_binary_label(output), output, metadata
