from __future__ import annotations

import time

from . import prompts
from .model_clients import ModelBackend
from .parsing import parse_binary_label
from .schemas import ModelRequest, NewsSample, PredictionRecord


def run_direct_sample(
    sample: NewsSample,
    backend: ModelBackend,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    prompt_profile: str = "paper",
) -> PredictionRecord:
    if prompt_profile == "paper":
        system_prompt = prompts.DIRECT_SYSTEM
        user_template = prompts.DIRECT_USER
    elif prompt_profile == "debiased":
        system_prompt = prompts.DIRECT_DEBIASED_SYSTEM
        user_template = prompts.DIRECT_DEBIASED_USER
    else:
        raise ValueError(f"Unknown direct prompt profile: {prompt_profile}")
    request = ModelRequest(
        task_name="direct_detection",
        system_prompt=system_prompt,
        user_prompt=user_template.format(
            modalities=sample.modality_key,
            content=sample.detector_text(),
        ),
        image_paths=tuple(path for path in sample.image_paths if path.is_file()),
        audio_paths=tuple(path for path in sample.audio_paths if path.is_file()),
        video_paths=tuple(path for path in sample.video_paths if path.is_file()),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    start = time.perf_counter()
    raw = ""
    error: str | None = None
    prediction = "ERROR"
    try:
        raw = backend.generate(request)
        prediction = parse_binary_label(raw)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return PredictionRecord(
        id=sample.id,
        gold=sample.binary_label,
        prediction=prediction,
        modality_key=sample.modality_key,
        raw_output=raw,
        backend=backend.name,
        latency_seconds=time.perf_counter() - start,
        error=error,
        model_metadata=backend.request_metadata(),
    )
