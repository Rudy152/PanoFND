from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from PIL import Image

from .schemas import NewsSample


MODEL_ID = "fanrui00/fact-r1-final-ckpt"
MODEL_REVISION = "06f54b231958d7e814ea35451f15d9bcac8feee2"
PAPER_URL = "https://arxiv.org/abs/2505.16836"
OFFICIAL_REPOSITORY = "https://github.com/zfr00/Fact-R1"
MODEL_URL = "https://huggingface.co/fanrui00/fact-r1-final-ckpt"
EXPECTED_WEIGHT_FILES = {
    "model-00001-of-00004.safetensors": {
        "size": 4_968_243_304,
        "sha256": "5d2e1622d9f7ede618f6a42f3b2209e9a3a9b72b3ea297b9e79d8c7e52a8e008",
    },
    "model-00002-of-00004.safetensors": {
        "size": 4_991_495_816,
        "sha256": "b089298a1f14aeb64243b3dda5a0ce7261e75ba3a21c7e1be9765876aae02a03",
    },
    "model-00003-of-00004.safetensors": {
        "size": 4_932_751_040,
        "sha256": "63c37e5d77f330b6908fdcc254f95283f8c76b9b72bb24e281f817cd0a6eccde",
    },
    "model-00004-of-00004.safetensors": {
        "size": 1_691_924_384,
        "sha256": "d91be46eb25a189cb01c453baa3bfce9cf13439f774dfcae2e25b40895d4ae11",
    },
}

# Verbatim from examples/qwen2_5_vl_7b_fact_r1_grpo.sh in the official repo.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think><answer> "
    "answer here </answer>"
)


def truncate_text(value: str, max_chars: int | None) -> str:
    value = value.strip()
    if max_chars is None or len(value) <= max_chars:
        return value
    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    marker = " ...[TRANSCRIPT MIDDLE TRUNCATED]... "
    remaining = max_chars - len(marker)
    head = remaining // 2
    return value[:head] + marker + value[-(remaining - head) :]


def build_user_prompt(
    sample: NewsSample,
    max_transcript_chars: int | None = 6_000,
) -> str:
    """Reproduce Fact-R1's released FakeVV test prompt without label leakage."""
    title = " ".join(sample.title.strip().split())
    transcript = truncate_text(sample.transcript, max_transcript_chars)
    evidence = f"The news title is '{title}'"
    if transcript:
        evidence += f", the news audio is '{transcript}'"
    return (
        f" {evidence}, is this multimodal content manipulated? Please answer yes "
        "or no. Output the thinking process in <think> </think> and final answer "
        "in <answer> </answer> tags."
    )


def parse_fact_r1_answer(output: str) -> tuple[str, bool]:
    """Return (binary label, official-format compliance). Yes means FAKE."""
    answer = re.search(r"<answer>\s*(yes|no)\s*[.!]?\s*</answer>", output, re.I)
    if answer:
        label = "FAKE" if answer.group(1).casefold() == "yes" else "REAL"
        has_thinking = bool(
            re.search(r"<think>.+?</think>", output, flags=re.I | re.S)
        )
        return label, has_thinking

    stripped = output.strip().casefold().rstrip(".!")
    if stripped in {"yes", "no"}:
        return ("FAKE" if stripped == "yes" else "REAL"), False
    return "ERROR", False


class FactR1VideoClassifier:
    """Local deterministic inference for the released Fact-R1 checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        quantization: str = "4bit",
        max_new_tokens: int = 384,
        min_pixels: int = 3_136,
        max_pixels: int = 25_088,
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        if quantization not in {"4bit", "bf16"}:
            raise ValueError("quantization must be '4bit' or 'bf16'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        model_kwargs: dict[str, object] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        }
        if quantization == "4bit":
            if device != "cuda":
                raise ValueError("4-bit inference requires CUDA")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **model_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=False,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.input_device = next(self.model.parameters()).device

    def classify(
        self,
        sample: NewsSample,
        video_frame_paths: Iterable[str | Path],
        *,
        max_transcript_chars: int | None = 6_000,
    ) -> tuple[str, str, dict[str, object]]:
        import torch

        frame_paths = tuple(Path(path) for path in video_frame_paths)
        if len(frame_paths) != 8:
            raise ValueError(
                f"Fact-R1's released protocol requires exactly 8 frames; got {len(frame_paths)}"
            )

        images: list[Image.Image] = []
        try:
            for path in frame_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())

            prompt = build_user_prompt(
                sample, max_transcript_chars=max_transcript_chars
            )
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[rendered], images=images, padding=True, return_tensors="pt"
            ).to(self.input_device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            prompt_length = inputs.input_ids.shape[1]
            output = self.processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            prediction, format_compliant = parse_fact_r1_answer(output)
            metadata: dict[str, object] = {
                "video_frames": len(images),
                "prompt_tokens": int(prompt_length),
                "completion_tokens": int(generated.shape[1] - prompt_length),
                "official_output_format": format_compliant,
                "transcript_present": bool(sample.transcript.strip()),
            }
            return prediction, output, metadata
        finally:
            for image in images:
                image.close()
