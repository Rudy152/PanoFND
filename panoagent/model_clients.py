from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable

from .schemas import ModelRequest


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, request: ModelRequest) -> str:
        raise NotImplementedError

    def request_metadata(self) -> dict[str, Any]:
        """Return non-secret metadata from the most recent request."""
        return {}


class MockBackend(ModelBackend):
    """Deterministic structural test backend. Never use its scores as results."""

    name = "mock"

    def generate(self, request: ModelRequest) -> str:
        task = request.task_name
        if task == "direct_detection":
            suspicious = any(
                marker in request.user_prompt.casefold()
                for marker in ("aliens invade", "dinosaurs living", "completely fabricated")
            )
            return "FAKE" if suspicious else "REAL"
        if task == "evidence_organizer":
            return json.dumps({
                "entities": [], "persons": [], "locations": [], "times": [], "events": [],
                "text_claims": [], "numeric_claims": [], "image_observations": [],
                "audio_claims": [], "video_observations": [],
            })
        if task == "unimodal_investigator":
            return json.dumps({"modalities": {
                kind: {"status": "unknown", "confidence": 0.0, "evidence": []}
                for kind in ("text", "image", "audio", "video")
            }})
        if task == "relation_investigator":
            return json.dumps({"relations": {}})
        if task == "evidence_sufficiency":
            return json.dumps({
                "sufficient": False,
                "reason": "Mock backend has no factual knowledge.",
                "claims_to_verify": [],
                "search_queries": [],
            })
        if task == "fused_investigator":
            return json.dumps({
                "central_claim": "An event happened.",
                "case_file": {"entities": [], "events": [], "claims": []},
                "modalities": {
                    kind: {"status": "unknown", "confidence": 0.0, "evidence": []}
                    for kind in ("text", "image", "audio", "video")
                },
                "relations": {},
                "atomic_checks": [],
                "signals": {
                    "claim_supported": False,
                    "claim_contradicted": False,
                    "identity_mismatch": False,
                    "attribution_mismatch": False,
                    "temporal_mismatch": False,
                    "jurisdiction_mismatch": False,
                    "location_mismatch": False,
                    "numeric_mismatch": False,
                    "manipulated_overlay": False,
                    "materiality": "none",
                    "confidence": 0.0,
                    "stage_valid": True,
                    "evidence_ids": [],
                },
                "retrieval": {"needed": False, "claims": [], "queries": []},
            })
        if task == "context_verifier":
            return json.dumps({"claims": [], "status": "insufficient", "sources": []})
        if task == "judge":
            return json.dumps({
                "central_claim": "An event happened.",
                "central_claim_status": "insufficient",
                "binary_label": "REAL",
                "misinformation_type": "REAL",
                "affected_modalities": [],
                "affected_relations": [],
                "confidence": 0.0,
                "evidence_ids": [],
                "explanation": "Mock orchestration result; not a research prediction.",
            })
        raise ValueError(f"Unsupported mock task: {task}")


class QwenOmniBackend(ModelBackend):
    """Lazy local Transformers backend following the official Qwen interface."""

    def __init__(
        self,
        model_id: str,
        use_audio_in_video: bool = True,
        attn_implementation: str | None = None,
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 128 * 28 * 28,
    ):
        self.model_id = model_id
        self.use_audio_in_video = use_audio_in_video
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.name = f"qwen-{model_id.rsplit('/', 1)[-1]}"
        self._model: Any = None
        self._processor: Any = None
        self._process_mm_info: Any = None
        self._thinker_only = False

    def _load(self) -> None:
        if self._model is not None:
            return
        # Long mixed-modality runs vary substantially in sequence length.
        # Expandable segments reduce allocator fragmentation between samples.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are missing. Install the project's qwen optional dependencies first."
            ) from exc
        kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        # Detection only needs the Thinker. Loading the full Omni model also
        # materializes the Talker and codec, which needlessly consumes several
        # gigabytes on a 16 GB research GPU.
        self._model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        self._thinker_only = True
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._process_mm_info = process_mm_info

    @staticmethod
    def _media_content(request: ModelRequest) -> list[dict[str, str]]:
        content: list[dict[str, str]] = []
        content.extend({"type": "image", "image": str(path)} for path in request.image_paths)
        content.extend({"type": "audio", "audio": str(path)} for path in request.audio_paths)
        content.extend({"type": "video", "video": str(path)} for path in request.video_paths)
        content.append({"type": "text", "text": request.user_prompt})
        return content

    def generate(self, request: ModelRequest) -> str:
        self._load()
        inputs: Any = None
        text_ids: Any = None
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": request.system_prompt}]},
            {"role": "user", "content": self._media_content(request)},
        ]
        try:
            rendered = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = self._process_mm_info(
                conversation, use_audio_in_video=self.use_audio_in_video
            )
            inputs = self._processor(
                text=rendered,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
            inputs = inputs.to(self._model.device).to(self._model.dtype)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "use_audio_in_video": self.use_audio_in_video,
                "do_sample": request.temperature > 0,
                # The 3B checkpoint occasionally continues with a synthetic next
                # dialogue turn instead of emitting EOS. Stop before that turn so
                # direct labels remain one word and agent JSON is not followed by
                # unrelated text.
                "stop_strings": ["\nHuman:"],
                "tokenizer": self._processor.tokenizer,
            }
            if request.temperature > 0:
                generation_kwargs["temperature"] = request.temperature
            text_ids = self._model.generate(**inputs, **generation_kwargs)
            # Decoder-only generation returns the prompt followed by the answer.
            # Only the answer may be parsed as the predicted label.
            input_length = inputs["input_ids"].shape[1]
            text_ids = text_ids[:, input_length:]
            decoded = self._processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # Some Omni checkpoints include the start of a synthetic next user
            # turn even when stopping criteria trigger. It is never part of the
            # requested answer and can make a one-word label ambiguous.
            return decoded[0].split("\nHuman:", 1)[0].strip()
        finally:
            # Release sample-specific activations before the next heterogeneous
            # example. Without this, reserved blocks accumulate and a later long
            # text/image sample can OOM even though it fits in a fresh process.
            del inputs
            del text_ids
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class DashScopeBackend(ModelBackend):
    """Alibaba Cloud Model Studio backend using its OpenAI-compatible API.

    Credentials are read only from environment variables. They are never
    accepted as CLI arguments, included in exceptions, or written to outputs.
    The implementation uses streaming for compatibility with older Qwen-Omni
    models and records the API-reported model, request ID, and token usage.
    """

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url_env: str = "DASHSCOPE_BASE_URL",
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
        urlopen: Callable[..., Any] | None = None,
    ):
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.name = f"dashscope-{model_id}"
        self._urlopen = urlopen or urllib.request.urlopen
        self._last_metadata: dict[str, Any] = {}

    def request_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def _configuration(self) -> tuple[str, str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing {self.api_key_env}. Set a rotated Alibaba Cloud API key "
                "in the local environment; never put it in a command or source file."
            )
        base_url = (self.base_url or os.environ.get(self.base_url_env, "")).strip().rstrip("/")
        if not base_url:
            raise RuntimeError(
                f"Missing {self.base_url_env}. Copy the API Host/base_url shown with "
                "the API key in Alibaba Cloud Model Studio."
            )
        if not base_url.startswith("https://"):
            raise RuntimeError("DashScope base URL must use https://")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.path in {"", "/"}:
            base_url = f"{base_url}/compatible-mode/v1"
        elif base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        return api_key, base_url

    @staticmethod
    def _data_url(path: Path, generic: bool = False) -> str:
        raw = path.read_bytes()
        media_type = "" if generic else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if not generic and media_type.startswith("image/"):
            try:
                from PIL import Image

                with Image.open(io.BytesIO(raw)) as image:
                    actual_format = (image.format or "").upper()
                    supported = {
                        "JPEG": "image/jpeg",
                        "PNG": "image/png",
                        "WEBP": "image/webp",
                        "BMP": "image/bmp",
                    }
                    if actual_format in supported:
                        media_type = supported[actual_format]
                    else:
                        output = io.BytesIO()
                        image.convert("RGB").save(output, format="JPEG", quality=90)
                        raw = output.getvalue()
                        media_type = "image/jpeg"
            except (ImportError, OSError):
                # Let the API report genuinely corrupt files. This fallback
                # keeps Pillow optional for text-only uses of the backend.
                pass
        data = base64.b64encode(raw).decode("ascii")
        return f"data:{media_type};base64,{data}"

    @classmethod
    def _media_content(cls, request: ModelRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for path in request.image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": cls._data_url(path)},
            })
        for path in request.audio_paths:
            content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": cls._data_url(path, generic=True),
                    "format": path.suffix.lower().lstrip(".") or "mp3",
                },
            })
        for path in request.video_paths:
            content.append({
                "type": "video_url",
                "video_url": {"url": cls._data_url(path, generic=True)},
            })
        return content

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": self._media_content(request)},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_new_tokens,
            "seed": 42,
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Qwen3-series models may enable thinking by default. Direct classification follows
        # the paper's one-word, temperature-zero protocol, so hidden reasoning
        # would only add latency/cost and can exhaust the tiny output budget.
        model_name = self.model_id.casefold()
        if request.task_name == "direct_detection" and model_name.startswith("qwen3"):
            if model_name.startswith("qwen3.8-"):
                payload["reasoning_effort"] = "none"
            else:
                payload["enable_thinking"] = False
        # Current Qwen3 cloud models support JSON object mode. Agent stages all
        # require a JSON object, so request it at the API boundary as well as in
        # the prompt. Local and legacy Qwen2.5 behavior remains unchanged.
        if request.task_name != "direct_detection" and model_name.startswith("qwen3"):
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _iter_sse_lines(response: Iterable[bytes]) -> Iterable[dict[str, Any]]:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                yield {"__stream_done__": True}
                break
            try:
                value = json.loads(data)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid SSE JSON from DashScope: {data[:200]!r}") from exc
            if isinstance(value, dict):
                yield value

    def _parse_response(self, response: Any) -> str:
        content_type = str(response.headers.get("Content-Type", "")).casefold()
        if "text/event-stream" not in content_type:
            payload = json.loads(response.read().decode("utf-8"))
            self._last_metadata = {
                "model": payload.get("model"),
                "request_id": payload.get("id"),
                "usage": payload.get("usage") or {},
            }
            return str(payload["choices"][0]["message"].get("content", "")).strip()

        chunks: list[str] = []
        metadata: dict[str, Any] = {"model": None, "request_id": None, "usage": {}}
        stream_done = False
        for event in self._iter_sse_lines(response):
            if event.get("__stream_done__"):
                stream_done = True
                continue
            metadata["model"] = event.get("model") or metadata["model"]
            metadata["request_id"] = event.get("id") or metadata["request_id"]
            if event.get("usage"):
                metadata["usage"] = event["usage"]
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    chunks.append(str(delta["content"]))
        if not stream_done:
            raise RuntimeError("DashScope SSE stream ended before [DONE]")
        self._last_metadata = metadata
        return "".join(chunks).strip()

    def generate(self, request: ModelRequest) -> str:
        api_key, base_url = self._configuration()
        payload = json.dumps(self._payload(request), ensure_ascii=False).encode("utf-8")
        endpoint = f"{base_url}/chat/completions"
        self._last_metadata = {}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            http_request = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                method="POST",
            )
            try:
                with self._urlopen(http_request, timeout=self.timeout_seconds) as response:
                    return self._parse_response(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"DashScope HTTP {exc.code}: {body}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"DashScope connection failed: {exc.reason}")
            except RuntimeError as exc:
                # Malformed or prematurely closed SSE streams are transient
                # transport failures and are safe to retry with the same request.
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(2**attempt)
        assert last_error is not None
        raise last_error


def make_backend(
    name: str,
    model_id: str,
    *,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str | None = None,
    timeout_seconds: float = 300.0,
    min_pixels: int = 16 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
) -> ModelBackend:
    if name == "mock":
        return MockBackend()
    if name == "qwen":
        return QwenOmniBackend(
            model_id=model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    if name == "dashscope":
        return DashScopeBackend(
            model_id=model_id,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unknown backend: {name}")
