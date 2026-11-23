"""DashScope (OpenAI-compatible) client for omni models with labelled attachments."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .media import Attachment


@dataclass(slots=True)
class ChatResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    first_token_logprobs: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    request_id: str | None = None
    latency_seconds: float = 0.0


def _image_data_url(path: Path, max_edge: int = 1280) -> str:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _audio_data_url(path: Path) -> str:
    return "data:;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_content(prompt: str, sections: list[tuple[str, list[Attachment]]], text_blocks: list[tuple[str, str]]) -> list[dict]:
    """Prompt, then labelled text blocks, then each attachment right after its label."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for label, value in text_blocks:
        content.append({"type": "text", "text": f"\n{label}\n{value}"})
    for heading, attachments in sections:
        content.append({"type": "text", "text": f"\n{heading}"})
        for item in attachments:
            content.append({"type": "text", "text": item.label})
            if item.kind == "image":
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(item.path)}})
            else:
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": _audio_data_url(item.path), "format": "mp3"},
                })
    return content


class OmniClient:
    def __init__(
        self,
        model_id: str = "qwen3.8-omni-flash",
        timeout_seconds: float = 300.0,
        max_retries: int = 4,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url_env: str = "DASHSCOPE_BASE_URL",
    ):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env

    def _endpoint(self) -> tuple[str, str]:
        key = os.environ.get(self.api_key_env, "").strip()
        base = os.environ.get(self.base_url_env, "").strip().rstrip("/")
        if not key or not base:
            raise RuntimeError(f"{self.api_key_env} and {self.base_url_env} must be set")
        if urllib.parse.urlsplit(base).path in {"", "/"}:
            base += "/compatible-mode/v1"
        return key, base + "/chat/completions"

    def chat(
        self,
        system: str,
        content: list[dict],
        max_tokens: int,
        json_mode: bool = False,
        logprobs: bool = False,
    ) -> ChatResult:
        key, endpoint = self._endpoint()
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.model_id.startswith("qwen3.8-"):
            payload["reasoning_effort"] = "none"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5  # DashScope accepts 0..5
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            request = urllib.request.Request(endpoint, data=body, method="POST", headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "Accept": "text/event-stream",
            })
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result = self._parse_stream(response)
                result.latency_seconds = time.perf_counter() - started
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
                last_error = exc
            time.sleep(min(30, 2 ** attempt * 2))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _parse_stream(response) -> ChatResult:
        chunks: list[str] = []
        result = ChatResult(text="")
        done = False
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            result.model = event.get("model") or result.model
            result.request_id = event.get("id") or result.request_id
            if event.get("usage"):
                result.usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
                for entry in ((choice.get("logprobs") or {}).get("content") or []):
                    if not result.first_token_logprobs and entry.get("top_logprobs"):
                        result.first_token_logprobs = entry["top_logprobs"]
        if not done:
            raise RuntimeError("stream ended before [DONE]")
        result.text = "".join(chunks).strip()
        return result
