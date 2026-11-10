from __future__ import annotations

import json
import os
import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panoagent.model_clients import DashScopeBackend
from panoagent.schemas import ModelRequest


class FakeSseResponse:
    def __init__(self, events: list[dict]):
        self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        self.lines = [f"data: {json.dumps(event)}\n".encode() for event in events]
        self.lines.append(b"data: [DONE]\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)


class PrematureSseResponse(FakeSseResponse):
    def __init__(self, events: list[dict]):
        super().__init__(events)
        self.lines = self.lines[:-1]


class DashScopeBackendTests(unittest.TestCase):
    def test_streaming_request_and_usage_are_recorded_without_key_leak(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeSseResponse([
                {
                    "id": "request-1",
                    "model": "qwen2.5-vl-7b-instruct",
                    "choices": [{"delta": {"content": "FA"}}],
                    "usage": None,
                },
                {
                    "id": "request-1",
                    "model": "qwen2.5-vl-7b-instruct",
                    "choices": [{"delta": {"content": "LSE"}}],
                    "usage": None,
                },
                {
                    "id": "request-1",
                    "model": "qwen2.5-vl-7b-instruct",
                    "choices": [],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
                },
            ])

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.jpg"
            image.write_bytes(b"not-a-real-jpeg")
            backend = DashScopeBackend(
                "qwen2.5-vl-7b-instruct",
                urlopen=fake_urlopen,
                timeout_seconds=12,
            )
            request = ModelRequest(
                task_name="direct_detection",
                system_prompt="system",
                user_prompt="user",
                image_paths=(image,),
                max_new_tokens=16,
                temperature=0.0,
            )
            with patch.dict(os.environ, {
                "DASHSCOPE_API_KEY": "secret-test-key",
                "DASHSCOPE_BASE_URL": "https://workspace.example/compatible-mode/v1",
            }):
                output = backend.generate(request)

        self.assertEqual(output, "FALSE")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(
            captured["request"].full_url,
            "https://workspace.example/compatible-mode/v1/chat/completions",
        )
        payload = json.loads(captured["request"].data.decode("utf-8"))
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["modalities"], ["text"])
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        self.assertNotIn("secret-test-key", captured["request"].data.decode("utf-8"))
        self.assertEqual(backend.request_metadata()["usage"]["total_tokens"], 101)

    def test_missing_environment_is_explicit(self) -> None:
        backend = DashScopeBackend("qwen2.5-omni-7b")
        request = ModelRequest("direct_detection", "system", "user")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Missing DASHSCOPE_API_KEY"):
                backend.generate(request)

    def test_bare_workspace_host_is_normalized(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return FakeSseResponse([{
                "id": "request-2",
                "model": "qwen2.5-omni-7b",
                "choices": [{"delta": {"content": "TRUE"}}],
                "usage": None,
            }])

        backend = DashScopeBackend("qwen2.5-omni-7b", urlopen=fake_urlopen)
        request = ModelRequest("direct_detection", "system", "user")
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "secret-test-key",
            "DASHSCOPE_BASE_URL": "https://workspace.example",
        }):
            self.assertEqual(backend.generate(request), "TRUE")
        self.assertEqual(
            captured["url"],
            "https://workspace.example/compatible-mode/v1/chat/completions",
        )

    def test_qwen35_disables_thinking_for_direct_benchmark(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeSseResponse([{
                "id": "request-3",
                "model": "qwen3.5-27b",
                "choices": [{"delta": {"content": "FALSE"}}],
                "usage": None,
            }])

        backend = DashScopeBackend("qwen3.5-27b", urlopen=fake_urlopen)
        request = ModelRequest("direct_detection", "system", "user")
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "secret-test-key",
            "DASHSCOPE_BASE_URL": "https://workspace.example",
        }):
            self.assertEqual(backend.generate(request), "FALSE")
        self.assertIs(captured["payload"]["enable_thinking"], False)

    def test_qwen3_agent_requests_json_object_mode(self) -> None:
        backend = DashScopeBackend("qwen3-vl-plus")
        request = ModelRequest("evidence_organizer", "system", "user")
        payload = backend._payload(request)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_premature_sse_stream_is_retried(self) -> None:
        calls = 0

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            events = [{
                "id": f"request-{calls}", "model": "qwen3-vl-plus",
                "choices": [{"delta": {"content": "{\"ok\": true}"}}],
                "usage": None,
            }]
            if calls == 1:
                return PrematureSseResponse(events)
            return FakeSseResponse(events)

        backend = DashScopeBackend("qwen3-vl-plus", urlopen=fake_urlopen, max_retries=2)
        request = ModelRequest("fused_investigator", "system", "user")
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "secret-test-key",
            "DASHSCOPE_BASE_URL": "https://workspace.example",
        }):
            self.assertEqual(backend.generate(request), '{"ok": true}')
        self.assertEqual(calls, 2)

    def test_qwen38_direct_disables_reasoning(self) -> None:
        backend = DashScopeBackend("qwen3.8-omni-flash")
        request = ModelRequest("direct_detection", "system", "user")
        payload = backend._payload(request)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("enable_thinking", payload)

    def test_unsupported_image_format_is_normalized_to_jpeg(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            mislabeled = Path(directory) / "sample.jpg"
            Image.new("RGB", (2, 2), "red").save(mislabeled, format="TIFF")
            url = DashScopeBackend._data_url(mislabeled)
            header, encoded = url.split(",", 1)
            self.assertEqual(header, "data:image/jpeg;base64")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(image.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
