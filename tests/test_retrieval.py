from __future__ import annotations

import unittest
import json

from panoagent.retrieval import NullSearchProvider, TavilySearchProvider


class FakeResponse:
    body = b'{"results": []}'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class RetrievalTests(unittest.TestCase):
    def test_null_provider_is_offline(self) -> None:
        self.assertEqual(NullSearchProvider().search(["claim"]), [])

    def test_tavily_uses_bearer_header_without_key_in_body(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        provider = TavilySearchProvider(
            api_key="tvly-secret", timeout_seconds=7, urlopen_fn=fake_urlopen
        )
        self.assertEqual(provider.search(["claim"]), [])
        request = captured["request"]
        self.assertEqual(request.get_header("Authorization"), "Bearer tvly-secret")
        self.assertNotIn("tvly-secret", request.data.decode("utf-8"))
        self.assertNotIn("api_key", json.loads(request.data.decode("utf-8")))
        self.assertEqual(captured["timeout"], 7)

    def test_tavily_bounds_raw_content(self) -> None:
        class ContentResponse(FakeResponse):
            body = json.dumps({
                "results": [{
                    "url": "https://example.com/a",
                    "title": "A",
                    "raw_content": "x" * 500,
                }]
            }).encode()

        provider = TavilySearchProvider(
            api_key="tvly-secret",
            urlopen_fn=lambda request, timeout: ContentResponse(),
            max_content_chars=120,
        )
        documents = provider.search(["claim"])
        self.assertEqual(len(documents[0].retrieved_text), 120)
        self.assertIn("TRUNCATED FOR RETRIEVAL CONTEXT BUDGET", documents[0].retrieved_text)


if __name__ == "__main__":
    unittest.main()
