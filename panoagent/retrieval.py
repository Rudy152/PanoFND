from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.error
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


# Hosts that may mirror benchmark data, papers, or annotation files. Retrieving them
# would leak labels rather than verify the news, so they are excluded by default.
DEFAULT_EXCLUDED_DOMAINS = (
    "huggingface.co",
    "github.com",
    "githubusercontent.com",
    "kaggle.com",
    "arxiv.org",
    "paperswithcode.com",
    "openreview.net",
    "aclanthology.org",
    "semanticscholar.org",
    "researchgate.net",
)

_STOPWORDS = frozenset(
    "about after again against also among because been before being between both could "
    "does doing down during each from further have having here into itself just more most "
    "only other over same should some such than that their them then there these they this "
    "those through under until very were what when where which while will with would your "
    "said says news report reported according".split()
)
_TOKEN_RE = re.compile(r"\d[\d.,:/]*%?|[A-Za-z][A-Za-z'\-]{3,}|[一-鿿]{2,}")


def normalize_url(url: str) -> str:
    """Scheme-, www-, query- and trailing-slash-insensitive URL key for audits."""
    parts = urlsplit(str(url or "").strip())
    host = parts.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{parts.path.rstrip('/')}".casefold()


def _focus_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for token in _TOKEN_RE.findall(text or ""):
        value = token.casefold().strip(".,:/")
        if not value or value in _STOPWORDS:
            continue
        # Exact numbers carry most of the signal for single-detail manipulations.
        terms[value] = 3.0 if value[0].isdigit() else 1.0
    return terms


def select_passages(text: str, focus_text: str, max_chars: int) -> str:
    """Keep the passages most relevant to the claims within a fixed budget.

    Head/tail truncation drops the middle of long articles, which is exactly where a
    single altered statistic or name usually sits. Passages are scored by overlap with
    the claim/query terms (numbers weighted higher), the lead passage is always kept for
    context, and the selected passages are returned in their original order.
    Returns an empty string when nothing overlaps so the caller can fall back.
    """

    text = str(text or "")
    if len(text) <= max_chars:
        return text
    terms = _focus_terms(focus_text)
    if not terms:
        return ""
    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = paragraph.strip()
        while len(paragraph) > 600:
            cut = paragraph.rfind(". ", 0, 600)
            cut = cut + 1 if cut > 200 else 600
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            chunks.append(paragraph)
    if not chunks:
        return ""

    scores: list[float] = []
    for chunk in chunks:
        lowered = chunk.casefold()
        score = sum(weight for term, weight in terms.items() if term in lowered)
        if score and re.search(r"\d", chunk):
            score += 0.5
        scores.append(score)
    if max(scores) <= 0:
        return ""

    separator = "\n...\n"
    chosen = {0}
    used = len(chunks[0])
    for index in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)):
        if scores[index] <= 0:
            break
        if index in chosen:
            continue
        cost = len(chunks[index]) + len(separator)
        if used + cost > max_chars:
            continue
        chosen.add(index)
        used += cost
    selected = separator.join(chunks[index] for index in sorted(chosen))
    return selected[:max_chars]


class _ArticleTextParser(HTMLParser):
    """Collect readable article text (paragraphs, headings, list items) from HTML."""

    _blocks = {"p", "h1", "h2", "h3", "li", "blockquote", "figcaption"}
    _skip = {"script", "style", "noscript", "svg", "nav", "footer", "form", "button", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._block_depth = 0
        self._current: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._skip:
            self._skip_depth += 1
        elif tag in self._blocks:
            self._block_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._blocks and self._block_depth:
            self._block_depth -= 1
            if self._block_depth == 0:
                text = " ".join("".join(self._current).split())
                if len(text) >= 40:
                    self.blocks.append(text)
                self._current = []

    def handle_data(self, data: str) -> None:
        if self._block_depth and not self._skip_depth:
            self._current.append(data)


def fetch_article_text(
    url: str,
    timeout_seconds: float = 20.0,
    urlopen_fn: Callable[..., Any] | None = None,
    max_bytes: int = 3_000_000,
) -> str:
    """Download one public page and return its paragraph text (empty on any failure)."""
    if urlsplit(url).scheme not in {"http", "https"}:
        return ""
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; PanoAgent-factcheck/0.1)",
        "Accept": "text/html,application/xhtml+xml",
    })
    try:
        with (urlopen_fn or urlopen)(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type", "")).casefold()
            if content_type and "html" not in content_type:
                return ""
            raw = response.read(max_bytes)
            charset = "utf-8"
            match = re.search(r"charset=([\w-]+)", content_type)
            if match:
                charset = match.group(1)
    except Exception:  # noqa: BLE001 - a missing page simply yields no evidence
        return ""
    parser = _ArticleTextParser()
    try:
        parser.feed(raw.decode(charset, errors="replace"))
    except Exception:  # noqa: BLE001 - malformed HTML
        pass
    return "\n".join(parser.blocks)


@dataclass(slots=True)
class EvidenceDocument:
    evidence_id: str
    query: str
    title: str
    url: str
    publisher: str
    publication_date: str
    retrieved_text: str
    retrieval_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchReport:
    documents: list[EvidenceDocument]
    errors: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


class SearchProvider(ABC):
    name = "custom"

    @abstractmethod
    def search(self, queries: list[str], max_results_per_query: int = 3) -> list[EvidenceDocument]:
        raise NotImplementedError

    def search_with_report(self, queries: list[str], focus_text: str = "") -> SearchReport:
        """Search without letting a network failure abort the whole sample."""
        try:
            return SearchReport(documents=self.search(queries), queries=list(queries))
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            return SearchReport(
                documents=[], errors=[f"{type(exc).__name__}: {exc}"], queries=list(queries)
            )

    @property
    def enabled(self) -> bool:
        return True


class NullSearchProvider(SearchProvider):
    name = "none"

    def search(self, queries: list[str], max_results_per_query: int = 3) -> list[EvidenceDocument]:
        return []

    @property
    def enabled(self) -> bool:
        return False


class TavilySearchProvider(SearchProvider):
    """Tavily REST client with retries, a query cache, URL dedup, and claim-focused excerpts."""

    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 45,
        urlopen_fn: Callable[..., Any] | None = None,
        max_content_chars: int = 6000,
        max_results_per_query: int = 3,
        max_queries: int = 5,
        max_documents: int = 10,
        exclude_domains: tuple[str, ...] | list[str] = DEFAULT_EXCLUDED_DOMAINS,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        extract_short_results: int = 3,
        short_result_chars: int = 600,
        fetch_fn: Callable[[str], str] | None = fetch_article_text,
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is required for Tavily retrieval")
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen_fn or urlopen
        self.max_content_chars = max_content_chars
        self.max_results_per_query = max_results_per_query
        self.max_queries = max_queries
        self.max_documents = max_documents
        self.exclude_domains = tuple(exclude_domains)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max(1, max_retries)
        self._sleep = sleep_fn
        # Some publishers return only a search snippet. The altered detail usually sits
        # in the article body, so fetch full text for the top short results.
        self.extract_short_results = max(0, extract_short_results)
        self.short_result_chars = short_result_chars
        self._fetch = fetch_fn

    def _bounded_content(self, value: Any, focus_text: str = "") -> str:
        text = str(value or "")
        if len(text) <= self.max_content_chars:
            return text
        focused = select_passages(text, focus_text, self.max_content_chars)
        if focused:
            return focused
        marker = "\n...[TRUNCATED FOR RETRIEVAL CONTEXT BUDGET]...\n"
        tail = min(500, max(0, self.max_content_chars // 5))
        head = self.max_content_chars - len(marker) - tail
        return text[:head] + marker + (text[-tail:] if tail else "")

    def _request_body(self, query: str, max_results: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            # `news` silently restricts results to the last few days, which would hide
            # the original reporting of older dataset items.
            "topic": "general",
            "max_results": max_results,
            "include_raw_content": True,
            "include_answer": False,
        }
        if self.exclude_domains:
            body["exclude_domains"] = list(self.exclude_domains)
        return body

    def _cache_path(self, body: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _post(self, body: dict[str, Any], endpoint: str = "search") -> dict[str, Any]:
        cache_path = self._cache_path({"endpoint": endpoint, **body} if endpoint != "search" else body)
        if cache_path is not None and cache_path.is_file():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))["response"]
            except (OSError, ValueError, KeyError):
                pass
        payload = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = Request(
                f"https://api.tavily.com/{endpoint}",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self._urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(f"Tavily HTTP {exc.code}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = RuntimeError(f"Tavily connection failed: {getattr(exc, 'reason', exc)}")
            if attempt + 1 < self.max_retries:
                self._sleep(2**attempt)
        else:
            assert last_error is not None
            raise last_error
        # Never freeze a failed extraction into the cache; a later run may succeed.
        if cache_path is not None and (endpoint != "extract" or result.get("results")):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "request": body,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "response": result,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        return result

    def _collect(
        self, queries: list[str], max_results_per_query: int, focus_text: str
    ) -> SearchReport:
        report = SearchReport(documents=[])
        seen_urls: set[str] = set()
        unique_queries: list[str] = []
        for query in queries:
            query = " ".join(str(query).split())[:400]
            if query and query.casefold() not in {item.casefold() for item in unique_queries}:
                unique_queries.append(query)
        report.queries = unique_queries[: self.max_queries]
        for query in report.queries:
            if len(report.documents) >= self.max_documents:
                break
            try:
                body = self._post(self._request_body(query, max_results_per_query))
            except Exception as exc:  # noqa: BLE001 - per-query failure is recorded
                report.errors.append(f"{query[:80]!r}: {type(exc).__name__}: {exc}")
                continue
            for item in body.get("results", []):
                url = str(item.get("url", "")).strip()
                key = normalize_url(url)
                if not url or key in seen_urls:
                    continue
                if any(
                    key.split("/", 1)[0] == domain or key.split("/", 1)[0].endswith("." + domain)
                    for domain in self.exclude_domains
                ):
                    continue
                seen_urls.add(key)
                report.documents.append(EvidenceDocument(
                    evidence_id=f"E{len(report.documents) + 1:03d}",
                    query=query,
                    title=str(item.get("title", "")),
                    url=url,
                    publisher=urlsplit(url).netloc,
                    publication_date=str(item.get("published_date", "") or ""),
                    retrieved_text=self._bounded_content(
                        item.get("raw_content") or item.get("content") or "",
                        focus_text=f"{focus_text}\n{query}",
                    ),
                    retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                if len(report.documents) >= self.max_documents:
                    break
        self._extract_short(report, focus_text)
        return report

    def _extract_short(self, report: SearchReport, focus_text: str) -> None:
        short = [
            doc for doc in report.documents
            if len(doc.retrieved_text) < self.short_result_chars
        ][: self.extract_short_results]
        if not short:
            return
        texts: dict[str, str] = {}
        try:
            body = self._post({"urls": [doc.url for doc in short]}, endpoint="extract")
            texts = {
                normalize_url(item.get("url", "")): str(item.get("raw_content") or "")
                for item in body.get("results", []) if isinstance(item, dict)
            }
        except Exception as exc:  # noqa: BLE001 - snippets remain usable evidence
            report.errors.append(f"extract: {type(exc).__name__}: {exc}")
        for doc in short:
            text = texts.get(normalize_url(doc.url), "")
            if len(text) <= len(doc.retrieved_text) and self._fetch is not None:
                # Some publishers block the search API's extractor but serve a plain GET.
                text = self._fetch_cached(doc.url)
            if len(text) > len(doc.retrieved_text):
                doc.retrieved_text = self._bounded_content(
                    text, focus_text=f"{focus_text}\n{doc.query}"
                )

    def _fetch_cached(self, url: str) -> str:
        cache_path = self._cache_path({"endpoint": "direct_fetch", "url": url})
        if cache_path is not None and cache_path.is_file():
            try:
                return str(json.loads(cache_path.read_text(encoding="utf-8"))["text"])
            except (OSError, ValueError, KeyError):
                pass
        text = self._fetch(url) if self._fetch is not None else ""
        if text and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        return text

    def search(self, queries: list[str], max_results_per_query: int | None = None) -> list[EvidenceDocument]:
        report = self._collect(
            queries, max_results_per_query or self.max_results_per_query, focus_text=""
        )
        if report.errors and not report.documents:
            raise RuntimeError("; ".join(report.errors))
        return report.documents

    def search_with_report(self, queries: list[str], focus_text: str = "") -> SearchReport:
        return self._collect(queries, self.max_results_per_query, focus_text=focus_text)


def make_search_provider(
    name: str,
    cache_dir: str | Path | None = None,
    max_results_per_query: int = 3,
    max_content_chars: int = 2000,
    max_documents: int = 8,
    exclude_domains: tuple[str, ...] | list[str] | None = None,
) -> SearchProvider:
    if name == "none":
        return NullSearchProvider()
    if name == "auto":
        name = "tavily" if os.environ.get("TAVILY_API_KEY", "").strip() else "none"
        if name == "none":
            return NullSearchProvider()
    if name == "tavily":
        return TavilySearchProvider(
            cache_dir=cache_dir,
            max_results_per_query=max_results_per_query,
            max_content_chars=max_content_chars,
            max_documents=max_documents,
            exclude_domains=(
                DEFAULT_EXCLUDED_DOMAINS if exclude_domains is None else tuple(exclude_domains)
            ),
        )
    raise ValueError(f"unknown search provider: {name}")
