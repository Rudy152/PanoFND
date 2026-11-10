"""Deterministic comparison of exact numbers between a news item and retrieved reports.

Most manipulated items in PanoFND change one detail of an otherwise real report
(e.g. "Estonia recorded a 192% rise" -> "199%"). An LLM verifier chooses which slots
to compare and can skip the altered one. This module aligns every number-bearing
sentence of the news title/article with its most similar sentence in the retrieved
evidence and reports numbers that were substituted while the surrounding wording
matches. Transcripts are excluded: speech recognition renders numbers unreliably.
"""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z一-鿿]{3,}")
_STOP = frozenset(
    "the and for with that this from were was are has have had its their into over "
    "than more most also which who what when where after before about said says".split()
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part and part.strip()]


def _numbers(sentence: str) -> set[str]:
    values = set()
    for raw in _NUMBER.findall(sentence):
        value = raw.replace(",", "")
        # Single digits are ordinals/list markers far more often than statistics.
        if len(value) == 1:
            continue
        values.add(value)
    return values


def _words(sentence: str) -> set[str]:
    return {word for word in _WORD.findall(sentence.casefold()) if word not in _STOP}


def numeric_diffs(
    news_text: str,
    evidence: list[dict[str, Any]],
    min_similarity: float = 0.6,
    min_shared_words: int = 5,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Return substituted numbers between closely matching sentences."""

    candidates: list[tuple[str, set[str], set[str], str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        for sentence in _sentences(str(item.get("retrieved_text") or "")):
            numbers = _numbers(sentence)
            if numbers:
                candidates.append((sentence, numbers, _words(sentence), str(item.get("evidence_id", ""))))

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(news_text):
        news_numbers = _numbers(sentence)
        news_words = _words(sentence)
        if not news_numbers or len(news_words) < min_shared_words:
            continue
        best = None
        best_score = 0.0
        for evidence_sentence, evidence_numbers, evidence_words, evidence_id in candidates:
            shared = news_words & evidence_words
            if len(shared) < min_shared_words:
                continue
            score = len(shared) / len(news_words | evidence_words)
            if score > best_score:
                best_score = score
                best = (evidence_sentence, evidence_numbers, evidence_id)
        if best is None or best_score < min_similarity:
            continue
        evidence_sentence, evidence_numbers, evidence_id = best
        missing = news_numbers - evidence_numbers
        extra = evidence_numbers - news_numbers
        # A substitution, not an omission: the source states a different number in the
        # same sentence slot.
        if not missing or not extra:
            continue
        key = (sentence, evidence_id)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "news_sentence": sentence[:300],
            "evidence_sentence": evidence_sentence[:300],
            "news_numbers": sorted(missing),
            "evidence_numbers": sorted(extra),
            "similarity": round(best_score, 3),
            "evidence_id": evidence_id,
        })
        if len(results) >= max_results:
            break
    return results
