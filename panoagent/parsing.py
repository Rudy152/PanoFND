from __future__ import annotations

import json
import re
from typing import Any


def parse_json_document(text: str) -> Any:
    """Parse one complete JSON document, optionally wrapped in a code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response is not one complete JSON document") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response does not contain a valid JSON object")


def parse_binary_label(text: str) -> str:
    cleaned = text.strip().upper()
    if cleaned in {"REAL", "TRUE"}:
        return "REAL"
    if cleaned in {"FAKE", "FALSE"}:
        return "FAKE"
    matches = re.findall(r"\b(REAL|FAKE|TRUE|FALSE)\b", cleaned)
    if not matches:
        raise ValueError(f"No binary label found in response: {text[:200]!r}")
    normalized = ["REAL" if item in {"REAL", "TRUE"} else "FAKE" for item in matches]
    if len(set(normalized)) != 1:
        raise ValueError(f"Ambiguous binary labels in response: {text[:200]!r}")
    return normalized[0]
