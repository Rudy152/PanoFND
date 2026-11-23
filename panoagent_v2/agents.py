"""L2: evidence agents. Each call sees only the modalities it is about."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import threading
from typing import Any

from panoagent.parsing import extract_json_object

from . import prompts
from .client import OmniClient, build_content
from .media import PreparedItem

PAIR_ORDER = ("text", "image", "audio", "video")
CONSISTENCY = {"consistent", "conflict", "unrelated", "insufficient"}
INTEGRITY = {"intact", "manipulated", "uncertain"}
INTERNAL = {"consistent", "contradictory", "uncertain"}
SLOT_STATUS = {"match", "mismatch", "unverifiable"}


class StageCache:
    """One JSON file per (sample, stage); valid only when the fingerprint matches."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, sample_id: str, stage: str) -> Path:
        return self.root / sample_id / f"{stage}.json"

    def load(self, sample_id: str, stage: str, fingerprint: str) -> dict | None:
        path = self._path(sample_id, stage)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if record.get("fingerprint") == fingerprint else None

    def save(self, sample_id: str, stage: str, record: dict) -> None:
        path = self._path(sample_id, stage)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)


def _fingerprint(model: str, system: str, user: str, item: PreparedItem, modalities: tuple[str, ...], extra: dict) -> str:
    payload = {
        "model": model, "system": system, "user": user, "extra": extra,
        "text": item.text if "text" in modalities else "",
        "media": [
            [m, a.label, a.digest] for m in modalities for a in item.attachments.get(m, [])
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _sections(item: PreparedItem, modalities: tuple[str, ...], tags: dict[str, str] | None = None):
    text_blocks = []
    sections = []
    for name in modalities:
        tag = (tags or {}).get(name, "")
        if name == "text":
            text_blocks.append((f"{tag}[NEWS TEXT]".strip(), item.text))
        else:
            sections.append((f"{tag}{prompts.MODALITY_NAMES[name].upper()}:".strip(), item.attachments[name]))
    return text_blocks, sections


def p_fake_from_logprobs(top: list[dict], text: str) -> float | None:
    """P(FALSE) / (P(TRUE) + P(FALSE)) over the first answer token's alternatives."""
    mass = {"TRUE": 0.0, "FALSE": 0.0}
    for entry in top or []:
        token = str(entry.get("token", "")).strip().upper()
        if token in mass:
            mass[token] += math.exp(float(entry.get("logprob", -100.0)))
    total = mass["TRUE"] + mass["FALSE"]
    if total > 0:
        return mass["FALSE"] / total
    answer = text.strip().upper()
    if answer.startswith("FALSE"):
        return 1.0
    if answer.startswith("TRUE"):
        return 0.0
    return None


def _normalize_pair(data: dict) -> dict:
    out = {
        "consistency": data.get("consistency") if data.get("consistency") in CONSISTENCY else "insufficient",
        "a_integrity": data.get("a_integrity") if data.get("a_integrity") in INTEGRITY else "uncertain",
        "b_integrity": data.get("b_integrity") if data.get("b_integrity") in INTEGRITY else "uncertain",
        "slot_checks": [],
        "cues": [str(x)[:200] for x in (data.get("cues") or [])][:3] if isinstance(data.get("cues"), list) else [],
        "summary": str(data.get("summary") or "")[:400],
    }
    for slot in (data.get("slot_checks") or [])[:6] if isinstance(data.get("slot_checks"), list) else []:
        if isinstance(slot, dict):
            out["slot_checks"].append({
                "field": str(slot.get("field") or "")[:30],
                "a_value": str(slot.get("a_value") or "")[:160],
                "b_value": str(slot.get("b_value") or "")[:160],
                "status": slot.get("status") if slot.get("status") in SLOT_STATUS else "unverifiable",
            })
    return out


def _normalize_uni(data: dict) -> dict:
    return {
        "integrity": data.get("integrity") if data.get("integrity") in INTEGRITY else "uncertain",
        "internal_consistency": (
            data.get("internal_consistency") if data.get("internal_consistency") in INTERNAL else "uncertain"
        ),
        "cues": [str(x)[:200] for x in (data.get("cues") or [])][:3] if isinstance(data.get("cues"), list) else [],
        "summary": str(data.get("summary") or "")[:400],
    }


class EvidenceAgents:
    def __init__(self, client: OmniClient, cache: StageCache, max_tokens: int = 700):
        self.client = client
        self.cache = cache
        self.max_tokens = max_tokens

    def _run(self, item: PreparedItem, stage: str, system: str, user: str, modalities: tuple[str, ...],
             tags: dict[str, str] | None, json_mode: bool, logprobs: bool, max_tokens: int) -> dict:
        extra = {"json": json_mode, "logprobs": logprobs, "max_tokens": max_tokens, "tags": tags or {}}
        fingerprint = _fingerprint(self.client.model_id, system, user, item, modalities, extra)
        cached = self.cache.load(item.sample.id, stage, fingerprint)
        if cached is not None and not cached.get("error"):
            cached["cached"] = True
            return cached
        text_blocks, sections = _sections(item, modalities, tags)
        record: dict[str, Any] = {"stage": stage, "fingerprint": fingerprint, "modalities": list(modalities)}
        try:
            result = self.client.chat(system, build_content(user, sections, text_blocks), max_tokens,
                                      json_mode=json_mode, logprobs=logprobs)
            record.update({
                "raw": result.text, "usage": result.usage, "model": result.model,
                "request_id": result.request_id, "latency_seconds": result.latency_seconds,
                "top_logprobs": result.first_token_logprobs, "error": None,
            })
            if json_mode:
                record["data"] = extract_json_object(result.text)
        except Exception as exc:  # noqa: BLE001 - recorded; features treat it as missing
            record["error"] = f"{type(exc).__name__}: {exc}"[:800]
        self.cache.save(item.sample.id, stage, record)
        record["cached"] = False
        return record

    def single_call(self, item: PreparedItem) -> dict:
        user = prompts.SINGLE_USER.format(modalities="+".join(item.modalities))
        record = self._run(item, "single_call", prompts.SINGLE_SYSTEM, user, item.modalities,
                           None, json_mode=False, logprobs=True, max_tokens=4)
        if not record.get("error"):
            record["p_fake"] = p_fake_from_logprobs(record.get("top_logprobs") or [], record.get("raw", ""))
        return record

    def pair(self, item: PreparedItem, a: str, b: str) -> dict:
        user = prompts.PAIR_USER.format(a_name=prompts.MODALITY_NAMES[a], b_name=prompts.MODALITY_NAMES[b])
        record = self._run(item, f"pair_{a}_{b}", prompts.PAIR_SYSTEM, user, (a, b),
                           {a: "SOURCE A ", b: "SOURCE B "}, json_mode=True, logprobs=False,
                           max_tokens=self.max_tokens)
        if not record.get("error"):
            record["result"] = _normalize_pair(record.get("data") or {})
        return record

    def unimodal(self, item: PreparedItem, a: str) -> dict:
        user = prompts.UNI_USER.format(a_name=prompts.MODALITY_NAMES[a])
        record = self._run(item, f"uni_{a}", prompts.UNI_SYSTEM, user, (a,), None, json_mode=True,
                           logprobs=False, max_tokens=self.max_tokens)
        if not record.get("error"):
            record["result"] = _normalize_uni(record.get("data") or {})
        return record

    def plan(self, item: PreparedItem) -> list[tuple[str, ...]]:
        mods = [m for m in PAIR_ORDER if m in item.modalities]
        if len(mods) == 1:
            return [("uni", mods[0])]
        return [("pair", a, b) for a, b in itertools.combinations(mods, 2)]
