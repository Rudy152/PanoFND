"""Fixed evidence features (defined before any result was seen), localisation, explanation."""

from __future__ import annotations

from typing import Any

FEATURES = (
    # modality presence
    "has_text", "has_image", "has_audio", "has_video",
    # single call with identical inputs (log-probability of FALSE)
    "single_p_fake",
    # local detectors
    "image_synth_max", "frame_synth_max",
    # pairwise / unimodal agents
    "frac_conflict", "frac_unrelated", "any_conflict",
    "slot_mismatch", "slot_match",
    "frac_manipulated", "any_manipulated", "frac_integrity_uncertain",
    "uni_contradictory", "failed_calls",
)

AGENT_FEATURES = (
    "frac_conflict", "frac_unrelated", "any_conflict", "slot_mismatch", "slot_match",
    "frac_manipulated", "any_manipulated", "frac_integrity_uncertain", "uni_contradictory",
)
DETECTOR_FEATURES = ("image_synth_max", "frame_synth_max")


def build_features(modalities: tuple[str, ...], single: dict | None, calls: list[dict],
                   detector: dict[str, list[float]]) -> dict[str, float]:
    f: dict[str, float] = {name: 0.0 for name in FEATURES}
    for name in ("text", "image", "audio", "video"):
        f[f"has_{name}"] = float(name in modalities)
    p = (single or {}).get("p_fake")
    f["single_p_fake"] = float(p) if isinstance(p, (int, float)) else 0.5
    f["image_synth_max"] = max(detector.get("image") or [0.0])
    f["frame_synth_max"] = max(detector.get("video") or [0.0])

    pairs = [c for c in calls if c.get("stage", "").startswith("pair_") and c.get("result")]
    unis = [c for c in calls if c.get("stage", "").startswith("uni_") and c.get("result")]
    f["failed_calls"] = float(sum(1 for c in calls if c.get("error")) + (1 if (single or {}).get("error") else 0))
    if pairs:
        f["frac_conflict"] = sum(c["result"]["consistency"] == "conflict" for c in pairs) / len(pairs)
        f["frac_unrelated"] = sum(c["result"]["consistency"] == "unrelated" for c in pairs) / len(pairs)
        f["any_conflict"] = float(f["frac_conflict"] > 0)
        slots = [s for c in pairs for s in c["result"]["slot_checks"]]
        f["slot_mismatch"] = float(min(6, sum(s["status"] == "mismatch" for s in slots)))
        f["slot_match"] = float(min(6, sum(s["status"] == "match" for s in slots)))
    votes = [c["result"][k] for c in pairs for k in ("a_integrity", "b_integrity")]
    votes += [c["result"]["integrity"] for c in unis]
    if votes:
        f["frac_manipulated"] = sum(v == "manipulated" for v in votes) / len(votes)
        f["any_manipulated"] = float(f["frac_manipulated"] > 0)
        f["frac_integrity_uncertain"] = sum(v == "uncertain" for v in votes) / len(votes)
    f["uni_contradictory"] = float(any(c["result"]["internal_consistency"] == "contradictory" for c in unis))
    return f


def localise(calls: list[dict]) -> dict[str, Any]:
    """Which modality looks manipulated and which pair conflicts (model-free summary)."""
    manipulated: dict[str, int] = {}
    conflicts: list[str] = []
    for c in calls:
        result = c.get("result") or {}
        mods = c.get("modalities") or []
        if c.get("stage", "").startswith("pair_") and result:
            if result["consistency"] == "conflict":
                conflicts.append("-".join(mods))
            for side, mod in zip(("a_integrity", "b_integrity"), mods):
                if result.get(side) == "manipulated":
                    manipulated[mod] = manipulated.get(mod, 0) + 1
        elif result.get("integrity") == "manipulated":
            manipulated[mods[0]] = manipulated.get(mods[0], 0) + 1
    ranked = sorted(manipulated, key=lambda m: -manipulated[m])
    return {"manipulated_modalities": ranked, "conflicting_pairs": conflicts}


def explain(label: str, probability: float, location: dict, calls: list[dict], single: dict | None) -> str:
    """Evidence-only explanation; no extra model call and nothing invented."""
    parts = [f"{label} (P(fake)={probability:.2f})."]
    if location["conflicting_pairs"]:
        parts.append("Conflicting sources: " + ", ".join(location["conflicting_pairs"]) + ".")
    if location["manipulated_modalities"]:
        parts.append("Flagged as manipulated: " + ", ".join(location["manipulated_modalities"]) + ".")
    for c in calls:
        result = c.get("result") or {}
        bad = [s for s in result.get("slot_checks", []) if s["status"] == "mismatch"]
        for s in bad[:2]:
            parts.append(f"{'-'.join(c.get('modalities') or [])} {s['field']}: '{s['a_value']}' vs '{s['b_value']}'.")
    p = (single or {}).get("p_fake")
    if isinstance(p, (int, float)):
        parts.append(f"Single-pass score P(fake)={p:.2f}.")
    return " ".join(parts)
