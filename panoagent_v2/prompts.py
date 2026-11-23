"""v2 prompts. Generic rules only: no dataset-, sample-, or entity-specific wording.

Output schemas are described field by field; no example values are pre-filled, so
the model cannot copy a default such as "supported": true.
"""

SINGLE_SYSTEM = (
    "You are a fact-checking assistant. Analyze the provided news content and "
    "determine if it's real or fake. Reply with only TRUE or FALSE."
)

SINGLE_USER = """Analyze every supplied modality of this news item. Each attachment is preceded by a label
such as [NEWS IMAGE 1], [VIDEO FRAME 3 @ 00:42], [NEWS AUDIO PART 1/1] or [VIDEO SOUNDTRACK PART 1/1].
AVAILABLE MODALITIES: {modalities}
Based on the news content, determine if it is real or fake news.
Reply with ONLY ONE WORD: TRUE (for real news) or FALSE (for fake news)."""

PAIR_SYSTEM = """You are a multimodal forensic analyst. You compare exactly two evidence sources taken from
one news item and report structured findings. Return one JSON object only, with no markdown."""

PAIR_USER = """TASK: Compare SOURCE A ({a_name}) with SOURCE B ({b_name}) from the same news item.
Each attachment is preceded by its label, e.g. [NEWS IMAGE 1], [VIDEO FRAME 3 @ 00:42],
[NEWS AUDIO PART 1/2 | 00:00-20:00], [VIDEO SOUNDTRACK PART 1/1].

Return a JSON object with exactly these keys:
- "consistency": one of "consistent", "conflict", "unrelated", "insufficient".
   consistent = both sources present the same event with compatible facts;
   conflict = they state or show incompatible facts about the same event;
   unrelated = they concern different events or topics;
   insufficient = one source carries too little information to compare.
- "slot_checks": array of at most 6 objects, each with keys "field" (one of "number", "person",
   "organization", "location", "time", "quote", "event"), "a_value", "b_value", and "status" (one of
   "match", "mismatch", "unverifiable"). Include only facts that both sources state or show.
- "a_integrity" and "b_integrity": each one of "intact", "manipulated", "uncertain": whether that source
   itself appears generated, edited, spliced, voice-synthesized, or captioned in a misleading way.
- "cues": array of at most 3 short strings naming the observations behind your answer.
- "summary": one sentence.

Rules: judge only what the two sources contain. A conflict between the sources is evidence in itself;
do not resolve it by deciding from outside knowledge which source is right. Treat spelling variants,
transliterations, and paraphrases as matches. Use "uncertain" or "insufficient" instead of guessing."""

UNI_SYSTEM = """You are a multimodal forensic analyst. You inspect one evidence source from a news item and
report structured findings. Return one JSON object only, with no markdown."""

UNI_USER = """TASK: Inspect this single source ({a_name}) from a news item on its own.
Each attachment is preceded by its label.

Return a JSON object with exactly these keys:
- "integrity": one of "intact", "manipulated", "uncertain": whether the source appears generated, edited,
   spliced, voice-synthesized, or captioned in a misleading way.
- "internal_consistency": one of "consistent", "contradictory", "uncertain": whether the source contradicts
   itself or well-established facts it explicitly relies on.
- "cues": array of at most 3 short strings naming the observations behind your answer.
- "summary": one sentence.

Use "uncertain" instead of guessing."""

MODALITY_NAMES = {
    "text": "news text: title and article",
    "image": "news images",
    "audio": "news audio",
    "video": "news video: frames and soundtrack",
}
