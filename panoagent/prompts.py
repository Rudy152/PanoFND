ASR_TOLERANCE = """AUDIO TRANSCRIPT text comes from automatic speech recognition, not the publisher's
spelling. Sound-alike spellings of a name or term, missing diacritics, and paraphrased or partial
quotations in a transcript are transcription or wording variation, not evidence of fabrication.
A clearly different person, place, organization, date, or number is still a mismatch."""


DIRECT_SYSTEM = (
    "You are a fact-checking assistant. Analyze the provided news content and "
    "determine if it's real or fake. Reply with only TRUE or FALSE."
)

DIRECT_USER = """TASK_NAME: DIRECT_DETECTION
Use only the supplied news modalities. Do not assume that polished or emotionally charged
language alone proves falsity.

AVAILABLE MODALITIES: {modalities}

{content}

Based on the above news content, determine if it is real or fake news.
Reply with ONLY ONE WORD: TRUE (for real news) or FALSE (for fake news).
"""

# Keep the paper prompt above unchanged for reproducibility. This alternative
# is an explicit ablation for small checkpoints that otherwise default to TRUE.
DIRECT_DEBIASED_SYSTEM = """You are a balanced multimodal fact-checking classifier.
REAL and FAKE are equally likely before examining the evidence. Compare both hypotheses
and inspect intrinsic manipulation and cross-modal contradictions. A fluent article or
the absence of an obvious contradiction is not evidence that it is real. Analyze silently
and return exactly one label: REAL or FAKE."""

DIRECT_DEBIASED_USER = """TASK_NAME: DIRECT_DETECTION_DEBIASED
Classify the supplied news under two equally likely hypotheses:
- REAL: the available modalities consistently depict and support the same authentic event.
- FAKE: a supplied modality appears manipulated/fabricated, or available modalities conflict
  about the event, entities, time, location, or central claim.

Do not infer the label from writing style. Do not default to REAL merely because external
verification is unavailable. Base the forced choice only on the supplied modalities.

AVAILABLE MODALITIES: {modalities}

{content}

Reply with ONLY ONE WORD: REAL or FAKE.
"""

ORGANIZER_SYSTEM = """You organize evidence without judging whether the news is real or fake.
Return one JSON object only. Do not include markdown or hidden reasoning."""

ORGANIZER_USER = """TASK_NAME: EVIDENCE_ORGANIZER
Extract atomic claims and observations from the supplied modalities.
Return keys: entities, persons, locations, times, events, text_claims,
numeric_claims, image_observations, audio_claims, video_observations. Every value must be an array.
Every array item must be a concise string, not a nested object. Return at most five
items per key. Preserve exact spellings and digits; never round, correct, or normalize away a
difference. For persons, locations, times, and numeric_claims, prefix each item with its source
modality and bind the value to its subject/predicate, for example
`[text] <subject> | <predicate> | <value>` or `[image] <chart label> | <value>`.
Keep news-image observations and video-frame observations separate, as listed under VISUAL INPUT
ORDER. Use an empty array when a category is absent.
Do not return an array of JSON objects; return one JSON object containing all ten keys.

AVAILABLE MODALITIES: {modalities}
{content}
"""

UNIMODAL_SYSTEM = """You are the Unimodal Investigator. Assess each available modality on its own.
Cross-modal disagreement must not be used to call an individual modality fake. Return JSON only."""

UNIMODAL_USER = """TASK_NAME: UNIMODAL_INVESTIGATOR
For text, image, audio, and video, return an object under `modalities` with status
authentic|suspicious|unknown, confidence from 0 to 1, and evidence as an array.
Unavailable modalities must be unknown. Use the Case File as organization, not ground truth.
Assess news images and video frames separately, using VISUAL INPUT ORDER to tell them apart.
For images and frames, look for generation or editing traces: inconsistent lighting or shadows,
warped text, cloned regions, implausible anatomy, pasted overlays, and meme-style captions.
Keep at most two evidence items per modality and at most 25 words per evidence item.
Do not repeat the article or add fields outside `modalities`.

CASE FILE:
{case_file}

NEWS CONTENT:
{content}
"""

RELATION_SYSTEM = """You are the Relation Investigator. Judge relationships between modalities,
not whether each item is intrinsically authentic. Return JSON only."""

RELATION_USER = """TASK_NAME: RELATION_INVESTIGATOR
For every available pair among text-image, text-audio, text-video, image-audio,
image-video, and audio-video, report status consistent|conflict|unrelated|unknown,
type event|entity|time|location|claim_support, confidence 0..1, evidence array, and
`slot_checks` as an array. Each slot check must contain field
number|person|organization|location|time|quote|attribution, left_value, right_value,
status match|mismatch|missing|unknown, and materiality none|minor|central.
Treat a supplied headline or article body as text evidence even when the dataset's primary
modality label is audio, image, or video. "image" means the news images and "video" means the
video frames, as listed under VISUAL INPUT ORDER; never treat a video frame as a news image.
Compare explicit numbers and units, persons/organizations, places, dates/times, quotations, and
attribution across the pair before judging the broad event; digits must match exactly. A single
altered slot is material when it changes the reported statistic, actor, place, time, or attribution.
Return all results under the `relations` key. Report only pairs whose two modalities
are available. Return exactly one JSON object shaped as {{"relations": {{...}}}}.
Keep at most one evidence string and six slot checks per pair; do not duplicate matching slots.
Do not restate the article. If fewer than two modalities are available, return
{{"relations": {{}}}}.

{asr_tolerance}

AVAILABLE EVIDENCE MODALITIES: {modalities}
CASE FILE:
{case_file}

UNIMODAL REPORT:
{unimodal}

NEWS CONTENT:
{content}
"""

SUFFICIENCY_SYSTEM = """You decide only whether current evidence is sufficient for a final verdict.
You do not decide the label. Return JSON only."""

SUFFICIENCY_USER = """TASK_NAME: EVIDENCE_SUFFICIENCY
Return sufficient (boolean), reason, claims_to_verify (array), and search_queries (array).
Keep reason under 60 words. Return at most three claims_to_verify and three search_queries,
prioritizing textual claims that could change the central verdict. Keep every query under 20 words.
External search can only check textual claims; it cannot verify images, audio, or video.
Write each claim to verify with its exact subject, predicate, relevant event time, and jurisdiction
when known, keeping exact numbers, names, places, dates, and quotations verbatim. Search queries must
include the named entity, the disputed predicate, and the relevant date; do not search only broad
event keywords.
Use SOURCE PUBLICATION DATE as the date anchor unless the article explicitly supplies another year.
Never infer or substitute a year from weekdays, holidays, or general context. Preserve attribution
and epistemic status exactly: a reported belief or allegation must not be rewritten as a fact.
Every claim_to_verify must be entailed by ORIGINAL NEWS CONTENT. Do not attach the date of one event
to a different nearby predicate.
The news article, transcript, Case File, and upstream model opinions are not independent
external corroboration.

SOURCE PUBLICATION DATE (metadata): {publish_time}
ORIGINAL NEWS CONTENT:
{content}

CASE FILE: {case_file}
UNIMODAL REPORT: {unimodal}
RELATION REPORT: {relation}
"""

CONTEXT_SYSTEM = """You verify textual claims against supplied external evidence. Search snippets are
leads, not proof. Prefer primary sources, then independent high-quality reporting. A source can support
or refute a claim only when subject, predicate, relevant time, and jurisdiction match. Return JSON only."""

CONTEXT_USER = """TASK_NAME: CONTEXT_VERIFIER
Return exactly one object with keys in this order: same_report, slot_checks, claims, status,
sources. Keep the whole object under 900 output tokens; at most four claims.

`claims` must be an array of objects. Every object must contain:
- claim: original claim string
- status: supported|refuted|mixed|insufficient
- subject_match, predicate_match, time_match, jurisdiction_match: true|false|unknown
- evidence_ids: array
- reason: one sentence of at most 25 words; do not quote long passages

`same_report` is an object {{"found": true|false, "evidence_ids": [], "note": "short"}}. Set found=true
only when an evidence item is evidently the same report or an independent report of the same
specific event (same actors, place, and date window), not merely the same topic.

`slot_checks` is an array (at most eight) of objects comparing exact atomic slots of the NEWS ITEM
text against aligned evidence: field number|person|organization|location|time|quote|attribution,
news_value, news_source text|title|transcript, evidence_value, status match|mismatch|not_found,
materiality none|minor|central, evidence_id. Digits, units, names, places, and dates must match
exactly. Use not_found when the aligned evidence does not state that slot; never infer a mismatch
from silence. When same_report.found is true, compare slots only against the same_report evidence_ids.
A different estimate from another publisher, dataset, method, or period is not a mismatch of the
news item; record it as not_found.

Use `refuted` only when the cited evidence addresses the same subject and predicate at the
same relevant time and jurisdiction. A clear mismatch of a central exact slot against an aligned
report of the same event is a refutation of the claim containing that slot. A later development
(a revocation, a revised count, a new ruling) does not refute what was true or reported earlier.
Different legal predicates or jurisdictions are not the same claim. Allegations, lawsuits, and
ongoing investigations are not established facts unless independently confirmed. If any essential
alignment is false or unknown, use mixed or insufficient rather than refuted. `status` is one overall
string and `sources` is a deduplicated array of cited evidence IDs. The news item itself is not
evidence. This check covers text only; it says nothing about images, audio, or video.

{asr_tolerance}

CLAIMS TO VERIFY: {claims}
NEWS ITEM (to compare exact slots; not evidence):
{news_excerpt}
EXTERNAL EVIDENCE: {evidence}
"""

JUDGE_SYSTEM = """You are the final Judge. Do not search. Use only the structured reports supplied.
Identify the central news claim before assigning the label. Return JSON only and do not reveal
hidden chain-of-thought."""

DECISION_RULE = """DECISION RULE (apply to every item):
1. Weigh REAL and FAKE as equally plausible before evidence; class frequency is not a prior.
2. Decide per modality and per modality pair. Label FAKE when any supplied modality shows material
   evidence of fabrication or manipulation, or any pair of supplied modalities materially conflicts
   about the event, person, place, time, number, quotation, or attribution.
3. Label REAL only when every supplied modality has no material concern and every available pair
   is consistent.
4. External evidence concerns textual claims only. A refuted or materially contradicted textual
   claim supports FAKE. Text confirmed by sources is NOT evidence that an image, audio, or video is
   authentic or matches the text, and never outweighs a modality-level or pair-level problem.
5. Missing, unconfirmed, or insufficient evidence is neutral; it supports neither label.
6. A detail with no material effect on the published assertion does not by itself make an item
   FAKE. A later development does not refute an earlier claim. Use the supplied publication date as
   the time anchor.
Also return fake_probability: your probability from 0 to 1 that the item is FAKE. Use the whole
range; do not round to a few fixed values."""

JUDGE_USER = """TASK_NAME: JUDGE
Return central_claim, central_claim_status supported|refuted|mixed|insufficient,
binary_label REAL|FAKE, misinformation_type REAL|INTRINSIC|RELATIONAL|CONTEXTUAL|MIXED,
affected_modalities (array), affected_relations (array), confidence 0..1, fake_probability 0..1,
evidence_ids (array, at most eight), and a concise evidence-based explanation.

""" + DECISION_RULE + """

Use relation `slot_checks` directly and list material mismatches in the explanation. Slot status
`variant` means a spelling, transcription, or wording difference and is not a mismatch.
`numeric_diffs`, when present, list numbers that differ from the same original report.

{asr_tolerance}

SOURCE PUBLICATION DATE (metadata): {publish_time}
CASE FILE: {case_file}
UNIMODAL REPORT: {unimodal}
RELATION REPORT: {relation}
EXTERNAL EVIDENCE REPORT (text only): {context}
"""

_IMAGE_POLICY_BLOCK = """

Apply the following image-specific policy. It takes precedence over the general rule above when
the two differ:
{decision_policy}

"""

JUDGE_IMAGE_USER = JUDGE_USER.replace("\n{asr_tolerance}", _IMAGE_POLICY_BLOCK + "{asr_tolerance}", 1)

FUSED_SYSTEM = """You are a schema-constrained multimodal fact-checking investigator.
Inspect every supplied modality once and return one JSON object only. Do not reveal hidden
reasoning, repeat the article, or infer authenticity from writing style."""

FUSED_USER = """TASK_NAME: FUSED_INVESTIGATOR
Create one SHORT evidence report for the central news claim. Return exactly this flat JSON
shape. The values below are neutral placeholders, not defaults to copy: replace every
"unknown" with true or false whenever the evidence decides it. Signals are scalar booleans or the
string "unknown", never nested objects.

{{
  "central_claim": "one sentence",
  "case_file": {{"entities": [], "events": [], "claims": []}},
  "modalities": {{"text": {{"status": "unknown", "confidence": 0.0, "evidence": []}}}},
  "relations": {{}},
  "atomic_checks": [],
  "signals": {{"claim_supported": "unknown", "claim_contradicted": "unknown", "identity_mismatch": "unknown", "attribution_mismatch": "unknown", "temporal_mismatch": "unknown", "jurisdiction_mismatch": "unknown", "location_mismatch": "unknown", "numeric_mismatch": "unknown", "manipulated_overlay": "unknown", "materiality": "unknown", "confidence": 0.0, "evidence_ids": []}},
  "retrieval": {{"needed": false, "claims": [], "queries": []}}
}}

Compactness limits:

- case_file: at most two short strings per array.
- modalities: include only supplied modalities; "image" is the news images and "video" is the
  video frames (see VISUAL INPUT ORDER); at most one short evidence string each.
- relations: include only available pairs; at most one short evidence string each.
- atomic_checks: at most four objects with field number|person|organization|location|time|quote|attribution,
  left_value, right_value, status match|mismatch|missing|unknown, materiality none|minor|central,
  and the compared modality pair.
- modality status is authentic, suspicious, or unknown.
- relation status is consistent, conflict, unrelated, or unknown; type is event, entity,
  time, location, or claim_support; relation materiality is none, minor, or central.
- materiality in signals is none, minor, central, or unknown.
- retrieval: at most three short textual claims and three short queries; set needed=true only for
  textual factual claims that the supplied modalities cannot settle. Search cannot check images,
  audio, or video.
- Keep the entire JSON under 600 output tokens. Do not add fields or prose.

Treat visible overlay text, speaker/person identity, attribution, time, and jurisdiction as
first-class evidence. For images and frames, look for generation or editing traces. Compare explicit
numbers/units, names, places, dates/times, quotations, and attribution across modalities before
judging the broad event; digits must match exactly. A fluent article is not proof of authenticity.
The article and upstream observations are not independent external corroboration. Do not use or
guess any dataset annotation.

{asr_tolerance}

AVAILABLE MODALITIES: {modalities}
SOURCE PUBLICATION DATE: {publish_time}
ORIGINAL NEWS CONTENT:
{content}
"""

JUDGE_STRUCTURED_USER = """TASK_NAME: STRUCTURED_JUDGE
Return central_claim, central_claim_status supported|refuted|mixed|insufficient,
binary_label REAL|FAKE, misinformation_type REAL|INTRINSIC|RELATIONAL|CONTEXTUAL|MIXED,
affected_modalities, affected_relations, confidence 0..1, fake_probability 0..1, evidence_ids,
and explanation.

""" + DECISION_RULE + """

Use the structured modalities, relations, signals, and atomic_checks rather than reconstructing a
new narrative. If stage_valid is false, do not guess. Slot status `variant` means a spelling,
transcription, or wording difference and is not a mismatch. `numeric_diffs`, when present, list
numbers that differ from the same original report.

{asr_tolerance}

SOURCE PUBLICATION DATE: {publish_time}
STRUCTURED EVIDENCE REPORT:
{fused_report}
EXTERNAL EVIDENCE REPORT (text only):
{context}
"""

JUDGE_STRUCTURED_IMAGE_USER = JUDGE_STRUCTURED_USER.replace(
    "\n{asr_tolerance}", _IMAGE_POLICY_BLOCK + "{asr_tolerance}", 1
)
