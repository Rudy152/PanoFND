from __future__ import annotations

from .schemas import NewsSample


IMAGE_ONLY_POLICY = """IMAGE_ONLY_POLICY:
The central claim is the complete assertion conveyed by the pixels plus any embedded or overlaid
text, quotation, person/entity attribution, and caption-like wording. Label FAKE when that visible
assertion contains a fabricated quote, wrong identity or attribution, material text-image mismatch,
or evidence of compositing/manipulation, even if the background event itself is real. Interpret
qualitative or rhetorical wording in its ordinary publication context; do not silently strengthen
it into a more literal claim or weaken it into a harmless claim. Tone alone is not evidence.
Prioritize exact visible facts: numbers and units, names, organizations, locations, dates/times,
quotations, and attribution. A material mismatch in one of these fields can make the item FAKE even
when the broad event is authentic. Mixed or missing external evidence is not itself a refutation,
and must not be upgraded to refuted without aligned evidence."""


def decision_policy_for(sample: NewsSample) -> str | None:
    """Return an override only for modalities that need specialized judging."""
    if sample.modalities == ("image",):
        return IMAGE_ONLY_POLICY
    return None
