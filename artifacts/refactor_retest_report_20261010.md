# PanoAgent small-refactor retest report

Date: 2026-10-10  
Backend: Alibaba Cloud `qwen3-vl-plus`  
Evaluation set: fixed 8-sample comparison set (`0011`, `0114`, `0941`, `0992`, `1539`, `1732`, `1835`, `1902`)  
External search during final retest: disabled; previously retrieved evidence was reused locally.

## Final result

- Evaluated: 8
- Correct: 7
- Accuracy: 0.875
- Balanced accuracy: 0.875
- Macro-F1: 0.8730
- Runtime/parse errors: 0
- Records with all stages complete: 8/8
- Final verification-run API usage: 23,230 tokens (22,574 prompt, 656 completion)
- Final verification-run wall time reported by records: 12.71 seconds

Compared with the pre-refactor aligned run:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Correct | 6/8 | 7/8 | +1 |
| Accuracy | 0.750 | 0.875 | +0.125 |
| Balanced accuracy | 0.750 | 0.875 | +0.125 |
| Macro-F1 | 0.750 | 0.873 | +0.123 |
| Stage-complete records | 8/8 | 8/8 | unchanged |

## Per-sample result

| ID | Modality | Gold | Prediction | Result |
|---|---|---|---|---|
| 0011 | audio | REAL | REAL | correct |
| 0114 | audio | FAKE | FAKE | correct |
| 0941 | text | REAL | REAL | correct |
| 0992 | text | FAKE | FAKE | correct |
| 1539 | text+image+video | REAL | REAL | correct |
| 1732 | text+image+video | FAKE | FAKE | correct |
| 1835 | image | REAL | FAKE | incorrect |
| 1902 | image | FAKE | FAKE | correct |

Accuracy by modality combination:

- audio: 2/2 (1.00)
- text: 2/2 (1.00)
- text+image+video: 2/2 (1.00)
- image: 1/2 (0.50)

## Refactor contents

1. A non-empty title or article body now counts as text evidence in relation analysis even when the dataset's primary modality is audio, image, or video. This preserves the benchmark modality label while enabling `text-audio` and `text-image` contradiction checks.
2. Source publication time and original news content are passed explicitly into evidence sufficiency. Claims may not invent a year, transfer a nearby date to a different predicate, or strengthen an attributed statement.
3. Image-only judging has a separate policy. General samples retain the general Judge prompt.
4. Context retrieval cache is checked before search. Cached evidence can be re-evaluated after a claim/prompt revision without another network request.
5. If the context model completes the claims array and overall status but truncates the redundant sources array, sources are safely derived from the completed claim evidence IDs.
6. Cached stage metadata is no longer counted as newly consumed API tokens.

## Confirmed fixes

- `0114`: relation analysis now checks the headline against the audio and reports `text-audio=conflict`; prediction changed from REAL to the correct FAKE.
- `1539`: the verifier now anchors to publication date `2025/07/05` and no longer invents a 2024 event year or a Wednesday remains-discovery date; prediction changed from FAKE to the correct REAL.
- `1902`: title/image entity conflict is explicit (`Johnson` vs `Jonny`) and the sample remains correctly FAKE.
- `0011`: a completion truncated only in the final sources array is recovered locally; the final run has no stage error.

## Remaining error

`1835` is still predicted FAKE. The model treats the Politico headline “The collapse of New Jersey's political machine” as a literal, completed institutional collapse, even though the context result is mixed, the image is assessed authentic, and the title-image relation is consistent. The image-specific policy was confirmed present in the actual Judge request, but the model still upgraded the qualitative claim to refuted.

Further prompt wording aimed only at this sample would risk overfitting. A better next experiment is calibration on a larger held-out image-only subset containing both qualitative headlines and fabricated overlays, or a separate calibrated claim-materiality classifier.

## Verification

- Unit/integration tests: 32 passed
- Final predictions: `runs/agent_cloud_qwen3_vl_plus_refactored_final_v2_8_20261010.jsonl`
- Final metrics: `runs/agent_cloud_qwen3_vl_plus_refactored_final_v2_8_20261010_metrics.json`
