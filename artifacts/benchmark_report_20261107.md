# Updated-dataset baseline benchmark — 2026-11-07

## Protocol

- Dataset: updated local PanoFND manifest, 1,950 unique samples.
- Audit: no duplicate IDs, missing media, unresolved paths, or invalid labels/modalities.
- Split: deterministic source-grouped split (seed 42); 1,373 train, 226 validation, 351 test.
- Pilot: 180 samples, balanced across REAL/FAKE and nine modality combinations (20 per combination).
- The new pilot overlaps the previous pilot on 168 IDs; 12 old IDs were replaced by 12 new IDs.
- Prompt: direct TRUE/FALSE fact-checking prompt, temperature 0.
- Audio is represented by its transcript; video is represented by 16 uniformly sampled frames.
- The dataset `description` field is excluded because it leaks gold-label information.
- Qwen3.8 reasoning is disabled (`reasoning_effort=none`) for a controlled one-word classification benchmark.

## Overall results

| Model | N | Errors | Accuracy | Balanced accuracy | Macro-F1 | Mean latency | Median latency | Estimated API cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-27b | 180 | 0 | 65.56% | 65.56% | 65.49% | 4.74 s | 3.15 s | CNY 0.34 |
| qwen3.8-omni-flash | 180 | 0 | 71.67% | 71.67% | 71.47% | 4.87 s | 3.44 s | CNY 0.45 |

Costs are estimated from API-reported token counts and public list prices, before free quota or discounts.

## Per-class results

| Model | Label | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| qwen3.5-27b | REAL | 67% | 61% | 64% | 90 |
| qwen3.5-27b | FAKE | 64% | 70% | 67% | 90 |
| qwen3.8-omni-flash | REAL | 69% | 80% | 74% | 90 |
| qwen3.8-omni-flash | FAKE | 76% | 63% | 69% | 90 |

## Accuracy by modality combination

| Modality combination | qwen3.5-27b | qwen3.8-omni-flash |
|---|---:|---:|
| audio | 50% | 65% |
| audio+image | 65% | 65% |
| image | 60% | 60% |
| text | 70% | 65% |
| text+audio | 55% | 85% |
| text+image | 60% | 80% |
| text+image+audio | 75% | 75% |
| text+image+video | 75% | 75% |
| text+video | 80% | 75% |

## Interpretation

- qwen3.8-omni-flash outperforms qwen3.5-27b by 6.11 percentage points in accuracy and about 5.98 points in Macro-F1.
- qwen3.8-omni-flash is strongest on text+audio (85%) and text+image (80%).
- qwen3.5-27b is strongest on text+video (80%).
- qwen3.5-27b reproduces its previous overall accuracy exactly (65.56%); its Macro-F1 changes only from 65.45% to 65.49%.
- qwen2.5-omni-7b could not be rerun: both the workspace endpoint and the shared Beijing endpoint now return HTTP 404 `model_not_found`. It was replaced, with user approval, by qwen3.8-omni-flash.

## Reproducibility notes

- Video frame cache keys now contain a SHA-256 fingerprint of the source video, preventing updated videos with unchanged filenames from reusing stale frames.
- One PyAV process crashed with a Windows native access violation while two experiments decoded media concurrently. The qwen3.5 run resumed from 101/180 after frame caching completed and finished with 180 unique successful predictions.
- Previous prediction and metric files were preserved; all new outputs use the `20261107` tag.
