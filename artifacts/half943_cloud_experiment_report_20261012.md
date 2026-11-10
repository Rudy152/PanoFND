# Half-dataset cloud experiment report

Date completed: 2026-10-12  
Backend: Alibaba Cloud `qwen3-vl-plus`  
Search provider: disabled  
Split: source-grouped seed-42 `train` partition, 943/1950 samples (48.36%)

## Data cleanup

The raw output contained 950 JSONL rows. Seven duplicate rows were created during an early
double-process launch. Results were deduplicated by sample ID, keeping the final complete row for
each ID. The cleaned result contains exactly 943 unique samples and no malformed JSONL lines.

## Overall metrics

| Metric | Result |
|---|---:|
| Unique samples | 943 |
| Evaluated predictions | 940 |
| Prediction/runtime errors | 3 |
| Accuracy | 0.6128 |
| Balanced accuracy | 0.6290 |
| Macro-F1 | 0.6055 |
| Records with stage errors | 28 |
| Fully stage-complete records | 915 |

Class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.4628 | 0.6829 | 0.5517 | 328 |
| FAKE | 0.7719 | 0.5752 | 0.6592 | 612 |

The model predicted 484 REAL, 456 FAKE, and 3 ERROR labels, while the test split contains 330 REAL
and 613 FAKE records. This indicates a substantial tendency to predict REAL, especially for fake
multimodal samples.

## Metrics by modality combination

| Modality | Evaluated | Accuracy | Balanced accuracy | Macro-F1 | Stage-error records |
|---|---:|---:|---:|---:|---:|
| audio | 76 | 0.5921 | 0.5921 | 0.5915 | 7 |
| audio+image | 121 | 0.6116 | 0.5770 | 0.5738 | 10 |
| image | 78 | 0.6538 | 0.6538 | 0.6538 | 0 |
| text | 66 | 0.7121 | 0.6724 | 0.6543 | 1 |
| text+audio | 110 | 0.7364 | 0.6971 | 0.6985 | 3 |
| text+image | 93 | 0.4409 | 0.5403 | 0.4329 | 4 |
| text+image+audio | 152 | 0.4737 | 0.5351 | 0.4624 | 0 |
| text+image+video | 147 | 0.6531 | 0.6803 | 0.6253 | 3 |
| text+video | 97 | 0.7113 | 0.7109 | 0.6908 | 0 |

The strongest raw accuracy was `text+audio` (73.64%), followed by `text` (71.21%) and `text+video`
(71.13%). The weakest combinations were `text+image` (44.09%) and `text+image+audio` (47.37%).

## Errors

Three samples produced no valid binary prediction:

- `0156` (REAL, text+audio): organizer JSON failure and Alibaba content inspection rejection at Judge.
- `0390` (REAL): Judge output did not normalize to REAL or FAKE, although no stage exception was recorded.
- `0571` (FAKE, audio+image): organizer and Judge JSON failures.

Stage errors affected 28 records (2.97%):

- evidence organizer: 14
- evidence sufficiency: 18
- relation investigator: 7
- unimodal investigator: 4
- Judge: 2

Counts overlap because one record can fail in multiple stages. The main causes were incomplete JSON
responses and Alibaba `DataInspectionFailed` content-review responses.

## API usage

| Token type | Count |
|---|---:|
| Prompt | 11,638,628 |
| Completion | 1,899,253 |
| Total | 13,537,881 |

Summed per-record model latency was 41,027 seconds. This is not wall-clock duration because the run
was paused while the computer slept and includes per-sample processing rather than a continuous timer.

## Interpretation

The 8-sample pilot score (87.5%) did not generalize to this source-grouped 943-sample experiment.
The large run provides a more credible estimate and reveals two main limitations:

1. The final decision stage favors REAL on fake multimodal records, reducing FAKE recall to 57.52%.
2. Text-image and text-image-audio fusion are substantially weaker than text, text-audio, and
   text-video combinations.

Because external search was disabled, context verification returned insufficient for uncached
samples. These numbers therefore measure the cloud multimodal pipeline without live retrieval, not
the complete retrieval-augmented configuration.

## Artifacts

- Raw predictions: `runs/agent_cloud_qwen3_vl_plus_half943_seed42_20261010.jsonl`
- Deduplicated predictions: `runs/agent_cloud_qwen3_vl_plus_half943_seed42_20261010_dedup.jsonl`
- Metrics: `runs/agent_cloud_qwen3_vl_plus_half943_seed42_20261010_metrics.json`
- Split manifest: `splits/half_eval_seed42.json`
