# PanoAgent

This repository contains a leakage-safe implementation scaffold for two
experiments on the PanoFND dataset:

1. **Direct MLLM baseline** - one model call maps the available modalities to
   `REAL` or `FAKE`.
2. **PanoAgent** - a six-stage workflow: Evidence Organizer, Unimodal
   Investigator, Relation Investigator, Evidence Sufficiency, optional Context
   Verification, and Judge.

The core package uses only the Python standard library. A local
Qwen2.5-Omni backend is optional and loaded lazily.

## Important dataset rules

- `description` is annotation/audit metadata and is **never** passed to the
  detector. Many fake rows explicitly reveal the label in this field.
- Splits are source-grouped so real articles and their generated fake variants
  cannot cross train/validation/test boundaries.
- The loader repairs filename zero-padding errors in the JSON at runtime while
  leaving the source dataset unchanged.
- Raw model responses and every agent-stage response are cached for
  reproducibility and resumability. A request fingerprint invalidates stale
  entries when prompts, upstream reports, media, or generation settings change.
- When retrieval returns no evidence, Context Verification deterministically
  returns `insufficient` with empty sources instead of asking the model to
  invent a verification result.

## Quick start (no model download required)

```powershell
python -m panoagent.cli audit --dataset .\fakenewsDataSet --output .\artifacts\dataset_audit.json
python -m panoagent.cli split --dataset .\fakenewsDataSet --output .\splits\source_grouped.json
python -m panoagent.cli direct --dataset .\fakenewsDataSet --split .\splits\source_grouped.json --partition test --backend mock --limit 20 --output .\runs\direct_mock.jsonl
python -m panoagent.cli agent --dataset .\fakenewsDataSet --split .\splits\source_grouped.json --partition test --backend mock --limit 5 --output .\runs\agent_mock.jsonl
python -m panoagent.cli evaluate --predictions .\runs\direct_mock.jsonl --output .\runs\direct_mock_metrics.json
python -m unittest discover -s tests -v
```

The mock backend validates orchestration only. Its metrics are not research
results.

## Alibaba Cloud API baselines

The `dashscope` backend calls the workspace-specific OpenAI-compatible API and
never accepts a literal API key as a command-line argument. Rotate any key that
has appeared in chat, source code, shell history, or logs. Set the replacement
key and the matching API Host/base URL only in the local environment:

```powershell
$env:DASHSCOPE_API_KEY = Read-Host "Rotated Model Studio API key" -MaskInput
$env:DASHSCOPE_BASE_URL = "https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
```

Create the balanced 180-item pilot from the already source-grouped test split:

```powershell
python -m panoagent.cli pilot-split --dataset .\fakenewsDataSet --split .\splits\source_grouped.json --partition test --per-cell 10 --output .\splits\pilot_180.json
```

Install video-frame extraction support and run the two controlled baselines.
Both models receive the same title/body, audio transcript, news images, and up
to 16 uniformly sampled video frames. The detector prompt is the paper's
zero-shot `TRUE`/`FALSE` prompt, and `description` remains excluded because it
leaks labels.

```powershell
pip install -e ".[frames]"
python -m panoagent.cli direct --dataset .\fakenewsDataSet --split .\splits\pilot_180.json --partition pilot --backend dashscope --model-id qwen2.5-vl-7b-instruct --media-mode frames --frame-count 16 --temperature 0 --max-new-tokens 16 --resume --output .\runs\qwen25_vl_7b_pilot.jsonl
python -m panoagent.cli direct --dataset .\fakenewsDataSet --split .\splits\pilot_180.json --partition pilot --backend dashscope --model-id qwen2.5-omni-7b --media-mode frames --frame-count 16 --temperature 0 --max-new-tokens 16 --resume --output .\runs\qwen25_omni_7b_controlled_pilot.jsonl
python -m panoagent.cli evaluate --predictions .\runs\qwen25_vl_7b_pilot.jsonl --output .\runs\qwen25_vl_7b_pilot_metrics.json
python -m panoagent.cli evaluate --predictions .\runs\qwen25_omni_7b_controlled_pilot.jsonl --output .\runs\qwen25_omni_7b_controlled_pilot_metrics.json
```

`qwen2.5-omni-7b` can also receive native audio and video. Treat a native-media
run as a separate ablation rather than comparing it directly with the
frame/transcript-controlled VL baseline. Smoke-test one item before a larger
native-media run:

```powershell
python -m panoagent.cli direct --dataset .\fakenewsDataSet --backend dashscope --model-id qwen2.5-omni-7b --media-mode native --ids 0001 --max-new-tokens 16 --output .\runs\qwen25_omni_native_smoke.jsonl
```

API-reported model IDs, request IDs, and token usage are stored under
`model_metadata` in each JSONL record. Secrets are never stored.

## Faster agent evaluation without reducing sample count

`--agent-profile fused` is an explicit speed/quality ablation. It combines
Evidence Organizer, Unimodal Investigator, Relation Investigator, and Evidence
Sufficiency into one schema-constrained multimodal call, then sends only the
structured report to a compact Judge. With retrieval disabled this reduces the
model calls per sample from five to two and uploads/encodes media once instead
of three times. The original `full` profile remains the comparison baseline.
`--agent-profile fused-rule` removes the second LLM Judge call as well and uses
a transparent fail-closed aggregator, reducing the no-retrieval path to one
model call. Ambiguous structured evidence becomes `ERROR`, never an implicit
`REAL`; compare its coverage as well as its accuracy.

DashScope samples can also run concurrently. Each worker owns a separate
backend and pipeline, so request metadata cannot leak between threads:

```powershell
python -m panoagent.cli agent --dataset .\fakenewsDataSet --split .\splits\source_grouped.json --partition validation --backend dashscope --model-id qwen3-vl-plus --media-mode frames --frame-count 4 --agent-profile fused --strict-stage-errors --workers 4 --resume --cache .\cache\qwen3_vl_fused_val --output .\runs\qwen3_vl_fused_val.jsonl
python -m panoagent.cli evaluate --predictions .\runs\qwen3_vl_fused_val.jsonl --output .\runs\qwen3_vl_fused_val_metrics.json
```

Concurrency lowers wall-clock time but not tokens or API cost. Start with two
workers and increase only within the provider's rate limits. `--workers` is
intentionally rejected for the local Qwen backend because loading multiple GPU
copies would exhaust VRAM. Frame extraction, request-fingerprint caching,
`--resume`, text/transcript budgets, pixel limits, and Qwen prompt caching are
complementary optimizations.

The historical 943-item run spent approximately 3.30M tokens in Relation,
3.06M in Unimodal, 2.72M in Sufficiency, 2.66M in Organizer, and 1.83M in Judge.
The fused profile targets that repeated-reading cost, but its quality and exact
token reduction must be measured on the same source-grouped validation split;
do not assume the 60% call reduction produces a 60% token reduction.

Evaluation now reports valid-prediction coverage, ordinary accuracy,
errors-as-wrong end-to-end accuracy, stage-complete subset metrics, and strict
stage-complete coverage. Use `--strict-stage-errors` when a failed upstream
stage must invalidate the whole prediction.

Every overall and per-modality slice also reports a 2-by-3 confusion matrix
(gold REAL/FAKE by predicted REAL/FAKE/ERROR), explicit `real_as_fake` and
`fake_as_real` counts, the majority-class baseline, and the model's
errors-as-wrong accuracy margin above that baseline. On imbalanced data,
compare balanced accuracy, Macro-F1, and both class recalls before ordinary
accuracy.

The evidence pipeline preserves exact atomic facts. Organizer output includes
`numeric_claims`; Relation output audits numbers/units, people/organizations,
locations, dates/times, quotations, and attribution with per-slot match status.
The Judge treats REAL and FAKE symmetrically and does not assume a real broad
event makes a materially altered number or entity authentic.

Class-bias calibration is offline and adds no model calls. Fit it only on a
source-grouped validation prediction file, freeze the resulting JSON, then
apply it to held-out test predictions:

```powershell
python -m panoagent.cli fit-calibration --predictions .\runs\qwen3_vl_fused_validation.jsonl --min-fake-recall 0.60 --per-modality --min-group-size 30 --output .\configs\qwen3_vl_fused_calibration.json
python -m panoagent.cli apply-calibration --predictions .\runs\qwen3_vl_fused_test_raw.jsonl --calibration .\configs\qwen3_vl_fused_calibration.json --output .\runs\qwen3_vl_fused_test_calibrated.jsonl
python -m panoagent.cli evaluate --predictions .\runs\qwen3_vl_fused_test_calibrated.jsonl --output .\runs\qwen3_vl_fused_test_calibrated_metrics.json
```

Never fit a threshold on the held-out test labels or on the 943-result file and
then report performance on that same file. Modality-specific thresholds are
omitted automatically when a validation group is too small or lacks one class.

Hybrid profile routing is selected offline from paired `fused` and `full`
predictions on the same source-grouped validation partition. The candidate
family is fixed in code before seeing validation results (modality fallback,
confidence fallback, and their combinations). By default, selection maximizes
Macro-F1 while requiring at least a 50% token reduction relative to `full`:

```powershell
python -m panoagent.cli select-routing --fused-predictions .\runs\qwen3_vl_fused_validation.jsonl --full-predictions .\runs\qwen3_vl_full_validation.jsonl --min-token-reduction 0.50 --output .\configs\qwen3_vl_hybrid_routing.json --hybrid-output .\runs\qwen3_vl_hybrid_validation.jsonl
python -m panoagent.cli evaluate --predictions .\runs\qwen3_vl_hybrid_validation.jsonl --output .\runs\qwen3_vl_hybrid_validation_metrics.json
```

Freeze the selected routing JSON before applying it to paired held-out test
predictions with `apply-routing`. Validation metrics are model-selection
metrics and must not be presented as final held-out test performance.

If earlier experiments touched part of a planned holdout, remove every related
source group before final testing:

```powershell
python -m panoagent.cli clean-holdout --dataset .\fakenewsDataSet --candidate-split .\splits\half_eval_seed42.json --candidate-partition test --exclude-split .\splits\source_grouped.json --exclude-partition validation --exclude-ids 0011 0114 0941 0992 1539 1732 1835 1902 --output .\splits\final_holdout_clean_765_20261031.json
```

## Qwen2.5-Omni backend

Install the optional dependencies in a dedicated environment, ensure `ffmpeg`
is available for video decoding, and run:

```powershell
pip install -e ".[qwen]"
python -m panoagent.cli direct --dataset .\fakenewsDataSet --ids 0901 --backend qwen --model-id Qwen/Qwen2.5-Omni-3B --media-mode transcript --limit 1 --output .\runs\direct_qwen_smoke.jsonl
```

The 3B checkpoint is already downloaded in this workspace. These commands run
fully offline from the local snapshot:

```powershell
$model = ".\models\modelscope_cache\models\Qwen--Qwen2.5-Omni-3B\snapshots\master"
python -m panoagent.cli direct --dataset .\fakenewsDataSet --ids 1051 --backend qwen --model-id $model --media-mode frames --frame-count 4 --max-new-tokens 4 --output .\runs\direct_qwen3b_video4.jsonl
python -m panoagent.cli agent --dataset .\fakenewsDataSet --ids 0901 --backend qwen --model-id $model --media-mode transcript --max-new-tokens 512 --cache .\cache\qwen3b_agent --output .\runs\agent_qwen3b.jsonl
```

The local machine has a 16GB GPU. Full-precision/BF16 7B video inference will
not fit. The default local profile therefore uses the Thinker-only 3B checkpoint
with `--media-mode frames`, a four-frame local default, and a 100,352-pixel cap
per image/frame. Article text and transcripts use deterministic 12,000- and
8,000-character head/tail budgets. Existing transcripts replace raw audio. A 16-frame run was
tested and exceeded this GPU's memory during attention, so use `--frame-count
16` for the paper/cloud protocol and keep the four-frame local run as a separate
resource-constrained ablation. `--media-mode native` remains available for a
larger GPU or remote worker. Always smoke-test one sample before scheduling the
dataset.

Verified local smoke outputs are in `runs/direct_qwen3b_image_lowres_smoke.jsonl`,
`runs/direct_qwen3b_video4_smoke.jsonl`, and
`runs/agent_qwen3b_final2_smoke.jsonl`. The last run completed the agent workflow
without a parse/runtime error; its optional Context Verifier was correctly
skipped because the Sufficiency stage did not request retrieval. These one-item
checks establish operability, not benchmark accuracy.

## Outputs

- `artifacts/dataset_audit.json`: integrity and leakage audit.
- `splits/source_grouped.json`: deterministic source-grouped partitions.
- `runs/*.jsonl`: one append-only prediction record per sample.
- `cache/<backend>/<sample-id>/<stage>.json`: resumable stage cache.

## Research sequence

1. Audit and freeze a source-grouped split.
2. Run Direct MLLM without chain-of-thought, retrieval, or descriptions.
3. Run PanoAgent without search.
4. Add a configured retrieval provider and run PanoAgent Full.
5. Compare overall and per-modality metrics; then annotate intrinsic versus
   relational misinformation using the original detailed records.

## Web search (联网搜索)

The agent's default is now `--search-provider auto`: Tavily is used whenever
`TAVILY_API_KEY` is set, otherwise no network search happens. Pass
`--search-provider none` for the no-retrieval ablation (all pre-2026-11-07 agent
results used `none`). Check the module on its own, without any model call:

```powershell
python -m panoagent.cli search --query "Higher costs and cramped conditions: the impact of Europe's housing crisis 2024" --focus "Estonia recorded a 199% rise"
```

Run the agent with search (loads the saved DashScope key; default is a 3-item
tuning-partition smoke test):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_agent_search.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_agent_search.ps1 -Split splits\half_eval_seed42.json -Partition train -Workers 2 -Tag tuning943
```

How it works:

- `--retrieval-policy text` (default) searches for every item with a title,
  article, or transcript, not only when the model asks. The first query is a
  deterministic title + publication-year query aimed at the original report;
  model-written slot queries follow. `model` restores the old behaviour.
- Long pages are trimmed to claim-relevant passages (numbers weighted highest)
  instead of head/tail, so an altered statistic in the middle of an article
  survives. Snippet-only hits are expanded with Tavily Extract, then a plain
  HTML fetch when the extractor is blocked.
- Context Verifier reports `same_report` (is this the original report of the
  same event?) and exact `slot_checks` (news value vs source value). Only slots
  checked against that same report count; another source's estimate is not a
  refutation.
- External slot guard: when the original report contradicts a central exact
  slot, the final label is FAKE even if the Judge said REAL (the Judge's label
  is kept under `stages.judge`). Disable with `--no-external-slot-guard`.
- The Judge receives the verifier's findings plus an evidence index, not the
  full retrieved texts.
- Search failures are recorded under `retrieval_errors`, never crash the run,
  and are never cached. Search responses and fetched pages are cached under
  `cache/search`. Dataset/paper hosts (Hugging Face, GitHub, arXiv, ...) are
  excluded from results.
- Every record carries `model_metadata.retrieval` (documents, whether the
  dataset's source URL/domain was retrieved, verifier status). The source URL is
  only used for this audit, never in a prompt. `--exclude-source-url` removes
  the original article from evidence for an ablation.

Search results are leads, not ground truth; report retrieval results separately
from the no-search ablation.

## Recommended full configuration

Configure secrets only in the current PowerShell process. Do not paste keys in
chat, command arguments, JSON configuration, source code, or committed files:

```powershell
$env:DASHSCOPE_API_KEY = Read-Host "Model Studio API key" -MaskInput
$env:DASHSCOPE_BASE_URL = "https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:TAVILY_API_KEY = Read-Host "Tavily API key" -MaskInput
```

The Model Studio key and base URL must belong to the same region. Replace the
Beijing host above with the workspace-specific Singapore, Virginia, Japan, or
other regional host shown in the Model Studio console when appropriate.

Use `qwen3-vl-plus` for the controlled protocol that converts audio to text and
video to uniformly sampled frames. This is the preferred first cloud run because
it keeps inputs comparable across models:

```powershell
python -m panoagent.cli direct --dataset .\fakenewsDataSet --split .\splits\formal_pilot_36.json --partition pilot --backend dashscope --model-id qwen3-vl-plus --media-mode frames --frame-count 16 --direct-prompt debiased --temperature 0 --max-new-tokens 16 --resume --output .\runs\direct_qwen3_vl_plus_formal36.jsonl

python -m panoagent.cli agent --dataset .\fakenewsDataSet --ids 1539 --backend dashscope --model-id qwen3-vl-plus --media-mode frames --frame-count 16 --search-provider tavily --max-new-tokens 768 --cache .\cache\qwen3_vl_plus_tavily_smoke --refresh-cache --output .\runs\agent_qwen3_vl_plus_tavily_smoke.jsonl
```

Run the one-item agent smoke test before expanding to the 36- or 180-item split.
Tavily is called only when Evidence Sufficiency returns `sufficient=false` and
provides non-empty search queries. Search results are sent to Context Verifier;
they are not treated as ground truth automatically.

For a separate native audio/video experiment, use the current omni-modal cloud
model and keep its results separate from the 16-frame controlled protocol:

```powershell
python -m panoagent.cli direct --dataset .\fakenewsDataSet --ids 1539 --backend dashscope --model-id qwen3.8-omni-flash --media-mode native --direct-prompt debiased --temperature 0 --max-new-tokens 16 --output .\runs\direct_qwen38_omni_native_smoke.jsonl
```

The backend disables thinking for one-word Direct classification and requests
JSON object output from Qwen3-series models during agent stages. Native Base64
media must still fit the provider's per-file limits; use the frame/transcript
protocol when a dataset file is too large.

The original `--direct-prompt paper` remains the reproducible baseline.
`--direct-prompt debiased` is an ablation designed to reduce the local 3B
checkpoint's observed TRUE/REAL default bias and must be reported separately.
