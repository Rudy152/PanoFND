# Aligned-8 speed-profile test — 2026-10-19

## Scope

- Model: `qwen3-vl-plus` through DashScope
- Samples: `0011 0114 0941 0992 1539 1732 1835 1902`
- Media mode: four video frames
- Concurrency: two workers
- Strict stage errors: enabled
- Live retrieval: disabled
- This balanced eight-sample set is a regression/latency diagnostic, not a generalization benchmark.

## Implementation issue found and fixed

The first cloud attempt exposed an ambiguous signal schema. Qwen emitted a verbose nested
object for every signal, and seven reports were truncated. The generic JSON repair path then
made a second call without recovering a valid report.

The fused contract now:

- requires flat scalar signal flags;
- limits the report to about 500 output tokens;
- accepts and normalizes the older nested signal shape;
- converts string booleans safely;
- disables the fused stage's second repair call and fails closed instead.

All local tests pass: **44/44**.

The failed diagnostic attempt consumed 36,929 tokens before the defect was identified. Those
tokens are excluded from the profile comparison below because the run produced no valid
predictions. The repair-call change prevents the same automatic double-call pattern.

## Results

| Profile | Valid coverage | Accuracy | Macro-F1 | Approx. model calls | Tokens |
|---|---:|---:|---:|---:|---:|
| Historical `full`, same 8 samples | 100% | 75.0% | 75.0% | 48 | 308,419 |
| New `fused` | 100% | 62.5% | 61.9% | 16 | 25,902 cold |
| New `fused-rule` | 0% final-label coverage | n/a | n/a | 8 | 19,563 |

The cold-token figure for `fused` is the evidence-stage total (19,563) plus the judge total
(6,339). This is a **91.6% token reduction** from `full`. Model calls fall by about **66.7%**.

The sum of per-stage model latency falls from 516.78 seconds for historical `full` to about
187.57 seconds for cold `fused`, a **63.7% reduction** before considering parallel execution.
The actual staged run used two workers: 120.26 seconds were spent obtaining all evidence and
all final judgments (including a sequential one-sample smoke test). Once fused evidence was
cached, rerunning only the eight final judgments took **27.18 seconds** and 6,339 tokens.

`fused-rule` completed every evidence stage but abstained on all eight cases. This is expected
from its fail-closed rule when contradiction fields remain unknown or evidence is not decisive.
It should be treated as a high-precision prefilter, not the sole classifier.

## Per-sample fused outcome

| ID | Modality | Gold | Fused | Correct |
|---|---|---|---|---:|
| 0011 | audio | REAL | FAKE | no |
| 0114 | audio | FAKE | REAL | no |
| 0941 | text | REAL | REAL | yes |
| 0992 | text | FAKE | FAKE | yes |
| 1539 | text+image+video | REAL | REAL | yes |
| 1732 | text+image+video | FAKE | FAKE | yes |
| 1835 | image | REAL | FAKE | no |
| 1902 | image | FAKE | FAKE | yes |

The two audio cases both failed, including one high-confidence wrong judgment. Confidence-only
fallback is therefore insufficient. Audio-specific routing should be evaluated on a separate
validation split before changing production policy.

## Recommended test-time architecture

1. Use `fused` as the normal fast path.
2. Reuse cached fused evidence for prompt, threshold, and judge experiments.
3. Route predefined difficult modality groups—especially audio—to `full` only after selecting
   the rule on validation data, never on this eight-sample test set.
4. Keep `fused-rule` as an optional decisive-case prefilter; send its `ERROR` abstentions to
   the learned judge or `full` profile.
5. Continue using two independent workers for DashScope. Increase concurrency only after
   measuring API rate limits and host upload bandwidth.

This design compresses test time through fewer model stages, one multimodal encoding pass,
parallel execution, and cache reuse rather than by evaluating fewer samples.
