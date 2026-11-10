# 实验问题诊断、改进与云端配置（2026-10-10）

## 问题定位

### Direct MLLM

正式 36 条结果中，35 条原始输出严格为 `TRUE`，另 1 条也以 `TRUE` 开头并附带新闻原文。模型最终将 36 条全部解析为 `REAL`。这表明主要问题是本地 Qwen2.5-Omni-3B 的判别偏置和任务能力不足，而不是标签解析、数据不平衡或显存错误。

可能成因：

1. 原论文的一词零样本提示对 3B 模型约束不足，模型将“没有明显矛盾”错误地当成真实。
2. 多数假新闻需要识别跨模态关系或核验外部事实，Direct 模型没有检索工具。
3. 受 16 GB 显存限制，本地实验仅使用 4 帧和压缩后的视觉输入，视频证据损失较大。
4. 3B 模型本身的视觉、关系推理与事实判别能力有限。

### PanoAgent

两条复杂样本的 `evidence_organizer` 均失败：一条生成过长嵌套对象并在 JSON 中途截断；另一条返回两个部分对象组成的 JSON 数组，而协议要求单个对象。旧评估器只统计最终标签是否可解析，因此曾显示 `parse_or_runtime_errors=0`，掩盖了中间阶段错误。

## 已完成改进

1. 保留 `--direct-prompt paper` 作为原始可复现基线，新增 `--direct-prompt debiased`。新提示明确 REAL/FAKE 先验相等、要求比较两个假设、检查内在伪造和跨模态冲突，并禁止因缺少反证而默认 REAL。
2. Evidence Organizer 明确要求九个键、字符串数组、每键最多五项，减少 3B 模型输出过长导致的截断。
3. Evidence Organizer 可以确定性合并“部分对象数组”，并用空数组补齐缺失类别；不会添加模型没有生成的事实。
4. Qwen3 云端代理阶段请求 `response_format={"type":"json_object"}`。
5. 评估输出新增 `records_with_stage_errors`、`stage_complete_records` 和 `stage_error_counts`。
6. Tavily 改用当前 API 的 `Authorization: Bearer ...` 请求头，密钥不再进入请求正文。
7. Qwen3.8 Direct 请求使用 `reasoning_effort=none`，避免一词分类被长推理占用输出预算。

回归测试：24/24 通过。

## 去偏提示小样本复测

从原正式子集中选择 8 条平衡样本，包含音频、文本、图像和 `text+image+video`：

| 指标 | 原提示在正式 36 条 | 去偏提示 smoke 8 条 |
| --- | ---: | ---: |
| REAL 预测占比 | 100% | 62.5% |
| Accuracy | 50.0% | 62.5% |
| Balanced accuracy | 50.0% | 62.5% |
| Macro-F1 | 33.3% | 61.9% |
| FAKE recall | 0% | 50% |

去偏提示打破了全部 REAL 的塌缩，但 8 条样本不足以证明统计显著提升。原提示和去偏提示属于两个实验条件，不应合并报告。

结果文件：

- `runs/direct_qwen3b_debiased_smoke8_20261010.jsonl`
- `runs/direct_qwen3b_debiased_smoke8_20261010_metrics.json`

## 外部搜索与云端模型配置

在当前 PowerShell 会话中安全输入密钥：

```powershell
$env:DASHSCOPE_API_KEY = Read-Host "Model Studio API key" -MaskInput
$env:DASHSCOPE_BASE_URL = "https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:TAVILY_API_KEY = Read-Host "Tavily API key" -MaskInput
```

API key 必须与 Model Studio base URL 属于同一区域。不要把真实密钥写入命令参数、配置 JSON、源码或实验输出。

推荐先使用 `qwen3-vl-plus` 运行转写文本 + 16 视频帧的受控方案：

```powershell
python -m panoagent.cli direct --dataset .\fakenewsDataSet --split .\splits\formal_pilot_36.json --partition pilot --backend dashscope --model-id qwen3-vl-plus --media-mode frames --frame-count 16 --direct-prompt debiased --temperature 0 --max-new-tokens 16 --resume --output .\runs\direct_qwen3_vl_plus_formal36.jsonl

python -m panoagent.cli agent --dataset .\fakenewsDataSet --ids 1539 --backend dashscope --model-id qwen3-vl-plus --media-mode frames --frame-count 16 --search-provider tavily --max-new-tokens 768 --cache .\cache\qwen3_vl_plus_tavily_smoke --refresh-cache --output .\runs\agent_qwen3_vl_plus_tavily_smoke.jsonl
```

单条 agent smoke test 成功后，再将 `--ids 1539` 替换为 `--split .\splits\formal_pilot_36.json --partition pilot --resume` 扩大运行。

原生音视频属于单独消融条件，可使用：

```powershell
python -m panoagent.cli direct --dataset .\fakenewsDataSet --ids 1539 --backend dashscope --model-id qwen3.8-omni-flash --media-mode native --direct-prompt debiased --temperature 0 --max-new-tokens 16 --output .\runs\direct_qwen38_omni_native_smoke.jsonl
```

Tavily 仅在 Evidence Sufficiency 返回 `sufficient=false` 且给出非空查询时调用。检索结果会进入 Context Verifier，并保留 URL、站点、文本和检索时间。原生 Base64 音视频仍受云端单文件大小限制；过大文件应继续使用转写文本和视频帧方案。

官方文档：

- Alibaba Cloud OpenAI-compatible API: https://help.aliyun.com/en/model-studio/compatibility-of-openai-with-dashscope
- Qwen VL API: https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai
- Qwen Omni: https://help.aliyun.com/en/model-studio/qwen-omni
- Qwen structured output: https://help.aliyun.com/en/model-studio/qwen-structured-output
- Tavily Search API: https://docs.tavily.com/documentation/api-reference/endpoint/search
