# PanoAgent 本地测试报告（2026-10-10）

## 测试环境

- 数据集：`fakenewsDataSet/data.json`
- 模型：本地 Qwen2.5-Omni-3B，Thinker-only
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU，15.99 GiB
- 媒体模式：`transcript`
- 温度：0
- 本次未配置 DashScope 或 Tavily 凭据，因此 Agent 测试属于无外部检索消融。

## 自动化测试

首次运行时，Windows 沙箱的默认临时目录拒绝写入，导致所有使用
`tempfile.TemporaryDirectory()` 的测试失败。将 `TEMP`/`TMP` 指向工作区可写的
`tmp/pytest-run` 并关闭 pytest 缓存后，结果为：

- 17 passed
- 0 failed
- 用时 2.42 秒

## Direct MLLM 数据测试

使用同一次模型加载测试一真一假两个文本样本：

| ID | 金标 | 预测 | 结果 | 推理耗时 |
|---|---|---|---|---:|
| 0901 | REAL | REAL | 正确 | 107.96 秒（含模型首次加载） |
| 0976 | FAKE | REAL | 错误 | 3.86 秒 |

指标：

- evaluated：2
- parse/runtime errors：0
- accuracy：0.5000
- balanced accuracy：0.5000
- macro-F1：0.3333
- FAKE recall：0.0000

样本量仅为 2，以上用于验证运行链路和基本判别行为，不能作为正式基准结果。

## 完整 PanoAgent 测试

对困难样本 0976 运行 Evidence Organizer、Unimodal Investigator、Relation
Investigator、Evidence Sufficiency、Context Verifier 和 Judge：

- 金标：FAKE
- 预测：REAL
- 冷启动完整流程耗时：382.45 秒
- 解析/运行错误：0
- Sufficiency 正确判定证据不足并触发 Context Verifier。

测试发现：当检索器返回零条证据时，旧实现仍调用 Context 模型，模型凭空生成了
NIH URL，Judge 随后把该 URL 当作证据。该问题已修复：零检索结果现在直接返回
`insufficient`、空 `sources` 和空 `retrieved_evidence`，不会调用模型补造来源。

修复后复用上游缓存重跑：

- Context status：insufficient
- Context sources：0
- Retrieved evidence：0
- 预测仍为 REAL
- Judge confidence：1.0
- 重跑耗时：128.88 秒（包含模型加载和 Judge，复用了上游缓存）

这说明来源幻觉漏洞已修复；剩余错误来自 3B Judge 将新闻文本自身当作真实性依据，
属于模型能力与置信度校准问题。正式实验应使用配置了真实检索结果的 PanoAgent Full，
并在平衡 pilot 上评测，不能根据这一条样本推断总体性能。

## 输出文件

- `runs/local_qwen3b_dataset_test_20261010.jsonl`
- `runs/local_qwen3b_dataset_test_20261010_metrics.json`
- `runs/panoagent_qwen3b_dataset_test_20261010.jsonl`
- `runs/panoagent_qwen3b_dataset_test_20261010_metrics.json`
- `runs/panoagent_qwen3b_dataset_test_20261010_fixed.jsonl`
- `runs/panoagent_qwen3b_dataset_test_20261010_fixed_metrics.json`

