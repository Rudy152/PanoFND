# 正式实验阶段报告（2026-10-10）

## 结论摘要

本轮完成了一个扩大样本量的本地正式试验：Direct MLLM 在 36 条类别与模态平衡样本上全部完成推理，但 36 条全部被预测为 `REAL`。因此准确率和 balanced accuracy 都是 0.5000，macro-F1 为 0.3333，说明当前本地 Qwen2.5-Omni-3B 配置发生了明显的单类塌缩，尚不能作为有效的假新闻检测器。

另外对完整 PanoAgent 流水线运行了 2 条 `text+image+video` 复杂样本。两条都生成了最终预测，结果为 1 对 1 错；但两条都在 `evidence_organizer` 阶段产生不符合 schema 的结构化输出。因此这两条只能视为流水线诊断实验，不能证明完整代理流程稳定。

## 实验设计

- 数据切分：沿用 source-grouped test split，避免同源样本跨集合泄漏。
- Direct 样本数：36。
- 标签分布：18 `REAL`，18 `FAKE`。
- 模态组合：9 种，每种 4 条；每种组合中 2 `REAL`、2 `FAKE`。
- 抽样随机种子：42。
- 本地模型：Qwen2.5-Omni-3B Thinker-only。
- 视频输入：均匀抽取 4 帧。
- 图像像素上限：100,352。
- 正文上限：12,000 字符。
- 转写文本上限：8,000 字符。
- Direct 解码：temperature 0，max new tokens 16。

平衡子集清单：`splits/formal_pilot_36.json`。

## Direct MLLM：36 条正式子集

| 指标 | 结果 |
| --- | ---: |
| 完成 / 计划 | 36 / 36 |
| 顶层解析或运行错误 | 0 |
| OOM | 0 |
| Accuracy | 0.5000 |
| Balanced accuracy | 0.5000 |
| Macro-F1 | 0.3333 |
| REAL recall | 1.0000 |
| FAKE recall | 0.0000 |
| REAL 预测数 | 36 |
| FAKE 预测数 | 0 |
| 推理总耗时 | 188.16 秒 |
| 平均耗时 | 5.23 秒/条 |

混淆结果等价于：18 条 `REAL` 全部正确，18 条 `FAKE` 全部错误。9 种模态组合的表现完全相同，每组准确率均为 0.5；这不是可靠的跨模态能力，而是统一预测 `REAL` 造成的表面分数。

输出文件：

- `runs/direct_qwen3b_formal36_budgeted_20261010.jsonl`
- `runs/direct_qwen3b_formal36_budgeted_20261010_metrics.json`

## PanoAgent：复杂多模态诊断

| 样本 | Gold | Prediction | 最终判定 | 延迟 | 阶段状态 |
| --- | --- | --- | --- | ---: | --- |
| 1539 | REAL | REAL | 正确 | 370.28 秒 | `evidence_organizer` schema 错误 |
| 1732 | FAKE | REAL | 错误 | 1058.65 秒 | `evidence_organizer` schema 错误 |

汇总：2 条均完成最终判定，accuracy 0.5000、balanced accuracy 0.5000、macro-F1 0.3333；但阶段完全成功数为 0/2。总耗时 1428.94 秒，平均 714.47 秒/条。

评估器中的 `parse_or_runtime_errors=0` 只表示最终预测字段可解析，不代表各代理阶段没有异常。原始记录中的 `error` 字段明确保留了两条 `evidence_organizer` 错误。

输出文件：

- `runs/agent_qwen3b_formal_multimodal2_20261010.jsonl`
- `runs/agent_qwen3b_formal_multimodal2_20261010_metrics.json`

## 本轮排障与验证

最初的 Direct 运行在第 13 条样本遇到显存不足。该样本包含 48,402 字符转写和 1400×1400 图像。随后加入正文/转写上下文预算、降低图像像素上限，并在每条样本后释放临时张量和 CUDA 缓存。修复后该样本单独通过，36 条正式运行也全部通过且没有 OOM。

旧的失败运行保留为诊断记录：`runs/direct_qwen3b_formal36_20261010.jsonl`，其中 13 条记录、1 条 OOM；它未混入正式指标。

代码回归测试：18/18 通过。

## 当前判断

工程稳定性问题已经从“无法跑完较长多模态输入”改善为“Direct 36 条稳定完成”。当前主要瓶颈已转为模型质量和代理输出约束：

1. Direct 模型存在严重的 `REAL` 单类偏置。
2. 3B 本地模型对 PanoAgent 的复杂 JSON schema 遵循不稳定。
3. 完整 PanoAgent 单条复杂样本耗时约 6–18 分钟，不适合在修正 schema 可靠性前直接扩展到大规模实验。
4. 本轮未启用外部搜索服务，也未使用 DashScope 大模型，因此结果是本地 3B 基线，不等同于论文级完整配置。

下一阶段应先修复/简化 `evidence_organizer` 输出协议并做小规模回归，再调整 Direct 判别提示、标签解析或改用更强模型，确认不再单类塌缩后才值得扩大正式样本量。
