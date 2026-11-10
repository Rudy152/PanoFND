# 对齐增强 PanoAgent 云端8样本实验（2026-10-10）

## 结构调整

本轮根据样本1539的误判进行了以下调整：

1. Evidence Sufficiency 生成查询时必须包含明确主体、争议谓词、事件时间和司法辖区。
2. Context Verifier 对每项外部证据分别检查 subject、predicate、time、jurisdiction 是否匹配。
3. 明确禁止用事后许可证撤销反推事发时无证，也禁止混淆地方许可、州许可证和联邦许可证。
4. Judge 必须先输出 `central_claim` 和 `central_claim_status`，再决定 REAL/FAKE。
5. 次要细节只有在构成实质性伪造或核心误导时才能改变整体标签。
6. Tavily 每条检索正文限制为6,000字符，减少无关网页内容、输入token和噪声。
7. PanoAgent 现在累计并保存每个阶段及整条样本的API token用量。

代码回归测试：25/25通过。

## 实验配置

- 模型：`qwen3-vl-plus`
- 后端：Alibaba Cloud Model Studio OpenAI-compatible API
- 外部搜索：Tavily
- 样本：8条，4 REAL / 4 FAKE
- 模态：audio、text、image、text+image+video，每类2条且标签平衡
- 视频：4帧资源约束方案
- 每阶段最大输出：1,024 tokens
- 温度：0

## 总体结果

| 指标 | 结果 |
| --- | ---: |
| 完成 / 计划 | 8 / 8 |
| 阶段完全成功 | 8 / 8 |
| 解析或运行错误 | 0 |
| Accuracy | 0.7500 |
| Balanced accuracy | 0.7500 |
| Macro-F1 | 0.7500 |
| REAL Precision / Recall / F1 | 0.75 / 0.75 / 0.75 |
| FAKE Precision / Recall / F1 | 0.75 / 0.75 / 0.75 |
| 总耗时 | 516.78秒 |
| 平均耗时 | 64.60秒/条 |
| Prompt tokens | 289,842 |
| Completion tokens | 18,577 |
| Total tokens | 308,419 |

## 逐样本结果

| ID | 模态 | Gold | Prediction | 正确 | 置信度 | Tokens | 延迟（秒） |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 0011 | audio | REAL | REAL | 是 | 0.98 | 65,625 | 69.97 |
| 0114 | audio | FAKE | FAKE | 是 | 0.95 | 51,038 | 61.23 |
| 0941 | text | REAL | REAL | 是 | 0.98 | 32,768 | 64.35 |
| 0992 | text | FAKE | FAKE | 是 | 0.92 | 24,998 | 55.29 |
| 1539 | text+image+video | REAL | REAL | 是 | 0.85 | 65,950 | 117.80 |
| 1732 | text+image+video | FAKE | FAKE | 是 | 0.95 | 23,818 | 51.28 |
| 1835 | image | REAL | FAKE | 否 | 0.65 | 20,778 | 48.89 |
| 1902 | image | FAKE | REAL | 否 | 0.85 | 23,444 | 47.98 |

## 分模态结果

| 模态 | 样本数 | Accuracy |
| --- | ---: | ---: |
| audio | 2 | 1.00 |
| text | 2 | 1.00 |
| text+image+video | 2 | 1.00 |
| image | 2 | 0.00 |

样本1539从上一版的错误 `FAKE` 修正为 `REAL`。新规则正确识别了“事后撤销许可证”不能证明“事发时没有许可证”，说明时间与谓词对齐产生了预期效果。

## 剩余问题

两条纯图像样本均判断错误，但错误方向相反：

- 1835：真实图像被判FAKE。模型将图片标题中的概括性“collapse”当成需要严格事实核验的完成态主张，过度惩罚修辞性标题。
- 1902：假图像被判REAL。模型确认了背景事件真实，却把可疑文字叠加和错误归因视为次要细节，没有把图像本身的篡改当成核心伪造。

这说明统一的“中央事件真实性”规则不适合直接用于纯图像样本。下一版应增加输入类型路由：纯图像时将图中可见文字、人物归因、拼接和编辑痕迹定义为核心主张；包含正文时再以事件级核心主张为主。由于每种模态只有2条，本轮分模态结果只用于诊断，不能作为稳定的统计结论。

## 输出

- `runs/agent_cloud_qwen3_vl_plus_aligned8_20261010.jsonl`
- `runs/agent_cloud_qwen3_vl_plus_aligned8_20261010_metrics.json`
- `cache/cloud_qwen3_vl_plus_aligned8_20261010/`
