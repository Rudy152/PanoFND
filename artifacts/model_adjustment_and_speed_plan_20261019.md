# PanoAgent 模型调整与测试提速记录（2026-10-19）

## 已实施调整

1. 保留 `full` 作为原始五/六阶段基线，新增 `fused` 消融配置。
2. `fused` 用一次多模态调用同时输出中心主张、单模态判断、跨模态关系、固定冲突信号和检索需求，再用一次结构化 Judge 完成二分类。
3. 固定信号包括 claim support/contradiction、identity、attribution、time、jurisdiction、overlay、materiality、confidence 和 stage validity；缺失信号规范化为 `unknown`，不会自动补成 `false`。
4. 新增 `--strict-stage-errors`：任一已执行阶段失败时最终预测记为 ERROR。
5. 新增 DashScope `--workers N` 样本级并发；每个 worker 独立持有 backend，避免 metadata 串线。本地 GPU 后端禁止并发加载。
6. 评测新增 coverage、ERROR 按错计的 end-to-end accuracy、stage-complete 子集和严格端到端指标。
7. 增加端到端泄漏回归测试，证明 `description` 不进入 full 或 fused 的任何 prompt。
8. 新增离线置信度校准命令：仅用 source-grouped validation 拟合全局或分模态 FAKE 阈值，再冻结应用到 held-out test；该步骤不增加模型调用。
9. 新增 `fused-rule`：融合取证后使用透明、fail-closed 的确定性聚合器，不再调用 LLM Judge；无检索时每样本只调用模型一次。歧义证据返回 ERROR，不默认 REAL。

## 历史 943 样本成本证据

从已有 JSONL 的阶段 metadata 离线汇总：

| 阶段 | 有 usage 的调用 | Tokens |
|---|---:|---:|
| Relation Investigator | 872 | 3,295,555 |
| Unimodal Investigator | 940 | 3,064,299 |
| Evidence Sufficiency | 940 | 2,722,827 |
| Evidence Organizer | 938 | 2,656,265 |
| Judge | 942 | 1,833,501 |

旧方案的大部分成本来自同一新闻和结构化摘要被反复输入。`fused` 把无检索配置的模型调用数从每样本 5 次降为 2 次（减少 60%），并把媒体编码次数从 3 次降为 1 次。实际 token 和准确率收益必须通过同划分消融测量。

## 更新后的严格评测口径

对 943 条历史去重结果重新计算：

- 有效二分类 coverage：940/943 = 99.68%
- 原有效样本 accuracy：61.28%
- ERROR 按错计的 end-to-end accuracy：61.08%
- 无阶段错误且有有效标签：914/943 = 96.92%
- 阶段错误也按错计的严格 accuracy：559/943 = 59.28%

这解释了为什么“最终有 REAL/FAKE”不能等同于完整成功。

## 推荐的可比实验

必须使用相同 source-grouped validation、相同媒体预算和相同错误计分：

1. Direct baseline（1 call）。
2. Full agent，无检索（5 calls）。
3. Fused agent，无检索（2 calls）。
4. Fused-rule，无检索（1 call）。
5. Fused agent + 固定缓存检索（通常 3 calls）。

验收同时比较 macro-F1、balanced accuracy、FAKE recall、REAL recall、strict coverage、tokens/sample 和 wall-clock seconds/sample。先在 validation 决定是否采用 fused；held-out test 只运行一次。

## 进一步缩短墙钟时间的方法

- DashScope 使用 2–4 个样本级 worker；收益近似受限于 API rate limit，而不是 GPU。
- 预先抽帧并复用 `cache/video_frames`，避免实验过程中反复解码几十 GB 视频。
- 使用 `--resume` 和请求指纹缓存；只重跑失败或 prompt 已改变的阶段。
- 保持 4 帧作为资源受限消融；若改为 8/16 帧，必须单独命名实验，不能与 4 帧结果合并。
- 对云端 VL 保持图片像素上限、正文/转写字符预算和紧凑 JSON schema。
- 若只改变 Judge 或聚合规则，复用缓存的 fused/full 证据，不重新上传媒体。

并发只减少墙钟时间，不减少 token 费用。减少阶段数、媒体重复编码与上下文长度，才会同时降低时间和成本。
