# 联网搜索模块启用与模型优化（2026-11-07）

## 背景

之前所有 agent 实验都用 `--search-provider none`。评审意见指出：大多数假新闻只改了一个细节
（如 1351 把 192% 改成 199%），只判断“核心事件是否真实”很难识别。纯文本样本里，只有外部原始报道
能揭示被改的数字。本轮把检索变成针对精确事实槽位的核查，并默认开启。

## 修改内容

| 问题 | 修改 |
|---|---|
| 检索只在模型主动要求时触发，fused 最多 1 条查询 | `--retrieval-policy text`（默认）：凡有标题/正文/转写都检索；首条为“标题 + 年份”的确定性查询，专门找原始报道；fused 最多 3 条查询 |
| 长文首尾截断，正文中段被改的数字会被丢掉 | 按 claim 词与数字打分选取段落，保留首段 |
| 部分媒体（如 Guardian）只返回摘要 | 先用 Tavily Extract，失败时用标准库 HTML 抓取正文 |
| Verifier 只看 claim，不对比原文槽位 | 新增 `same_report` 与逐槽位 `slot_checks`（新闻值 vs 来源值）；只对比同一篇原始报道，其他来源的不同估计不算矛盾 |
| Judge 收到全部检索全文，token 浪费 | Judge 只收到核查结论和证据索引 |
| Judge 把“改了一个数字”视为次要细节 | 外部槽位保护：原始报道与核心槽位矛盾时，最终标签为 FAKE；Judge 原标签仍保留在记录里，可用 `--no-external-slot-guard` 消融 |
| 网络错误会让整个样本崩溃 | 错误写入 `retrieval_errors`，不会写入缓存；查询带重试，结果按查询缓存 |
| 检索可能命中数据集或论文站点 | 默认排除 Hugging Face、GitHub、arXiv 等站点 |
| 无法评估“是否搜到了原文” | 每条记录都有 `model_metadata.retrieval`（是否命中数据集 source URL/域名，仅用于审计，不进入提示词）；`--exclude-source-url` 可用于消融 |
| Verifier 输出被截断 | 关键字段放在前面，预算提高到 1500 tokens；fused 预算提高到 1200 tokens；不规范的 status 会归一化 |

`description` 和 `source` 仍然不会进入任何提示词，相关测试已覆盖。本地测试：**78/78 通过**。

## 实测（qwen3-vl-plus + Tavily，fused，943 条调参集中的 3 条，未触碰 765 条 holdout）

| ID | 金标 | Judge | 最终 | 依据 |
|---|---|---|---|---|
| 1351 | FAKE | REAL | **FAKE** | 搜到 Guardian 原文；`199% vs 192%` 为核心槽位不一致 |
| 1276 | REAL | REAL | **REAL** | 搜到同一篇原文，所有核心槽位一致 |
| 0979 | FAKE | REAL | **FAKE** | 由保护规则判定，但依据是 `data_source` 差异，属于较弱的理由 |

每条约 1.2–1.4 万 tokens，耗时约 40 秒。

**这 3 条只能证明链路可用，不能说明准确率。** 1351 证明设计是有效的；0979 也提示保护规则可能
增加 REAL→FAKE 误判，必须在调参集上测量。

## 下一步

1. 在 943 条调参集上运行（`scripts\run_agent_search.ps1 -Split splits\half_eval_seed42.json -Partition train -Workers 2 -Tag tuning943`）。
   预计约 1,300 万 DashScope tokens，Tavily 每条约 2–4 次查询。
2. 用缓存零成本重跑 `--no-external-slot-guard`，对比 Balanced Acc.、Macro-F1、两类召回率，以及
   text+image / text+image+audio 中 FAKE→REAL 的数量；同时报告 `source_url_retrieved` 命中率。
3. 冻结后再对 765 条 holdout 只运行一次。

## 均衡 90 条对照实验（2026-11-07，qwen3-vl-plus，fused）

- 子集：`splits/tuning90_balanced_seed42_20261107.json`。从 943 条调参集中抽取，每个“真假 × 9 种模态”格子 5 条，与 765 条 holdout 无重叠；其中 3 条（0003、0007、0044）在设计语音识别修复时用过。
- 两组共用同一份 fused 报告（90/90 来自缓存），差别只在检索核查和 Judge。ERROR 按判错计算。
- 第一次不检索组的结果作废：共享缓存导致检索证据串入，已修复，作废文件为 `runs/agent_fused_nosearch_tuning90_20261107_INVALID_leaked_evidence.jsonl`。

| 组 | Balanced Acc. | Macro-F1 | REAL 召回 | FAKE 召回 | FAKE→REAL | REAL→FAKE | ERROR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 不检索 | **61.1%** | **61.0%** | 44.4% | 77.8% | 9 | 23 | 3 |
| 检索 | 54.4% | 56.0% | 57.8% | 51.1% | 20 | 16 | 5 |

- 配对比较：只有检索组判对 15 条，只有不检索组判对 21 条，精确 McNemar 检验 p = 0.405，差异不显著。
- 检索组：87 条执行了检索，其中 61 条找到同一篇报道，30 条命中数据集记录的 source URL；数字比对触发 1 次，保护规则触发 1 次（0963，金标 REAL，误判）。
- 主要失败模式：检索确认了假新闻背后的真实事件，Judge 随之判 REAL。共有 14 条假新闻在不检索时判对、检索后改判为 REAL，多数是音频或视频与其他模态不一致的样本。
- 0963 误判的原因：数据集正文去掉了标点，“11.2 percent”变成了“112 percent”。
- 结论：当前形式的检索没有带来提升，不应据此运行完整的 943 条。
