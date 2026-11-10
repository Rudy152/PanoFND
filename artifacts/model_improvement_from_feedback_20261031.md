# 根据评审意见进行的模型改进（2026-10-31）

## 先看重新统计后的基线

10 月 12 日的 943 条 source-grouped 调参集包含 613 条 FAKE、330 条 REAL，FAKE 占
65.01%。全部预测 FAKE 的普通准确率就是 65.01%。原模型的 ERROR-as-wrong 准确率为
61.08%，低于多数类基线 3.92 个百分点；但 Balanced Accuracy 为 62.90%，Macro-F1
为 60.55%，说明模型不是完全无效，而是普通 accuracy 在类别不平衡下具有误导性。

原模型的关键混淆计数：

| Gold → Prediction | 数量 |
|---|---:|
| REAL → REAL | 224 |
| REAL → FAKE | 104 |
| REAL → ERROR | 2 |
| FAKE → FAKE | 352 |
| FAKE → REAL | **260** |
| FAKE → ERROR | 1 |

主要错误确实是 **FAKE 被判成 REAL**。REAL recall 为 68.29%，FAKE recall 只有
57.52%。

分模态错误方向进一步确认了问题：

| 模态 | N | REAL→FAKE | FAKE→REAL | ERROR | Balanced Acc. | Macro-F1 | 多数类 Acc. | 模型 E2E Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audio | 76 | 17 | 14 | 0 | 59.21% | 59.15% | 50.00% | 59.21% |
| audio+image | 123 | 21 | 26 | 2 | 57.70% | 57.38% | 66.67% | 60.16% |
| image | 78 | 14 | 13 | 0 | 65.38% | 65.38% | 50.00% | 65.38% |
| text | 66 | 0 | 19 | 0 | 67.24% | 65.43% | 56.06% | 71.21% |
| text+audio | 111 | 15 | 14 | 1 | 69.71% | 69.85% | 66.67% | 72.97% |
| text+image | 93 | 5 | **47** | 0 | 54.03% | 43.29% | 66.67% | 44.09% |
| text+image+audio | 152 | 13 | **67** | 0 | 53.51% | 46.24% | 75.00% | 47.37% |
| text+image+video | 147 | 10 | **41** | 0 | 68.03% | 62.53% | 74.15% | 65.31% |
| text+video | 97 | 9 | 19 | 0 | 71.09% | 69.08% | 68.04% | 71.13% |

## 检索设定

10 月 12 日的 943 条实验和 10 月 19 日的 validation 实验都使用
`--search-provider none`，没有开启 Tavily 或其他实时检索。ContextVerifier 在没有缓存外部
证据时只能返回 `insufficient`。这些结果衡量的是无实时检索的多模态流水线，不能当作
完整检索增强系统的结果。

## 已完成的模型修改

1. **消除 Judge 的默认 REAL 倾向。** 删除 “Return FAKE only when…” 的非对称措辞，
   明确 REAL 与 FAKE 先验相等。REAL 必须有材料事实槽的一致性与支持；缺乏外部证据
   不能自动变成 REAL。
2. **加入精确原子事实抽取。** Evidence Organizer 新增 `numeric_claims`，并要求保留
   原始数字、单位和拼写，不得四舍五入或自动纠正。
3. **逐项跨模态核对。** Relation Investigator 现在检查 number、person、organization、
   location、time、quote 和 attribution，输出 `slot_checks`。例如 192% 对 199% 会明确
   记录为 mismatch，而不是因为“同一个核心事件”被忽略。
4. **允许单个材料细节决定 FAKE。** Judge 和 fused Judge 都明确：一个数字、人名、
   地点或日期的变化，只要改变了发布断言，就足以构成 FAKE，即使大事件本身真实。
5. **扩展 fused 信号。** 新增 `numeric_mismatch`、`location_mismatch` 和 `atomic_checks`；
   确定性聚合器也识别这些矛盾。
6. **移除 1835 过拟合规则。** 纯图像策略中不再出现 collapse、teetering、weakened。
   新规则只保留通用原则：按语境解释修辞，但优先核对精确数字、实体、时间与归因。
7. **完善评估。** overall 和每种 modality combination 都自动输出混淆矩阵、多数类
   基线、FAKE→REAL、REAL→FAKE，以及高于多数类基线的 E2E accuracy margin。

所有本地测试通过：**53/53**。

## 数据划分修正

评审建议用已跑过的 943 条做调参，另一半做最终测试。由于之前的 228 条 validation 和
aligned-8 已经触碰了原 1007 条候选 test 中的 236 个 ID，原 1007 条不能再全部声称为
未见数据。按 source group 整组排除污染后，已经冻结：

- `splits/final_holdout_clean_765_20261031.json`
- 765 条：260 REAL / 505 FAKE
- SHA-256：`2F6E5CCB5D2C586F8B70C53AE0DE0C26E557921F917EC17DDFB4B6EEFB78B5D7`

后续提示词和阈值只能使用 943 条 tuning partition。765 条 holdout 只允许在模型与路由
完全冻结后运行一次，不再根据其标签修改模型。

## 下一步验证协议

新提示词会改变请求指纹，所以旧缓存不能代表改进后的模型。应先在 943 条 tuning 集上
重跑，比较以下主指标：Balanced Accuracy、Macro-F1、REAL recall、FAKE recall，以及
text+image、text+image+audio 的 FAKE→REAL 数量。只有这些指标改善且没有明显损害其他
模态，才冻结模型并运行 765 条最终 holdout。
