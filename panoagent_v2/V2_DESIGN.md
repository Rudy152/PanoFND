# PanoAgent v2 设计说明

v2 是一个独立的包，原来的 `panoagent` 和历史结果都没有改动。v2 只从 `panoagent` 引用数据加载、划分和 JSON 解析这几个工具函数。

## 架构

```
样本 → L0 媒体规范化 → L1 本地取证检测 → L2 两两/单模态证据智能体 → L4 可训练打分器 → 标签 + 概率 + 定位 + 解释
```

| 层 | 文件 | 作用 | 对应的评审意见 |
|---|---|---|---|
| L0 | `media.py`、`media_worker.py` | 每个附件都带标签，例如 `[NEWS IMAGE 1]`、`[VIDEO FRAME 3 @ 00:42]`、`[NEWS AUDIO PART 1/2]`、`[VIDEO SOUNDTRACK PART 1/1]`。音频以原始声音输入（压缩为单声道 MP3，按 20 分钟分段），不再使用转写稿。视频抽 8 帧并标注时间点，同时提取音轨。所有解码在子进程里完成，解码器崩溃不会中断整个运行。缓存按文件内容的 SHA-256 生成，而不是按文件路径。 | 意见 2、意见 6（缓存指纹只记路径） |
| L1 | `detectors.py` | 用 CLIP ViT-L/14-336 做零样本判断（AI 生成或编辑过的图片 vs 相机拍摄的照片），权重已在本地。它只提供分数，不直接下结论。 | 意见 3（目前是较弱的版本，见下文） |
| L2 | `agents.py`、`prompts.py` | 先做一次单次调用（输入与后续相同），取出 TRUE/FALSE 的 log 概率，作为连续的 P(fake)。之后对每一对模态各调用一次，每次只给模型看这两个模态；只有一个模态的样本做单模态审查。输出只能从固定枚举值里选，模板不预填默认值，也没有针对个别样本的补丁。 | 意见 2、意见 4、意见 5 |
| L4 | `features.py`、`scorer.py` | 17 个特征，在看到任何结果之前就固定下来。打分器是带标准化和 L2 正则的逻辑回归，使用嵌套交叉验证：外层 5 折、内层 4 折，按新闻来源分组并按标签分层；正则强度和阈值只在训练折里选择。报告的每一条预测都来自没有见过该样本、也没有见过同一来源样本的模型。 | 意见 5、意见 6（校准抵消槽位保护） |
| 输出 | `features.py`、`cli.py` | 定位：给出被判为篡改的模态，以及存在矛盾的模态对。解释：只根据已有证据按模板拼出，不额外调用模型。 | 意见 7（可解释性需要能评估） |

## 已删除的内容

v1 里的数字逐句比对、槽位保护强制改判、离线校准，以及所有针对个别样本或个别实体写的提示词，v2 都不再使用。检索（L3）在这一版里没有启用：11/07 的对照实验显示它会把假新闻推向判真。以后如果再加，只能核查文本，必须只用新闻发布日期之前的证据，并排除辟谣网站。

## 数据纪律

- 训练和调参只用 943 条调参集（`splits/half_eval_seed42.json` 的 train 部分）。180 条实验集 `splits/v2_tuning180_seed42_20261121.json` 是它的子集，与 765 条最终测试集没有相同的样本 ID，来源组也不重叠；但 943 条调参集与最终测试集有标题相同的视频样本，涉及 94 条最终测试样本（2026-11-24 更正）。
- `description` 和来源 URL 不进入任何提示词，也不作为特征，有测试覆盖这一点。
- 765 条最终测试集只能在 v2 冻结之后运行一次。

## 已知限制

- 图片检测器偏弱：CLIP 零样本对图表这类本身"像合成图"的图片会给出高分。MiRAGe-Img 需要 BLIP-2 flan-t5-xl（约 15 GB），音频防伪需要 AASIST 这类模型，都需要下载，这一版没有包含。
- 180 条上的交叉验证样本量小，结论的置信区间较宽。

## 运行方式

```
powershell -ExecutionPolicy Bypass -File scripts\run_v2.ps1 -Split splits\v2_tuning180_seed42_20261121.json -Partition pilot -Output runs\v2_tuning180_features_20261121.jsonl
python -m panoagent_v2.cli score --dataset fakenewsDataSet --features runs\v2_tuning180_features_20261121.jsonl --output-dir runs\v2_tuning180_score_20261121 --baseline "single call qwen3.8-omni-flash (v1 inputs)" runs\qwen38_omni_flash_full_1950_20261107.jsonl
python -m pytest tests_v2
```
