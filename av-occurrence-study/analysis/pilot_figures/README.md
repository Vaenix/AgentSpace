# replication_1124 论文图表

从 [replication_1124](../../run/replication_1124/README.md) 已冻结的 18 条原始回复、提示映射（`manifest/prompts.jsonl`）离线复算，生成英文论文图、计数表和图注。只读取历史工件，不重新推理，也不导入推理入口的解析函数：预设严格规则和事后格式敏感性规则都在脚本内独立实现，再与冻结字段逐条核对。

## 生成与核对

只需要 Python 3.11 和 matplotlib（本次为 3.9.0），无需网络、媒体或模型权重。

```sh
# 重新生成全部输出（覆盖本目录下的生成文件）
python3 av-occurrence-study/analysis/pilot_figures/make_pilot_figures.py

# 在临时目录重新生成，并与已提交文件逐字节比较；任一复算核对不符即失败
python3 av-occurrence-study/analysis/pilot_figures/make_pilot_figures.py --check
```

脚本在写出任何文件前执行 38 项核对，全部通过才写出结果：

- `prompts.jsonl` 的 SHA-256 与冻结的 `input_hashes.json` 一致；恰有 18 个不同的问题-条件键，与提示清单一一对应。
- 独立实现的严格解析逐条复现 `predictions.jsonl` 中的 `answer_id`、`interval_s`、`physical_window`、`result_format`。
- 严格口径的逐题比较与 `comparisons.csv` 一致，总计与 `summary.json` 一致。
- 事后口径恢复的 6 条回复与 `format_sensitivity.json` 的 `recovered_outputs` 一致；两种口径的总计与逐视频计数均与该文件一致。
- 每个分母满足：全部题数 = 可比 + 弃答导致不可比 + 格式导致不可比。

matplotlib 版本不同时，PDF/SVG 可能出现字节差异，数据表和图注不受影响。

## 输出

| 文件 | 内容 |
| --- | --- |
| `fig_pilot_per_question.pdf` | 矢量图：3 部视频 × 2 问在原顺序、选项反转、片段代号反转下的逐题回复，以及两种口径的比较 |
| `fig_pilot_per_question.svg` | 同一图的可编辑 SVG（文字保留为文本） |
| `make_pilot_figures.py` | 图、表和图注的生成源 |
| `table_pilot_counts.csv` | 机器可读计数表：口径 × 干预 × 视频/合计 |
| `table_pilot_counts.md`、`table_pilot_counts.tex` | 英文表；严格统计与事后敏感性分开，附 P03_117 两个实际引用区间的实例（LaTeX 需 `booktabs`） |
| `pilot_responses.csv` | 18 条回复：原始字母和代号、映射、两种口径下的状态与物理时间窗 |
| `pilot_comparisons.csv` | 每种口径 12 个题-干预比较，含不可比原因 |
| `captions.md` | 英文图注和表注，计数由脚本填入 |
| `verification.json` | 输入文件 SHA-256、回复状态计数、分母和核对清单 |

## 口径

- **严格（预设、主结果）**：回复须是仅含 `answer`、`event` 的 JSON，`answer` 为选项字母或 null，`event` 为单独片段代号或 null。
- **事后格式敏感性（补充）**：严格口径判为格式无效、但事件字段形如 `E01 530.566–533.272s` 时，仅当代号与显示的起止秒数都和该条件的冻结映射精确一致，才恢复物理时间窗。该规则在看到回复后制定。
- 一题只有在原顺序和干预两侧都得到动作与物理时间窗时才可比；不可比题保留在分母中，并按弃答或格式区分（两侧兼有时记为弃答）。
- 每部视频的两问共用同一片段、帧和选项，结果相关。没有人工金标，所有计数只描述行为一致性，不代表答案或引用是否正确。
