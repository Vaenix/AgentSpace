# 人工金标与两选一回复的离线对齐

本目录把三片独立批（confirmation_1124）的 18 条两选一回复，对齐到 `eval/freeze_reviews.py` 冻结的人工金标，再用 `eval/score.py` 计分。金标使用新题号和文本动作 ID（`act_…`）。模型回复是当时提示里的选项字母和片段代号。入口不生成模型回复，也不读写人工复核原件。

**当前状态：未跑。** 三片尚无人工金标，[预检报告](precheck.json)只核对键和冻结输入，真实正确率和定位命中率均为 `null`。

## 复现

在 AgentSpace 仓库根目录运行，需要 Python 3.10 或更高版本和完整 Git 历史。输出路径须尚不存在。

```sh
# 真实三片：只做键和输入预检，与存档逐字节比较
python3 av-occurrence-study/run/confirmation_1124/gold_alignment/align.py align \
  --output /tmp/confirmation-1124-precheck.json
cmp av-occurrence-study/run/confirmation_1124/gold_alignment/precheck.json \
  /tmp/confirmation-1124-precheck.json

# 自测：全部使用标注 SYNTHETIC 的构造样例，另含一项真实预检
python3 -m unittest discover -s av-occurrence-study/run/confirmation_1124/gold_alignment -v
```

人工金标冻结后，按以下两步对齐。`frozen-v1` 是 `freeze_reviews.py freeze` 的输出目录，内含 `gold.jsonl`、`questions.jsonl` 和 `freeze.json`。

```sh
python3 av-occurrence-study/run/confirmation_1124/gold_alignment/align.py template \
  --gold-dir frozen-v1 --output mapping.tsv
# 在 mapping.tsv 中填写需人工判断的“映射”“映射人”“理由”，保持其余列不变
python3 av-occurrence-study/run/confirmation_1124/gold_alignment/align.py align \
  --gold-dir frozen-v1 --mapping mapping.tsv --output alignment-report.json
```

默认只接受存档回复账本：SHA-256 须为 `3e465d57ad01f4a09ed31beb0c85711c79872cb3dfaa78c1b3431a712a2d6aa9`，且不附技术失败文件。其他账本（改动过的副本、不完整账本，或附 `--technical-failures` 的账本）会被拒绝。如需诊断，须显式加 `--diagnostic-ledger`，报告随之标为 `"ledger": "diagnostic_custom_ledger"`，`result_scope` 写明“diagnostic only: not a frozen confirmation_1124 result”。这类报告不作为本批冻结结果。

## 冻结输入

| 输入 | 公开引用 | SHA-256 |
| --- | --- | --- |
| 协议 | [86c0044 `协议.md`](https://github.com/Vaenix/AgentSpace/blob/86c00449b09e2540c0d2c8f52b197055142da3e8/av-occurrence-study/run/confirmation_1124/协议.md) | `9abcb58d…d3481` |
| 三片选择 | [86c0044 `manifest/selected.jsonl`](https://github.com/Vaenix/AgentSpace/blob/86c00449b09e2540c0d2c8f52b197055142da3e8/av-occurrence-study/run/confirmation_1124/manifest/selected.jsonl) | `5d9a9158…915a5` |
| 18 条提示与各条件代号映射 | [86c0044 `manifest/prompts.jsonl`](https://github.com/Vaenix/AgentSpace/blob/86c00449b09e2540c0d2c8f52b197055142da3e8/av-occurrence-study/run/confirmation_1124/manifest/prompts.jsonl) | `93880eb7…f28e8` |
| 原始回复 | [89fc8a5 `predictions.jsonl`](https://github.com/Vaenix/AgentSpace/blob/89fc8a5a8a4dd7ee0574663e4566e96224ce524c/av-occurrence-study/run/confirmation_1124/predictions.jsonl) | `3e465d57…a6aa9` |
| 解析器 | 协议锁定 [f691998](https://github.com/Vaenix/AgentSpace/blob/f6919987b6fe3ecf9746f7d9a5d7ab1edb8db3c4/av-occurrence-study/run/infer_mlx.py)，从主分支合并提交 87ee601 读取 | `4ebbd521…fc027` |

运行时，前四个文本文件须与冻结提交逐字节一致。回复账本的指纹固定写在 `align.py` 中，并须与 `statistics.json` 的记录相同，否则按上节规定拒绝或标为诊断。完整哈希、复用的 `score.py`、`freeze_reviews.py` 和 `summarize.py` 指纹均写入报告的 `inputs`。

## 对齐规则

1. **解码**：复用本批 `summarize.decode_rows`，按每条提示自己的 `option_codes`（A/B）与 `segment_codes`（E01/E02）还原为原选项语义 ID 和原片物理窗。仅换选项和仅换片段代号两条件因此各按其映射解码。严格回复须与推理记录逐字段一致。
2. **键检查**：每条提示的 pair_id、video_id、序数和条件顺序与三片选择一致。原选项 ID 等于所选 EPIC 动作编号的 blake2s-8，文字等于来源描述。仅换选项只交换 A/B，仅换片段代号只交换 E01/E02。物理窗等于来源发声区间。
3. **题目**：金标题对按 `pair_id` 匹配，`video_id` 不符即报错。金标 `occurrence` 对应原提示的序数（第一或第二次声音）。原题号 `…-smoke-N` 与金标题号 `…-N` 并列保留在报告中。
4. **动作**：只有一种自动等价。原选项去掉提示构造时追加的句点后，与金标动作文字经 `freeze_reviews.action_id` 同一规范化（NFKC、大小写折叠、空白合并）完全相同。其余措辞一律标为“待人工映射”，不做同义词或模糊匹配。映射表中每个原选项可填 `金标1`、`金标2` 或 `无对应`，并须填写映射人和理由。人工映射与自动相同判定冲突、两个原选项映射到同一金标动作、映射表与当前金标指纹或原选项不符，均报错。
5. **区间**：目标区间是金标中裁决后的原片发声区间，原样保留，另附裁决后的动作区间。预测区间是回复所指物理窗的原片发声区间。报告逐题给出裁决区间与两个物理窗的 IoU，供核查人工区间与原候选窗的关系。

## 分母与计分条件

每个题对归入一种互斥状态，见 `denominators.pairs`：

| 状态 | 含义 | 是否计分 |
| --- | --- | --- |
| `no_gold` | 金标冻结记录中没有此题对 | 否 |
| `excluded` | 仲裁为排除 | 否 |
| `undecided` | 仲裁为待定 | 否 |
| `mapping_pending` | 至少一个原选项待人工映射 | 否 |
| `answer_not_in_original_options` | 原选项均已判定，人工确认某题金标动作不在原两个选项中；逐题另列于 `answer_not_in_original_options` | 否 |
| `aligned` | 两题的金标动作都能由原选项对应 | 是 |

`denominators.questions` 逐题计数。无金标、排除和待定题对中的两题都计入题对的状态。纳入题对按各题自己的语义映射计为 `aligned`、`mapping_pending` 或 `answer_not_in_original_options`。例如一题已对应、另一题确认不在原选项时，两类各计 1 题。`aligned_questions_in_unscored_pairs` 另列这类情况：题目本身已对应，但所在题对整体未进入计分，并按题对状态区分待映射和缺选项。`scored_pairs` 和 `scored_questions` 是实际交给计分器的数量。

只有 `aligned` 题对进入 `eval/score.py` 的 `score_group` 与 `compare_groups`，主解析 `matched_seconds` 和严格解析 `strict` 分别计分，三个条件各成一组。动作按金标 `act_…` 比较。若模型选了判定为“无对应”的原选项，记作答错。定位为预测物理窗区间与裁决区间的 IoU 不低于 0.5。

动作和证据两个字段分别判定状态：`valid`、`explicit_null`、`invalid_field`、`code_seconds_conflict`（仅证据）、`format_error` 或 `technical_missing`。每组 `reply_statuses` 列出已收回复数、技术缺失数、原有互斥状态和 `fields` 下两个字段各自的计数。

| 情况 | 交给 `score.py` 的预测 | 计分含义 |
| --- | --- | --- |
| 技术缺失 | 不提交 | 唯一记为缺预测的情况，`prediction_present` 不满足 |
| JSON 或键格式错误 | 提交，动作为不可用标记，区间为空 | 已收到回复，答错且未定位，不算弃答 |
| 动作有效 | 映射后的金标 `act_…` | 不受证据字段影响，照常判定答案 |
| 动作为显式 `null` | `null` | 计入 `abstained` |
| 动作字段无效 | 不可用标记 | 答错，不算弃答 |
| 证据有效 | 该物理窗的原片区间 | 照常判定定位 |
| 证据为 `null`、无效或代号与秒数冲突 | 空区间 | 定位不成立 |

题对联合成功仍要求两题的动作与定位均有效且正确。例如真实三片在严格解析下，换片段代号的 6 条回复动作均有效、证据均为无效字段（见 `precheck.json` 的 `field_statuses_all_keys`），有金标时动作照常计分，定位和联合成功不成立。没有对齐题对时，报告和各组状态为 `未跑`，指标为 `null`。

## 测试证据

`test_align.py` 共 13 项测试。其中 11 项只用标注 `SYNTHETIC` 的题对、视频、动作和复核人代号，另 2 项读取真实账本。合成金标由真实的 `freeze_reviews.py prepare/freeze` 从合成复核记录生成。

- **正常**：同措辞自动对齐到新题号和 `act_…`，三个条件均答对、定位正确。
- **选项或证据反转**：在反转条件中仍沿用原字母或原代号，分别得到答案互换和证据互换。代号附秒数时，只有严格解析判证据字段无效，动作仍照常计分。
- **未决映射**：措辞不同时进入待人工映射，不计分，人工填写后对齐。交叉映射按金标而非来源序数计分。
- **人工确认不在原选项**：该题对不计分，并单列该题。
- **映射冲突、过期，或含重复行（包括空白重复）、缺行**：均拒绝。
- **逐题分母**：一题已对应、另一题缺选项或待映射时，逐题分别计数，并另列未进入计分的已对应题。
- **字段状态**：动作有效但证据无效，或动作无效但证据有效时，各保留有效字段。格式错误算已收回复，不算弃答。只有技术缺失算缺预测。
- **分母**：排除、待定、无金标、缺预测与弃答各自保留。
- **真实预检**：真实三片只核对 18 个键和冻结输入，所有正确率为 `null`，并与存档的 `precheck.json` 逐字节一致。
- **账本防护**：在真实账本首条回复前加一个换行，默认被拒绝；加 `--diagnostic-ledger` 后可运行，报告标为诊断。
