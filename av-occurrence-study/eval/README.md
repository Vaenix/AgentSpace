# 重复声画事件评分

评分器以人工复核后冻结的题对为分母，输出中文运行状态、按题和按题对汇总、逐题明细，以及条件之间的成对变化。依赖 Python 3.10 或更高版本的标准库。

## 从双人复核冻结题单

1. 两位复核人用盲核页独立记录，分别下载 `av-review-A.json` 和 `av-review-B.json`。将两份文件放在研究目录下。复核时对照原视频，用原片时间填写发声和随后动作的起止秒数，并留下听到什么、看到什么或排除理由。
2. 从研究目录运行下面的第一条命令，生成仲裁表。用 Numbers 或 Excel 打开表格，保留全部题对、现有列标题和材料指纹。两个复核列保留各自的秒数、描述、结论和疑问。逐对填 `纳入`、`排除` 或 `待定`，并写下理由，包括任何分歧与裁决依据。
3. 纳入时，用原片确认是两次独立同类发声、顺序正确、两次之后各有不同动作，且作答同时需要声音和画面。填写确认后的声音类别、两段发声及动作区间，以及两条动作描述和两条同场景干扰动作。区间按 `12.3~13.4` 填写原片秒数。保留文字和映射意义一致的规范动作表述，选项避开透露先后的措辞。视觉对照可另留题清单。将文件保存为 UTF-8、制表符分隔的 `adjudication.tsv`。
4. 用真实仲裁人代号运行第二条命令，输出至一个新目录。工具用两份原始复核记录及候选队列的哈希校验仲裁表，生成纳入题对的 `gold.jsonl`、仅有题目和选项的 `questions.jsonl`，以及记录材料版本、人工决策与分歧理由的 `freeze.json`。把三个文件与原始复核记录一并归档。保留至少一对经确认的样本才能冻结本批。

```sh
python3 eval/freeze_reviews.py prepare --queue epic/review_queue.jsonl \
  --review-a av-review-A.json --review-b av-review-B.json --output adjudication.tsv
python3 eval/freeze_reviews.py freeze --queue epic/review_queue.jsonl \
  --review-a av-review-A.json --review-b av-review-B.json \
  --adjudication adjudication.tsv --adjudicator R03 --output frozen-v1
```

同对题目共用四个动作及同一顺序，正确位置跨题对轮换。选项 ID 由规范动作文本生成，与选项字母无关。组装推理输入时另附该条件允许的媒体内容，保留同一题号及选项映射。评分端使用 `frozen-v1/gold.jsonl`。变更规范动作、题单或标注时使用新的冻结版本。

## 输入

两个输入均为 UTF-8 JSON Lines 文件，每行一个 JSON 对象。以下为构造的格式示例。

金标每行一题。一个 `pair_id` 包含同一 `video_id` 的两题，`occurrence` 分别为 `1` 和 `2`，题号和正确答案各异。`interval_s` 为原视频上主要发声锚点的开始与结束秒数，两次发声按先后排列，区间互不重叠。将已完成人工复核的题标为 `verified`。

```jsonl
{"pair_id":"demo_01","video_id":"demo_video","qid":"demo_q1","occurrence":1,"answer_id":"close_oven","interval_s":[13.0,14.5],"review_status":"verified"}
{"pair_id":"demo_01","video_id":"demo_video","qid":"demo_q2","occurrence":2,"answer_id":"carry_plate","interval_s":[53.5,55.0],"review_status":"verified"}
```

`answer_id` 是跨条件通用的答案语义编号。选项顺序可变，生成预测文件时需从当次选项位置还原为对应编号。拒答填 `null`，不可定位时区间也填 `null`。额外日志字段可随预测保留。

```jsonl
{"condition":"chronological","pair_id":"demo_01","qid":"demo_q1","answer_id":"close_oven","interval_s":[13.0,14.5]}
{"condition":"chronological","pair_id":"demo_01","qid":"demo_q2","answer_id":null,"interval_s":null}
```

## 运行与解读

在研究目录 `av-occurrence-study/` 下运行。可重复声明 `--condition`，包括暂未产生预测的条件。省略参数时从预测文件读取条件名。尚无真实预测文件时可省略 `--predictions`。

```sh
python3 eval/score.py --gold /path/to/gold.jsonl \
  --predictions /path/to/predictions.jsonl \
  --condition chronological --condition av_grouped
```

- **按题计分** `answer_correct` 统计答案正确。`localization_correct` 统计预测区间与目标区间的交集长度除以并集长度（IoU）达到 0.5。若预测同时覆盖两次发声且二者 IoU 相等并达到 0.5，计为定位含糊。`answer_is_other_occurrence` 和 `localized_other_occurrence` 分别统计答成或定位成同对的另一次。
- **按题对计分** `joint_pair_success` 要求两题均回答正确且定位正确。`answer_swap` 与 `localization_swap` 分别统计两题答案互换和证据互换，两类都发生记为 `full_swap`。
- **分母与运行状态** 题目指标与题对指标分别用全部金标题和全部金标题对为分母。部分预测缺失时，状态为 `部分完成`，缺失题计入分母。输入没有金标题对或该组没有预测记录时，状态为 `未跑`，指标为 `null`。完整提交的拒答也计入分母。
- **成对变化** 两组提交齐全时，报告只在其中一组成功的题对数和两组联合成功率的百分点差。输出保留显式声明的条件顺序，省略声明时按条件首次出现的顺序比较。条件变化为描述性统计，先导批次重点看逐题证据和错误类型。
