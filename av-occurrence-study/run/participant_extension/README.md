# 剩余参与者的下一轮候选顺序

冻结于 2026-11-24。本目录只冻结候选顺序，不含媒体、提示或模型输出。来源为 EPIC 官方公开训练标注及视频元数据，与 [11 月 24 日训练来源队列](../replication_1124/manifest/data_inputs.json) 使用同一组 CSV 和 SHA-256。排序没有读取任何回答，也没有检查媒体能否下载或解码。

## 规则

1. 用未改动的 [epic/screen.py](../../epic/screen.py) 在全部训练标注上生成候选对。声类、时长、间隔和动作配对条件均与既有筛选一致。
2. 排除以下来源中出现过的全部参与者及其所有视频：原验证队列 `epic/review_queue.jsonl`、第一批训练来源视频 `replication_1124/manifest/selected.jsonl`、第二批训练来源视频（冻结提交 `86c00449b09e2540c0d2c8f52b197055142da3e8` 的 `confirmation_1124/manifest/selected.jsonl`，原样复制于 `inputs/`，哈希一致），以及两份冒烟推理记录。两批合起来正好是六项训练来源队列的全部六个视频。
3. 在剩余候选对中逐项选择。每步只看本批尚未入选的参与者，先取本批该声类已选次数最少者（从零计），再按 `pair_id` 的 SHA-256 升序。每名参与者至多一对，上限 12 人。

脚本在建队前会核对 5 个 CSV、`screen.py` 和全部排除来源的 SHA-256。它还会用同一规则重算 11 月 24 日的六项训练队列，并逐字节对比，不符即中止。

## 人数与排除核对

| 项目 | 数量 |
| --- | --- |
| 训练声音标注中的参与者 | 32 |
| 有候选对的参与者 | 27（119 对，84 个视频） |
| 排除的参与者 | 15（92 对）：验证队列 9 人，两批训练来源视频共 6 人 |
| 符合条件的参与者 | **12**（27 对，24 个视频） |
| 入选 | **12 人，每人 1 对** |

其余 5 名训练参与者没有满足筛选条件的候选对。符合条件的参与者恰好 12 人，所以上限没有截掉任何人。入选者与 15 名已排除参与者没有交集。另在仓库所有远程分支中检索视频编号，出现过的参与者正好是这 15 人。逐人、逐对的决定和原因见 `manifest/exclusion_audit.json`。

## 冻结队列

| 入选顺序 | 题对 | 参与者 | 原片长（秒） | 声类 | 第一次声音（秒）→ 后续动作 | 第二次声音（秒）→ 后续动作 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | P11_104-7 | P11 | 861.7 | footstep | 713.868–715.659 → put vegetables in jar | 746.157–746.852 → rinse hands |
| 2 | P06_102-5 | P06 | 733.2 | water | 260.097–274.679 → close kettle | 333.326–334.514 → mix dough |
| 3 | P01_05-4 | P01 | 1272.0 | rustle | 197.901–199.088 → close drawer | 439.634–456.992 → throw plastic into bin |
| 4 | P13_08-2 | P13 | 2465.1 | cut / chop | 127.070–127.716 → wash plate | 149.772–150.772 → pick up onion |
| 5 | P35_105-5 | P35 | 2148.3 | water | 121.005–122.967 → put down container | 168.454–183.253 → clean counter top |
| 6 | P21_01-7 | P21 | 692.9 | footstep | 1.315–2.648 → opening the fridge | 128.307–129.727 → taking out groceries |
| 7 | P12_101-5 | P12 | 2045.9 | water | 5.381–17.070 → open fridge | 68.437–71.049 → pick up trash from stove |
| 8 | P28_104-7 | P28 | 719.3 | footstep | 392.133–393.608 → adjust temperature | 538.051–539.509 → open lid |
| 9 | P16_01-5 | P16 | 1243.1 | water | 4.846–15.086 → turn on the cooker | 27.477–33.190 → take out the knife |
| 10 | P07_04-7 | P07 | 825.2 | footstep | 0.981–3.961 → wash forks | 109.714–111.367 → turn on stove |
| 11 | P26_119-5 | P26 | 694.7 | water | 96.378–112.978 → pick up washing up liquid | 215.295–219.274 → put down onion |
| 12 | P09_103-5 | P09 | 743.6 | water | 0.000–4.252 → turn on stove | 150.665–155.540 → add condiments to chicken |

`manifest/queue.jsonl` 逐行保存题对 ID、参与者、视频、原片时长、声类、两次声音与动作的标注 ID 和原片秒数、动作文字、建议问句、官方媒体地址、入选顺序及每步的排序键。各项仍是 `annotation_candidate`，没有经过人工核验。

声类分布为流水 6、脚步 4、窸窣 1、切剁 1。三对倾倒声候选属于 P01 和 P06，这两人已先以其他声类入选，所以本批没有倾倒声。P09_103 的第一次声音从 0 秒开始，P13_08、P35_105 等的后续动作持续较长；取帧或裁窗时需按既定规则处理，本队列不做替换。

## 复建

在仓库根目录运行。缓存目录不存在时用 curl 从官方地址下载 5 个 CSV 并核对哈希；输出目录须尚不存在。`--check` 会把全部输出与本目录 `manifest/` 逐字节比对。

```sh
python3.12 av-occurrence-study/run/participant_extension/build_queue.py \
  --cache /tmp/epic-train-csv \
  --output /tmp/participant-extension-rebuild/manifest --check
```

缓存默认放在已被 Git 忽略的 `av-occurrence-study/epic/annotations/official_train/`。`manifest/fingerprint.json` 记录三个输出文件各自的 SHA-256、合并指纹及入选顺序。合并指纹是按文件名顺序拼接三行 `<sha256>  <文件名>` 后的 SHA-256。

| 指纹 | SHA-256 |
| --- | --- |
| `queue.jsonl` | `363bc1dac211ae71ba6d6a5259401af35f130c24a2bc58b3e02af0f845e1d2fe` |
| 合并指纹 | `faabf53d91a888e59844ff5b22565635aad30ddc040809328aa6fee82c5f7084` |

`source_hashes.json` 含本脚本自身的 SHA-256，修改脚本后合并指纹会随之改变，需重新冻结。
