# 跨模型对照：SmolVLM2-2.2B-Instruct 与 #11 Qwen2.5-VL-7B 逐题配对

本对照按 Issue #7 的请求开展（评论 7154672580，2026-11-25）。主问题是：Qwen2.5-VL-7B 出现的“动作答案相同、所引物理时间窗变化”是否也出现在另一个模型上。本对照为探索性实验。**本方案在阅读 #11 的 7B 结果之后制定**，对参照模型的结果并不盲。

**结论：按冻结协议，本次运行回答不了这个问题。** SmolVLM2 对 66 条可运行提示均给出了首次回复，但在主解析和严格解析下完整有效的回复都是 0 条，从未引用证据代号。57 条是 JSON：选项字母有效，`event` 字段填的却是动作文字（如 `"event": "turn on stove"`），不是 E01/E02，因此记为无效字段。另外 9 条只回复了 `Answer: A`，没有 JSON，记为格式失败。两项干预下，SmolVLM2 的可比较题都是 0/24，两个模型的共同可比较题也是 0，SmolVLM2 的“动作稳定且换窗”无法定义。Qwen 一侧沿用 #11 的结果，不变。

[协议](protocol.md) · [锁定参数](model_lock.json) · [六图检查](six_image_check.json) · [六图诊断](six_image_diagnostics.json) · [原始回复](replies.jsonl) · [运行日志](attempts.jsonl) · [配对统计](statistics.json)

## 顺序与冻结

| 步骤 | 提交 | 内容 |
| --- | --- | --- |
| 1 | `2fac798` | 冻结协议与 `model_lock.json`。此前没有对任何研究输入生成 SmolVLM2 回复。 |
| 2 | `b61eb82` | 六图接收检查，使用合成纯色图，不含研究图片 |
| 3 | `f39677e` | 运行器与配对复算脚本 |
| 4 | 本提交 | 正式运行 66 条（UTC 06:48–06:51）、统计与诊断。运行后，复算脚本只增加了 JSON 字段状态的描述性计数，配对规则和状态判定均未改。 |

冻结前只做过一项不加载模型的测量：用锁定的分词器和图片处理器，把 66 条冻结提示文本配上六张合成的 448 × 448 图做分词，用来确定图片处理设置（见下表）。

## 冻结的运行设置

| 项目 | 记录 |
| --- | --- |
| 模型 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`，修订 `be92141c8d0ceb98596a923c308ac2bbb4a7c466`，Apache-2.0。加载前核对 14 个文件的 SHA-256。 |
| 精度 | 官方权重以 float32 存储，未量化加载后整体转为 bfloat16（模型卡示例所用精度）。日志确认所有浮点参数均为 bfloat16。 |
| 聊天模板 | 原生 `chat_template.json`，未改动。用户消息依冻结顺序放六个图片项，后接冻结的提示文本，加生成提示，无系统消息和历史。每次调用都核对渲染结果与直接 Jinja 渲染逐字一致。 |
| 图片处理 | 原生 `SmolVLMImageProcessor` 配置，只覆盖 `do_image_splitting=false`。每帧缩放为一块 384 × 384，对应 81 个图片 token，每次调用共 486 个。若保留默认切块，每帧为 17 块，提示长度为 8769–8791 token，超过文本模型的 8192 位置上限。 |
| 解码 | 温度 0（贪心），不设其他采样或惩罚参数，最多生成 64 个 token，遇 `<end_of_utterance>` 停止。每次调用独立上下文，按 `calls.jsonl` 的 `run_order` 执行。 |
| 环境 | Apple M3 Pro，Metal `Device(gpu, 0)`；Python 3.12.0、MLX 0.32.4、mlx-vlm 0.8.0、transformers 5.23.0，与 #11 为同一环境；离线运行。 |
| 输入 | 扩展批冻结输入（039e834 / 9b528fc），经 `results_1124/frozen.py` 核对：72 个调用键，66 张帧按 SHA-256 核对。P11_104-7 的 6 个键沿用冻结时的取帧缺失，不运行。 |
| 解析 | 与 #11 相同的 `infer_mlx.py`（87ee601，`4ebbd521…`）。主解析为 `matched_seconds`，严格解析作敏感性参考。 |
| 运行结果 | 生成 66、复用 0、失败 0、冻结缺失 6。加载耗时 2.189 秒，单次耗时 1.637–3.284 秒。提示长度 777–799 token，与处理器自行计数逐条一致（不一致 0 条）。生成 4–22 token，没有回复达到 64 上限。MLX 峰值内存 5.45 GB。 |

## 六图接收检查（非研究图片）

六张 448 × 448 纯色图依次为红、绿、蓝、黄、黑、白。**结构检查通过**：6 个图片块、486 个图片 token，生成时的提示 token 数与处理器计数一致（542）。行为上，模型列出的颜色为 `Yellow, Green, Blue, Yellow, Green, Yellow`，6 个中只对 2 个（第 2、3 张）。

检查后补做了诊断，不改协议，结果见 `six_image_diagnostics.json`：

- 六图调用中，每张图的 81 个输入特征与该图单独输入时一致（最大差 0.016–0.063，为 bfloat16 舍入），与其他图差 12.6 以上。图片按顺序送到了对应位置。
- 每张图单独提问，6 张全部答对。两张图（红、蓝）分别答对；三张图（黄、黑、绿）把第 2、3 张答反；六张图问第 3 张、第 5 张，都答成 yellow。

据此，输入按协议送达模型。但在这项检查中，该模型按编号区分多张图的能力弱，三张图以上就会出错。凡依赖“image n”编号的行为，解读时都应考虑这一点。

## 回复状态（每条件 24 个计划键）

状态互斥，依次判定：技术缺失、格式失败、代号与秒数冲突、无效字段、双弃答、只弃答动作、只弃答证据、完整。两种解析下 SmolVLM2 的结果相同。

| 模型 | 条件 | 实收／计划 | 完整 | 无效字段 | 格式失败 | 弃答（null） | 技术缺失 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2 | 原顺序 | 22/24 | 0 | 18 | 4 | 0 | 2 |
| SmolVLM2 | 仅换选项 | 22/24 | 0 | 21 | 1 | 0 | 2 |
| SmolVLM2 | 仅换证据代号 | 22/24 | 0 | 18 | 4 | 0 | 2 |
| Qwen（#11，主解析） | 原顺序 | 22/24 | 15 | 0 | 0 | 7 | 2 |
| Qwen（#11，主解析） | 仅换选项 | 22/24 | 15 | 0 | 0 | 7 | 2 |
| Qwen（#11，主解析） | 仅换证据代号 | 22/24 | 14 | 0 | 0 | 8 | 2 |

- **拒答**：SmolVLM2 没有用协议允许的 null 弃答（0/66），也没有自由文本拒答。
- **格式失败**：9 条，原文都是 ` Answer: A`，逐条列在 `statistics.json` 的 `format_error_replies` 中。涉及 P21_01 两问的原顺序与换代号条件、P12_101 第一问三个条件，以及 P12_101 第二问的原顺序与换代号条件。
- **无效字段**：57 条，均为答案有效、事件无效。`event` 字段内容都是所选选项的动作文字（忽略大小写和句末句点后 57/57 一致），不是证据代号，也不是时间。
- **Qwen 严格解析**另有 28 条无效字段，见 #11。

## 逐项干预的配对结果

| 解析 | 干预 | 计划题数 | Qwen 可比较 | Qwen 动作稳定且换窗 | Qwen 涉及视频 | SmolVLM2 可比较 | SmolVLM2 动作稳定且换窗 | 共同可比较 | 共同集上 Qwen／SmolVLM2 换窗 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 主解析 | 仅换选项 | 24 | 14 | 4 | 2（P21_01、P35_105） | 0 | 未定义（0/0） | 0 | 未定义 |
| 主解析 | 仅换证据代号 | 24 | 14 | 4 | 2（P06_102、P07_04） | 0 | 未定义（0/0） | 0 | 未定义 |
| 严格 | 仅换选项 | 24 | 10 | 2 | 2（P21_01、P35_105） | 0 | 未定义（0/0） | 0 | 未定义 |
| 严格 | 仅换证据代号 | 24 | 0 | 0 | 0 | 0 | 未定义（0/0） | 0 | 未定义 |

SmolVLM2 不可比较题的（原顺序, 干预）状态对：

- 仅换选项：技术缺失×技术缺失 2，无效字段×无效字段 18，格式失败×无效字段 3，格式失败×格式失败 1。
- 仅换证据代号：技术缺失×技术缺失 2，无效字段×无效字段 18，格式失败×格式失败 4。

24 题的逐题配对（两模型的状态、动作、物理窗和四格归类）见 `statistics.json` 中 `rules.<解析>.contrasts.<干预>.questions`。

## 解读边界

- 本结果只说明：在相同文本、同一组帧、同一解析器下，SmolVLM2-2.2B 没有按要求输出证据代号，因此无法检验它是否会出现“动作稳定且换窗”。它不说明该模型会或不会出现这种模式。
- 若改提示、加格式约束或放宽解析，就不再是冻结的“同一文本、同一解析器”对照，需要另立协议。本次没有做这些尝试。
- 不报告正确率（没有人工参考答案），不做显著性检验，不把差异归因于模型大小。两个模型还在量化（4-bit 对 bfloat16）、每帧图片 token（256 对 81）、实际分辨率、分词器、模板和训练数据上不同。
- 同一片的两问共享帧、选项和候选窗，彼此不独立。
- Qwen 的原始回复与 #11 存档逐字节相同（SHA-256 `b87936a0…6aaa`），没有重新生成或改动。

## 复跑

离线复算只需 Git、Python 3.11 或更高版本和已归档的文本，不需媒体、GPU 或模型。脚本会核对协议文件、冻结输入、Qwen 账本的 SHA-256，从 `87ee601` 读取解析器并核对指纹，对两个模型的原文重放两种解析，并逐字段核对 SmolVLM2 账本中的严格解码结果。

```sh
python3 av-occurrence-study/run/extension_cross_model/summarize_cross_model.py \
  --output /tmp/cross-model-replay.json
cmp av-occurrence-study/run/extension_cross_model/statistics.json /tmp/cross-model-replay.json
python3 -m unittest discover -s av-occurrence-study/run/extension_cross_model -p 'test_*.py'
```

本机用 Python 3.11.7 和 3.12.0 各复算一次，输出与存档逐字节相同。

本机重新生成需要按[冻结输入说明](../participant_extension/frozen_inputs/README.md)重建 66 张帧，并下载锁定修订的检查点（约 9 GB）。脚本在加载模型前完成全部指纹核对。已有原文的键直接复用，不会重新生成。

```sh
hf download HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --revision be92141c8d0ceb98596a923c308ac2bbb4a7c466 --local-dir /path/to/smolvlm2-be92141
export HF_HUB_OFFLINE=1
PY=/path/to/mlx-env/bin/python   # Python 3.12.0, MLX 0.32.4, mlx-vlm 0.8.0, transformers 5.23.0
$PY av-occurrence-study/run/extension_cross_model/check_six_images.py \
  --model /path/to/smolvlm2-be92141 --images-dir /tmp/six-colour --output /tmp/six_image_check.json
$PY av-occurrence-study/run/extension_cross_model/diagnose_six_images.py \
  --model /path/to/smolvlm2-be92141 --images-dir /tmp/six-colour --output /tmp/six_image_diagnostics.json
$PY av-occurrence-study/run/extension_cross_model/run_cross_model.py \
  --model /path/to/smolvlm2-be92141 --frames-root /path/to/frozen-inputs
```

`--frames-root` 指向含 `frames/<题对>/` 的目录。要得到新的首次回复，请用 `--results-dir` 指向一个空目录，否则会复用本目录中的 66 条回复。
