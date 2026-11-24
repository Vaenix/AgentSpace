# 本地推理与媒体处理

回答器固定为 Qwen2.5-VL-7B-Instruct 的 MLX 社区 4 位版。实验输入是 448 × 448 的 RGB 帧与文字。声音描述由源标注或外部工具准备。推理阶段由 Qwen 读取文字描述，声学模型的选型和原始音轨编码属于更早的数据处理步骤。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [MLX 权重固定版本](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit/tree/fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b)

## 条件

| 参数 | 提供给模型的内容 |
| --- | --- |
| `blind` | 仅题目和选项 |
| `visual` | 同一批候选时间区间的帧与时间戳 |
| `flat` | 按时间排列全部帧条目，然后按时间排列全部声音描述条目 |
| `paired` | 每个候选片段内交替排列对应的帧和声音描述 |

`flat` 和 `paired` 只调整文字条目顺序，帧顺序、内容、帧数、时间戳、描述内容和选项相同。记录实际提示 token 数与时延。`visual` 用于度量声音描述的影响，所有已提供候选的条件都知道其区间。若需评估纯视觉检索或人工标注的正确片段，分别从相应检索或干预程序制作候选清单，并在实验记录中注明其来源。

## 输入和输出

每一行为一个独立问题，使用以下字段。两个问题属于同一个 `pair_id`，`qid` 独立。来自同一题对的选项共用稳定 ID 和顺序，正式批次的选项顺序须在运行前按视频划分固定随机化。

```json
{
  "pair_id": "example-pair",
  "qid": "example-q1",
  "question": "What does the person do just after the first bell sound?",
  "options": {"option-x": "Picks up a cup.", "option-y": "Opens a drawer."},
  "segments": [
    {
      "interval_s": [10.2, 10.7],
      "sound_description": "A short bell rings.",
      "frames": [{"time_s": 11.2, "path": "images/e0.png"}]
    },
    {
      "interval_s": [82.1, 82.8],
      "sound_description": "A short bell rings.",
      "frames": [{"time_s": 83.0, "path": "images/e1.png"}]
    }
  ]
}
```

区间采用原视频时间，按时间排序且不重叠。图像路径可相对于清单存放。帧时间严格递增，可延伸至声音后的动作窗口。模型通过选项展示字母和事件代号作答，文件路径、内部题号、真实答案 ID 和审核状态不出现在提示中。描述字段保留声学证据，人工排查是否提前泄漏了视觉动作答案。

输出每行记录条件、`pair_id`、`qid`、预测的稳定 `answer_id`、选中候选的 `interval_s`、原始回复、解析状态、帧数、提示及生成 token 数、模型位置、加载与逐题耗时，以及 MLX 记录的峰值内存。无法映射的字段写为 `null`。评价按独立盲审核的金标计算。

## 运行

在 Apple Silicon Mac 使用独立 Python 环境。本轮版本为 MLX-VLM 0.8.0、MLX 0.32.4。模型快照约 5.65 GB。

```sh
python3 -m venv /tmp/av-occurrence-mlx-venv
/tmp/av-occurrence-mlx-venv/bin/python -m pip install 'mlx-vlm==0.8.0' 'mlx==0.32.4'
/tmp/av-occurrence-mlx-venv/bin/hf download mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --revision fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b --local-dir /tmp/av-occurrence-qwen25-vl-7b-ready
```

已有候选视频时，可用 FFmpeg 定帧，再对同一份冻结清单逐条运行。

```sh
python3 av-occurrence-study/run/extract_frame.py review-window.mp4 2.35 frame.png
/tmp/av-occurrence-mlx-venv/bin/python av-occurrence-study/run/infer_mlx.py input.jsonl \
  --model /tmp/av-occurrence-qwen25-vl-7b-ready \
  --condition blind --condition visual --condition flat --condition paired \
  --output predictions.jsonl
```

当前设备的受限命令环境无法发现 Metal。获得本机执行授权后，实际运行检测通过，设备返回 `Device(gpu, 0)`。任何输出文件只新建一次。失败保留已经完成的逐题结果，重跑使用新的输出路径，由批次日志记录差异。

## EPIC 条件式冒烟

`prepare_epic_smoke.py` 从完成两个短窗的候选中获取一对，从标注时间抽取每窗 3 帧，用两个来源动作提示组成选项。它用该视频的两个**已提供候选**提问，以便检查真实样本推理、答案和片段代号的映射。候选窗与声音类别沿用公开标注，得分需要另行完成声画复核与冻结题单。

```sh
python3 av-occurrence-study/run/prepare_epic_smoke.py \
  --queue av-occurrence-study/epic/review_queue.jsonl \
  --media-manifest av-occurrence-study/epic/available_media.jsonl \
  --output-dir /tmp/av-occurrence-epic-smoke --pair P22_03-4
```

## 2026-11-19 的实际运行

本机使用 Metal GPU 成功载入 Qwen2.5-VL-7B-Instruct 4 位快照，固定修订 `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`。两个发布方权重分片的 SHA-256 均核对过。进程离线运行，温度为 0，最多生成 64 token。

公开视频 `P22_03` 的开发候选对 `P22_03-4` 提供两段短窗。声音区间沿用源标注的 53.290–53.847 秒与 377.226–384.082 秒。模型分别收到 53.5685、55.465、56.295 秒，以及 380.654、386.7425、387.8075 秒的画面。候选问题只改 first/second。初始选项 A 是打开水龙头，B 是放下杯子。两次开发推理均沿用源标注生成的声音描述。首窗关键动作在运动模糊画面中有歧义，环境声及全片真实序号尚待复核。

| 条件 | 问第一次之后 | 问第二次之后 |
| --- | --- | --- |
| 仅问题和选项 | 不选择动作或片段 | 不选择动作或片段 |
| 平排证据 | B，放下杯子，E02 | B，放下杯子，E02 |
| 显式配对 | B，放下杯子，E02 | B，放下杯子，E02 |

两组图文输入各为 6 张图像和 1836 个提示 token。两题四次图文推理耗时 7.706–8.907 秒，MLX 记录峰值内存约 6.57 GB，模型加载耗时 2.628 秒。原始回复及其到事件区间的映射见 `epic_smoke_predictions.jsonl`。这是给定候选短窗的链路验证，本次候选没有冻结金标。结果未计入研究指标，也未用于调整输入条件。

随后只反转 A/B 排列，同素材重跑平排和显式配对。核对两题的帧、时间、选项字词、稳定动作 ID 及其他提示都与原运行一致。四条输出均为「A，放下杯子，E01」。语义动作始终是放下杯子，所选候选却从 E02 变为 E01，说明在这次开发试跑中，单独改变选项次序就改变了定位代号。该现象仍需在独立、人工复核的题对上验证，无法由这一对题判断普遍性、机制或准确率。反转记录见 `epic_smoke_reversed_options.jsonl`。
