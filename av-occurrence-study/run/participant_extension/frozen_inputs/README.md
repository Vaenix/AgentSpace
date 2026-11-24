# 扩展批推理前冻结输入

2026-11-24 冻结，尚未生成任何回答。按[预设分析计划](https://github.com/Vaenix/AgentSpace/blob/f653ba6fceef7b16a733349041a33823834d0eca/av-occurrence-study/run/participant_extension_preflight/analysis_plan.md)（提交 `f653ba6f`）的 72 个调用键，固化 66 条可运行提示与 6 条技术缺失。入口是 [`../freeze_inputs.py`](../freeze_inputs.py)。

## 来源与规则

- **调用键**：`f653ba6f` 的 `planned_calls.csv` 原样复制为 [`../inputs/planned_calls.csv`](../inputs/planned_calls.csv)，SHA-256 `a69bf82c…`，与计划所记一致。脚本核对其顺序是否等于冻结队列（合并指纹 `faabf53d…`）× 第一、第二问 × 三个条件。
- **媒体**：使用 [`../media_prep/windows.jsonl`](../media_prep/windows.jsonl) 记录的 24 个裁片。运行前逐一核对本机裁片的 SHA-256。
- **提示与条件映射**：直接调用未改动的 `replication_1124/replicate.py` 的 freeze 步骤，即前两批的提示构造。其中 flat 证据条目按原片时间排列六帧，两窗使用同一来源声类文字。三个条件依次为原顺序、仅换选项、仅换片段代号。该步骤经 `prepare_epic_smoke.py` → `extract_frame.py` 从裁片重新取帧。脚本要求每帧及每份 `smoke-input.jsonl` 与 `windows.jsonl` 所记的 SHA-256 逐字节相同。
- **模型与解析**：沿用前批。模型为 MLX 4 位 Qwen2.5-VL-7B-Instruct，修订号 `fdcc572e…`，两块权重的指纹见 `provenance.json`。温度 0，最多 64 个生成 token，每次调用独立、无对话历史。主解析为 `infer_mlx.py` 的 `parse_reply(..., rule="matched_seconds")`，严格敏感性分析为 `decode(..., rule="strict")`。当前 `infer_mlx.py` 的 SHA-256 `4ebbd521…` 与协议锁定的 PR #2 版本相同，脚本会核对。

## 结果

| 项目 | 数量 |
| --- | --- |
| 计划调用键 | 72（12 名参与者 × 2 问 × 3 条件） |
| 可运行提示 | **66**（11 片、22 问） |
| 技术缺失 | **6**（P11_104-7 的全部调用键，入选顺序 1，原排名保留） |
| 已冻结帧 | 66 张，均为 448×448 PNG |

**缺失原因**：P11_104-7 第 2 窗的动作 75% 帧请求裁片内 9.5255 秒，小于视频轨时长 9.6 秒，所以不触发窗内替代。但 5 fps 裁片的最后一帧在 9.4 秒，此后没有帧可取。`extract_frame.py` 不输出图像，`prepare_epic_smoke.py` 随即停止，不写该对的问题输入。两问共用同一组六帧，所以 2 问 × 3 条件全部缺失。冻结时脚本对这对重新运行该程序，确认它仍在同一帧停止，且停止前写出的 5 帧与 `windows.jsonl` 一致。按复核决定，沿用冻结取帧规则，保留这名参与者，不补帧、不换样。可用的 5 帧及两个裁片的哈希记在 `provenance.json` 的 `unavailable` 中。

## 文件

| 文件 | 内容 |
| --- | --- |
| `calls.jsonl` | 72 行，按 `run_order` 排序，即计划的生成顺序。可运行行给出 `prompts.jsonl` 行号、提示文本 SHA-256、六帧 SHA-256、选项映射 `option_codes` 和片段代号映射 `segment_codes`。缺失行给出 `technical_missing`、所在步骤和原因。 |
| `prompts.jsonl` | 66 行，格式与前两批相同：提示全文、帧路径、语义动作 ID、按原片时间排序的物理窗、当条件的两种映射。 |
| `input_hashes.json` | 提示文件与 66 帧的 SHA-256、`replicate.py` 的 SHA-256、模型修订号与条件顺序。 |
| `frames/*/smoke-input.jsonl` | 11 份逐题输入（帧坐标与声类文字）。PNG 只留在本机，Git 忽略。 |
| `selected.jsonl`、`media.jsonl` | 可运行的 11 个队列行，以及 22 个裁片的窗口、实测时长与 SHA-256（路径只记文件名）。 |
| `provenance.json` | 计数、缺失明细、计划和队列指纹、`windows.jsonl` 哈希、模型、生成参数、解析器、各程序 SHA-256、各清单 SHA-256，以及工具版本。 |

## 复现

先用 `prepare_media.py` 在工作目录中准备裁片（见 [媒体准备](../media_prep/README.md)），再在仓库根目录运行。输出目录须尚不存在，`--check` 会将除 PNG 和本说明外的全部文件与本目录逐字节比较。

```sh
python3.12 av-occurrence-study/run/participant_extension/prepare_media.py \
  --work-dir /tmp/av-participant-media --output /tmp/av-participant-media/manifest \
  --check av-occurrence-study/run/participant_extension/media_prep
python3.12 av-occurrence-study/run/participant_extension/freeze_inputs.py \
  --work-dir /tmp/av-participant-media --output /tmp/av-participant-media/frozen-inputs \
  --check av-occurrence-study/run/participant_extension/frozen_inputs
python3.12 -m unittest discover -s av-occurrence-study/run/participant_extension -p 'test_freeze_inputs.py'
```

离线测试不需要媒体或模型。它核对 72 个键与计划一致，并核对指纹、帧哈希与 `windows.jsonl` 的对应。它还检查三个条件只改变各自的映射，并用锁定的两种解析器对每条提示的每个选项和代号做往返解码。

本步不加载模型。生成时应在第一次调用前核对 `input_hashes.json` 与权重指纹，再按 `calls.jsonl` 的顺序只运行 `runnable` 行，并保留 6 个缺失键。帧和裁片经 x264 编码，哈希依赖 FFmpeg 版本（本次 8.1.2）和机器。
