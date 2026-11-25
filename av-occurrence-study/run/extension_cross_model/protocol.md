# Cross-model control: SmolVLM2-2.2B-Instruct on the extension questions

Exploratory follow-up requested on Issue #7 (comment 7154672580, 2026-11-25). The question is whether the pattern observed with Qwen2.5-VL-7B — the same semantic action chosen while the cited physical window moves — also appears with another model on the same inputs.

## What was known before this plan

This plan was written after the Qwen2.5-VL-7B results of the extension batch (PR #11, commit 3df02a8e1b84d876ebdf8b54b9826d803860fcf1) had been read, including the per-question and per-recording tables. The open gray-frame (#13) and joint-reversal (#15) PRs were also known to exist. The comparison is therefore not blind to the reference model's outcome. Before this commit no SmolVLM2 reply to any research input had been generated. One pre-freeze measurement was made without the model: the pinned tokenizer and image processor tokenized the 66 frozen prompt texts with six synthetic 448 × 448 images to count prompt tokens (see Image processing). The model weights were not loaded for it.

## Model, precision and template

`model_lock.json` fixes every item below and is part of this commit.

- Checkpoint: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` at revision `be92141c8d0ceb98596a923c308ac2bbb4a7c466`, with the SHA-256 of all 14 files verified before loading.
- Precision: the published float32 weights are loaded unquantized and cast to bfloat16 before the first call, the precision of the model card's examples.
- Chat template: the revision's own `chat_template.json`, unmodified. One user message holds six image items in the frozen frame order, then the frozen prompt text. The generation prompt is added. No system message and no history. The runner checks that the rendered string equals a direct Jinja rendering of the stored template.
- Image processing: the revision's `SmolVLMImageProcessor` settings with a single override, `do_image_splitting = false`. Each 448 × 448 frame becomes one 384 × 384 tile of 81 image tokens, so each call has 486 image tokens. With the stored default (splitting on) each frame becomes 17 tiles and the measured prompts are 8769–8791 tokens, beyond the text model's 8192 positions. Without splitting they are 777–799 tokens. The model's own video setting also feeds frames at 384 without splitting.
- Decoding: temperature 0 (greedy), no other sampling or penalty options, at most 64 new tokens (the budget of the #11 run and of the model card examples). Generation stops at `<end_of_utterance>` or the budget. Every call has a fresh context.
- Runtime: the Python 3.12.0 / MLX 0.32.4 / mlx-vlm 0.8.0 environment of the #11 run, offline after download.

## Six-image reception check

Before the formal run, load the locked model and settings and give it six synthetic, non-research images: plain fields in six distinct named colours. The check passes when the processed input has 6 image blocks and 486 image tokens, and the prompt token count equals the processor's own count. Record the model's reply listing the colours in order as a behavioural note. A wrong colour list is reported but does not change the protocol. A structural failure stops the run.

## Inputs and formal run

Use the extension inputs frozen at 039e834 / 9b528fc, verified by `participant_extension/results_1124/frozen.py`: 72 call keys, 12 recordings × 2 questions × 3 conditions (original, options reversed, evidence codes reversed). Use the 66 runnable prompts exactly, with their 66 frames verified by SHA-256 before loading. The six P11_104-7 keys keep the technical-missing status recorded at freeze time and are not run. Run order follows `calls.jsonl`. Save the first returned text of each key unchanged, with token counts, timing and versions. A key with a stored reply is never regenerated. Log every attempt. Retry only keys without a reply, on identical inputs. A reply that reaches the 64-token budget is kept and flagged.

The Qwen2.5-VL-7B reference is the first reply per key in `participant_extension/results_1124/replies.jsonl` at 3df02a8 (SHA-256 `b87936a0…6aaa`). It is not regenerated or altered.

## Parsing and reply statuses

Decode both models' replies with the same parser source, `infer_mlx.py` at 87ee601 (SHA-256 `4ebbd521…`), and each prompt's own option and evidence maps. The primary rule is `matched_seconds`; `strict` is a sensitivity check, as in #11. Statuses are mutually exclusive, in this order: technical missing, format error (invalid JSON or keys), code–seconds conflict, invalid field, both fields null, answer null only, event null only, complete.

Abstentions (the prompt's explicit null) are reported apart from format failures. A free-text refusal with no JSON object is a format error under the parser; every format-error reply is listed verbatim so that refusals can be read directly, without a second automatic label. Generation failures count as technical missing, never as abstentions.

## Paired analysis

For each intervention — options reversed (the primary contrast in #11) and evidence codes reversed — the 24 planned questions are compared within each model, original against intervention. A question is comparable for a model when both of its replies are complete. Among comparable questions the four cells are semantic action stable/changed × physical window stable/changed. J, the event of interest, is action stable with physical window changed.

Report per intervention and parser:

1. Planned questions (24).
2. For each model: comparable count, the four cells, J and the number of distinct videos with at least one J.
3. Jointly comparable questions (comparable for both models). On this common set: each model's J, its videos with J, and the paired classification J in both / Qwen only / SmolVLM2 only / neither.
4. A 24-row per-question table with both models' statuses, actions, windows and cells, and the ordered status pairs of non-comparable questions per model.
5. Reply status distributions per model and condition (planned, received, each status), including verbatim format-error replies and budget-hit replies.

All results are descriptive counts with their denominators. Zero denominators are reported as undefined. No accuracy is measured (there are no human reference answers) and no significance test is used. The two questions of a recording share frames, options and windows, so they are not independent. Differences between the models are not attributed to model size. The models also differ in quantization (4-bit versus bfloat16), image tokens per frame (256 versus 81), image resolution seen, tokenizer, template and training data.
