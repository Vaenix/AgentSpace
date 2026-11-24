# Real-frame group swap in the participant extension

This exploratory experiment reuses the 24 original-order questions. The [protocol](protocol.md) and [ordered manifest](planned.jsonl) were published at `d1bd1a73f714d4ef4c4884134f5c64d178673b07` before generating any swapped replies. Both groups of three frames exchange slots within each recording. The displayed timestamps, interval labels and question remain unchanged. Under the intervention, a citation to the first displayed interval points at images from the second source group.

## Results

Both conditions returned text on 22 of 24 planned questions. The 22 prompt-token counts matched across conditions. The generator used the same checkpoint file digests as the original extension. Under the primary matched-seconds parser, 9 pairs gave complete replies in both conditions, 6 only in the original and 7 only after swapping. Four of the nine comparable pairs changed both action and displayed interval (the two questions each in P09_103 and P21_01), four changed only the action, and one changed neither. None changed only the interval. Thus the primary changed-interval fraction was 4/9 across five comparable recordings. Only one comparable question retained its action; it kept its displayed interval, so the secondary same-image-group count was 0/1. Strict parsing gave a changed-interval fraction of 1/6 and a same-image-group fraction of 0/1. P11_104 retained its two missing-input questions.

## Replay from stored text

Use a full Git clone containing the extension history and Python 3.11 or newer. Run from this directory and choose a new output filename.

```sh
python3 summarize_swap.py --output /tmp/frame-swap-report.json
cmp /tmp/frame-swap-report.json statistics.json
python3 -m unittest -v test_summarize_swap.py
```

The [primary results](statistics.json) include both parser rules, the full 24-question denominator, the disposition of each condition, four paired outcomes, recording-level summaries and question-level statuses. The reports read the [first replies](replies.jsonl) and the [attempts ledger](attempts.jsonl) alongside the archived original condition. Source actions and intervals have not been verified by human reviewers.

## Reconstruct the inference setup

The original execution used Python 3.12.0, MLX 0.32.4, MLX-VLM 0.8.0 and 4-bit checkpoint `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`. A separate reproduction requires the checkpoint and the frozen extension frames. The frames root must contain the original `frames/` directory. All runtime inputs are local. Choose a fresh directory for any separate generation and retain the archived first replies above as the reference.

```sh
python3 run_swap.py --frames-root /path/to/extension/inputs --model /path/to/checkpoint --output-dir /path/to/new-output
```
