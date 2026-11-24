# Same-host original-input repeat

The [frozen protocol](protocol.md) defines this exploratory repeat separately from the first replies used in the extension. Run offline replay from this directory inside a full clone with Python 3.11 or newer:

```sh
python3 summarize_repeat.py --output /tmp/original-repeat-report.json
cmp /tmp/original-repeat-report.json statistics.json
```

The 24 planned questions correspond to the original order only. The [repeat ledger](replies.jsonl) contains newly generated text; [attempts.jsonl](attempts.jsonl) records the full traversal of the unmodified three-condition runner. The latter includes skip entries for the other two conditions. These contribute no new model outputs.

## Observed agreement

All 22 returned strings, including whitespace, reproduced the archived first responses exactly. The token count agreed in every pair. For both parsers, all 22 pairs agreed on status, action and physical interval. On this repeat baseline, the published swapped-frame table was reproduced exactly: the displayed interval moved on four of nine complete primary pairs, or one of six with strict parsing. The two remaining original questions retained the frozen input failure. The 44 skip entries were not generated in this run. This is one additional model invocation per runnable original question.

## Separate reproduction

Use the original MLX environment and weights specified in the [extension results](../participant_extension/results_1124/README.md) and the original extension frame directory. Prepare a fresh results directory with only the two single-label reversal conditions in the skip ledger:

```sh
python3 - /path/to/new-run <<'PY'
from pathlib import Path
import json, sys
source = Path("../participant_extension/results_1124/replies.jsonl")
destination = Path(sys.argv[1])
destination.mkdir()
seed = [line for line in source.read_bytes().splitlines(keepends=True)
        if json.loads(line)["condition"] != "original"]
assert len(seed) == 44
(destination / "replies.jsonl").write_bytes(b"".join(seed))
PY
python3 ../participant_extension/results_1124/run_generation.py --model /path/to/checkpoint --frames-root /path/to/extension/inputs --results-dir /path/to/new-run
```

The runner skips the 44 seeded keys by identity. It does not pass their text to the model. Filter the new ledger to rows with `condition == "original"` to obtain the repeated original texts. The 22 surviving questions are generated once in the original queue order. Two more original keys retain the input failure from frame extraction.
