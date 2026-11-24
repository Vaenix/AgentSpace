"""Load and verify the extension inputs exactly as frozen in commit 039e834.

Only reads text files. Every frozen file is compared with the SHA-256 it had in
the reviewed freeze commit, so generation and replay refuse altered inputs.
"""
import csv
from hashlib import sha256
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parent
FROZEN = EXTENSION / "frozen_inputs"
RUN = EXTENSION.parent

# The reviewer accepted 039e834; PR #10 later rebased the same patch to 9b528fc.
# Both commits hold byte-identical frozen_inputs/ and inputs/planned_calls.csv.
APPROVED_FREEZE_COMMIT = "039e8347bbafb9a5efa4f476366a73cd0513f29c"
REBASED_FREEZE_COMMIT = "9b528fc2f03c7e793e3dab34c9e52fbf8160d0cb"
PLAN_COMMIT = "f653ba6fceef7b16a733349041a33823834d0eca"
FROZEN_SHA256 = {
    "frozen_inputs/calls.jsonl": "d8efdc41e7d8c835c09e4c320829d83679d0fef825577b8c6305532a4d4c9ace",
    "frozen_inputs/prompts.jsonl": "6dd56b0eadaaf1f792d2c639a3a35226b2f6b900b5aa65e052d6c4baea158d8c",
    "frozen_inputs/input_hashes.json": "7c6e048df5ceac1f4b716d050fb9f914e44b347224fce2123524c07c82b64e47",
    "frozen_inputs/provenance.json": "6df5554c22812c7a48cc2bb29affdbe9a31fd8976c8f4ff36bcf1609796955dd",
    "frozen_inputs/selected.jsonl": "2ee5dc07a64fedb35f5c1984ccc090c494991ff175cba3fc2d58d0e70a472cde",
    "inputs/planned_calls.csv": "a69bf82c89961a310096750d2b78f47efb27e1d0042ad9d6ce1d0e15f6202139",
}
PARSER_SHA256 = "4ebbd521d030f6cfdf9f4b4bead886bd53d076ffb6893cc542202680219fc027"
MODEL_REVISION = "fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b"
SHARD_SHA256 = {
    "model-00001-of-00002.safetensors": "f80072ac0e82a2ace0740eab995f6f9ae2623adb55e6906e21c2f8edca537803",
    "model-00002-of-00002.safetensors": "016bd9d3596e9b824f5a527875e5830414552c950eee794e98efb44c1cd8dc6a",
}
TEMPERATURE = 0.0
MAX_TOKENS = 64
CONDITIONS = ("original", "options_reversed", "segment_labels_reversed")
PLANNED_CALLS = 72
PLANNED_RECORDINGS = 12


def file_sha256(path):
    total = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            total.update(block)
    return total.hexdigest()


def read_jsonl(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def key_of(row):
    return row["qid"], row["condition"]


def load():
    """Return the 72 frozen calls in run order, with the prompt row attached to runnable ones."""
    raw = {}
    for relative, expected in FROZEN_SHA256.items():
        data = (EXTENSION / relative).read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"{relative} differs from the frozen commit {APPROVED_FREEZE_COMMIT[:7]}")
        raw[relative] = data
    calls = read_jsonl(raw["frozen_inputs/calls.jsonl"])
    prompts = read_jsonl(raw["frozen_inputs/prompts.jsonl"])
    hashes = json.loads(raw["frozen_inputs/input_hashes.json"])
    provenance = json.loads(raw["frozen_inputs/provenance.json"])
    selected = read_jsonl(raw["frozen_inputs/selected.jsonl"])
    planned = list(csv.DictReader(io.StringIO(raw["inputs/planned_calls.csv"].decode("utf-8"))))

    if sha256(raw["frozen_inputs/prompts.jsonl"]).hexdigest() != hashes["prompts_sha256"]:
        raise ValueError("prompts.jsonl does not match input_hashes.json")
    if hashes["checkpoint_revision"] != MODEL_REVISION or tuple(hashes["condition_order"]) != CONDITIONS:
        raise ValueError("Frozen checkpoint revision or condition order differs")
    if provenance["checkpoint"]["expected_shard_sha256"] != SHARD_SHA256:
        raise ValueError("Frozen shard fingerprints differ")
    generation = provenance["generation"]
    if generation["temperature"] != TEMPERATURE or generation["max_tokens"] != MAX_TOKENS:
        raise ValueError("Frozen generation parameters differ")
    if provenance["parser"]["sha256"] != PARSER_SHA256:
        raise ValueError("Frozen parser fingerprint differs")

    if len(calls) != PLANNED_CALLS or [row["run_order"] for row in calls] != list(range(1, PLANNED_CALLS + 1)):
        raise ValueError("calls.jsonl must list run orders 1..72")
    fields = ("run_order", "extension_rank", "participant_id", "video_id", "pair_id", "qid", "ordinal", "condition")
    if [tuple(str(call[name]) for name in fields) for call in calls] != [tuple(row[name] for name in fields) for row in planned]:
        raise ValueError("calls.jsonl does not follow planned_calls.csv")
    if len({key_of(call) for call in calls}) != PLANNED_CALLS:
        raise ValueError("Duplicate call key")

    used_lines = []
    for call in calls:
        if call["status"] == "technical_missing":
            if not call.get("reason") or not call.get("stage") or call.get("prompt_line") is not None:
                raise ValueError(f"{key_of(call)}: frozen technical_missing needs a stage and reason")
            call["prompt"] = None
            continue
        if call["status"] != "runnable":
            raise ValueError(f"{key_of(call)}: unknown frozen status {call['status']}")
        prompt = prompts[call["prompt_line"] - 1]
        used_lines.append(call["prompt_line"])
        for name in ("qid", "condition", "pair_id", "video_id", "ordinal", "option_codes", "segment_codes"):
            if prompt[name] != call[name]:
                raise ValueError(f"{key_of(call)}: prompt row disagrees on {name}")
        if sha256(prompt["prompt"].encode("utf-8")).hexdigest() != call["prompt_sha256"]:
            raise ValueError(f"{key_of(call)}: prompt text fingerprint differs")
        frame_hashes = [hashes["frame_sha256"][path] for path in prompt["frames"]]
        if frame_hashes != call["frames_sha256"] or len(frame_hashes) != 6:
            raise ValueError(f"{key_of(call)}: frame fingerprints differ between calls and input_hashes")
        call["prompt"] = prompt
    if sorted(used_lines) != list(range(1, len(prompts) + 1)):
        raise ValueError("Each frozen prompt must be used by exactly one call")
    recordings = list(dict.fromkeys(call["pair_id"] for call in calls))
    if len(recordings) != PLANNED_RECORDINGS:
        raise ValueError("Expected twelve recordings")
    if [row["pair_id"] for row in selected] != [pair for pair in recordings
                                               if any(c["pair_id"] == pair and c["status"] == "runnable" for c in calls)]:
        raise ValueError("selected.jsonl disagrees with the runnable recordings")
    return {
        "calls": calls,
        "frame_sha256": hashes["frame_sha256"],
        "recordings": recordings,
        "fingerprints": {relative: sha256(data).hexdigest() for relative, data in raw.items()},
    }
