#!/usr/bin/env python3
"""Load and verify the joint-reversal inputs frozen in commit af01c3d.

Reads text only; needs Python 3.11+ and no media, GPU or model. The two
protocol files (protocol.md, planned.jsonl) are compared with their SHA-256 at
the freeze commit. The extension's three archived conditions come from
results_1124 at 3df02a8: the frozen inputs (checked by results_1124/frozen.py)
and the first-reply ledgers, which are never regenerated.

For each of the 24 planned questions the joint prompt is built twice:
  A. the answer-reversed prompt with its evidence entries replaced by those of
     the code-reversed prompt;
  B. the code-reversed prompt with its options replaced by those of the
     answer-reversed prompt.
Both paths must give the same text, option map, evidence map, frame order and
physical windows, and that result must equal the planned.jsonl row byte for
byte. Options and evidence lines are also checked against their maps.

Run as a script to write the text-only input check report:
  python3 crossed_inputs.py --output input_check.json
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EXTENSION_RESULTS = RUN / "participant_extension" / "results_1124"
sys.path.insert(0, str(EXTENSION_RESULTS))

import frozen  # noqa: E402  (extension inputs frozen in 039e834)

PROTOCOL_COMMIT = "af01c3dd12d78ad42fa75c17bebfb02366a34bea"
ARCHIVE_COMMIT = "3df02a8e1b84d876ebdf8b54b9826d803860fcf1"
PROTOCOL_SHA256 = {
    "protocol.md": "b399f8f8e8035f39942bd269c49ff17d690dcbbed017103e7a85f6c84ef15fa6",
    "planned.jsonl": "e78f54bb21d898aa7c802a4a541f4066d2125532c760bce63d24f30b6aa1a533",
}
ARCHIVE_SHA256 = {
    "replies.jsonl": "b87936a0fd0d2b0d5d1f75f93f2d31dd81ad7fbc942fe2491d12ed21adc86aaa",
    "attempts.jsonl": "0bd2342c5ae8b04f41e46193b61c010d3c6b286eb3a4a730a23836235bff422b",
    "frozen.py": "d6df4a6c91bb60549e375b25f8864fa8281fece4b26b5b901e2fc4529757b67c",
    "summarize.py": "f5d7a6b5340da45189a651fa6223679456616e7163c635124581dc72c66fe00f",
}
CONDITION = "both_reversed"
ANSWER_REVERSED = "options_reversed"
CODES_REVERSED = "segment_labels_reversed"
CONDITIONS = ("original", ANSWER_REVERSED, CODES_REVERSED, CONDITION)
PLANNED_KEYS = 24
RUNNABLE_KEYS = 22
SLOTS = 6
IMAGE_SIZE = (448, 448)

OPTIONS_HEAD = "\n\nOptions:\n"
INSTRUCTION_HEAD = "\n\nReturn only a JSON object"
EVIDENCE_HEAD = "\n\nEvidence entries (image numbers follow the supplied image order):\n"
PLANNED_FIELDS = ("qid", "participant_id", "video_id", "extension_rank", "ordinal", "pair_id", "status",
                  "condition", "order")
RUNNABLE_FIELDS = ("prompt", "prompt_sha256", "option_codes", "segment_codes", "frames", "frames_sha256",
                   "physical_window_order", "semantic_actions")


def split_prompt(text):
    """Return (stem, options, instruction, evidence) with each marker occurring exactly once."""
    for marker in (OPTIONS_HEAD, INSTRUCTION_HEAD, EVIDENCE_HEAD):
        if text.count(marker) != 1:
            raise ValueError(f"Prompt must contain {marker.strip()!r} exactly once")
    stem, rest = text.split(OPTIONS_HEAD)
    options, rest = rest.split(INSTRUCTION_HEAD)
    instruction, evidence = rest.split(EVIDENCE_HEAD)
    if not (stem and options and instruction and evidence):
        raise ValueError("Prompt has an empty section")
    return stem, options, INSTRUCTION_HEAD + instruction, evidence


def join_prompt(stem, options, instruction, evidence):
    return stem + OPTIONS_HEAD + options + instruction + EVIDENCE_HEAD + evidence


def seconds(value):
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def check_maps(text, option_codes, segment_codes, semantic_actions):
    """Options lines must name the mapped actions in letter order; evidence lines must carry the mapped intervals."""
    _, options, _, evidence = split_prompt(text)
    expected = [f"{letter}. {semantic_actions[code]}" for letter, code in sorted(option_codes.items())]
    if options.split("\n") != expected:
        raise ValueError("Options lines disagree with the option map")
    lines = evidence.split("\n")
    if len(lines) != 2 * len(segment_codes):
        raise ValueError("Expected one visual and one sound line per evidence code")
    labels = []
    for line in lines:
        code, _, rest = line.partition(" ")
        interval = segment_codes[code]
        if not rest.startswith(f"{seconds(interval[0])}–{seconds(interval[1])}s | "):
            raise ValueError(f"Evidence line for {code} disagrees with the evidence map")
        labels.append(code)
    if sorted(labels[:len(segment_codes)]) != sorted(segment_codes) or labels[:len(segment_codes)] != labels[len(segment_codes):]:
        raise ValueError("Each evidence code must have one visual line and one sound line in the same order")
    return labels[:len(segment_codes)]


def construct(answer_row, codes_row):
    """Build the joint prompt along both paths and return the shared result."""
    a_stem, a_options, a_instruction, a_evidence = split_prompt(answer_row["prompt"])
    c_stem, c_options, c_instruction, c_evidence = split_prompt(codes_row["prompt"])
    if (a_stem, a_instruction) != (c_stem, c_instruction):
        raise ValueError("Answer-reversed and code-reversed prompts differ outside options and evidence")
    path_a = {  # answer-reversed prompt, evidence entries taken from the code-reversed prompt
        "prompt": join_prompt(a_stem, a_options, a_instruction, c_evidence),
        "option_codes": answer_row["option_codes"], "segment_codes": codes_row["segment_codes"],
        "frames": answer_row["frames"], "physical_window_order": answer_row["physical_window_order"],
        "semantic_actions": answer_row["semantic_actions"],
    }
    path_b = {  # code-reversed prompt, options taken from the answer-reversed prompt
        "prompt": join_prompt(c_stem, a_options, c_instruction, c_evidence),
        "option_codes": answer_row["option_codes"], "segment_codes": codes_row["segment_codes"],
        "frames": codes_row["frames"], "physical_window_order": codes_row["physical_window_order"],
        "semantic_actions": codes_row["semantic_actions"],
    }
    differing = sorted(name for name in path_a if path_a[name] != path_b[name])
    if differing:
        raise ValueError(f"The two construction paths differ in {', '.join(differing)}")
    check_maps(path_a["prompt"], path_a["option_codes"], path_a["segment_codes"], path_a["semantic_actions"])
    return path_a


def read_protocol():
    raw = {}
    for name, expected in PROTOCOL_SHA256.items():
        data = (HERE / name).read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"{name} differs from the frozen protocol commit {PROTOCOL_COMMIT[:7]}")
        raw[name] = data
    return raw


def read_archive():
    raw = {}
    for name, expected in ARCHIVE_SHA256.items():
        data = (EXTENSION_RESULTS / name).read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"results_1124/{name} differs from the archive commit {ARCHIVE_COMMIT[:7]}")
        raw[name] = data
    return raw


def load():
    """Return the 24 joint-reversal keys in planned order, the archive and every fingerprint used."""
    protocol = read_protocol()
    archive = read_archive()
    extension = frozen.load()
    planned = frozen.read_jsonl(protocol["planned.jsonl"])
    if len(planned) != PLANNED_KEYS or [row["order"] for row in planned] != list(range(1, PLANNED_KEYS + 1)):
        raise ValueError("planned.jsonl must list orders 1..24")
    calls = {frozen.key_of(call): call for call in extension["calls"]}
    queue = [call for call in extension["calls"] if call["condition"] == "original"]
    if len(queue) != PLANNED_KEYS:
        raise ValueError("Expected 24 frozen questions")

    keys, checks = [], []
    for row, base in zip(planned, queue):
        qid = row["qid"]
        if row["condition"] != CONDITION:
            raise ValueError(f"{qid}: planned condition must be {CONDITION}")
        for name in ("qid", "participant_id", "video_id", "extension_rank", "ordinal", "pair_id"):
            if row[name] != base[name]:
                raise ValueError(f"{qid}: planned {name} does not follow the frozen queue order")
        components = {condition: calls[qid, condition] for condition in CONDITIONS[:3]}
        statuses = {condition: call["status"] for condition, call in components.items()}
        key = {name: row[name] for name in PLANNED_FIELDS}
        key["run_order"] = row["order"]
        if row["status"] == "technical_missing":
            if set(statuses.values()) != {"technical_missing"}:
                raise ValueError(f"{qid}: a planned technical miss must be missing in all three archived conditions")
            if (row["stage"], row["reason"]) != (base["stage"], base["reason"]) or set(RUNNABLE_FIELDS) & set(row):
                raise ValueError(f"{qid}: planned technical miss differs from the frozen record")
            key.update(stage=row["stage"], reason=row["reason"], prompt=None)
            keys.append(key)
            checks.append({"order": row["order"], "qid": qid, "status": "technical_missing",
                           "stage": row["stage"], "archived_statuses": statuses})
            continue
        if row["status"] != "runnable" or set(statuses.values()) != {"runnable"}:
            raise ValueError(f"{qid}: planned runnable key needs three runnable archived conditions")
        prompts = {condition: call["prompt"] for condition, call in components.items()}
        original = prompts["original"]
        built = construct(prompts[ANSWER_REVERSED], prompts[CODES_REVERSED])
        built["prompt_sha256"] = sha256(built["prompt"].encode("utf-8")).hexdigest()
        built["frames_sha256"] = [extension["frame_sha256"][path] for path in built["frames"]]
        differing = sorted(name for name in RUNNABLE_FIELDS if row[name] != built[name])
        if differing:
            raise ValueError(f"{qid}: planned.jsonl differs from the constructed prompt in {', '.join(differing)}")
        if sorted(row) != sorted(set(PLANNED_FIELDS) | set(RUNNABLE_FIELDS)):
            raise ValueError(f"{qid}: unexpected planned fields")
        if len(built["frames"]) != SLOTS or any(call["frames_sha256"] != built["frames_sha256"]
                                                for call in components.values()):
            raise ValueError(f"{qid}: frames must be the same six archived images in all conditions")
        o_stem, o_options, o_instruction, o_evidence = split_prompt(original["prompt"])
        a_parts = split_prompt(prompts[ANSWER_REVERSED]["prompt"])
        c_parts = split_prompt(prompts[CODES_REVERSED]["prompt"])
        j_parts = split_prompt(built["prompt"])
        if (a_parts[3], c_parts[1]) != (o_evidence, o_options) or a_parts[1] == o_options or c_parts[3] == o_evidence:
            raise ValueError(f"{qid}: archived single-factor prompts do not each change exactly one block")
        if (j_parts[0], j_parts[2]) != (o_stem, o_instruction):
            raise ValueError(f"{qid}: joint prompt changes the question stem or instruction")
        for condition, prompt in prompts.items():
            check_maps(prompt["prompt"], prompt["option_codes"], prompt["segment_codes"], prompt["semantic_actions"])
            if (prompt["frames"], prompt["physical_window_order"], prompt["semantic_actions"]) != (
                    built["frames"], built["physical_window_order"], built["semantic_actions"]):
                raise ValueError(f"{qid}: {condition} differs in frames, windows or actions")
        key.update(built, prompt={name: built[name] for name in (
            "prompt", "frames", "physical_window_order", "semantic_actions", "option_codes", "segment_codes")})
        key["option_codes"], key["segment_codes"] = built["option_codes"], built["segment_codes"]
        keys.append(key)
        letters = {condition: {letter: prompt["semantic_actions"][code]
                               for letter, code in sorted(prompt["option_codes"].items())}
                   for condition, prompt in list(prompts.items()) + [(CONDITION, built)]}
        evidence_order = {condition: list(check_maps(prompt["prompt"], prompt["option_codes"], prompt["segment_codes"],
                                                     prompt["semantic_actions"]))
                          for condition, prompt in list(prompts.items()) + [(CONDITION, built)]}
        checks.append({
            "order": row["order"], "qid": qid, "status": "runnable",
            "prompt_sha256": built["prompt_sha256"],
            "archived_prompt_sha256": {condition: components[condition]["prompt_sha256"]
                                       for condition in CONDITIONS[:3]},
            "construction_paths_identical": True,
            "planned_row_identical": True,
            "options_by_letter": letters,
            "evidence_codes_in_window_order": evidence_order,
            "physical_window_order": built["physical_window_order"],
            "frames": built["frames"],
            "frames_sha256": built["frames_sha256"],
        })
    runnable = sum(key["status"] == "runnable" for key in keys)
    if runnable != RUNNABLE_KEYS:
        raise ValueError(f"Expected {RUNNABLE_KEYS} runnable keys, found {runnable}")
    if len({frozen.key_of(key) for key in keys}) != PLANNED_KEYS:
        raise ValueError("Duplicate joint-reversal key")
    return {
        "keys": keys,
        "checks": checks,
        "recordings": extension["recordings"],
        "archived_calls": extension["calls"],
        "archived_replies": frozen.read_jsonl(archive["replies.jsonl"]),
        "archived_attempts": frozen.read_jsonl(archive["attempts.jsonl"]),
        "fingerprints": {
            "protocol": dict(PROTOCOL_SHA256),
            "archive": dict(ARCHIVE_SHA256),
            "extension_frozen_inputs": extension["fingerprints"],
        },
    }


def png_size(path):
    with Path(path).open("rb") as stream:
        head = stream.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file: {path}")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def check_frames(frames_root, keys):
    """Verify the SHA-256 and 448 x 448 size of every image a runnable key loads, in slot order."""
    verified = {}
    for key in keys:
        if key["status"] != "runnable":
            continue
        for relative, expected in zip(key["frames"], key["frames_sha256"]):
            path = frames_root / relative
            measured = frozen.file_sha256(path)
            if measured != expected:
                raise ValueError(f"Frame differs from its frozen fingerprint: {relative}")
            if png_size(path) != IMAGE_SIZE:
                raise ValueError(f"Frame is not {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}: {relative}")
            verified[relative] = measured
    return verified


def report(inputs):
    return {
        "protocol_commit": PROTOCOL_COMMIT,
        "archive_commit": ARCHIVE_COMMIT,
        "fingerprints": inputs["fingerprints"],
        "planned_keys": PLANNED_KEYS,
        "runnable_keys": sum(check["status"] == "runnable" for check in inputs["checks"]),
        "technical_missing_keys": sum(check["status"] == "technical_missing" for check in inputs["checks"]),
        "keys": inputs["checks"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON input check report")
    args = cli.parse_args()
    result = report(load())
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {result['runnable_keys']} runnable and {result['technical_missing_keys']} technical-missing "
          f"keys along both construction paths. Wrote {args.output}.")


if __name__ == "__main__":
    main()
